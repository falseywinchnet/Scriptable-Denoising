"""Regress separate pre-AGC evidence and post-AGC mask-source routing."""
import ctypes as C
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'tools'), str(ROOT/'runtime')]
from native_client import Engine
from viewer import snapshot
from compile_filter import compile_candidate, release
LIB = os.environ.get('CLEANUP_TEST_LIBRARY', str(ROOT/'build/libCleanupNative.dylib'))


def script(probe=False):
    source = (ROOT/'Filter.py').read_text().replace('VIEWER_AUTOSTART = True', 'VIEWER_AUTOSTART = False')
    if not probe:
        return source
    return source.split('# COMPILED METHODS')[0] + '''
class FilterBank:
    @staticmethod
    @njit(FILTER_SIGNATURE, nogil=True, cache=False)
    def run(analysis, content, state, config):
        # Encode the entire current guide into a measurable, bounded gain.
        guide = state[GUIDE_SLOT, GUIDE_OFFSET:GUIDE_OFFSET+GUIDE_PLANE]
        gain = 1. / (1. + np.max(guide)*1000.)
        # Diagnostic evidence slots deliberately expose input provenance.
        state[0,8] = np.max(state[3,64:64+REFERENCE_PLANE])
        state[0,9] = np.max(np.abs(state[3,AUDIO_HISTORY_OFFSET:AUDIO_HISTORY_OFFSET+24576]))
        content[:,:,:] *= gain
        return PROCESS
_run_bank = FilterBank.run
@njit(FILTER_SIGNATURE, nogil=True, cache=False)
def Filter(analysis, content, state, config):
    return _run_bank(analysis, content, state, config)
'''


class MaskSource(unittest.TestCase):
    def test_parent_separates_reference_guide_and_channel_reuse(self):
        rng = np.random.default_rng(1511)
        x = rng.normal(0,.08,24576).astype('float32')
        early = rng.normal(0,.00002,24576).astype('float32')
        with tempfile.TemporaryDirectory() as td:
            path = Path(td)/'Filter.py'; path.write_text(script(True))
            with Engine(LIB,path,channels=2) as engine:
                status=engine.wait(); self.assertTrue(status['success'],status)
                self.assertEqual(status['mask_source'],'content')
                for at in range(0,len(x),8192):
                    content=np.column_stack((x[at:at+8192],x[at:at+8192]*3))
                    analysis=np.column_stack((early[at:at+8192],early[at:at+8192]))
                    _,code=engine.process(content,analysis);self.assertEqual(code,0)
                self.assertEqual(engine.status()['last_calls']['guide'],2)
                d=snapshot(status['port'])
                # Independent literal central double inverse, all 48 centers.
                window=.5-.5*np.cos(2*np.pi*np.arange(2048)/2048)
                expected=[]
                audio=x.astype('float64')
                padded=np.pad(audio,(1024,1024))
                for center in range(0,24576,512):
                    frame=padded[center:center+2048]*window
                    expected.append(abs(np.fft.irfft(np.fft.irfft(np.fft.rfft(frame)))))
                expected=np.array(expected)[:,:2048]
                np.testing.assert_allclose(np.asarray(d['guide']).reshape(16,2048),expected[16:32],rtol=2e-6,atol=2e-10)
                gain=1/(1+expected.max()*1000)
                np.testing.assert_allclose(d['filtered'],np.asarray(d['input'])*gain,rtol=3e-6,atol=2e-8)
                self.assertAlmostEqual(d['evidence']['reference_available'],max(abs(audio)),places=7)
                # Reference FFT must still see the tiny early stream, not x.
                reference=[];padded=np.pad(early.astype('float64'),(256,256))
                for center in range(0,24576,128):
                    reference.append(abs(np.fft.rfft(padded[center:center+512]*np.hanning(512))))
                self.assertAlmostEqual(d['evidence']['reference_count'],np.max(reference),places=8)

    def test_real_mask_independent_of_preagc_level_but_squelch_uses_it(self):
        rate=48000; n=8192*8;t=np.arange(n)/rate
        envelope=.5+.5*np.sin(2*np.pi*3*t)**2
        x=(envelope*(.12*np.sin(2*np.pi*711*t)+.04*np.sin(2*np.pi*1422*t))+
           np.random.default_rng(715).normal(0,.006,n)).astype('float32')
        records=[]
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'Filter.py';path.write_text(script())
            with Engine(LIB,path,channels=2) as engine:
                status=engine.wait();self.assertTrue(status['success'],status)
                outputs=[]
                for at in range(0,n,8192):
                    block=x[at:at+8192]
                    out,code=engine.process(np.column_stack((block,block)),np.column_stack((block*.001,block)))
                    self.assertEqual(code,0);outputs.append(out)
                output=np.concatenate(outputs)
                np.testing.assert_allclose(output[:,0],output[:,1],rtol=2e-5,atol=1e-7)
                self.assertEqual(engine.status()['last_calls']['guide'],1)
                self.assertEqual(engine.status()['last_calls']['filter'],2)
                records.append(dict(case='1000x_preagc_difference',max_channel_error=float(np.max(abs(output[:,0]-output[:,1])))))
                engine.lib.cleanup_set_squelch(engine.handle,1)
                # Sustained content must not replace absent early gate evidence.
                for _ in range(8):
                    block=np.column_stack((x[:8192],x[:8192]))
                    out,code=engine.process(block,np.zeros_like(block));self.assertEqual(code,0)
                np.testing.assert_array_equal(out,0.)
                records.append(dict(case='silent_early_with_loud_content',output_peak=float(abs(out).max())))
        (ROOT/'build/mask-source-results.json').write_text(json.dumps(records,indent=2))
        print(records)

    def test_invalid_source_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'Filter.py';path.write_text(script(True).replace('MASK_SOURCE = "content"','MASK_SOURCE = "typo"'))
            with self.assertRaisesRegex(ValueError,'MASK_SOURCE'):
                compile_candidate(path)


if __name__=='__main__':unittest.main(verbosity=2)
