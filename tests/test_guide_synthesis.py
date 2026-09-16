"""Parent wiring checks, independent of the experimental mask's decisions."""
import ctypes as C
import os
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'tools'), str(ROOT/'runtime')]
from native_client import Engine
from compile_filter import compile_candidate, release
from viewer import snapshot
LIB = os.environ.get('CLEANUP_TEST_LIBRARY', str(ROOT/'build/libCleanupNative.dylib'))


def fixture(mode='rfft', gain=1., fft=512, hop=128):
    header = (ROOT/'Filter.py').read_text().replace('CLEANUP_POPULATION = "rolloff"','CLEANUP_POPULATION = "receiver"').replace('GUIDE_REGISTRATION = False','GUIDE_REGISTRATION = True').replace('GUIDE_BINS = GUIDE_FFT','GUIDE_BINS = 512',1).split('# COMPILED METHODS')[0]
    header = header.replace('ANALYSIS_MODE = "baseline"', 'ANALYSIS_MODE = "registered"')
    header = header.replace('VIEWER = False', 'VIEWER = True').replace('VIEWER_AUTOSTART = True', 'VIEWER_AUTOSTART = False')
    header = header.replace('FFT_SIZE = 512', f'FFT_SIZE = {fft}').replace('HOP = 128', f'HOP = {hop}')
    if mode == 'odft':
        header = header.replace('BINS = FFT_SIZE // 2 + 1', 'TRANSFORM = "odft"\nBINS = FFT_SIZE // 2')
    return header + f'''
class FilterBank:
    @staticmethod
    @njit(FILTER_SIGNATURE, nogil=True, cache=False)
    def run(analysis, content, state, config):
        content[:, :, :] *= {gain}
        return PROCESS
_run_bank = FilterBank.run
@njit(FILTER_SIGNATURE, nogil=True, cache=False)
def Filter(analysis, content, state, config):
    return _run_bank(analysis, content, state, config)
'''


class GuideSynthesis(unittest.TestCase):
    def test_reload_smoke_uses_actual_host_geometry(self):
        # At 96kHz this crop covers 5979Hz; a hardcoded 48k smoke incorrectly
        # rejected the same valid 3400Hz receiver configuration.
        source = fixture().replace('GUIDE_BINS = 512', 'GUIDE_BINS = 256')
        with tempfile.TemporaryDirectory() as td:
            path = Path(td)/'Filter.py'; path.write_text(source)
            with self.assertRaisesRegex(ValueError, 'does not cover receive bandwidth'):
                compile_candidate(path, sample_rate=48000., bandwidth=3400.)
            with Engine(LIB, path, sample_rate=96000) as engine:
                status = engine.wait()
                self.assertTrue(status['success'], status)
                self.assertEqual(status['sample_rate'], 96000)
                self.assertEqual(status['guide_bins'], 256)

    def test_identity_and_constant_gain_with_registered_parent(self):
        # The guide may not perturb synthesis when the script chooses a known
        # gain. Both channels, transform symmetries and awkward host chunks run.
        x = np.random.default_rng(844).normal(0, .02, (32768, 2)).astype('float32')
        with tempfile.TemporaryDirectory() as td:
            path = Path(td)/'Filter.py'
            for mode, gain in (('rfft', 1.), ('odft', .25)):
                with self.subTest(mode=mode, gain=gain):
                    path.write_text(fixture(mode, gain))
                    with Engine(LIB, path, channels=2) as engine:
                        status = engine.wait()
                        self.assertTrue(status['success'], status)
                        delay = status['latency_samples']
                        source = np.concatenate((x, np.zeros((delay+8192, 2), 'float32')))
                        chunks = []; offset = 0; sizes = (137, 8193, 503, 4096)
                        while offset < len(source):
                            end = min(offset+sizes[len(chunks) % len(sizes)], len(source))
                            out, code = engine.process(source[offset:end].copy(), inplace=True)
                            self.assertEqual(code, 0, engine.status())
                            chunks.append(out.copy()); offset = end
                        actual = np.concatenate(chunks)[delay:delay+len(x)]
                        np.testing.assert_allclose(actual, gain*x, atol=2e-7, rtol=2e-6)
                        self.assertEqual(engine.status()['faults'], 0)
                        # Changing synthesis geometry reconstructs framing and
                        # clears old audio, while retaining independent guide.
                        path.write_text(fixture(mode, gain, fft=1024, hop=256))
                        engine.reload(); changed = engine.wait()
                        self.assertTrue(changed['success'], changed)
                        self.assertEqual(changed['fft'], 1024)
                        self.assertEqual(changed['hop'], 256)
                        self.assertEqual(changed['generation'], status['generation']+1)
                        out, code = engine.process(np.zeros((8192, 2), 'float32'))
                        self.assertEqual(code, 0)
                        np.testing.assert_array_equal(out, 0.)

    def test_viewer_has_actual_guide_and_final_preinverse_coefficients(self):
        source = np.random.default_rng(995).normal(0, .02, 24576).astype('float32')
        source += (.13*np.sin(2*np.pi*1031*np.arange(len(source))/48000)).astype('float32')
        with tempfile.TemporaryDirectory() as td:
            path = Path(td)/'Filter.py'; path.write_text(fixture('rfft', .25))
            with Engine(LIB, path) as engine:
                status = engine.wait(); self.assertTrue(status['success'], status)
                for at in range(0, len(source), 8192):
                    _, code = engine.process(source[at:at+8192]); self.assertEqual(code, 0)
                display = snapshot(status['port'])
                lib = engine.lib
                lib.cleanup_guide_create.argtypes = [C.c_size_t]*4+[C.c_long]
                lib.cleanup_guide_create.restype = C.c_void_p
                lib.cleanup_guide_destroy.argtypes = [C.c_void_p]
                lib.cleanup_guide_run.argtypes = [C.c_void_p, C.c_void_p, C.c_size_t, C.c_void_p]
                guide = np.zeros((48, 512)); audio = source.astype('float64')
                plan = lib.cleanup_guide_create(2048, 512, 48, 512, 0)
                self.assertTrue(plan)
                try:
                    self.assertEqual(lib.cleanup_guide_run(plan, audio.ctypes.data, len(audio), guide.ctypes.data), 0)
                finally:
                    lib.cleanup_guide_destroy(plan)
                # FIRST=63 synthesis centers; guide's first displayed center is
                # ceil(63*128/512)=16, with the 128-sample offset in diagnostics.
                np.testing.assert_allclose(np.asarray(display['guide']).reshape(16,512), guide[16:32], rtol=1e-6, atol=1e-10)
                self.assertEqual(display['guide_center_offset'], 128)
                expected = []
                for t in range(63, 127):
                    frame = audio[t*128-256:t*128+256]*np.hanning(512)
                    expected.append(.25*np.abs(np.fft.rfft(frame)))
                np.testing.assert_allclose(np.asarray(display['filtered']).reshape(64,257), expected, rtol=2e-6, atol=2e-7)


if __name__ == '__main__':
    unittest.main(verbosity=2)
