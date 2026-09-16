import copy
import json
from pathlib import Path
import socket
import sys
import unittest
from unittest.mock import patch
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'tests'),str(ROOT/'runtime')]
from test_engine import NativeTests,fixture
from viewer import History,texture_rgba,launch


class DiagnosticTests(NativeTests):
    def engine(self,source,channels=1):
        source=source.replace('VIEWER = False','VIEWER = True').replace('VIEWER_AUTOSTART = True','VIEWER_AUTOSTART = False')
        return super().engine(source,channels)

    def read(self,e):
        with socket.create_connection(('127.0.0.1',e.status()['port'])) as sock:
            sock.sendall(b'diagnostic\n')
            return json.loads(sock.makefile().readline())

    def test_published_spectra_gate_and_reload(self):
        with self.engine(fixture(action='return SKIP')) as e:
            e.lib.cleanup_set_squelch(e.handle,1)
            self.stream(e,np.ones((40000,1),'float32')*.1)
            first=self.read(e)
            self.assertTrue(first['gate_closed'])
            self.assertEqual(first['action'],2)
            self.assertTrue(first['valid'])
            self.assertGreater(max(first['guide']),0)
            self.assertGreater(max(first['input']),0)
            self.assertEqual(max(first['filtered']),0)
            e.lib.cleanup_set_bypass(e.handle,1)
            self.stream(e,np.ones((8192,1),'float32')*.1)
            bypass=self.read(e)
            self.assertFalse(bypass['gate_closed'])
            self.assertTrue(bypass['bypass'])
            self.assertGreater(max(bypass['filtered']),0)
            np.testing.assert_array_equal(bypass['input'],bypass['filtered'])
            self.assertEqual(e.status()['stage_ns']['guide'],0)
            self.assertEqual(e.status()['stage_ns']['filter'],0)
            if bypass.get('analysis_paused',False):
                self.assertEqual(max(bypass['guide']),0)
            self.path.write_text(fixture().replace('VIEWER = True','VIEWER = False')) # explicitly disable regardless of demo defaults
            e.reload();self.assertTrue(e.wait()['success'])
            self.assertFalse(self.read(e)['enabled'])

    def test_input_snapshot_uses_content_and_same_synthesis_grid(self):
        with self.engine(fixture(gain=.5)) as e:
            content=np.random.default_rng(802).normal(0,.03,(40000,1)).astype('float32')
            # The analysis input is deliberately different. The new diagnostic
            # must report the pre-filter CONTENT on the final synthesis grid.
            _,code=e.process(content,analysis=content*7)
            self.assertEqual(code,0)
            item=self.read(e)
            self.assertEqual(len(item['input']),item['emit']*item['bins'])
            np.testing.assert_allclose(item['filtered'],np.array(item['input'])*.5,rtol=2e-7,atol=1e-8)

    def test_custom_filter_state_publication(self):
        source=fixture(action='state[3, :] = 0.125\n        return PROCESS').replace('VIEWER_GUIDE = None',
            'VIEWER_GUIDE = dict(slot=3, offset=0, bins=16, frames=FRAMES, first=FIRST, emit=EMIT, hop=HOP, frequency_denominator=4094, half_bin=False)')
        # Geometry names are declared below VIEWER_GUIDE in the example header;
        # put the custom declaration after all constants, before compiled code.
        line=next(line for line in source.splitlines() if line.startswith('VIEWER_GUIDE = dict'))
        source=source.replace(line,'VIEWER_GUIDE = None').replace('class FilterBank:',line+'\nclass FilterBank:')
        with self.engine(source) as e:
            self.stream(e,np.ones((30000,1),'float32')*.1)
            item=self.read(e)
            self.assertEqual(item['guide_bins'],16)
            self.assertEqual(item['guide_denominator'],4094)
            np.testing.assert_array_equal(item['guide'],.125)


class HistoryTests(unittest.TestCase):
    def item(self,seq=1,generation=1):
        return dict(enabled=True,generation=generation,sequence=seq,sample_rate=8,hop=1,
                    guide_hop=2,guide_bins=2,bins=2,guide_emit=2,emit=4,
                    guide=[1.,2.]*2,filtered=[3.,4.]*4,gate_closed=True)

    def test_launch_reports_child_failure_before_first_frame(self):
        with patch('viewer.Path.is_file',return_value=True), \
             patch('viewer.subprocess.Popen') as popen:
            popen.return_value.poll.return_value=5
            popen.return_value.returncode=5
            with self.assertRaisesRegex(RuntimeError,'exited during startup'):
                launch(52381,1)

    def test_texture_frequency_direction_and_missing_data(self):
        values=np.array([[1e-6,10.],[np.nan,0.]],dtype=np.float32)
        rgba=texture_rgba(values).reshape(2,2,4)
        self.assertTrue(rgba.flags.c_contiguous)
        self.assertEqual(rgba.dtype,np.float32)
        self.assertGreater(rgba[0,0,0],rgba[1,0,0]) # high frequency bright, top row
        np.testing.assert_allclose(rgba[1,1],(.08,.08,.09,1.))
        np.testing.assert_array_equal(rgba[:,:,3],1.)
        self.assertTrue(np.isfinite(texture_rgba(np.array([[np.inf,-np.inf]],dtype=np.float32))).all())

    def test_duplicates_gaps_and_generation_reset(self):
        h=History(2.)
        self.assertTrue(h.accept(self.item()))
        self.assertFalse(h.accept(self.item()))
        self.assertTrue(h.accept(self.item(3)))
        np.testing.assert_array_equal(h.gate[-4:],1.)
        np.testing.assert_array_equal(h.gate[-8:-4],-1.)
        self.assertTrue(np.isnan(h.filtered[-8:-4]).all())
        changed=self.item(1,2);changed['bins']=3;changed['filtered']=[1.,2.,3.]*4
        self.assertTrue(h.accept(changed))
        self.assertEqual(h.filtered.shape,(16,3))
        self.assertTrue(np.isnan(h.filtered[:-4]).all())


if __name__=='__main__':unittest.main(verbosity=2)
