import ctypes as C
import json
import os
from pathlib import Path
import sys
import time
import unittest
import numpy as np
from .bridge import OverlapEvidence, odft_frequencies, project_decisions, unrolled_frequencies
from .oracle.uncertainty_fusion import double_irfft_low_rows

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'tests'),str(ROOT/'tools')]
from test_engine import NativeTests, fixture


class ODFTNativeTests(NativeTests):
    # Inherit all disposition/reload/error tests, replacing only synthesis geometry.
    def engine(self, source, channels=1):
        return super().engine(source.replace('BINS = FFT_SIZE // 2 + 1',
                                              'TRANSFORM = "odft"\nBINS = FFT_SIZE // 2'), channels)

    def test_odft_to_rfft_reload(self):
        with self.engine(fixture()) as e:
            self.assertEqual(e.status()['transform'],'odft')
            self.assertEqual(e.status()['bins'],256)
            self.path.write_text(fixture())
            e.reload()
            self.assertEqual(e.wait()['transform'],'rfft')
            self.assertEqual(e.status()['bins'],257)


class ResearchTests(unittest.TestCase):
    def test_overlapping_block_total_and_contiguous_evidence(self):
        x=np.zeros(192,dtype=bool)
        x[32:78:2]=True  # 23 scattered hits, no contiguous speech run
        evidence=OverlapEvidence.measure(x,(32,161),(0,192))
        self.assertEqual((evidence.occupied,evidence.longest),(23,1))
        self.assertTrue(evidence.baseline_512())
        x[:]=False; x[40:57]=True  # 17 contiguous hits, below total threshold
        evidence=OverlapEvidence.measure(x,(32,161),(0,192))
        self.assertTrue(evidence.baseline_512())
        x[56]=False
        self.assertFalse(OverlapEvidence.measure(x,(32,161),(0,192)).baseline_512())
        x[:]=False;x[:17]=True
        self.assertTrue(OverlapEvidence.measure(x,(32,161),(0,192)).baseline_512())
        self.assertEqual(OverlapEvidence.measure(x,(32,160),(32,160)).longest,0)
        doubled=OverlapEvidence(46,2,258,384)
        rule=doubled.calibrated(reference_event_intervals=24,target_event_intervals=48)
        self.assertEqual((rule['total_threshold'],rule['run_threshold']),(44,32))
        self.assertTrue(rule['open'])

    def test_coordinate_mapping_and_neutral_uncovered_bins(self):
        sf=unrolled_frequencies(2048,256,48000)
        tf=odft_frequencies(512,48000)
        self.assertEqual(len(tf),256)
        self.assertEqual(tf[0],46.875)
        self.assertEqual(tf[-1],23953.125)
        st=np.array([0.,.1,.2])
        gain=np.repeat((.2+.5*sf/sf[-1])[:,None],3,axis=1)
        result=project_decisions(gain,sf,st,tf,np.array([.05,.15]))
        valid=tf<=sf[-1]
        np.testing.assert_allclose(result[:,valid],np.tile(.2+.5*tf[valid]/sf[-1],(2,1)))
        np.testing.assert_array_equal(result[:,~valid],1.)
        for invalid in [np.full_like(gain,np.nan),gain*10]:
            with self.assertRaises(ValueError):project_decisions(invalid,sf,st,tf,st)

    def test_bfft_exact_double_inverse(self):
        lib=C.CDLL(os.environ.get('CLEANUP_UNROLLED_LIBRARY',str(ROOT/'build/libCleanupUnrolled.dylib')))
        lib.cleanup_unrolled_create.argtypes=[C.c_size_t,C.c_size_t]
        lib.cleanup_unrolled_create.restype=C.c_void_p
        lib.cleanup_unrolled_destroy.argtypes=[C.c_void_p]
        ptr=C.POINTER(C.c_double)
        lib.cleanup_unrolled_frame.argtypes=[C.c_void_p,ptr,ptr]
        lib.cleanup_unrolled_envelope.argtypes=[C.c_void_p,ptr,ptr]
        errors=[]; timings=[]
        for n,rows in [(512,128),(1024,256),(2048,256),(4096,512)]:
            plan=lib.cleanup_unrolled_create(n,rows)
            self.assertTrue(plan)
            try:
                x=np.random.default_rng(71).normal(size=(n,5))
                x[:,0]=0.;x[:,1]=1.;x[:,2]=(-1.)**np.arange(n)
                window=.5-.5*np.cos(2*np.pi*np.arange(n)/n)
                oracle=double_irfft_low_rows(np.fft.rfft(x*window[:,None],axis=0),n_fft=n,crop_rows=rows)
                recovered=np.fft.irfft(np.fft.rfft(x*window[:,None],axis=0),n=n,axis=0)
                envelope=np.hypot(np.fft.irfft(recovered,axis=0),np.fft.irfft(-1j*recovered,axis=0))[:rows]
                output=np.empty(rows)
                for col in range(5):
                    signal=np.ascontiguousarray(x[:,col])
                    code=lib.cleanup_unrolled_frame(plan,signal.ctypes.data_as(ptr),output.ctypes.data_as(ptr))
                    self.assertEqual(code,0)
                    errors.append(float(np.max(np.abs(output-oracle[:,col]))))
                    np.testing.assert_allclose(output,oracle[:,col],atol=2e-12,rtol=2e-10)
                    code=lib.cleanup_unrolled_envelope(plan,signal.ctypes.data_as(ptr),output.ctypes.data_as(ptr))
                    self.assertEqual(code,0)
                    errors.append(float(np.max(np.abs(output-envelope[:,col]))))
                    np.testing.assert_allclose(output,envelope[:,col],atol=2e-12,rtol=2e-10)
                start=time.perf_counter()
                for _ in range(200):lib.cleanup_unrolled_frame(plan,signal.ctypes.data_as(ptr),output.ctypes.data_as(ptr))
                timings.append(dict(fft=n,rows=rows,us_per_frame=(time.perf_counter()-start)*1e6/200))
            finally:lib.cleanup_unrolled_destroy(plan)
        report=dict(max_absolute_error=max(errors),timings=timings)
        (ROOT/'build/unrolled-parity.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report))

if __name__=='__main__':unittest.main(verbosity=2)
