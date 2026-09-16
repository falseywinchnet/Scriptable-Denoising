"""Literal crossing selection, estimator characterization, and native publication."""
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'runtime'),str(ROOT/'tools')]
from compile_filter import bind_native_dsp, _geometry
from native_client import Engine
from listening_demo import diagnostic
LIB=os.environ.get('CLEANUP_TEST_LIBRARY',str(ROOT/'build/libCleanupNative.dylib'))
REPORT={}

def source():
    return (ROOT/'Filter.py').read_text().replace('CLEANUP_POPULATION = "rolloff"','CLEANUP_POPULATION = "receiver"').replace('GUIDE_REGISTRATION = False','GUIDE_REGISTRATION = True').replace('GUIDE_BINS = GUIDE_FFT','GUIDE_BINS = 512',1).replace('REGISTERED_FLOOR = "legacy"','REGISTERED_FLOOR = "zero_crossing"').replace('VIEWER_AUTOSTART = True','VIEWER_AUTOSTART = False')

def oracle(x):
    nonzero=x[x!=0.]
    ids=np.flatnonzero(np.signbit(nonzero[1:])!=np.signbit(nonzero[:-1]))
    values=np.abs(np.stack((nonzero[ids],nonzero[ids+1]))).ravel()
    return (float(values.mean()) if len(values) else 0.,len(values))

class CrossingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec=importlib.util.spec_from_file_location('crossing_test',ROOT/'Filter.py')
        cls.m=importlib.util.module_from_spec(spec);bind_native_dsp(cls.m)
        exec(compile(source(),str(ROOT/'Filter.py'),'exec'),cls.m.__dict__)

    def test_exact_selection_and_zero_runs(self):
        for x in [[],[0,0,0],[1,2,3],[-1,-2],[1,0,0,-2],[1,0,2],[-1,2,-3,4],[0,-1,0,2,0]]:
            a=np.array(x,dtype=float)
            np.testing.assert_allclose(self.m.crossing_neighbors(a),oracle(a),atol=0)
        self.assertEqual(self.m.crossing_neighbors(np.array([1.,0.,-2.])),(1.5,2.))

    def test_distribution_and_scale_characterization(self):
        rng=np.random.default_rng(934);count=240000;t=np.arange(count)/48000.
        cases={'white_noise':rng.normal(0,.01,count),'tone_200':.2*np.sin(2*np.pi*200*t+.31),
               'tone_2000':.2*np.sin(2*np.pi*2000*t+.31)}
        cases['tone_noise']=cases['tone_200']+cases['white_noise']
        cases['impulsive']=cases['tone_noise'].copy();cases['impulsive'][::701]+=.8
        rows={}
        for name,x in cases.items():
            mean,n=self.m.crossing_neighbors(x)
            np.testing.assert_allclose((mean,n),oracle(x),rtol=1e-12)
            doubled=self.m.crossing_neighbors(2*x)
            self.assertAlmostEqual(doubled[0],2*mean,places=12);self.assertEqual(doubled[1],n)
            rows[name]=dict(mean=mean,crossings=n/2,guide_floor=mean*self.m.ZERO_CROSSING_GUIDE_SCALE)
        self.assertAlmostEqual(rows['white_noise']['mean'],.01*np.sqrt(2/np.pi),delta=.00008)
        self.assertGreater(rows['tone_2000']['mean'],8*rows['tone_200']['mean'])
        REPORT['characterization']=rows

    def test_floor_mapping_and_smoothing(self):
        m=self.m;data=np.array([[.1,.3,.7],[.4,.2,.8]])
        mask=np.zeros_like(data);threshold,fraction=m.crossing_peaks(data,mask,3,.3)
        np.testing.assert_array_equal(mask,[[0,0,1],[1,0,1]])
        self.assertEqual((threshold,fraction),(.3,.5))
        # Independent white-noise cosine-inverse check, before registration.
        rng=np.random.default_rng(8);w=.5-.5*np.cos(2*np.pi*np.arange(m.GUIDE_FFT)/m.GUIDE_FFT)
        noise=rng.normal(size=(200,m.GUIDE_FFT));a=np.abs(np.fft.irfft(noise*w,axis=1))[:,10:500]
        measured=a.mean()*m.GUIDE_MAGNITUDE_SCALE / np.abs(noise).mean()
        self.assertAlmostEqual(measured/m.ZERO_CROSSING_GUIDE_SCALE,1.,delta=.02)
        REPORT['unit_conversion']=dict(analytic=m.ZERO_CROSSING_GUIDE_SCALE,measured=measured)

    def test_storage_validation(self):
        m=self.m;original=m.AUDIO_HISTORY
        try:
            for bad in [None,dict(slot=0,offset=65536),dict(slot=3,offset=64),dict(slot=2,offset=0),dict(slot=3,offset=m.ENTITY_STORAGE-1)]:
                m.AUDIO_HISTORY=bad
                with self.assertRaises(ValueError):_geometry(m)
        finally:m.AUDIO_HISTORY=original

    def test_native_audio_history_and_actual_filter(self):
        rng=np.random.default_rng(518)
        x=rng.normal(0,.01,(32768,1)).astype('float32')
        x[:,0]+=(.15*np.sin(2*np.pi*730*np.arange(len(x))/48000)).astype('float32')
        x[:,0]+=(.05*np.sin(2*np.pi*12000*np.arange(len(x))/48000)).astype('float32')
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'Filter.py';path.write_text(source())
            with Engine(LIB,path,channels=1) as engine:
                status=engine.wait();self.assertTrue(status['success'],status)
                self.assertEqual(status['guide_coverage_hz'],24000.)
                history=np.zeros(24576);records=[];reference_output=[]
                for pos in range(0,len(x),8192):
                    block=x[pos:pos+8192]
                    history=np.r_[history[8192:],block[:,0].astype(float)]
                    y,code=engine.process(block);self.assertEqual(code,0);self.assertTrue(np.isfinite(y).all())
                    reference_output.append(y.copy())
                    d=diagnostic(status['port']);z=d['crossing_stats'];mean,n=oracle(history)
                    np.testing.assert_allclose(z[:3],[mean,n,mean*self.m.ZERO_CROSSING_GUIDE_SCALE],rtol=6e-6)
                    self.assertEqual(z[3],1);self.assertEqual(d['action'],1)
                    self.assertEqual(d['mask_stats'][0],0);self.assertEqual(d['mask_stats'][1],0)
                    self.assertEqual(d['mask_stats'][2],d['mask_stats'][6])
                    records.append(z)
                self.assertEqual(engine.status()['faults'],0)
                self.assertGreater(np.asarray(d['filtered']).reshape(-1,257)[:,128].max(),.1,
                                   'The experimental mask must retain supported content above the receive-band setting')
                comparisons=[]
                for bandwidth in (500.,22000.):
                    engine.lib.cleanup_set_bandwidth(engine.handle,bandwidth)
                    engine.reload();self.assertTrue(engine.wait()['success'])
                    actual=[]
                    for pos in range(0,len(x),8192):
                        y,code=engine.process(x[pos:pos+8192]);self.assertEqual(code,0);actual.append(y.copy())
                    np.testing.assert_array_equal(np.concatenate(actual),np.concatenate(reference_output))
                    comparisons.append(dict(bandwidth=bandwidth,max_audio_difference=0.))
                REPORT['bandwidth_invariance']=comparisons
                engine.reload();self.assertTrue(engine.wait()['success'])
                y,code=engine.process(np.zeros((8192,1),'float32'));self.assertEqual(code,0)
                np.testing.assert_array_equal(diagnostic(status['port'])['crossing_stats'][:3],0)
                REPORT['native']=records

if __name__=='__main__':
    result=unittest.main(exit=False)
    (ROOT/'build/crossing-floor-tests.json').write_text(json.dumps(REPORT,indent=2))
    sys.exit(not result.result.wasSuccessful())
