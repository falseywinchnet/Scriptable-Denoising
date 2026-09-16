"""Bounded surface invariants and native pipeline publication, not quality claims."""
import importlib.util
import json
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
LIB=ROOT/'build/libCleanupNative.dylib'
REPORT={}

def source():
    return (ROOT/'Filter.py').read_text().replace('CLEANUP_POPULATION = "rolloff"','CLEANUP_POPULATION = "receiver"').replace('GUIDE_REGISTRATION = False','GUIDE_REGISTRATION = True').replace('GUIDE_BINS = GUIDE_FFT','GUIDE_BINS = 512',1).replace('REGISTERED_FLOOR = "legacy"','REGISTERED_FLOOR = "variance_surface"').replace('VIEWER_AUTOSTART = True','VIEWER_AUTOSTART = False')

class SurfaceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec=importlib.util.spec_from_file_location('surface_test',ROOT/'Filter.py')
        cls.m=importlib.util.module_from_spec(spec);bind_native_dsp(cls.m)
        exec(compile(source(),str(ROOT/'Filter.py'),'exec'),cls.m.__dict__)

    def run_surface(self,x,checkpoint=None,meta=None,origin=0.):
        x=np.ascontiguousarray(x,dtype=float)
        fields=[np.zeros_like(x) for _ in range(4)]
        if checkpoint is None:checkpoint=np.zeros((3,x.shape[1]))
        if meta is None:meta=np.zeros(2)
        self.m.variance_surface(x,*fields,checkpoint,meta,np.zeros(9),48000.,origin, np.zeros(192))
        return fields,checkpoint,meta

    def test_formula_constant_zero_gain_and_outlier_bound(self):
        fields,_,_=self.run_surface(np.ones((192,128)))
        np.testing.assert_allclose(fields[0],1);np.testing.assert_allclose(fields[1],0,atol=1e-14)
        np.testing.assert_allclose(fields[2],1)
        zeros,_,_=self.run_surface(np.zeros((192,128)))
        for a in zeros:np.testing.assert_array_equal(a,0)
        rng=np.random.default_rng(5);x=np.exp(rng.normal(size=(192,128))*.4)
        a,_,_=self.run_surface(x);b,_,_=self.run_surface(x*7.)
        for k,power in [(0,1),(1,2),(2,1),(3,1)]:np.testing.assert_allclose(b[k],a[k]*7.**power,rtol=2e-11,atol=1e-11)
        x=np.ones((192,128));x[85,60]=1e3
        small,_,_=self.run_surface(x);x[85,60]=1e12
        huge,_,_=self.run_surface(x)
        for a,b in zip(small,huge):np.testing.assert_array_equal(a,b)
        self.assertLessEqual(huge[3][85,60],4.)
        REPORT['outlier']=dict(input_ratio=1e9,max_floor=float(huge[2].max()),clipped_observation=float(huge[3][85,60]))

    def test_penalty_matches_measured_moments(self):
        x=np.tile(np.linspace(.5,1.5,128),(192,1))
        (mu,var,floor,_),_,_=self.run_surface(x)
        expected=mu/(1+var/mu**2)
        np.testing.assert_allclose(floor,expected,rtol=1e-11,atol=1e-12)
        self.assertTrue(np.all(floor<=mu))
        self.assertGreater(float(var.max()),0.)

    def test_mean_variance_and_floor_innovations_are_bounded(self):
        x=np.ones((192,128));x[90:]=1e6
        (mu,var,floor,_),_,_=self.run_surface(x)
        alpha=1-np.exp(-128/(48000*.04));scale=np.maximum(mu[:-1],np.sqrt(var[:-1]))
        self.assertTrue(np.all(np.abs(np.diff(mu,axis=0))<=alpha*scale*(1+1e-10)))
        self.assertTrue(np.all(np.abs(np.diff(var,axis=0))<=4*alpha*scale**2*(1+1e-10)))
        self.assertTrue(np.all(np.abs(np.diff(floor,axis=0))<=alpha*scale*(1+1e-10)))

    def test_overlapping_context_not_retrained(self):
        rng=np.random.default_rng(84);data=np.exp(.4*rng.normal(size=(256,128)))
        first,checkpoint,meta=self.run_surface(data[:192])
        second,_,_=self.run_surface(data[64:],checkpoint,meta,8192.)
        # Forward stencil has 14 rows of lookahead. Compare complete anchors.
        for k in range(3):np.testing.assert_allclose(first[k][64:176],second[k][:112],rtol=1e-11,atol=1e-12)
        REPORT['overlap']='same observations updated once; complete anchor region agrees'

    def test_geometry_rejects_bad_publication(self):
        m=self.m;_geometry(m);saved=m.SURFACE_VIEW.copy()
        try:
            m.SURFACE_VIEW['offset']=m.ENTITY_STORAGE
            with self.assertRaisesRegex(ValueError,'SURFACE_VIEW'):_geometry(m)
        finally:m.SURFACE_VIEW=saved
        saved_view=m.VIEWER_GUIDE.copy()
        try:
            m.VIEWER_GUIDE['bins'] += 1
            with self.assertRaisesRegex(ValueError,'SURFACE_VIEW'):_geometry(m)
        finally:m.VIEWER_GUIDE=saved_view

    def test_native_continuous_fm_and_reload(self):
        with tempfile.TemporaryDirectory() as td:
            script=Path(td)/'Filter.py';script.write_text(source())
            with Engine(LIB,script,channels=1) as e:
                status=e.wait();self.assertTrue(status['success'],status)
                e.lib.cleanup_set_harmonics(e.handle,0)
                rng=np.random.default_rng(71);t=np.arange(8192*6)/48000.
                # Uninterrupted real FM waveform, not a speech/pause fixture.
                x=.1*np.cos(2*np.pi*1800*t+3*np.sin(2*np.pi*170*t))+.005*rng.normal(size=t.size)
                outputs=[]
                for chunk in np.split(x.astype('float32'),6):
                    y,code=e.process(chunk);self.assertEqual(code,0);self.assertTrue(np.isfinite(y).all());outputs.append(y)
                d=diagnostic(status['port']);self.assertTrue(d['valid'])
                for key in ('surface_mean','surface_variance','surface_floor','surface_mask'):
                    self.assertEqual(len(d[key]),d['guide_emit']*d['guide_bins'])
                    self.assertTrue(np.isfinite(d[key]).all());self.assertGreaterEqual(min(d[key]),0.)
                self.assertLessEqual(max(d['surface_mask']),1.)
                self.assertGreater(max(d['surface_floor']),0.)
                e.lib.cleanup_set_bandwidth(e.handle,500.)
                y,code=e.process(x[:8192].astype('float32'));self.assertEqual(code,0)
                REPORT['native']=dict(status=e.status(),floor_max=max(d['surface_floor']),mask_mean=float(np.mean(d['surface_mask'])),output_peak=float(np.max(np.abs(outputs))))
                e.reload();fresh=e.wait();self.assertTrue(fresh['success'],fresh)
                y,code=e.process(np.zeros(8192,dtype='float32'));self.assertEqual(code,0)
                d=diagnostic(fresh['port']);self.assertEqual(max(d['surface_floor']),0.)

if __name__=='__main__':
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(SurfaceTests)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    (ROOT/'build/variance-surface-tests.json').write_text(json.dumps(REPORT,indent=2))
    sys.exit(not result.wasSuccessful())
