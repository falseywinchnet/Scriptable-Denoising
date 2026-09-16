"""Population membership invariants and unchanged Cleanup decision operator."""
import importlib.util
import json
from pathlib import Path
import sys
import unittest
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'runtime')]
from compile_filter import bind_native_dsp

def module(population):
    text=(ROOT/'Filter.py').read_text().replace('CLEANUP_POPULATION = "rolloff"','CLEANUP_POPULATION = "receiver"').replace('GUIDE_REGISTRATION = False','GUIDE_REGISTRATION = True').replace('GUIDE_BINS = GUIDE_FFT','GUIDE_BINS = 512',1).replace('CLEANUP_POPULATION = "receiver"',f'CLEANUP_POPULATION = "{population}"')
    text=text.replace('CLEANUP_AUDIT_BITS = 0','CLEANUP_AUDIT_BITS = 64')
    spec=importlib.util.spec_from_file_location('population_'+population,ROOT/'Filter.py')
    m=importlib.util.module_from_spec(spec);bind_native_dsp(m);exec(compile(text,str(ROOT/'Filter.py'),'exec'),m.__dict__)
    return m

class PopulationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reference=module('receiver');cls.oracle=module('oracle');cls.inferred=module('rolloff')

    def seed(self,m):
        rng=np.random.default_rng(713)
        state=np.zeros((m.MAX_ENTITIES,m.ENTITY_STORAGE))
        guide=state[2,:m.GUIDE_PLANE].reshape(m.GUIDE_FRAMES,m.GUIDE_BINS)
        guide[:,:257]=rng.uniform(1e-6,1e-5,(m.GUIDE_FRAMES,257))
        guide[:,35]+=.002;guide[14:34,80]+=.001
        ref=state[3,64:64+m.REFERENCE_PLANE].reshape(m.REFERENCE_FRAMES,m.REFERENCE_BINS)
        ref[:]=rng.uniform(1e-5,1e-4,ref.shape);ref[:,5]+=.8
        x=rng.normal(0,.1,(m.FRAMES,m.BINS,2))
        return state,x

    def test_oracle_preserves_interior_and_has_no_audio_cut(self):
        outputs=[];masks=[]
        for m in (self.reference,self.oracle):
            state,x=self.seed(m);y=x.copy()
            self.assertEqual(m.Filter(x,y,state,np.array([48000.,3000.,0.,0.,0.])),m.PROCESS)
            masks.append(state[0,64+2*m.MASK_PLANE:64+3*m.MASK_PLANE].reshape(m.MASK_FRAMES,m.GUIDE_BINS).copy())
            outputs.append(y)
        np.testing.assert_allclose(masks[0][:,:253],masks[1][:,:253],atol=1e-13,rtol=1e-13)
        # A statistical prefix must not automatically erase an out-of-prefix peak.
        m=self.oracle;state,x=self.seed(m)
        state[2,:m.GUIDE_PLANE].reshape(m.GUIDE_FRAMES,m.GUIDE_BINS)[:,500]=.002
        y=x.copy();self.assertEqual(m.Filter(x,y,state,np.array([48000.,3000.,0.,0.,0.])),m.PROCESS)
        self.assertGreater(np.linalg.norm(y[m.FIRST:m.FIRST+m.EMIT,62:64]),0.)

    def test_padding_does_not_change_population_statistics_or_peak_decisions(self):
        m=self.oracle;rng=np.random.default_rng(913);n=137
        data=rng.uniform(0.,3.,(192,512));data[:,30:60]+=4.
        wide=np.zeros((192,2048));wide[:,:512]=data
        values=np.zeros(wide.size);dev=values.copy();raw=np.full(192,.2);det=np.ones(192)
        a=m.statistics(data,n,values,dev);b=m.statistics(wide,n,values,dev)
        np.testing.assert_array_equal(a,b)
        mask=np.zeros_like(data);extended=np.zeros_like(wide)
        m.population_peaks(data,mask,512,raw,det,.9,False,a[0],a[1],values,dev,n)
        m.population_peaks(wide,extended,2048,raw,det,.9,False,b[0],b[1],values,dev,n)
        np.testing.assert_array_equal(mask,extended[:,:512])
        # Second pass aliases its input in both domains; residual values survive.
        m.population_peaks(data,data,512,raw,det,.9,False,a[0],a[1],values,dev,n)
        m.population_peaks(wide,wide,2048,raw,det,.9,False,b[0],b[1],values,dev,n)
        np.testing.assert_array_equal(data,wide[:,:512])

    def test_inferred_ignores_receiver_bandwidth(self):
        m=self.inferred;answers=[];edges=[]
        guide=np.load(ROOT/'build/population/probe.npz')['voiced3000']
        for bandwidth in (1200.,3400.,16000.):
            state,x=self.seed(m);state[2,:m.GUIDE_PLANE]=guide.ravel();y=x.copy()
            self.assertEqual(m.Filter(x,y,state,np.array([48000.,bandwidth,0.,0.,0.])),m.PROCESS)
            self.assertTrue(np.isfinite(y).all());answers.append(y);edges.append(state[0,28])
        np.testing.assert_array_equal(answers[0],answers[1]);np.testing.assert_array_equal(answers[0],answers[2])
        self.assertEqual(len(set(edges)),1)

if __name__=='__main__':unittest.main(verbosity=2)
