"""Exact paired Cleanup recurrence: independent oracle, edges and native timing."""
import ctypes as C
import json
import os
from pathlib import Path
import sys
import time
import unittest
import numpy as np
from numba import njit, types

ROOT=Path(__file__).resolve().parents[1]
F64=types.float64;I64=types.int64;MAT=F64[:,::1];VEC=F64[::1]


def literal_oracle(mask,nb,time_pad,freq_pad,iterations):
    """Independent NumPy transcription, summing every original window in order.

    Both branches read the previous round's plane, then merge. There is no
    rolling-sum recurrence and no call to the production implementation.
    """
    vertical=np.pad(mask,((time_pad,time_pad),(freq_pad,freq_pad)))
    horizontal=vertical.copy()
    rows,cols=vertical.shape;active=nb+2*freq_pad
    for _ in range(iterations):
        frequency=vertical.copy()
        frequency[:,1:]+=vertical[:,:-1]
        frequency[:,:-1]+=vertical[:,1:]
        frequency/=3.
        temporal=horizontal.copy();temporal[:,:active]=0.
        for k in range(-6,7):
            lo=max(0,-k);hi=min(rows,rows-k)
            if lo<hi:temporal[lo:hi,:active]+=horizontal[lo+k:hi+k,:active]
        temporal[:,:active]/=13.
        vertical=(frequency+temporal)*.5
        horizontal=vertical.copy()
    return vertical[time_pad:time_pad+mask.shape[0],freq_pad:freq_pad+mask.shape[1]].copy(),vertical,horizontal


# Frozen pre-native Filter.py recurrence. Only dimensions/pads/round count were
# parameterized for standalone checks; arithmetic and loop order are unchanged.
@njit(types.void(MAT,MAT,MAT,I64,VEC,I64,I64,I64),nogil=True,cache=False)
def prior_numba(mask,vertical,horizontal,nb,scratch,time_pad,freq_pad,iterations):
    vertical[:]=0.;horizontal[:]=0.
    for t in range(mask.shape[0]):
        for b in range(mask.shape[1]):
            vertical[t+time_pad,b+freq_pad]=mask[t,b]
            horizontal[t+time_pad,b+freq_pad]=mask[t,b]
    for iteration in range(iterations):
        for t in range(vertical.shape[0]):
            for b in range(vertical.shape[1]):
                v=vertical[t,b]
                if b>0:v+=vertical[t,b-1]
                if b<vertical.shape[1]-1:v+=vertical[t,b+1]
                scratch[b]=v/3.
            for b in range(vertical.shape[1]):vertical[t,b]=scratch[b]
        for b in range(nb+2*freq_pad):
            for t in range(vertical.shape[0]):
                v=0.
                for k in range(-6,7):
                    if 0<=t+k<vertical.shape[0]:v+=horizontal[t+k,b]
                scratch[t]=v/13.
            for t in range(vertical.shape[0]):horizontal[t,b]=scratch[t]
        for t in range(vertical.shape[0]):
            for b in range(vertical.shape[1]):
                v=(vertical[t,b]+horizontal[t,b])*.5
                vertical[t,b]=v;horizontal[t,b]=v
    for t in range(mask.shape[0]):
        for b in range(mask.shape[1]):mask[t,b]=vertical[t+time_pad,b+freq_pad]


class NativeSmoothing(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fallback=ROOT/('build/windows/CleanupNative.dll' if os.name=='nt' else 'build/libCleanupNative.dylib')
        cls.library=C.CDLL(os.environ.get('CLEANUP_SMOOTH_LIBRARY',os.environ.get('CLEANUP_TEST_LIBRARY',str(fallback))))
        cls.call=cls.library.cleanup_recursive_smooth
        cls.call.argtypes=[C.c_void_p]*4+[C.c_int64]*6;cls.call.restype=C.c_int32
        cls.report=dict(parity=[],benchmark=[])

    @classmethod
    def tearDownClass(cls):
        (ROOT/'build/native-smoothing-results.json').write_text(json.dumps(cls.report,indent=2)+'\n')

    def arrays(self,shape,pads):
        frames,bins=shape;tp,fp=pads;rows,cols=frames+2*tp,bins+2*fp
        # Sentinel values sit immediately before/after each promised capacity.
        capacities=(frames*bins,rows*cols,rows*cols,max(rows,cols))
        owners=[np.full(n+2,9.87654321e123) for n in capacities]
        views=[a[1:-1] for a in owners]
        return owners,(views[0].reshape(shape),views[1].reshape(rows,cols),views[2].reshape(rows,cols),views[3])

    def invoke(self,arrays,nb,tp,fp,iterations=3):
        mask,vertical,horizontal,scratch=arrays
        return self.call(*(a.ctypes.data for a in arrays),*mask.shape,nb,tp,fp,iterations)

    def test_literal_oracle_and_original_numba(self):
        rng=np.random.default_rng(473)
        geometries=[((192,257),(13,3),(0,1,37,129,257)),
                    ((48,512),(13,3),(0,7,291,512)),
                    ((1,1),(0,0),(0,1)),
                    ((3,7),(0,0),(0,3,7)),
                    ((11,19),(2,4),(0,8,19))]
        worst=0.
        for shape,pads,accepted in geometries:
            cases={'random':rng.random(shape),'zeros':np.zeros(shape),'ones':np.ones(shape)}
            corners=np.zeros(shape);corners[0,0]=1.;corners[-1,-1]=.75;corners[0,-1]=.3;corners[-1,0]=.125
            impulse=np.zeros(shape);impulse[shape[0]//2,shape[1]//2]=1.
            cases.update(corners=corners,impulse=impulse)
            for nb in accepted:
                for name,source in cases.items():
                    rounds=(0,1,3,5) if shape[0]<12 else (3,)
                    for iterations in rounds:
                        with self.subTest(shape=shape,pads=pads,nb=nb,signal=name,iterations=iterations):
                            owners,arrays=self.arrays(shape,pads);arrays[0][:]=source
                            expected=literal_oracle(source,nb,*pads,iterations)
                            self.assertEqual(self.invoke(arrays,nb,*pads,iterations),0)
                            error=max(float(np.max(abs(a-b))) for a,b in zip(arrays[:3],expected))
                            worst=max(worst,error)
                            for actual,desired in zip(arrays[:3],expected):np.testing.assert_allclose(actual,desired,atol=1e-12,rtol=0.)
                            self.assertTrue(np.isfinite(arrays[0]).all());self.assertGreaterEqual(arrays[0].min(),0.)
                            for owner in owners:np.testing.assert_array_equal(owner[[0,-1]],9.87654321e123)
                            old=[a.copy() for a in arrays];old[0][:]=source
                            prior_numba(old[0],old[1],old[2],nb,old[3],*pads,iterations)
                            for a,b in zip(old[:3],expected):np.testing.assert_allclose(a,b,atol=1e-14,rtol=0.)
            self.report['parity'].append(dict(shape=shape,pads=pads,nb=list(accepted)))
        # Directly execute the frozen Python loops once too, independently of LLVM.
        _,arrays=self.arrays((9,13),(2,3));source=rng.random((9,13));arrays[0][:]=source
        prior_numba.py_func(arrays[0],arrays[1],arrays[2],5,arrays[3],2,3,3)
        for a,b in zip(arrays[:3],literal_oracle(source,5,2,3,3)):np.testing.assert_allclose(a,b,atol=1e-14,rtol=0.)
        self.report['maximum_absolute_drift']=worst
        print('native smoothing maximum absolute drift:',worst)

    def test_invalid_geometry_and_input_leave_mask_unchanged(self):
        _,arrays=self.arrays((8,9),(13,3));arrays[0][:]=.5;before=arrays[0].copy()
        for nb,tp,fp,rounds in [(-1,13,3,3),(10,13,3,3),(3,-1,3,3),(3,13,-1,3),(3,13,3,-1)]:
            self.assertEqual(self.invoke(arrays,nb,tp,fp,rounds),-1)
            np.testing.assert_array_equal(arrays[0],before)
        addresses=[a.ctypes.data for a in arrays]
        self.assertEqual(self.call(addresses[0],addresses[1],addresses[1],addresses[3],8,9,3,13,3,3),-1)
        self.assertEqual(self.call(0,*addresses[1:],8,9,3,13,3,3),-1)
        for value in (np.nan,np.inf,-.01):
            arrays[0][:]=before;arrays[0][3,4]=value
            self.assertEqual(self.invoke(arrays,3,13,3),-2)
            self.assertEqual(arrays[0][0,0],.5)

    def test_native_vs_prior_numba_timing(self):
        rng=np.random.default_rng(982)
        for shape,nb in [((192,257),37),((48,512),291),((48,512),512)]:
            _,arrays=self.arrays(shape,(13,3));source=rng.random(shape)
            for method in ('native','prior_numba'):
                samples=[]
                for repeat in range(35):
                    arrays[0][:]=source
                    start=time.perf_counter_ns()
                    if method=='native':self.assertEqual(self.invoke(arrays,nb,13,3),0)
                    else:prior_numba(arrays[0],arrays[1],arrays[2],nb,arrays[3],13,3,3)
                    elapsed=time.perf_counter_ns()-start
                    if repeat>=5:samples.append(elapsed)
                self.report['benchmark'].append(dict(shape=shape,nb=nb,method=method,
                    median_ns=int(np.median(samples)),minimum_ns=min(samples),maximum_ns=max(samples)))
        print('smoothing benchmark:',json.dumps(self.report['benchmark']))

if __name__=='__main__':unittest.main(verbosity=2)
