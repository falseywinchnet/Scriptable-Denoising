"""Exact cache traversal parity, including aliasing and minimum workspace."""
import ast
from pathlib import Path
import time
import unittest
import numpy as np
from numba import njit, types

ROOT=Path(__file__).resolve().parents[1]
# Compile just this typed method, avoiding unrelated Filter compilation cost.
tree=ast.parse((ROOT/'Filter.py').read_text())
nodes=[n for n in tree.body if (isinstance(n,ast.FunctionDef) and n.name=='sawtooth') or
       (isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='TRIANGLE' for t in n.targets))]
scope=dict(np=np,njit=njit,VOID=types.void,MAT=types.float64[:,::1],I64=types.int64,VEC=types.float64[::1])
exec(compile(ast.Module(body=nodes,type_ignores=[]),'smoothing-extract','exec'),scope)
sawtooth=scope['sawtooth'];triangle=scope['TRIANGLE']

@njit
def original(data,out,nb,scratch):
    for b in range(nb):
        for t in range(data.shape[0]):
            v=0.
            for k in range(15):
                j=t+k-7
                if 0<=j<data.shape[0]:v+=data[j,b]*triangle[k]
            scratch[t]=v/6.916666666666666666
        for t in range(data.shape[0]):out[t,b]=scratch[t]

class SmoothingTests(unittest.TestCase):
    def test_exact_output_and_aliasing(self):
        rng=np.random.default_rng(914)
        for rows,bins,nb in ((1,8,8),(7,37,31),(192,257,129),(192,2048,2048),(192,67,65)):
            data=rng.normal(size=(rows,bins));data[::5]=0.
            for workspace in (rows,rows*17,rows*32):
                for alias in (False,True):
                    a=data.copy();b=data.copy()
                    expected=a if alias else np.full_like(a,-77.)
                    actual=b if alias else np.full_like(b,-77.)
                    original(a,expected,nb,np.empty(workspace))
                    sawtooth(b,actual,nb,np.empty(workspace))
                    np.testing.assert_array_equal(actual,expected)

    def test_report_timing(self):
        data=np.random.default_rng(813).uniform(size=(192,2048));out=np.empty_like(data);scratch=np.empty(192*32)
        records={}
        for name,method in (('original',original),('tiled',sawtooth)):
            method(data,out,2048,scratch)
            timings=[]
            for _ in range(32):
                start=time.perf_counter();method(data,out,2048,scratch);timings.append(time.perf_counter()-start)
            records[name]=float(np.median(timings))*1000
        print('Time smoother median ms:',records)

if __name__=='__main__':unittest.main(verbosity=2)
