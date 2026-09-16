"""Finite-value sorting parity, workspace bounds, and adversarial depth fallback."""
import ast
from pathlib import Path
import numpy as np
from numba import njit, types

source=(Path(__file__).resolve().parents[1]/'Filter.py').read_text()
scope=dict(np=np,njit=njit,F64=types.float64,VEC=types.float64[::1],I64=types.int64,VOID=types.void)
nodes=[n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name in ('heap_sort','sort_values','median_select')]
exec(compile(ast.Module(body=nodes,type_ignores=[]),'<sort-workspace-test>','exec'),scope)
rng=np.random.default_rng(956)
cases=0
for n in (0,1,2,15,16,17,257,513,2048,6144,393216):
    for values in (rng.normal(size=n),np.zeros(n),np.arange(n,dtype=float),np.arange(n,dtype=float)[::-1].copy(),
                   rng.integers(-3,4,n).astype(float),np.abs(np.arange(n,dtype=float)-n//2)):
        expected=np.sort(values)
        stack=np.full(194,12345.)
        actual=values.copy();scope['sort_values'](actual,n,stack[1:-1])
        np.testing.assert_array_equal(actual,expected)
        assert stack[0]==stack[-1]==12345.
        heap=values.copy();scope['heap_sort'](heap,0,n-1)
        np.testing.assert_array_equal(heap,expected)
        median=scope['median_select'](values.copy(),n)
        assert median==(float(np.median(values)) if n else 0.)
        cases+=1
print(f'{cases} sorting/heap/median cases passed; workspace canaries intact')
