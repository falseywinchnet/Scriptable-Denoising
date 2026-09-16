"""Research-only statement timings in the real engine callback.

Run with the same arguments as benchmark_filter.py. Build the clock helper as
build/profile_filter_clock.dylib first. This instruments an in-memory AST after
the normal script audit; neither Filter.py nor the live host is modified.
"""
import ast
import builtins
import ctypes as C
import json
from pathlib import Path
import runpy
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'runtime'))
import compile_filter
import native_client

clock=C.CDLL(str(ROOT/'build/profile_filter_clock.dylib'))
clock.profile_mark.argtypes=[C.c_int];clock.profile_mark.restype=None
clock.profile_reset.argtypes=[];clock.profile_reset.restype=None
for name in ('profile_elapsed','profile_calls'):
    getattr(clock,name).argtypes=[C.c_int];getattr(clock,name).restype=C.c_uint64
labels=[]
original_bind=compile_filter.bind_native_dsp
def bind(module,*args):
    original_bind(module,*args)
    module._profile_mark=clock.profile_mark
compile_filter.bind_native_dsp=bind

def mark(index):return ast.parse(f'_profile_mark({index})').body[0]
class EndReturns(ast.NodeTransformer):
    def visit_Return(self,node):return [mark(-1),node]

def compile_instrumented(tree,filename,mode):
    for function in tree.body:
        if isinstance(function,ast.FunctionDef) and function.name=='cleanup_registered':
            body=[]
            for statement in function.body:
                index=len(labels)
                if index>=256:raise ValueError('Too many profiling sites')
                labels.append(dict(line=statement.lineno,statement=ast.unparse(statement).splitlines()[0]))
                rewritten=EndReturns().visit(statement)
                body.append(mark(index))
                body.extend(rewritten if isinstance(rewritten,list) else [rewritten])
            function.body=body
    return builtins.compile(ast.fix_missing_locations(tree),filename,mode)
compile_filter.compile=compile_instrumented
original_wait=native_client.Engine.wait
def wait(self,*args,**kwargs):
    result=original_wait(self,*args,**kwargs)
    clock.profile_reset()
    return result
native_client.Engine.wait=wait

runpy.run_path(str(ROOT/'tools/benchmark_filter.py'),run_name='__main__')
rows=[dict(**label,total_ms=clock.profile_elapsed(i)/1e6,calls=clock.profile_calls(i)) for i,label in enumerate(labels)]
rows.sort(key=lambda r:r['total_ms'],reverse=True)
report=Path(sys.argv[sys.argv.index('--report')+1]).with_suffix('.statements.json')
report.write_text(json.dumps(rows,indent=2)+'\n')
print(json.dumps(rows[:15],indent=2))
