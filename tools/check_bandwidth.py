"""Exercise live bandwidth changes on the full FFT guide; compare exact medians.

The control changes only median selection back to the former full-sort method.
Both engines see identical samples and receiver settings in alternating order.
"""
import argparse
import ast
import json
from pathlib import Path
import tempfile
import time
import numpy as np
from native_client import Engine

ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--library',type=Path,required=True)
p.add_argument('--report',type=Path,required=True)
p.add_argument('--population',action='store_true')
p.add_argument('--sort-workspace',action='store_true',help='Compare caller-workspace sorts against the previous Numba sorts')
a=p.parse_args()
source=(ROOT/'Filter.py').read_text().replace('VIEWER = True','VIEWER = False')
if a.population:source=source.replace('CLEANUP_POPULATION = "receiver"','CLEANUP_POPULATION = "rolloff"')
reference=source.replace('med = median_select(values, n)','values[:n].sort()\n    med = median_sorted(values, n)').replace('man = median_select(deviations, n)','deviations[:n].sort()\n    man = median_sorted(deviations, n)')
if a.sort_workspace:
    reference=source
    tree=ast.parse(source);lines=source.splitlines(keepends=True);offsets=[0]
    for line in lines:offsets.append(offsets[-1]+len(line))
    changes=[]
    for node in ast.walk(tree):
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id=='sort_values':
            changes.append((offsets[node.lineno-1]+node.col_offset,offsets[node.end_lineno-1]+node.end_col_offset,
                            ast.get_source_segment(source,node.args[0])+'.sort()'))
    for start,end,value in sorted(changes,reverse=True):reference=reference[:start]+value+reference[end:]
assert source!=reference
# Exercise the selector independently on repeated/order-adversarial finite data.
from numba import njit,types
scope=dict(np=np,njit=njit,F64=types.float64,VEC=types.float64[::1],I64=types.int64,VOID=types.void)
tree=ast.parse(source)
exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('heap_sort','median_select')],type_ignores=[]),'<median-test>','exec'),scope)
rng=np.random.default_rng(2881);cases=0
for n in (0,1,2,3,15,16,513,2048,393216):
    for values in (rng.normal(size=n),np.zeros(n),np.arange(n,dtype=float),np.arange(n,dtype=float)[::-1].copy(),rng.integers(-3,4,n).astype(float),np.abs(np.arange(n,dtype=float)-n//2)):
        expected=float(np.median(values)) if n else 0.
        actual=scope['median_select'](values.copy(),n)
        assert actual==expected,(n,actual,expected)
        cases+=1
records=[];maximum_error=0.
with tempfile.TemporaryDirectory(prefix='cleanup-bandwidth-') as tmp:
    paths=[Path(tmp)/'sorted.py',Path(tmp)/'selected.py']
    for path,text in zip(paths,(reference,source)):path.write_text(text)
    with Engine(a.library,paths[0],channels=2) as old,Engine(a.library,paths[1],channels=2) as new:
        states=[e.wait() for e in (old,new)]
        for s in states:
            assert s['success'] and s['guide_bins']==s['guide_fft'] and not s['guide_registration'],s
        generation=[s['generation'] for s in states]
        block=states[0]['block'];rate=states[0]['sample_rate'];sample=0
        for bandwidth in (3400.,6000.,8000.,16000.,24000.,48000.,6000.):
            for engine in (old,new):engine.lib.cleanup_set_bandwidth(engine.handle,bandwidth)
            elapsed=[[],[]]
            for step in range(6):
                if a.sort_workspace:
                    for engine in (old,new):
                        engine.lib.cleanup_set_squelch(engine.handle,int(step in (2,3)))
                        engine.lib.cleanup_set_harmonics(engine.handle,int(step in (4,5)))
                t=(sample+np.arange(block))/rate;sample+=block
                mono=(.03*rng.normal(size=block)+.07*np.sin(2*np.pi*713*t)+.015*np.sin(2*np.pi*min(bandwidth*.7,19000.)*t)).astype('float32')
                data=np.column_stack((mono,mono));outputs=[None,None]
                for i in ((0,1) if step%2==0 else (1,0)):
                    begin=time.perf_counter();y,code=(old,new)[i].process(data);elapsed[i].append((time.perf_counter()-begin)*1000)
                    assert code==0 and np.isfinite(y).all(),(old,new)[i].status()
                    outputs[i]=y
                maximum_error=max(maximum_error,float(abs(outputs[0]-outputs[1]).max()))
                np.testing.assert_array_equal(outputs[0],outputs[1])
            statuses=[e.status() for e in (old,new)]
            assert [s['generation'] for s in statuses]==generation
            assert all(s['faults']==0 and not s['runtime_fault'] for s in statuses)
            records.append(dict(requested_hz=bandwidth,effective_hz=statuses[1]['bandwidth'],
                sorted_median_ms=float(np.median(elapsed[0])),selected_median_ms=float(np.median(elapsed[1])),
                sorted_ms=elapsed[0],selected_ms=elapsed[1]))
report=dict(population=a.population,sort_workspace=a.sort_workspace,median_cases=cases,array_equal=True,max_error=maximum_error,registration=False,guide_bins=states[1]['guide_bins'],records=records)
a.report.parent.mkdir(parents=True,exist_ok=True);a.report.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
