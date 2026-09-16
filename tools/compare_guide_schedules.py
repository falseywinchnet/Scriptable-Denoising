"""Interleave guide implementations on identical streaming input; require parity."""
import argparse
import ctypes as C
import json
from pathlib import Path
import time
import wave
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--library',action='append',required=True,help='label=library path; first is reference')
p.add_argument('--report',type=Path,required=True)
p.add_argument('--blocks',type=int,default=32)
p.add_argument('--bins',type=int,default=2048)
a=p.parse_args()
ptr=C.POINTER(C.c_double)
entries=[]
try:
    for item in a.library:
        label,path=item.split('=',1)
        lib=C.CDLL(str(Path(path).resolve()))
        lib.cleanup_guide_create.argtypes=[C.c_size_t]*4+[C.c_long];lib.cleanup_guide_create.restype=C.c_void_p
        lib.cleanup_guide_run_stream.argtypes=[C.c_void_p,ptr,C.c_size_t,C.c_size_t,ptr]
        lib.cleanup_guide_destroy.argtypes=[C.c_void_p]
        handle=lib.cleanup_guide_create(2048,a.bins,48,512,0)
        if not handle:raise RuntimeError(label+' failed to construct')
        lib.cleanup_guide_lane_count.argtypes=[C.c_void_p]
        lib.cleanup_guide_lane_count.restype=C.c_size_t
        lanes=lib.cleanup_guide_lane_count(handle)
        entries.append(dict(label=label,lib=lib,handle=handle,lanes=lanes,out=np.empty((48,a.bins)),seconds=[],cpu_seconds=[],max_error=0.,bitwise=True))
        if not 1<=lanes<=4:raise AssertionError(f'{label}: {lanes} lanes exceeds budget')
    with wave.open(str(ROOT/'research/true_superresolution/daveandsimon-first-second.wav')) as wav:
        source=np.frombuffer(wav.readframes(wav.getnframes()),'<i2').astype('float64')/32768.
    audio=np.zeros(24576)
    for k in range(a.blocks):
        audio[:-8192]=audio[8192:];audio[-8192:]=source[np.arange(k*8192,(k+1)*8192)%len(source)]
        order=np.roll(np.arange(len(entries)),k%len(entries))
        if k%2:order=order[::-1]
        for index in order:
            e=entries[index];cpu_start=time.process_time();start=time.perf_counter()
            code=e['lib'].cleanup_guide_run_stream(e['handle'],audio.ctypes.data_as(ptr),len(audio),8192,e['out'].ctypes.data_as(ptr))
            e['seconds'].append(time.perf_counter()-start)
            e['cpu_seconds'].append(time.process_time()-cpu_start)
            if code or not np.isfinite(e['out']).all():raise RuntimeError(e['label']+' failed')
        reference=entries[0]['out']
        for e in entries:
            e['max_error']=max(e['max_error'],float(abs(e['out']-reference).max()))
            e['bitwise'] &= bool(np.array_equal(e['out'],reference))
    results=[]
    for e in entries:
        samples=np.array(e['seconds'][2:])*1000
        results.append(dict(label=e['label'],lanes=e['lanes'],median_ms=float(np.median(samples)),p95_ms=float(np.quantile(samples,.95)),
            median_cpu_ms=float(np.median(e['cpu_seconds'][2:])*1000),
            max_error=e['max_error'],array_equal=e['bitwise'],seconds=e['seconds'],cpu_seconds=e['cpu_seconds']))
    report=dict(blocks=a.blocks,bins=a.bins,warmup_blocks=2,results=results,
        method='Sequential calls interleaved by rotating/reversing library order; each plan sees identical streaming samples.')
    a.report.parent.mkdir(parents=True,exist_ok=True);a.report.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps([{k:v for k,v in r.items() if k not in ('seconds','cpu_seconds')} for r in results],indent=2))
    if not all(e['bitwise'] for e in entries):raise AssertionError('Scheduling changed guide samples')
finally:
    for e in entries:e['lib'].cleanup_guide_destroy(e['handle'])
