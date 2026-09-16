"""Measure registered-guide internals in a CLEANUP_PROFILE_GUIDE=ON build.

Work counters sum elapsed time across lanes; they are not audio wall latency.
"""
import argparse
import ctypes as C
import json
from pathlib import Path
import time
import wave
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--library',type=Path,required=True)
p.add_argument('--report',type=Path,required=True)
p.add_argument('--bins',type=int,default=2048)
p.add_argument('--blocks',type=int,default=32)
args=p.parse_args()
lib=C.CDLL(str(args.library.resolve()));ptr=C.POINTER(C.c_double)
lib.cleanup_guide_create.argtypes=[C.c_size_t]*4+[C.c_long];lib.cleanup_guide_create.restype=C.c_void_p
lib.cleanup_guide_run_stream.argtypes=[C.c_void_p,ptr,C.c_size_t,C.c_size_t,ptr]
lib.cleanup_guide_destroy.argtypes=[C.c_void_p]
for name in ('cleanup_guide_timings','cleanup_guide_work_timings'):
    getattr(lib,name).argtypes=[C.c_void_p,C.POINTER(C.c_uint64)]
with wave.open(str(ROOT/'research/true_superresolution/daveandsimon-first-second.wav')) as wav:
    source=np.frombuffer(wav.readframes(wav.getnframes()),'<i2').astype('float64')/32768.
audio=np.zeros(24576);output=np.empty((48,args.bins))
handle=lib.cleanup_guide_create(2048,args.bins,48,512,0)
if not handle:raise RuntimeError('Cannot create guide')
stage=(C.c_uint64*3)();work=(C.c_uint64*8)();records=[]
lane_data=None
if hasattr(lib,'cleanup_guide_lane_timings'):
    lib.cleanup_guide_lane_count.argtypes=[C.c_void_p];lib.cleanup_guide_lane_count.restype=C.c_size_t
    lanes=lib.cleanup_guide_lane_count(handle)
    lane_data=(C.c_uint64*(3*lanes))()
    lib.cleanup_guide_lane_timings.argtypes=[C.c_void_p,C.POINTER(C.c_uint64)]
labels=('shared_reference','moving_patch_fft','cross_products','ring_products','ring_inverse','peak_search','chart_confidence_blend','final_warp')
try:
    for k in range(args.blocks):
        audio[:-8192]=audio[8192:];audio[-8192:]=source[np.arange(k*8192,(k+1)*8192)%len(source)]
        begin=time.perf_counter()
        code=lib.cleanup_guide_run_stream(handle,audio.ctypes.data_as(ptr),len(audio),8192,output.ctypes.data_as(ptr))
        wall=time.perf_counter()-begin
        if code or not np.isfinite(output).all():raise RuntimeError(f'Guide failed: {code}')
        lib.cleanup_guide_timings(handle,stage);lib.cleanup_guide_work_timings(handle,work)
        records.append(dict(wall_ms=wall*1e3,stage_ms=[v/1e6 for v in stage],work_ms=dict(zip(labels,[v/1e6 for v in work]))))
        if lane_data is not None:
            lib.cleanup_guide_lane_timings(handle,lane_data)
            records[-1]['lane_ms']=np.asarray(lane_data,dtype=float).reshape(lanes,3).tolist()
            records[-1]['lane_ms']=(np.asarray(records[-1]['lane_ms'])/1e6).tolist()
finally:lib.cleanup_guide_destroy(handle)
report=dict(bins=args.bins,blocks=args.blocks,median_wall_ms=float(np.median([r['wall_ms'] for r in records])),
    median_work_ms={name:float(np.median([r['work_ms'][name] for r in records])) for name in labels},
    meaning='Work times sum lanes; use their fractions to locate cost, not as wall latency.',records=records)
if lane_data is not None:
    report['lane_stage_labels']=['views','reference','registration_warp']
    report['median_lane_ms']=np.median([r['lane_ms'] for r in records],axis=0).tolist()
args.report.parent.mkdir(exist_ok=True,parents=True);args.report.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k!='records'},indent=2))
