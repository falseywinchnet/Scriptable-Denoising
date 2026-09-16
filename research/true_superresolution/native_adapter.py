"""Offline adapter for the extracted bfft cosine-field kernel."""
import ctypes as C
from pathlib import Path
import numpy as np

def native_observations(samples, centers, offsets, *, n_fft=2048, rows=256, library=None):
    root=Path(__file__).resolve().parents[2]
    lib=C.CDLL(str(library or root/'build/libCleanupUnrolled.dylib'))
    ptr=C.POINTER(C.c_double)
    lib.cleanup_unrolled_create.argtypes=[C.c_size_t,C.c_size_t]
    lib.cleanup_unrolled_create.restype=C.c_void_p
    lib.cleanup_unrolled_destroy.argtypes=[C.c_void_p]
    lib.cleanup_unrolled_frame.argtypes=[C.c_void_p,ptr,ptr]
    plan=lib.cleanup_unrolled_create(n_fft,rows)
    if not plan:raise RuntimeError('could not construct bfft unrolled plan')
    out=np.empty((len(offsets),rows,len(centers)))
    column=np.empty(rows)
    try:
        for lattice,offset in enumerate(offsets):
            pad=n_fft//2+abs(offset)
            audio=np.pad(np.asarray(samples,dtype=np.float64),(pad,pad))
            for t,center in enumerate(centers):
                begin=pad-n_fft//2+offset+int(center)
                frame=audio[begin:begin+n_fft]
                if len(frame)!=n_fft:raise ValueError('center lies outside available context')
                result=lib.cleanup_unrolled_frame(plan,frame.ctypes.data_as(ptr),column.ctypes.data_as(ptr))
                if result:raise RuntimeError(f'bfft unrolled failed: {result}')
                out[lattice,:,t]=column
    finally:lib.cleanup_unrolled_destroy(plan)
    return out
