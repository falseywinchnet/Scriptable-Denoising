"""ctypes test/experiment client for the public DLL ABI (also works on macOS)."""
import ctypes as C
import json
from pathlib import Path
import time
import numpy as np


class Engine:
    def __init__(self, library, script, channels=1, sample_rate=48000, runtime=None):
        self.lib = C.CDLL(str(Path(library).resolve()))
        self.channels = channels
        ptr = C.POINTER(C.c_float)
        self.lib.cleanup_create.argtypes = [C.c_char_p, C.c_char_p, C.c_double, C.c_uint32]
        self.lib.cleanup_create.restype = C.c_void_p
        self.lib.cleanup_destroy.argtypes = [C.c_void_p]
        self.lib.cleanup_process.argtypes = [C.c_void_p, ptr, ptr, C.c_uint32]
        self.lib.cleanup_process_pair.argtypes = [C.c_void_p, ptr, ptr, ptr, C.c_uint32]
        self.lib.cleanup_status.argtypes = [C.c_void_p, C.c_void_p, C.c_uint32]
        self.lib.cleanup_status.restype = C.c_uint32
        self.lib.cleanup_reload.argtypes = [C.c_void_p]
        for name in ('bypass', 'squelch', 'harmonics'):
            getattr(self.lib, 'cleanup_set_' + name).argtypes = [C.c_void_p, C.c_int]
        self.lib.cleanup_set_bandwidth.argtypes = [C.c_void_p, C.c_double]
        runtime = runtime or Path(__file__).resolve().parents[1] / 'runtime'
        self.handle = self.lib.cleanup_create(str(Path(script).resolve()).encode(), str(Path(runtime).resolve()).encode(), sample_rate, channels)
        if not self.handle:
            raise RuntimeError('cleanup_create failed')

    def status(self):
        # Retry if a concurrent compiler error made the JSON longer.
        capacity = 65536
        for _ in range(3):
            buffer = C.create_string_buffer(capacity)
            needed = self.lib.cleanup_status(self.handle, buffer, capacity)
            if needed <= capacity:
                return json.loads(buffer.value)
            capacity = needed
        raise RuntimeError('Status changed repeatedly while reading')

    def wait(self, timeout=120):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            state = self.status()
            if not state['loading']:
                return state
            time.sleep(.05)
        raise TimeoutError(self.status())

    def reload(self):
        return self.lib.cleanup_reload(self.handle)

    def process(self, samples, analysis=None, inplace=False):
        data = np.ascontiguousarray(samples, dtype=np.float32).reshape((-1, self.channels))
        out = data if inplace else np.empty_like(data)
        ptr = C.POINTER(C.c_float)
        if analysis is None:
            result = self.lib.cleanup_process(self.handle, data.ctypes.data_as(ptr), out.ctypes.data_as(ptr), len(data))
        else:
            early = np.ascontiguousarray(analysis, dtype=np.float32).reshape(data.shape)
            result = self.lib.cleanup_process_pair(self.handle, early.ctypes.data_as(ptr), data.ctypes.data_as(ptr), out.ctypes.data_as(ptr), len(data))
        return out, result

    def close(self):
        if self.handle:
            self.lib.cleanup_destroy(self.handle)
            self.handle = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
