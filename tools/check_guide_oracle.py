"""Verify a native target against frozen Python-oracle arrays without SciPy.

tests/test_registered_guide.py creates the fixture from the research oracle.
This runner allows the Windows/Wine target to verify the same exact arrays.
"""
import argparse
import ctypes as C
import json
from pathlib import Path
import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--library', type=Path, required=True)
    p.add_argument('--fixture', type=Path, required=True)
    p.add_argument('--report', type=Path, required=True)
    args = p.parse_args()
    lib = C.CDLL(str(args.library.resolve()))
    lib.cleanup_guide_create.argtypes = [C.c_size_t]*4+[C.c_long]
    lib.cleanup_guide_create.restype = C.c_void_p
    lib.cleanup_guide_run.argtypes = [C.c_void_p, C.c_void_p, C.c_size_t, C.c_void_p]
    lib.cleanup_guide_destroy.argtypes = [C.c_void_p]
    records = []
    with np.load(args.fixture) as data:
        geometry = [int(data[key]) for key in ('aperture', 'rows', 'columns', 'hop')]
        plan = lib.cleanup_guide_create(*geometry, 0)
        if not plan:
            raise RuntimeError('Guide plan construction failed')
        try:
            for key in data.files:
                if not key.startswith('audio_'):
                    continue
                name = key[len('audio_'):]
                audio = np.ascontiguousarray(data[key], dtype=np.float64)
                expected = data['expected_'+name]
                actual = np.empty(expected.shape, dtype=np.float64, order='C')
                code = lib.cleanup_guide_run(plan, audio.ctypes.data, len(audio), actual.ctypes.data)
                if code:
                    raise RuntimeError(f'Guide failed for {name}: {code}')
                np.testing.assert_allclose(actual, expected, atol=2e-10, rtol=2e-7)
                records.append(dict(signal=name, max_absolute_error=float(np.max(abs(actual-expected)))))
        finally:
            lib.cleanup_guide_destroy(plan)
    report = dict(geometry=geometry, cases=records)
    args.report.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
