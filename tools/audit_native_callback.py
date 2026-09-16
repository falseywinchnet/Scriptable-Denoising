"""Audit the real Filter C entry: native call graph, buffer aliasing, NRT traffic.

Run with NUMBA_NRT_STATS=1. This compiles the actual Filter without editing it.
Allocation counters cover Numba's runtime, not C++, libc, or managed allocations.
"""
import argparse
import ctypes
import json
import os
from pathlib import Path
import re
import sys

os.environ['NUMBA_NRT_STATS'] = '1'
import numpy as np
from numba import carray, cfunc, types
from numba.core.runtime import rtsys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'runtime'))
import compile_filter as compiler


def call_graph(ir, root):
    """Only traverse native definitions reachable from the exported C wrapper.

    Python-dispatcher wrappers elsewhere in an LLVM module are not audio calls.
    External/indirect callees and exception paths remain explicit in the report.
    """
    definitions = {}
    current = None
    for line in ir.splitlines():
        if line.startswith('define '):
            match = re.search(r'@(?:"([^"]+)"|([^ (]+))\(', line)
            current = (match[1] or match[2]) if match else None
            if current:
                definitions[current] = []
        elif line == '}':
            current = None
        elif current and re.search(r'\b(call|invoke)\b', line):
            definitions[current].append(line.strip())
    pending = [root]
    visited = set()
    external, indirect, sensitive = set(), [], []
    while pending:
        name = pending.pop()
        if name in visited:
            continue
        visited.add(name)
        for line in definitions.get(name, []):
            match = re.search(r'@(?:"([^"]+)"|([^ (]+))\(', line)
            if not match:
                indirect.append(dict(function=name, instruction=line))
                continue
            target = match[1] or match[2]
            if target in definitions:
                pending.append(target)
            else:
                external.add(target)
            if re.search(r'Py[A-Z_]|numba_gil|NRT_.*alloc|NRT_.*free|NRT_.*new_varsize', target):
                sensitive.append(dict(function=name, target=target, instruction=line))
    assert root in definitions, ('C entry absent', root)
    return dict(entry=root, reachable_functions=len(visited), external_calls=sorted(external),
                indirect_calls=indirect, runtime_calls=sensitive)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--require-no-allocations', action='store_true')
    args = parser.parse_args()
    # Distinguish array-view construction from sort's internal partition stack.
    @cfunc(types.uint64(types.CPointer(types.float64), types.int64), nopython=True)
    def view_probe(ptr, count):
        array = carray(ptr, (count,))
        array[0] += 1.
        return array.ctypes.data

    @cfunc(types.void(types.CPointer(types.float64), types.int64), nopython=True)
    def sort_probe(ptr, count):
        carray(ptr, (count,)).sort()

    probe = np.arange(100., 0., -1.)
    probe_pointer = probe.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    probes = {}
    for name, func in (('view', view_probe.ctypes), ('sort', sort_probe.ctypes)):
        before = rtsys.get_allocation_stats()
        result = func(probe_pointer, probe.size)
        after = rtsys.get_allocation_stats()
        probes[name] = {k: getattr(after,k)-getattr(before,k) for k in before._fields}
        if name == 'view':
            assert result == probe.ctypes.data and probe[0] == 101.
    probes['same_pointer_and_write_visible'] = True
    meta = compiler.compile_candidate(ROOT / 'Filter.py')
    module, callback = compiler._generations[meta['token']]
    try:
        ir = callback.inspect_llvm()
        graph = call_graph(ir, callback.native_name)
        arrays = [np.zeros((meta['frames'], meta['bins'], 2)),
                  np.zeros((meta['frames'], meta['bins'], 2)),
                  np.zeros((meta['slots'], meta['storage'])),
                  np.array([48000., 6000., 0., 0., 0.])]
        analysis, content, state, config = arrays
        pointer = ctypes.POINTER(ctypes.c_double)
        pointers = tuple(a.ctypes.data_as(pointer) for a in arrays)
        native = callback.ctypes
        rng = np.random.default_rng(4241)
        records = []
        for kind in ('silence', 'random', 'peaked'):
            for squelch, harmonics in ((0, 0), (1, 0), (0, 1)):
                state.fill(0.)
                analysis[:] = 0. if kind == 'silence' else rng.normal(0., .01, analysis.shape)
                reference = state[3, 64:64+module.REFERENCE_PLANE].reshape((192,257))
                guide = state[2, :module.GUIDE_PLANE].reshape((module.GUIDE_FRAMES, module.GUIDE_BINS))
                if kind != 'silence':
                    reference[:] = np.abs(rng.normal(0., .01, reference.shape))
                    guide[:] = np.abs(rng.normal(0., .0001, guide.shape))
                if kind == 'peaked':
                    reference[:, 7] += 1.
                    guide[:, 56] += .02
                    analysis[:, 7, 0] += 1.
                content[:] = analysis
                config[2:4] = squelch, harmonics
                before = rtsys.get_allocation_stats()
                actions = [native(*pointers) for _ in range(3)]
                after = rtsys.get_allocation_stats()
                delta = {k: getattr(after, k)-getattr(before, k) for k in before._fields}
                if args.require_no_allocations:
                    assert all(v == 0 for v in delta.values()), (kind, squelch, harmonics, delta)
                assert all(x in (0,1,2) for x in actions), actions
                assert np.isfinite(content).all() and np.isfinite(state).all()
                records.append(dict(stimulus=kind, squelch=squelch, harmonics=harmonics,
                                    calls=3, actions=actions, nrt=delta,
                                    state_counter=float(state[0,0]),
                                    content_changed=bool(np.any(content != analysis))))
        methods = {name: dict(signatures=len(value.nopython_signatures),
                             objectmode=any(v.objectmode for v in value.overloads.values()))
                   for name in meta['methods']
                   for value in [getattr(module.FilterBank, name.split('.')[1])
                                 if name.startswith('FilterBank.') else getattr(module,name)]}
        report = dict(sha256=meta['sha256'], methods=methods, graph=graph, records=records, probes=probes,
                      note='Synthetic spectra exercise C callback only; timings and audio quality are not measured.')
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2)+'\n')
        args.report.with_suffix('.llvm').write_text(ir)
        print(json.dumps(dict(report=str(args.report), methods=len(methods),
                             reachable=graph['reachable_functions'], records=records), indent=2))
    finally:
        compiler.release(meta['token'])


if __name__ == '__main__':
    main()
