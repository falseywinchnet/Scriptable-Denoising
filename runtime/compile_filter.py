"""Control-thread-only compiler for the single-file Cleanup experiment.

No import/reload or Python dispatcher calls happen on the native audio thread.
Generations retain their module and cfunc until the native owner releases them.
"""
import ast
import ctypes
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback
import types as pytypes
import uuid

import numpy as np
from numba import carray, cfunc, types
from numba.core.registry import CPUDispatcher

_generations = {}
_ALLOWED_IMPORTS = {'numpy', 'numba', 'numba.experimental', 'math'}


def _name(node):
    if isinstance(node, ast.Call):
        return _name(node.func)
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return ''


def _audit_tree(tree):
    parents = {child: parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}
    for node in ast.walk(tree):
        if isinstance(node, (ast.AsyncFunctionDef, ast.Lambda)):
            raise ValueError('Only decorated, statically named Numba functions are supported')
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name not in _ALLOWED_IMPORTS:
                    raise ValueError(f'Unsupported import: {alias.name}; keep DSP in Filter.py')
        if isinstance(node, ast.ImportFrom) and node.module not in _ALLOWED_IMPORTS:
            raise ValueError(f'Unsupported import: {node.module}; keep DSP in Filter.py')
        if isinstance(node, ast.With):
            if any(_name(item.context_expr) == 'objmode' for item in node.items):
                raise ValueError('objmode blocks are forbidden on the audio path')
        if isinstance(node, ast.FunctionDef):
            if not isinstance(parents[node], (ast.Module, ast.ClassDef)):
                raise ValueError(f'{node.name}: nested or conditional method declarations are unsupported')
            names = {_name(d) for d in node.decorator_list}
            if not names.intersection({'njit', 'jit'}):
                raise ValueError(f'{node.name}:{node.lineno}: every method must be Numba decorated')
            jit_decorators = [d for d in node.decorator_list if _name(d) in {'njit', 'jit'}]
            if any(not isinstance(d, ast.Call) or not d.args for d in jit_decorators):
                raise ValueError(f'{node.name}:{node.lineno}: an explicit Numba signature is required')
        if isinstance(node, ast.ClassDef):
            if not isinstance(parents[node], ast.Module):
                raise ValueError('Classes must be declared at module scope')
            for child in node.body:
                if isinstance(child, ast.FunctionDef) and not any(_name(d) == 'staticmethod' for d in child.decorator_list):
                    raise ValueError(f'{node.name}.{child.name}: use compiled static methods with native fixed storage')
    return tree


def _integer(module, key, minimum, maximum):
    value = getattr(module, key)
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f'{key} must be an integer in [{minimum}, {maximum}]')
    return value


def _geometry(m):
    share_identical = getattr(m, 'SHARE_IDENTICAL_CHANNELS', False)
    if type(share_identical) is not bool:
        raise ValueError('SHARE_IDENTICAL_CHANNELS must be a boolean deterministic-channel contract')
    if m.ABI_VERSION not in (1, 2):
        raise ValueError('Unsupported ABI_VERSION')
    fft = _integer(m, 'FFT_SIZE', 16, 8192)
    hop = _integer(m, 'HOP', 1, fft)
    block = _integer(m, 'BLOCK_SIZE', fft, 65536)
    contexts = _integer(m, 'CONTEXT_BLOCKS', 3, 9)
    if fft & (fft - 1) or fft % hop or block % hop:
        raise ValueError('FFT_SIZE must be a power of two; HOP must divide FFT_SIZE and BLOCK_SIZE')
    transform = getattr(m, 'TRANSFORM', 'rfft')
    if transform not in ('rfft', 'odft'):
        raise ValueError('TRANSFORM must be "rfft" or "odft" (invertible bfft plans)')
    frames, bins, emit = contexts * block // hop, fft // 2 + (transform == 'rfft'), block // hop
    first = _integer(m, 'FIRST', 0, frames - emit)
    if first * hop < fft // 2 or (first + emit - 1) * hop + fft // 2 > contexts * block:
        raise ValueError('Emitted frames must have complete context (no reflected boundary audio)')
    if (m.FRAMES, m.BINS, m.EMIT) != (frames, bins, emit):
        raise ValueError('FRAMES/BINS/EMIT do not match the declared STFT geometry')
    if m.WINDOW != 'hann':
        raise ValueError('Stage one supports WINDOW="hann" from bfft with its canonical dual')
    slots = _integer(m, 'MAX_ENTITIES', 1, 16)
    storage = _integer(m, 'ENTITY_STORAGE', 1, 32000000)
    if not isinstance(m.CHAIN, tuple) or not 1 <= len(m.CHAIN) <= slots:
        raise ValueError('CHAIN must be a nonempty tuple with at most MAX_ENTITIES entries')
    if slots * storage * 8 + frames * bins * 32 > 512 * 1024 * 1024:
        raise ValueError('Generation exceeds the 512 MiB per-channel memory budget')
    if (m.PASSTHROUGH, m.PROCESS, m.SKIP, m.ERROR) != (0, 1, 2, -1):
        raise ValueError('Disposition constants must be PASSTHROUGH=0 PROCESS=1 SKIP=2 ERROR=-1')
    viewer = getattr(m, 'VIEWER', False)
    autostart = getattr(m, 'VIEWER_AUTOSTART', True)
    if type(viewer) is not bool or type(autostart) is not bool:
        raise ValueError('VIEWER and VIEWER_AUTOSTART must be bool')
    guide = getattr(m, 'VIEWER_GUIDE', None)
    view = dict(slot=0, offset=0, bins=bins, frames=frames, first=first, emit=emit,
                hop=hop, frequency_denominator=fft, half_bin=transform == 'odft')
    if guide is not None:
        if not isinstance(guide, dict) or set(guide) != set(view):
            raise ValueError('VIEWER_GUIDE requires slot, offset, bins, frames, first, emit, hop, frequency_denominator, half_bin')
        view = dict(guide)
        if any(type(view[k]) is not int or view[k] < 0 for k in view if k != 'half_bin') or type(view['half_bin']) is not bool:
            raise ValueError('Invalid VIEWER_GUIDE field types')
        if (view['slot'] >= slots or min(view['bins'], view['frames'], view['emit'], view['hop'], view['frequency_denominator']) < 1
                or view['first'] + view['emit'] > view['frames']
                or view['emit'] * view['hop'] != block
                or view['offset'] + view['bins'] * view['frames'] > storage):
            raise ValueError('VIEWER_GUIDE must fit fixed storage and emit one parent block')
    if viewer and emit*bins + view['emit']*view['bins'] > 2000000:
        raise ValueError('Viewer snapshot exceeds two million magnitude values')
    analysis_mode = getattr(m, 'ANALYSIS_MODE', 'baseline')
    if analysis_mode not in ('baseline', 'registered'):
        raise ValueError('ANALYSIS_MODE must be baseline or registered')
    native_guide = dict(guide=int(analysis_mode == 'registered'), guide_fft=0, guide_bins=0,
                        guide_frames=0, guide_hop=0, guide_slot=0, guide_offset=0)
    registration = getattr(m, 'GUIDE_REGISTRATION', True)
    if type(registration) is not bool:
        raise ValueError('GUIDE_REGISTRATION must be a bool')
    native_guide['guide_registration'] = int(registration)
    mask_source = getattr(m, 'MASK_SOURCE', 'analysis')
    if mask_source not in ('analysis', 'content'):
        raise ValueError('MASK_SOURCE must be analysis or content')
    native_guide['mask_source_content'] = int(mask_source == 'content')
    if analysis_mode == 'registered':
        if m.CHAIN not in ((0,), (0, 1)):
            raise ValueError('Registered mode requires Cleanup in slot 0 and optional Harmonics in slot 1')
        if block != 8192 or contexts != 3 or slots < 4 or len(m.CHAIN) > 2:
            raise ValueError('Registered mode requires 8192-sample blocks, three contexts, and reserved slots 2/3')
        if storage < 64+192*257+3*192+7*257:
            raise ValueError('Fixed reference evidence scratch exceeds storage')
        for key, low, high in [('FFT',4,8192), ('BINS',12,16382), ('FRAMES',12,2048),
                               ('HOP',1,8192), ('SLOT',0,slots-1), ('OFFSET',0,storage-1)]:
            native_guide['guide_'+key.lower()] = _integer(m,'GUIDE_'+key,low,high)
        gn,gb,gf,gh,gs,go = (native_guide['guide_'+k] for k in ('fft','bins','frames','hop','slot','offset'))
        if gb > 2*(gn-1) or go+gb*gf > storage or gb*gf > 2000000 or slots*storage*8+frames*bins*32+gb*gf*1000 > 512*1024*1024:
            raise ValueError('Registered guide exceeds fixed storage or geometry limit')
        if gs != 2 or gs < len(m.CHAIN):
            raise ValueError('Native guide input must own reserved state slot 2; slot 3 holds fixed reference evidence')
        if (gf-1)*gh < (first+emit-1)*hop:
            raise ValueError('Registered guide must cover every emitted synthesis frame center')
    audio = getattr(m, 'AUDIO_HISTORY', None)
    audio_metadata = dict(audio_history=0, audio_slot=0, audio_offset=0)
    if audio is not None:
        if (not isinstance(audio, dict) or set(audio) != {'slot','offset'} or
                any(type(v) is not int or v < 0 for v in audio.values()) or
                audio['slot'] >= slots or audio['slot'] < len(m.CHAIN) or
                audio['offset'] + contexts*block > storage):
            raise ValueError('AUDIO_HISTORY must fit an unoccupied fixed storage row')
        if analysis_mode == 'registered':
            if audio['slot'] == gs and audio['offset'] < go+gb*gf and audio['offset']+contexts*block > go:
                raise ValueError('AUDIO_HISTORY overlaps native guide')
            if audio['slot'] == 3 and audio['offset'] < 64+192*257+3*192+7*257:
                raise ValueError('AUDIO_HISTORY overlaps reference evidence scratch')
        audio_metadata = dict(audio_history=1, audio_slot=audio['slot'], audio_offset=audio['offset'])
    floor = getattr(m, 'REGISTERED_FLOOR', 'legacy')
    if floor not in ('legacy','zero_crossing','variance_surface','occupancy_surface'):
        raise ValueError('REGISTERED_FLOOR must be legacy, zero_crossing, variance_surface or occupancy_surface')
    if analysis_mode == 'registered' and floor == 'zero_crossing' and audio is None:
        raise ValueError('The zero-crossing floor requires AUDIO_HISTORY')
    population = getattr(m, 'CLEANUP_POPULATION', 'receiver')
    if population not in ('receiver', 'oracle', 'rolloff'):
        raise ValueError('CLEANUP_POPULATION must be receiver, oracle or rolloff')
    if population != 'receiver' and (analysis_mode != 'registered' or floor != 'legacy'):
        raise ValueError('Population selection requires the intact registered legacy Cleanup path')
    harmonic_model=getattr(m,'HARMONIC_MODEL','packet')
    if harmonic_model not in ('packet','excitation') or (harmonic_model=='excitation' and analysis_mode!='registered'):
        raise ValueError('Excitation harmonics requires registered guide analysis')
    surface = getattr(m, 'SURFACE_VIEW', None)
    surface_metadata = dict(surface=0)
    if surface is not None:
        if analysis_mode != 'registered' or floor not in ('variance_surface','occupancy_surface'):
            raise ValueError('SURFACE_VIEW requires registered variance_surface')
        for key in ('offset','mask_offset','frames','bins','hop'):
            if type(surface.get(key)) is not int or surface[key] < (0 if key in ('offset','mask_offset') else 1):
                raise ValueError('Invalid SURFACE_VIEW '+key)
        count=surface['frames']*surface['bins']
        if (surface['offset']+4*count+3*surface['bins']+2 > storage or
                surface['mask_offset']+count > surface['offset'] or
                surface['bins'] != gb or view['bins'] != gb or view['half_bin'] or
                view['frequency_denominator'] != 2*(gn-1) or
                surface['frames']*surface['hop'] != contexts*block or
                view['hop'] % surface['hop'] or
                (view['first']+view['emit'])*view['hop'] > surface['frames']*surface['hop']):
            raise ValueError('SURFACE_VIEW exceeds storage or disagrees with guide lattice')
        extra={'low_offset','occupancy_offset','reference_offset'} & set(surface)
        if extra and len(extra)!=3:raise ValueError('SURFACE_VIEW extra fields must be declared together')
        for key in ('low_offset','occupancy_offset','reference_offset','shape_offset'):
            if key in surface and (type(surface[key]) is not int or surface[key]<0 or surface[key]+count>storage):
                raise ValueError('SURFACE_VIEW optional field exceeds storage')
        surface_metadata=dict(surface=1, **{'surface_'+key:value for key,value in surface.items()})
    view_metadata = {'view_'+key: int(value) for key, value in view.items()}
    return dict(share_identical=int(share_identical), config_size=5 if m.ABI_VERSION == 2 else 4, fft=fft, hop=hop, block=block, contexts=contexts, frames=frames,
                bins=bins, emit=emit, first=first, slots=slots, storage=storage,
                transform=0 if transform == 'rfft' else 1,
                viewer=int(viewer), viewer_autostart=int(autostart), view_custom=int(guide is not None),
                **view_metadata, **native_guide, **audio_metadata, **surface_metadata)


def bind_native_dsp(module, address=0, smoothing_address=0):
    """Inject typed native primitives, never Python callbacks on the audio path."""
    signature=ctypes.CFUNCTYPE(ctypes.c_int32, *([ctypes.c_void_p]*7),
                              ctypes.c_int64,ctypes.c_int64,ctypes.c_int64)
    if not address or not smoothing_address:
        root=Path(__file__).resolve().parents[1]
        override=os.environ.get('CLEANUP_TEST_LIBRARY')
        paths=[Path(override)] if override else ([root/'build/windows/CleanupNative.dll',root/'CleanupNative.dll']
                    if os.name=='nt' else [root/'build/libCleanupNative.dylib',root/'build/libCleanupNative.so'])
        library=next((p for p in paths if p.is_file()),None)
        if library is None:raise RuntimeError('Build CleanupNative before compiling native DSP primitives')
        module._native_library=ctypes.CDLL(str(library.resolve()))
        if not address:
            address=ctypes.cast(module._native_library.cleanup_dip_transport,ctypes.c_void_p).value
        if not smoothing_address:
            smoothing_address=ctypes.cast(module._native_library.cleanup_recursive_smooth,ctypes.c_void_p).value
    module._dip_transport=signature(address)
    smoothing_signature=ctypes.CFUNCTYPE(ctypes.c_int32,*([ctypes.c_void_p]*4),*([ctypes.c_int64]*6))
    module._recursive_smooth=smoothing_signature(smoothing_address)


def compile_candidate(path, native_dip_address=0, sample_rate=48000., bandwidth=3400., native_smooth_address=0):
    if not np.isfinite(sample_rate) or sample_rate <= 0 or not np.isfinite(bandwidth) or bandwidth <= 0:
        raise ValueError('Smoke-test sample rate and receive bandwidth must be finite and positive')
    source_path = Path(path).resolve(strict=True)
    source = source_path.read_text(encoding='utf-8-sig')
    tree = _audit_tree(ast.parse(source, filename=str(source_path)))
    name = '_cleanup_' + uuid.uuid4().hex
    module = pytypes.ModuleType(name)
    module.__file__ = str(source_path)
    sys.modules[name] = module
    try:
        if any(isinstance(n,ast.Name) and n.id in ('_dip_transport','_recursive_smooth') for n in ast.walk(tree)):
            bind_native_dsp(module,native_dip_address,native_smooth_address)
        # Compile the bytes just read: no .pyc timestamp race during rapid reloads.
        exec(compile(tree, str(source_path), 'exec'), module.__dict__)
        geometry = _geometry(module)
        if geometry['guide'] and bandwidth > (geometry['guide_bins']-1)*sample_rate/(2*(geometry['guide_fft']-1)):
            raise ValueError('Registered guide does not cover receive bandwidth; enlarge GUIDE_BINS and Reload')
        entry = module.Filter
        if not isinstance(entry, CPUDispatcher):
            raise TypeError('Filter must be a Numba dispatcher')
        if not isinstance(module.FilterBank, type):
            raise TypeError('FilterBank must be a class of compiled static methods')
        frames, bins = geometry['frames'], geometry['bins']
        slots, storage = geometry['slots'], geometry['storage']
        config_size = geometry['config_size']
        signature = types.int32(*(types.CPointer(types.float64),) * 4)

        @cfunc(signature, nopython=True)
        def callback(analysis_ptr, content_ptr, state_ptr, config_ptr):
            analysis = carray(analysis_ptr, (frames, bins, 2))
            content = carray(content_ptr, (frames, bins, 2))
            state = carray(state_ptr, (slots, storage))
            config = carray(config_ptr, (config_size,))
            try:
                return entry(analysis, content, state, config)
            except Exception:
                return -1

        methods = []
        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                methods.append((node.name, getattr(module, node.name)))
            elif isinstance(node, ast.ClassDef):
                cls = getattr(module, node.name)
                for child in node.body:
                    if isinstance(child, ast.FunctionDef):
                        methods.append((f'{node.name}.{child.name}', getattr(cls, child.name)))
        for method_name, method in methods:
            if not isinstance(method, CPUDispatcher) or not method.nopython_signatures:
                raise ValueError(f'{method_name}: not compiled in nopython mode (unused helpers need an explicit signature)')
            if len(method.nopython_signatures) != 1:
                raise ValueError(f'{method_name}: declare exactly one explicit specialization')
            method.disable_compile()  # The live callback cannot grow an implicit specialization.
            if any(result.objectmode for result in method.overloads.values()):
                raise ValueError(f'{method_name}: object mode is forbidden')
        # Call the exact C ABI with disposable state; initialization never warms live storage.
        analysis = np.zeros((frames, bins, 2), dtype=np.float64)
        content = analysis.copy()
        state = np.zeros((slots, storage), dtype=np.float64)
        config = np.array([float(sample_rate), float(bandwidth), 0., 0.,
                           -(geometry['contexts']-1.) * geometry['block']][:config_size])
        pointer = ctypes.POINTER(ctypes.c_double)
        for squelch in (0., 1.):
            for stimulus in (0, 1):
                config[2] = squelch
                if stimulus:
                    analysis[:] = np.random.default_rng(7).normal(0., .01, analysis.shape)
                    analysis[:, min(7, bins - 1), 0] += 1.
                else:
                    analysis[:] = 0.
                content[:] = analysis
                result = callback.ctypes(*(a.ctypes.data_as(pointer) for a in (analysis, content, state, config)))
                if result not in (0, 1, 2):
                    raise ValueError(f'Filter returned invalid disposition {result} during native smoke test')
                if not np.isfinite(content).all() or not np.isfinite(state).all():
                    raise ValueError('Filter produced nonfinite samples or state during native smoke test')
        # Only archived scripts contain the retired harmonic entity.
        if 1 in module.CHAIN:
            config[2]=0.;config[3]=1.;state[:]=0.;content[:]=analysis
            result=callback.ctypes(*(a.ctypes.data_as(pointer) for a in (analysis,content,state,config)))
            if result not in (0,1,2) or not np.isfinite(content).all() or not np.isfinite(state).all():
                raise ValueError('Filter failed the Harmonics-enabled native smoke test')
        token = uuid.uuid4().hex
        metadata = dict(geometry, token=token, address=callback.address,
                        sha256=hashlib.sha256(source.encode()).hexdigest(),
                        methods=[name for name, _ in methods], source=str(source_path))
        _generations[token] = (module, callback)
        return metadata
    except BaseException:
        sys.modules.pop(name, None)
        raise


def release(token):
    generation = _generations.pop(token, None)
    if generation:
        sys.modules.pop(generation[0].__name__, None)


if __name__ == '__main__':
    try:
        print(json.dumps(compile_candidate(sys.argv[1]), indent=2))
    except Exception:
        traceback.print_exc()
        sys.exit(1)
