# Scriptable Denoising — experimental wrapper

This is a Windows x64 **experimental prerelease**, tested in SDR# Next under Wine
with WASAPI. Native Windows performance is not yet qualified.
Its first demonstrator is modified Cleanup v2 in the editable `Filter.py`.
It includes the compiled SDR# Next wrapper, the native
engine and viewer, and a private CPython 3.12 / NumPy / Numba / LLVM runtime.
No system Python installation, package download or PATH change is required.

The wrapper supplies reusable transform, compiled-filter, persistent-state and
viewer capabilities. The enhanced magnitude representation is an optional
analysis facility; its potential for ML vision methods is untested. Current DLL
and installation folder names retain `Cleanup` for integration.

## SDR# Next

Close SDR# before replacing DLLs. Copy the entire `Cleanup` folder into the
SDR# `Plugins` directory. Use the matching current .NET 10 x64 SDR# Next host.
Keep `_python`, `runtime` and the DLLs together. The underscore on `_python`
prevents SDR# from scanning its DLLs as managed plugins.

On first load, the plugin compiles `Filter.py`; this can take a while. Reload
compiles a replacement off the audio path. Success installs it and resets its
STFT geometry, filter state and overlap history. An error keeps the previous
working generation and reports the Python traceback. Bypass and Squelch operate the native engine. The viewer displays
analysis above the final coefficients before inversion and can be paused.

The current script covers the full analysis grid through Nyquist, using
receiver-aware statistics and the original two-pass Cleanup separation. Early
input supplies squelch evidence; the enhanced mask guide uses the later audio
(`MASK_SOURCE="content"`), restoring the original default After relationship.
Packaging preserves the chosen statistical policy; Population analysis is an
explicit `CLEANUP_POPULATION="rolloff"` experiment. **Registration defaults off**: a single central
enhanced-magnitude observation guides Cleanup. The seven-view registration and
maximum fusion remain available by setting `GUIDE_REGISTRATION = True` in
`Filter.py` and pressing Reload. This rebuilds the guide and resets its history;
registration uses at most four active DSP lanes, including the caller. With it
off, guide construction uses the caller alone. Harmonics has been retired from
this version; its code and historical comparisons live under research/harmonics.
Receiver-aware and other
research modes remain in the single editable `Filter.py`; they are experiments,
not adaptive noise/speech classifiers. Read `_source/docs/population-cleanup.md`
for the estimator's limitations.
See `_source/docs/optional-registration.md` for the on/off listening comparison.

For Wine, use **WASAPI** audio output. MME crashed in the development host.
Wine measurements do not establish performance on a native Windows receiver.

## Native host integration

Load `Cleanup.dll` by absolute path. Use `_sdk/include/cleanup_loader.h` and
`cleanup.h`. The loader discovers its bundled engine and interpreter; call
`cleanup_create(NULL, NULL, sample_rate, channels)` to use adjacent defaults.
Check `cleanup_loader_error` if creation returns NULL. Poll `cleanup_status`
for asynchronous compilation, faults, configured geometry and latency.

`cleanup_process` accepts interleaved float32 mono/stereo buffers of arbitrary
length and supports in-place operation. `cleanup_process_pair` accepts aligned
early-analysis and later-content inputs. Compensate for the reported latency
when mixing paths. Stop audio calls before destroy. Keep Cleanup.dll loaded
until process exit; the embedded interpreter is process-wide. Hosts already
embedding an incompatible Python runtime should use process isolation.

`_sdk/example/bundle_host.cpp` is an example standalone loader and smoke host;
it needs no Python headers or import library. A MinGW import library is also
included. MSVC users can call GetProcAddress as in the example or generate an
import library from the exported C symbols. This is a documented new C ABI,
not a tested drop-in replacement for SDR Console's NR5 interface.

`cleanupctl.exe PORT status`, `reload`, `bypass toggle`, and `squelch toggle`
control the same DLL. The legacy harmonics ABI remains for archived scripts;
the current Filter ignores it. The plugin writes the port to
`cleanup-control.json`. The native viewer runs separately from the host.

## Source and runtime

`Filter.py` holds the DSP experiment. `_source/viewer` contains the cross-platform
C++ viewer and its pinned CMake dependencies. `_source/native`, bfft, and the
root CMake file contain the native engine and Windows bootstrap source. The C#
wrapper is distributed precompiled. Python package license notices are retained
in `_python/Lib/site-packages/*.dist-info`, CPython's in `_python/LICENSE.txt`,
and the viewer's in `VIEWER-LICENSES.txt`. Original source notices are preserved.

Windows, macOS and Linux require separate native/runtime archives for each CPU
architecture; a Windows runtime cannot serve as a universal Python bundle.
`MANIFEST.json` records every packaged file's SHA256 and the pinned runtime
versions. See `_source/docs/release-2026-09-16.md` for the verified version and its limits.
