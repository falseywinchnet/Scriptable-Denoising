# Scriptable Denoising — an open experiment

An **experimental scriptable denoising wrapper**: a native C++ audio engine runs
reloadable, explicitly compiled Numba filters, with an SDR# Next plugin and a
cross-platform spectral viewer. **Modified Cleanup v2 is the first demonstrator**
supplied as `Filter.py`; the wrapper is intended to support further DSP experiments.

The SDR# panel has **Reload**, a green **success** / red Python error label,
**Bypass**, and **Squelch**. There are no sliders. Filter scripts declare transform
geometry, own persistent fixed-size state, chain compiled entities, and return
passthrough, process, or skip for each segment. Reload compiles a replacement and
reconstructs the native plans, buffers, and state.

## First demonstrator: modified Cleanup v2

The supplied script preserves Cleanup's recursive two-pass separation while
exploring enhanced-magnitude analysis. It derives from the complete older SDR#
plugin source supplied locally; it is an experimental modification, not a claim
that it supersedes the original Cleanup in every setting.

The reusable facilities include bfft FFT/inverse FFT, STFT and reconstruction,
RFFT/half-bin ODFT support, fixed scratch storage, native recursive smoothing,
a spectral viewer, and a command-line control interface. There is also an optional
**enhanced magnitude representation**, with optional sub-hop registration. Other
methods may find these facilities useful. In particular, the enhanced representation
may be worth investigating as input to ML vision methods; **that usefulness is
unknown and has not been demonstrated here**. It is an analysis representation,
not evidence of recovered missing information.

The current Cleanup v2 demonstrator uses that representation with registration
disabled. The wrapper can also run the ordinary baseline analysis path. Historical
harmonic synthesis experiments are archived and inactive in this version.

## Download

The [Windows x64 experimental release](https://github.com/falseywinchnet/Scriptable-Denoising/releases)
includes the compiled SDR# Next wrapper, native DLL, viewer, and a private Python /
NumPy / Numba runtime. Extract the `Cleanup` folder into SDR#’s `Plugins` directory
with the host closed. See [installation and ABI instructions](docs/windows-bundle.md)
and [the verified version](docs/release-2026-09-16.md).

## Start here

- Edit **Filter.py**. Its header contains geometry, constants, fixed storage sizes,
  and the required explicit Numba typing style.
- Press **Reload**. Compilation happens in a background thread. A successful reload
  reconstructs every native STFT plan, buffer, queue, history and entity state.
- A red compile error leaves the previous working generation running. A runtime
  callback fault uses bypass until a successful Reload.
- **Bypass** returns original samples with the same engine delay. **Squelch** allows
  the filter's activity decision to drain the overlap tail and then emit silence.

Read [the algorithm and ABI notes](docs/algorithm.md) before changing the baseline.
They distinguish the preserved recursive processing from the intentional window
change and numerical repairs.

The live default uses the enhanced magnitude guide with **registration off**,
the complete recursive Cleanup mask, and full guide coverage through Nyquist.
[Mask source routing](docs/mask-source.md) preserves the original default After
relationship: early input determines whole-block squelch; later audio supplies
the magnitude field used for filtering. The [mono analysis tap](docs/mono-analysis-tap.md)
reads SDR#'s early stream without modifying it.

Harmonics is retired from the active filter and panel. Its source remains in
`research/harmonics/Filter-retired.py` for reproducible historical comparisons.
The [population experiment](docs/population-cleanup.md) and earlier floor trials
remain documented separately. Live integration and listening still require
validation on the particular host and reception conditions.

## CLI control, without SDR# accessibility

The DLL starts a loopback-only command server. The default port is 52381; if occupied,
an instance chooses an available port and exposes it through its API and the
plugin's `cleanup-control.json` discovery file. All button state lives in the DLL,
so the panel reflects changes made by a CLI.

```sh
python3 tools/control.py status
python3 tools/control.py bypass on
python3 tools/control.py bypass off
python3 tools/control.py squelch toggle
python3 tools/control.py --wait reload
```

For another instance:

```sh
python3 tools/control.py --discovery /path/to/Plugins/Cleanup/cleanup-control.json status
```

`cleanupctl.exe` is a small native Windows client with the same command vocabulary:
`cleanupctl.exe --port 52381 bypass toggle`. The macOS `build/cleanupctl` talks to
SDR# running under Wine through the same loopback port. The Python CLI's `--wait`
option additionally waits for Reload and returns failure on a compiler error.

The server is part of the DLL and lives as long as the host's engine instance.
For an independent background experiment host, run:

```sh
python tools/experiment.py serve input.wav --library build/libCleanupNative.dylib
```

It feeds a repeating WAV at its sample rate; it does not open a sound device.
While it runs, the same commands control it. Offline, latency-compensated output:

```sh
python tools/experiment.py render input.wav output.wav \
  --library build/libCleanupNative.dylib
```

Use 16-bit PCM mono/stereo WAV inputs. Each render writes a JSON report containing
source paths, source hash, active geometry, timings and RMS alongside the WAV.
On Windows use the packaged `_python/python.exe` and `CleanupNative.dll` paths.

## Build and stage

The C# wrapper is supplied precompiled under `plugin/precompiled/windows-x64/`.
Its source and the SDR# SDK are excluded from this public snapshot. The native
engine, script compiler, viewer, and DSP experiment are included as source.

Native builds need CMake, a C++17 compiler, and Python 3.12 development files with
`requirements.txt` installed. Research tests additionally use SciPy. On macOS/Linux:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel 4
python3 tests/test_mask_source.py
```

On Linux set `CLEANUP_TEST_LIBRARY=build/libCleanupNative.so` for tests whose default
is the macOS library. On Windows with Visual Studio C++ build tools:

```powershell
./tools/build-windows.ps1 -SdrSharpDirectory C:/SDRSharp -PythonDirectory C:/Python312
```

For MinGW cross compilation:

```sh
CLEANUP_WINDOWS_PYTHON=/path/to/windows-python312 \
CLEANUP_SDRSHARP=/path/to/sdrsharp-next \
sh tools/build-cross.sh
```

The build scripts use the precompiled wrapper when its private C# project is absent.
The wrapper targets the tested SDR# Next .NET 10 x64 interfaces.

For the MinGW artifacts, install with:

```sh
python3 tools/stage.py --sdrsharp /path/to/sdrsharp-next \
  --python-runtime /path/to/windows-python312 --development
```

`--development` makes the deployed Filter.py a link to this checkout on macOS,
so there is one authoritative experimental script. Omit it for a portable Windows
copy. The private runtime is named `_python` to exclude it from SDR#'s recursive
plugin assembly scan. Existing plugin installations are backed up under
`_cleanup_backups` before staging. DLL/compiler changes need a host restart;
Filter.py changes only need Reload.

**Wine audio:** select WASAPI for SDR# output. MME crashes in this user's setup.
The tested Wine prefix has .NET Core and Windows Desktop runtimes 10.0.10.

## Verification and scope

Automated checks cover:

- Explicit typing, missing decorators, implicit/uncompiled functions, illegal
  geometry, invalid dispositions and nonfinite output.
- bfft identity with stereo, irregular callback sizes and in-place input/output.
- Exact delayed passthrough and tapered overlap draining before zero output.
- Reload failure retention, geometry changes, reset of old histories, native
  runtime failures, separate state rows, and reload concurrent with audio.
- The initial Cleanup path producing finite output and attenuation on synthetic
  tone-plus-noise input.
- Native mask/smoothing parity against the original Downloads C++ methods.

The Windows DLL and managed plugin have also been loaded inside the downloaded
SDR# Next under Wine 11.17, with real audio callbacks and WASAPI output. Evidence
is retained in [docs/evidence](docs/evidence).

Default engine latency is **16,768 samples / 349.333 ms at 48 kHz**, before host
resampling and output buffering. Baseline stereo processing has been observed within
the block interval on this Mac under Wine; this is an experimental audio path,
not a hard real-time guarantee. Sorting uses explicit caller-owned scratch; the
current 27-call audit reports zero Numba-runtime allocations. See
[the allocation audit](docs/scratch-workspace.md) for its scope.

Impulse reconstruction, reverb correction, and improved VAD/slider-free suppression
are stage-two experiments. The bank provides fixed per-entity storage and compiled
chaining now; the initial enabled entity is Cleanup. SDRconsole integration remains
an untested consumer of the new plain C ABI, not a claimed drop-in NR5 replacement.

## Sources and provenance

- [Cleanup repository](https://github.com/falseywinchnet/Cleanup) — located as requested;
  its contents were not used.
- [SDR# Cleanup plugin repository](https://github.com/falseywinchnet/SDRSharp-Cleanup-Plugin).
  Working source: the complete local Downloads plugin. Its native reference header
  is retained under `reference/`; the C# wrapper sources are not distributed.
- [bfft](https://github.com/falseywinchnet/bfft) — copied from the existing local checkout;
  file hashes and commit are in `vendor/bfft/PROVENANCE.json`.
- The older PyITD streamcleaner ZIP supplied Python design context; its retained
  source/manual are in `reference/streamcleaner-master/`.

Existing source copyright/license notices are preserved in the reference files and
vendored bfft LICENSE. This stage does not invent a replacement license for the
author's older source. This experiment has its own repository; the original
Cleanup and SDR# plugin repositories are unchanged.

## Native cross-platform viewer

See [viewer setup and controls](docs/viewer.md). `CleanupViewer` is a C++ Dear
ImGui/ImPlot application with an OpenGL renderer, built from the same source on
Windows, Linux, and macOS. Reload launches it directly. Pause freezes the display
while processing and polling continue. The Windows executable now renders inside
Wine without the former host-side Python bridge. See the documented execution
coverage and the live verification in the release notes.

The [experimental plan](docs/experiment-plan.md) separates the measured Cleanup
baseline, guide-driven masks, and optional DIP harmonic extension.

## Complete Cleanup in registered analysis space

The full two-pass Cleanup mask and all three recursive smoothing rounds now run
on the registered sub-hop inverse field when `ANALYSIS_MODE = "registered"`.
The resulting real gain is interpolated onto the synthesis grid, preserving its
complex phase. `"baseline"` retains the ordinary-analysis comparison. The native
parent creates guide plans, persistent workers and all fixed state at Reload.

See [registered Cleanup](docs/registered-cleanup.md) for geometry, the fixed
512/128 whole-block admission reference, numerical parity, and measured timing.
The native Mini configuration meets the measured block budget; the Windows/Wine
configuration must pass its own full callback budget before live promotion.
These performance results do not establish improved denoising quality.

The optional [Harmonics experiment](docs/harmonics.md) has a separate button and
native command, is off by default, and leaves measured lower-band coefficients
unchanged. Evaluate it separately from the guide-driven denoiser.

## Bounded mean/variance floor experiment

`REGISTERED_FLOOR = "variance_surface"` selects a time–frequency surface whose
mean is penalized by relative variance. Bounded observations, statistical updates
and floor updates restrain large excursions; no quiet interval or source-presence
classifier is required. The full-band floor feeds both Cleanup mask passes.
See [variance surface](docs/variance-surface.md) for the exact construction,
overlapping-context checkpoints, tests, native diagnostics and listening demo.
This full-band path currently exceeds the live callback budget on the tested Mini.
