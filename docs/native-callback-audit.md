# Audio boundary audit — 2026-09-16

The successful audio path is C# → C++ → Numba-generated native C entry →
compiled Filter/FilterBank/helpers → C++, not a Python dispatcher round trip.
Python loads, compiles, validates, and retains generations on the control worker.
The normal callback does not acquire the Python GIL. This is native DSP, but is
neither one monolithic function nor an allocation-free whole application.

## Pointer and ownership contract

| Boundary | Actual behavior |
| --- | --- |
| SDR# → managed wrapper | Early samples enter a fixed ring; late samples and aligned early samples are copied into reusable float arrays. Larger callbacks can grow those arrays on the audio thread. SDR# resamplers are outside this audit. |
| C# → C ABI | `fixed` pins those arrays. Pointer arguments avoid array marshaling. The late input/output pointer is identical: public API in-place operation is already used. |
| Native input → DSP histories | C++ reads both inputs before writing the corresponding output. It converts interleaved float32 to per-channel float64 buffers, then shifts contextual histories. Input/output aliasing does not eliminate required contextual storage. |
| C++ → compiled Filter | Four raw double pointers: analysis, writable content, persistent state, config. Numba `carray` constructs views. No array-data serialization, conversion, or copy occurs at this boundary. |
| Filter → helpers | Typed native calls, sometimes inlined. Contiguous storage slices/reshapes are views. Analysis/content are deliberately distinct buffers: the analysis must survive content modification. |
| Filter → native primitives | Smoothing and DIP receive typed C function addresses and array data pointers. The ctypes objects are resolved during compilation; they are not Python callbacks at audio time. |
| Filter return | An integer disposition. Modified coefficients and persistent state are already in C++-owned memory. No result array is copied back from Python. |

Source: `native/src/engine.cpp` (`Callback`, `block`, `cleanup_process_pair`),
`runtime/compile_filter.py` (`compile_candidate`), `plugin/CleanupPlugin.cs`
(`ProcessEarly`, `ProcessLate`).

The complete code is not Numba: the wrapper is managed C#, orchestration and
Fourier transforms are C++, and Filter.py's methods compile with Numba/LLVM.
All 49 declared Filter methods were verified to have one nopython specialization,
without object mode. The compiler disables further implicit specialization.

## Measured allocation finding

`tools/audit_native_callback.py` uses the exact current C callback and Numba
runtime allocation counters on the M4. It also walks native LLVM definitions
reachable from the exported C entry, excluding unrelated Python dispatcher
wrappers. Evidence: `evidence/native-callback/mini.json`.

For each of silence, random magnitudes, and a peaked spectrum, three calls were
made with squelch off, squelch on, and harmonics on (27 calls total):

| Configuration | NRT allocations per call | NRT frees per call | MemInfo objects per call |
| --- | ---: | ---: | ---: |
| Harmonics off | 1536 | 1536 | 768 |
| Harmonics on | 1540 | 1540 | 770 |

MemInfo counts are a separate view of these allocations, not additional counts
to add to the allocation column. Equal allocations/frees indicate no retained
NRT allocation in these cases; they do not make the path realtime deterministic.

The LLVM points to `numba.misc.quicksort...run_quicksort1`: its native partition
stack uses two dynamically allocated lists. The reference and guide entropy
functions each sort once for each of 192 time columns. Thus 384 sorts × four NRT
allocations accounts for 1536 allocations. The enabled packet-harmonic path adds
one sort in these tests. Median selection's depth-limit fallback can also sort;
the observed count is not an upper bound for all inputs or experiment settings.

An independent raw-pointer probe confirms that `carray` preserves the exact
input address, writes through to the original array, and allocates nothing in
NRT. A separate sort probe measures its allocation traffic. The issue is the
generated algorithm's workspace, not moving entire arrays through Python.

These counters do not observe C++/libc, managed allocations, or bfft internals.
They were collected on macOS arm64; Windows x64 code generation remains a
separate allocation-audit target. Synthetic spectra test execution, not hearing
quality or the correctness of a real received-signal squelch decision.

## Error-path qualification

The generated C wrapper contains an exception-status branch that acquires the
GIL and invokes Python's error reporting (`PyErr_WriteUnraisable`, etc.). Its
successful branch returns directly. Filter exceptions are normally caught by
our compiled callback and converted to ERROR, but the generated outer safety
path still exists. Do not claim that all possible callback paths are independent
of Python merely because compilation used `nopython=True`.

## Remaining native copies and work

- bfft's frequency-major complex arrays are transposed into Filter's time-major
  real/imaginary arrays, then emitted coefficients are packed back for inversion.
- The enhanced guide cache is copied into its reserved state row each block.
  Direct publication into that row is a candidate after checking cache ownership
  and the identical-channel reuse contract.
- Signed audio history is copied into state even in the current legacy-floor
  configuration; its consumer is the optional crossing-floor experiment.
- Identical-channel sharing copies the first channel's output coefficients and
  materializes persistent state if the channels diverge. Those copies preserve
  independent histories; removing them requires ownership changes.
- Current default geometry computes the analysis STFT and fixed-reference STFT
  separately although both have 512/128 geometry and the same analysis input.
  Reuse is a candidate only when all transform/window/geometry contracts agree.
- Diagnostic magnitudes are copied into a separate snapshot, allowing the viewer
  to poll without retaining or mutating the live DSP buffers.

Priority: replace the native quicksort's dynamic partition workspace with bounded
preallocated/stack workspace while preserving exact sorted values and decisions;
then measure copy and repeated-transform savings. A zero-copy boundary already
exists, so a Python-dispatch rewrite would target a cost this path does not have.

## Squelch remains an open live issue

The C callback returned SKIP for synthetic silence and random magnitudes with
squelch enabled, and PROCESS for the peaked case. This verifies the flag and
disposition path, not the user's live noise-only input. In enhanced mode,
whole-frame admission still uses the fixed 512/128 reference evidence, its total
count and contiguous-run rule. Diagnose the live evidence before changing that
rule. The user reports smooth playback but insufficient/no squelching.

Reproduce (from this repository):

```sh
/Users/ultimussecundai/.local/bin/m4build -- /Users/joshuahkuttenkuler/miniforge3/bin/python tools/audit_native_callback.py --report build/native-callback-audit.json
```
