# Cleanup: stage-one algorithm and native contract

This is an open denoising project experiment. The initial filter ports the complete
spectral denoising pass from the user's Downloads plugin. Stage two will improve
VAD and suppression behavior; it has not been substituted into this baseline.

## What remains from Cleanup

- Three audio blocks of context, default 8192 samples per block at 48 kHz.
- A 512-point real FFT, 128-sample hop, 192 contextual spectra, 64 emitted spectra.
- Logistic reference distributions, a three-frame sorted-spectrum correlation
  score, three-frame score averaging, activity-run detection, and run cleanup.
- Global and per-frame MAN/ATD statistics and the original weighted peak threshold.
- Two mask passes, triangular time smoothing, and three recursive rounds combining
  frequency and time smoothing. These recursive rounds remain active.
- Early audio determines squelch. With the default `MASK_SOURCE="content"`,
  mask statistics use the later audio, restoring the original default After
  relationship. See [the source-routing correction](mask-source.md).
- Squelch chooses the skip disposition. Unsquelched noise regions retain the
  original attenuation-floor rule.

`tests/test_legacy.py` feeds the same spectra and VAD decisions into the actual
Downloads C++ spectral-mask methods and the Python port. The emitted mask rows
agree to approximately 5.6e-17 for 26, 37, and 129 bins, squelched and unsquelched.
This certifies those masking/smoothing stages. It is not a bitwise claim about the
whole old multi-window audio pipeline or an independent VAD quality score.

## Intentional changes and defect repairs

1. **One window pair.** bfft's symmetric Hann analysis window and its internally
   calculated canonical dual synthesis window replace the old logistic/Hann
   analysis switching and precomputed synthesis coefficients. bfft owns both
   transforms and streaming overlap-add. This is the tested window choice for
   reconstruction; no universal perceptual-optimality claim is made.
2. **Finite flat inputs.** Empty MAN/ATD input returns immediately, and constant
   spectra give zero non-noise score. The old header could index -1 or divide by
   zero after its attempted empty-input guard.
3. **Correlation normalization.** Pearson correlation is affine invariant, so
   sorted spectra are correlated directly. The old normalization overwrote its
   minimum during the loop, then subtracted that changed value from later bins.
4. **Extrema are per call.** Maxima and minima no longer accidentally accumulate
   across unrelated segments or hold uninitialized values.
5. **VAD run termination.** The longest activity run includes a run at the final
   context frame. The original helper failed to finalize that trailing run.
6. **Preserved quirks are explicit.** The original first-n deviation convention,
   the run-length-plus-one test, the 6.916666... triangular normalization, the
   second peak mask's in-place behavior, and the unsquelched output-row VAD indexing
   remain. Changing them belongs to a measured stage-two experiment.
7. **No sliders.** `VAD_THRESHOLD=0.057` and `MASK_STRENGTH=1.0` are explicit
   experiment constants. Bandwidth follows the SDR# demodulator automatically.

## Strict script style

Every script function and static class method must have an explicit Numba signature,
for example `@njit(F64(VEC, I64), nogil=True, cache=False)`. `VEC`, `MAT`, and `CUBE`
are C-contiguous float64 arrays. Indices are int64; disposition returns are int32.
The example header documents the style inline. Ordinary Python annotations alone
are insufficient. The loader verifies one nopython specialization per function,
rejects object-mode blocks, implicit signatures, and uncompiled functions, and
turns off further dispatcher specialization before publishing the callback.

LLVM still generates machine code at Reload. Explicit signatures bound the type
contract and expose errors at import/compile time; no measured reload speedup is
claimed. The compiler runs on a background control thread.

## FilterBank and fixed persistent storage

`FilterBank` contains compiled static methods. Its `run` dispatcher chains the
entity IDs in `CHAIN`; slot k always receives `state[k]`. Native code allocates
`MAX_ENTITIES × ENTITY_STORAGE` float64 values per channel once per successful
reload. Each slot owns its row even when two slots use the same entity type.
Methods can divide that row into fixed array views. The initial entity is Cleanup;
impulse reconstruction and reverb correction are future entity implementations,
not pretend enabled features. New entity branches must call explicitly typed
compiled methods and retain the same disposition contract.

There is no Python class instantiation, interpreter call, or GIL acquisition on the
audio path. NumPy operations supported by Numba run as compiled code. Numba's sort
implementation may use native scratch allocation; this is not a certified hard
real-time, allocation-free runtime. Persistent storage and C++ STFT buffers are fixed.

## Reload transaction

`Filter.py` declares FFT_SIZE, HOP, BLOCK_SIZE, CONTEXT_BLOCKS, WINDOW, geometry,
chain, and storage sizes. The loader audits the source, eagerly compiles all
explicit signatures, builds a C callback, and tests that callback on disposable
silence and signal/noise buffers with squelch off/on. Geometry and memory limits
are checked before activation; invalid results or nonfinite output/state fail.

C++ then allocates every channel's bfft plans, histories, queue, spectra, overlap
tail and entity storage. Under a short synchronization boundary, the new generation
replaces the old one between host callbacks. All DSP state begins at zero. The
previous generation's code and storage are released on the control thread after
no audio callback can reference them. A rejected candidate leaves the old DSP
state and code intact. Reload deliberately incurs fresh pipeline startup latency;
it is not a seamless crossfade between algorithms.

The stable compiler in `runtime/compile_filter.py` is infrastructure. Updating that
file, the DLL, or the managed plugin requires a host restart. Ordinary experiments
change only Filter.py and use Reload.

## Dispositions and latency

| Return | Parent action |
| --- | --- |
| PASSTHROUGH = 0 | Exact original **delayed** content samples; update the synthesis tail with unmodified spectra so later processing transitions have valid overlap. |
| PROCESS = 1 | Synthesize the script's middle spectra through bfft's overlap-add. |
| SKIP = 2 | Synthesize zero spectra; add the preceding tapered overlap, then output exact zeros. |
| ERROR = -1 or invalid/nonfinite output | Mark a runtime fault and use delayed bypass until a successful Reload. |

The engine accepts arbitrary interleaved float callback sizes and supports in-place
processing. Each stereo channel has separate history and state. For the defaults,
engine latency is 16,768 samples = 349.333 ms at 48 kHz, plus host/resampler/audio
buffering. This follows `contexts*block - first*hop + fft/2`; the tests check the
actual sample alignment. Bypass keeps this delay to avoid changing synchronization.
Before the first successful generation exists, audio passes through directly.

## Host portability

`native/include/cleanup.h` is a plain C ABI with create/destroy, process/process_pair,
reload, bypass, squelch, bandwidth, status and command-port calls. It can be wrapped
by another Windows audio host. SDRconsole/NR5 integration has not been tested;
its old binary ABI is not assumed to match this new API. The DLL uses Python 3.12
and retains the compiled callback's runtime while it is active. On macOS, the
library resolves Python from its embedding process (the test/experiment Python
client provides it). Windows embeds the packaged python312.dll itself.
