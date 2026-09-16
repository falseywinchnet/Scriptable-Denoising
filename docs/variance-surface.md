# Bounded time–frequency variance surface

Experimental `REGISTERED_FLOOR = "variance_surface"` in the single compiled
`Filter.py`. Reload selects it and reconstructs/reset its fixed native storage.
The root default remains `legacy`; the listening renderer produces an additional
`surface.py` / `surface.wav` comparison. This implements the user's mean/variance
proposal, not the rejected temporal noise/not-noise classifier.

## Surface construction

The input is the registered magnitude field in Cleanup's reference amplitude
units, on the existing hop128 mask lattice after its triangular pre-smoothing.
All nonnegative frequencies are included; receive bandwidth is not an estimator
input. No speech, pause, stationarity or source-presence decision controls the
surface update. Time interpolation does not add independent observations.

1. At each cell, collect nine anchors: three frequencies (center and ±150 Hz,
   rounded to guide bins) at three times (current and approximately +20/+40 ms).
   Median/MAD anchors bound the current observation at
   `median + 3 * max(median, MAD)`. Only its statistical contribution is clipped;
   the measured magnitude used for mask decisions is untouched.
2. Sliding frequency windows of ±150 Hz form local first and second moments from
   these bounded observations. This stage is linear in the number of cells.
3. Track their mean and variance in time with
   `alpha = 1 - exp(-mask_hop / (sample_rate * 0.04))`. For previous mean `mu`,
   variance `v` and `scale = max(mu, sqrt(v))`, limit the mean innovation to ±scale
   and the variance innovation to ±4*scale² before multiplying by alpha. The
   variance target includes local variance plus `(1-alpha)*bounded_delta²`, the
   between-mean term of the exponentially weighted mixture.
4. Form the candidate `mu / (1 + v/mu²)` (zero mean yields zero). This is the
   requested variance penalty, with experimental coefficient 1. Then limit floor
   movement to ±alpha*scale per mask step. Floor values remain nonnegative. During
   a downward transient, the rate-limited floor can temporarily exceed the new
   mean; the unbounded candidate itself is always between zero and the mean.

These are bounded/winsorized moment estimates, not unbiased estimates of a
physical noise PSD. The width, time constants and clipping multipliers are
explicit provisional code constants, not new UI sliders. The initial positive
local observation establishes a scale: a previous exact zero has no relative
scale for limiting first acquisition. Nine-anchor clipping still applies there.
A sustained broad change can move the estimate over time; the bounds prevent an
arbitrary one-step displacement, not all eventual adaptation.

The anchors look forward within the context already available to this engine.
They require no extra audio buffering. State is checkpointed one block into each
context and reused at the start of the next context. Lookahead does not commit
future state. The same overlapping observations therefore do not repeatedly
train the tracker. A discontinuous origin discards that checkpoint and reseeds.
The native registered field can itself change with context registration; exact
checkpoint equivalence is tested for identical observations, not asserted for
context-dependent registration fields.

## Connection to Cleanup

The surface replaces MAN/ATD and entropy-weighted peak thresholds for this
experiment. It is measured once from the unmasked field and reused in both peak
passes; zeros inserted by the first mask cannot drag the second floor down.
Cleanup's original masking of magnitudes, second triangular pass, multiplier,
mask intersection, final triangular smoothing and three native recursive
smoothing rounds remain. The bounded final gain is projected onto synthesis
coefficients at their physical centers before the existing inverse transform.

No blanket passband cut or non-squelch MAN/ATD fallback is applied to this mode.
The separate existing whole-block squelch, when explicitly enabled, still uses
its fixed 512/128 reference and original total/contiguous interval rules. This
change does not recalibrate those rules. The listening comparison has squelch
and harmonics off. Silence advances the estimator and clears its mask; stale
surface diagnostics are not reused across silent blocks.

## Diagnostics

`SURFACE_VIEW` declares fixed row-0 offsets for mean, variance, floor and existing
final mask. The compiler validates bounds and lattice correspondence. C++
preallocates four publication arrays at Reload and copies only emitted guide
centers under the existing nonblocking diagnostic lock. JSON serialization stays
on the control server. No Python dispatch or display allocation enters the audio
callback. Reload destroys the prior generation's publication and moments.

The existing listening page adds **Variance surface**, selected initially. Its
original top analysis / bottom final-before-inversion pair stays intact. Extra
maps expose the floor (selectable mean or standard deviation) and final recursive
mask. Cursor curves show input analysis, mean, standard deviation, floor and
actual output. Amplitude fields share the existing coherent-window dB convention;
the mask uses a separately labeled linear 0–1 scale. Variance is stored as variance
in the NPZ and displayed as its square root, so the amplitude scale is honest.

## Verification and reproduction

```
/Users/ultimussecundai/.local/bin/m4build -- sh tools/verify-surface-mini.sh
/Users/ultimussecundai/.local/bin/m4build -- sh tools/render-surface-mini.sh
```

The render script expects the retained `build/listening-demo-v5` on the Mini and
uses its exact original WAV. All previous variants are copied, not re-rendered.
Retrieve evidence before another m4build sync.

Tests cover the nonzero-variance formula, zero and flat fields, amplitude scaling,
an isolated outlier increased from 1e3 to 1e12 with identical bounded contribution,
mean/variance/floor innovation bounds, overlapping contexts, invalid publication
geometry, uninterrupted FM native processing, finite bounded mask, and Reload
reset. Registered legacy mask/projection parity and ordinary/custom diagnostic
publication are checked separately. These establish implementation behavior,
not perceptual denoising quality or universal robustness.

## Measured first rendering

The 30-second Dave/Simon comparison produced finite audio, zero native faults,
zero clipped samples, peak 0.62242 and RMS 0.09560 at unchanged recording gain.
All six preexisting comparison WAVs were verified byte-identical. The exact
variant script hash matches the native generation hash in the render manifest.

Inspection at 2.7 seconds shows the surface following the occupied envelope and
rolloff near 3 kHz. There is in-band attenuation but still patchy output gaps.
The gain mask also opens in very low-energy stopband regions; a bright gain-map
cell does not imply bright output energy. At that cursor, the mean gain over
200–2800 Hz is 0.264, and the median penalized-floor/mean ratio is 0.792. These are
observations, not speech-quality scores or reasons to declare the tuning optimal.

Worst callback: 433.14 ms for a 170.67 ms block on the Mini. Total render with
snapshot export: 86.14 s for 30 s of input. The full-band registered path plus
this estimator **does not meet the live deadline**; the rendered browser platform
allows the algorithm to be evaluated before optimization. The Windows native DLL
cross-build succeeded; this round does not claim a Windows/Wine or SDR# live run.

Evidence: `docs/evidence/variance-surface-tests.json`,
`variance-surface-tests.log`, `variance-surface-render.json`,
`variance-surface-analysis.json`, and `variance-surface-browser.json`.
The raw float32 surface archive remains at the Mini path recorded in the analysis
report; the local demo contains the complete dB/gain display rasters and WAV.
The browser check exercises loading, field selection, variant switching, seek,
play/pause, complete raster dimensions and console errors. It uses an isolated
headless browser, leaving the user's playback state alone.

One existing viewer regression test implicitly relied on the demo header having
`VIEWER=False`. The header currently enables it; the test now explicitly disables
it for its reload assertion. That regression passed after the fixture correction.
