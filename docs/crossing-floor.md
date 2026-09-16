# Zero-crossing neighbor floor experiment

User proposal, September 15, 2026: replace the registered representation's
MAN/ATD/row-fit floor with the mean of samples next to zero crossings.

## Exact trial

The interpretation used here is **signed input audio**. For each sign change,
collect the absolute values of the two nonzero samples on either side. Average
all those observations over the parent's current three-block contextual history
(24576 samples, 0.512 seconds at 48 kHz). A sample bounding two sign changes is
counted twice. Exact-zero runs contribute their nonzero bounding samples only
when those endpoints have opposite signs. No mean removal, clipping, outlier
rejection, median, deviation, or signal-dependent fitting is applied.

`REGISTERED_FLOOR = "zero_crossing"` selects the trial in `Filter.py`.
`"legacy"` retains the previous registered result for comparison. These are competing experimental hypotheses; legacy behavior is not a ground-truth floor. The ordinary baseline Cleanup path is unchanged. The listening renderer explicitly renders
both settings; the new button is **Crossing floor**. Harmonics is off in this
comparison so it cannot obscure the floor experiment.

The experimental path evaluates the entire nonnegative frequency axis through
Nyquist. Its guide has `GUIDE_FFT` rows (2048 at the current aperture), with no
receiver-bandwidth cutoff in seed selection, reference evidence, gain projection
or output. It does not accept a presumed signal bandwidth. Sample rate supplies
frequency coordinates only. A regression checks identical audio under receiver
bandwidth settings500Hz,3400Hz and22000Hz, including a supported12000Hz tone.
This full-band extension changes the amount of registered analysis work; measure
its callback cost separately from the earlier cropped experiment.

The parent optionally publishes its signed analysis history to a reserved fixed
state region specified by `AUDIO_HISTORY`. Compiler validation rejects storage
overflow and overlap with bank entities, the registered field or reference
scratch. Publication is a native copy; the actual estimator is explicitly typed,
compiled Numba in `Filter.py`. Reload resets the history as well as entity state.

## Units, deliberately exposed

A time-domain amplitude cannot be compared directly with a differently normalized
transform amplitude. For the literal cosine inverse C=irfft(window*x), the
interior-bin white-noise approximation is

```
E|C| / E|x| = sqrt(2 * sum(window**2)) / (2 * (N-1)).
```

For a periodic Hann, sum(window**2)=3N/8. Multiply by the existing
`GUIDE_MAGNITUDE_SCALE` to enter the registered mask's units. The resulting
factor at aperture2048 is **4.88941117**. Independent white-noise simulation
measured **4.90006701**. This is an amplitude-units approximation for the
pre-registration cosine field, not a fit to the recording. It does **not** correct
registration's maximum-selection bias, window boundary effects, DC/end-bin
variance, noise coloration, or conditional sampling bias near speech crossings.

Both passes compare their temporally smoothed magnitudes against this same
floor. They retain Cleanup's binary first seed, second-pass residual convention,
triangle smoothing, padded recursive smoothing, and projection onto synthesis
bins. The experimental branch does not use MAN/ATD, the entropy-weighted row
threshold, or the MAN-based nonsquelched fallback gain. Whole-frame squelch
admission retains the reference lattice and count rule but now evaluates all
reference bins; its statistical calibration is itself experimental; squelch is OFF
in the listening demo. No-crossing input provides no floor observation and
passes through unless the independent whole-frame squelch has already closed.

The raw mean, crossing count, converted floor, and both pre-smoothing selected
fractions are shown in the demo. State0[24:28] stores raw mean, observation count
(two per crossing), converted floor, and active-experiment flag.

## Characterization, not a quality claim

`tests/test_crossing_floor.py` compares compiled selection with an independent
vectorized oracle, tests exact zeros and no crossings, checks amplitude scaling,
checks white-noise units, rejects invalid storage, verifies actual native history
publication over overlapping blocks, and verifies reset on Reload and invariance to receiver bandwidth.

At 48 kHz, 0.2-amplitude clean tones gave means0.002618 at200Hz and0.026016
at2000Hz. Noise alone with sigma0.01 gave0.007980. Adding sparse positive0.8
impulses to the noisy200Hz tone raised the mean from0.009534 to0.036550.
These cases show slope and impulsive contamination explicitly; the trial keeps
the proposed arithmetic mean intact so listening and statistics can test it.
The exact results are in `docs/evidence/crossing-floor-tests.json`.

## Full-band listening result

The corrected full-band render has mean crossing-neighbor amplitude0.0133426
and converted floor0.0652375 across the evaluated source blocks. It selects
12.2475% and12.1975% of the **entire Nyquist-band** raster in the two passes.
These percentages use2048 frequency rows and must not be compared directly with
the receive-band-restricted legacy fractions. The previously reported roughly
86% came from the first internal, still-cropped experiment.

The30-second output has no clipping or engine faults. Original, baseline,
registered and both existing harmonic comparison WAVs are byte-identical to the
preceding demo. The new full-band guide costs more: the Mini render took53.58s
including diagnostic capture/compression, with a worst audio block296.40ms
versus its170.67ms deadline. This is a listening experiment awaiting native
optimization, not a live-real-time deployment. The same five focused tests pass
on the Mini and in the Windows DLL under Wine; receiver bandwidth invariance is
bit-exact. Evidence: `crossing-fullband-tests.json`, `crossing-fullband-wine.json`,
`crossing-fullband-demo.json` and `crossing-browser.json` under `docs/evidence`.
