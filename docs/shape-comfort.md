# Cleanup distribution weighting and centroid comfort noise

This trial follows the accepted occupancy-plus-excitation comparison. The old
recordings stay frozen. DSP policy remains in `Filter.py`; native additions only
publish the current-distribution score for inspection.

## What the previous occupancy trial omitted

The Downloads source `reference/legacy-plugin/CleanupNative/CleanupNative.h`
contains three relevant uses of its distribution statistic:

1. `fast_entropy` sorts magnitudes from three neighboring spectra and compares
   their order statistics with a logistic quantile curve using Pearson
   correlation. Its score is `e = 1 - correlation`. It is a shape comparison,
   despite the name entropy; it is not Shannon entropy or a calibrated probability.
2. `fast_peaks` skips rows without sufficient smoothed/raw score. For admitted
   rows it scales contextual MAN/ATD by `f = 1 - e / MAXIMUM`, then blends each
   scaled statistic with the corresponding current-row statistic using
   `exp(-0.5 * abs(context - row))`. The blended pair determines the peak threshold.
3. With squelch off, a score-dependent residual gain can remain after recursive
   smoothing. The second peak pass also intentionally retains its in-place
   residual values at cells that are not set to one.

Thus the statistic controls seed admission, the threshold mixture, and residual
support. Occupancy alone omitted those connections. The source's comment about
"higher noise" is ambiguous; the implemented equations above are the reference.
Neither the logistic reference nor the amplitude-dependent exponential weight
has become a universal physical noise law by being retained here.

## Local counterpart for unknown passband

Enable `OCCUPANCY_SHAPE = True` with `REGISTERED_FLOOR = "occupancy_surface"`.

The current magnitude observations are divided by their bounded mean curve
before the shape comparison. This removes local receiver-envelope scale without
requiring a supplied low-pass cutoff. Neighborhoods span ±600 Hz and three
actual registered guide centers. Scores are evaluated at frequency centers
16 guide bins apart, then interpolated; this does not create additional evidence.
Reflection handles frequency edges. Exactly absent observations remain zero.

The low-pass stopband therefore cannot dominate a distant in-band neighborhood.
An abrupt transition inside a neighborhood can still influence its score;
local equalization is an experimental treatment, not proof of complete receiver
invariance. Changes to that local curve can change the distribution. There is no
quiet-frame training, temporal minimum, or requirement that the source stop.

For the threshold blend, the contextual pair is the tracked low reference `L`
and residual RMS `T-L` from occupancy. The current-row pair uses a lower-tail
mean and RMS distance in ±150 Hz neighborhoods. The first pass uses bounded
observations for that fit; the second uses its masked/smoothed input. We retain
Cleanup's exponential row/context blend and its raw/smoothed seed-admission
fallback. The final trial softens that binary admission: with
`x = clip(max(raw, smoothed)/0.057, 0, 1)`, seed strength is
`x*x*(3-2*x)`. Zero departure still provides no seed, while a departure at or
above the original knee provides full strength. `SHAPE_SOFT_ADMISSION=False`
reproduces the hard-gate ablation. This is a continuous relaxation, not a
probability calibration. The selected statistic is weighted seed mass in this
mode. The fixed 0.057 threshold is a **baseline experiment constant**, not
validated calibration for this registered, locally equalized distribution.

With squelch off, weak residual support has the analogous normalized form
`L / context_peak * smoothed_shape / maximum_shape` in low-score cells.
Whole-frame count/contiguous-interval squelch rules are unchanged.

The diagnostic decision contour is the first-pass blended threshold. The
reference field now shows the unweighted occupancy contour for this variant;
other variants retain their previous interpretation. The new shape field shows
raw `1-correlation` on a 0–0.2 display range; larger means greater departure from
the reference shape. Mean, variance, low reference and occupancy remain visible.

## Comfort noise between energy peaks

Enable `CENTROID_COMFORT_GAIN = 0.035` with `HARMONIC_MODEL = "excitation"`.
The default zero preserves the reference experiment.

Small complex aperiodic values are placed directly on upper synthesis bins.
Their amplitude follows the measured upper envelope, a falling frequency taper,
retained lower-band energy, and the square root of the existing centroid erosion
field. They fill between partials as well as within the temporal dip. The field
is zero outside supported erosion regions, and zero retained lower content means
no addition. Missing pitch does not independently veto this noise contribution;
the existing contextual energy-peak evidence supplies its support.

Random values are deterministic functions of absolute sample time and bin, with
independent circular complex Gaussian values at each hop center. They do not copy the donor's formant spectrum or
share the periodic components' harmonic phase. This is shaped comfort noise,
not a claim of recovered excitation, calibrated blue noise or a measured upper
speech band. The existing harmonic on/off ramp also controls this contribution.

The joint addition remains bounded: periodic amplitude may use 90% of the
original addition norm budget and comfort noise at most 10%. By the triangle
inequality, their sum remains within the original 12% lower-band power budget.
Noise alone can use at most 0.12% of retained lower-band coefficient power.
This is an upper bound; the chosen envelope gain normally yields less.
Protected lower coefficients are untouched.

## Reproduction and comparison

```
/Users/ultimussecundai/.local/bin/m4build -- sh tools/verify-shape-mini.sh
/Users/ultimussecundai/.local/bin/m4build -- sh tools/render-shape-mini.sh
```

The first command checks locality, persistent-line evidence, score scale
invariance, the exact row/context weight expression, second-pass aliasing,
comfort-noise locality, determinism, ramp and power bound; it also runs the
previous occupancy/excitation checks. The second evaluates the same known
clean/noise fixtures as the prior trial, then adds three listening variants:

- **Shape-weighted contour:** new distribution weighting, harmonics off.
- **Occupancy + comfort:** accepted occupancy/excitation plus centroid noise.
- **Shape + comfort:** both changes with excitation.

Compare occupancy-plus-excitation with occupancy-plus-comfort to isolate the
noise contribution. Compare occupancy contour with shape-weighted contour to
isolate distribution weighting. A second comparison of the two comfort variants
keeps the addon consistent while changing the denoiser.

## Hard-gate ablation

The direct hard-admission trial recovered substantial noise rejection but also
increased waveform distortion. At 3 kHz on the synthetic voiced fixture,
occupancy gave −8.02 dB clean / −10.10 dB noise, while hard shape weighting gave
−9.91 dB clean / −17.54 dB noise. Its output/input clean correlation after removal
of overall gain fell from 0.727 to 0.655 (baseline 0.877). Thus the loss was not
merely uniform attenuation. The 1.2 kHz weak chirp was almost removed
(−44.21 dB clean), a clear reason not to promote the literal hard cutoff.

`docs/evidence/shape-pairs.json` preserves that ablation; the softened comparison
uses the same fixture generator, seed and physical LPFs. These labels are only
available to evaluation, never to the estimator. The aborted hard-gate combined
render was superseded by the soft-admission combination before publication.

For gain-independent inspection, let P be output/input clean energy and D the
recorded normalized squared error. The least-squares scalar gain is
`g = (1 + P - D)/2`; remaining error after gain compensation is `P/g² - 1`.
This separates a level reduction from loss/change of waveform shape; it is still
not a natural-speech quality or intelligibility score.

The final soft trial is rendered with
`/Users/ultimussecundai/.local/bin/m4build -- sh tools/render-shape-soft-mini.sh`.
Its source keeps the accepted occupancy/excitation render as the initial browser
selection. The additional frequency-neighborhood sort was changed to a sliding
sorted window; independent threshold/aliasing tests pass after that optimization.

## Soft-admission result

On the 3 kHz synthetic voiced fixture:

| Method | Clean energy change | Noise energy change | SNR improvement |
| --- | ---: | ---: | ---: |
| Baseline | −3.86 dB | −9.60 dB | 5.74 dB |
| Occupancy | −8.02 dB | −10.10 dB | 2.08 dB |
| Hard shape trial | −9.91 dB | −17.54 dB | 7.64 dB |
| Soft shape trial | −8.12 dB | −12.74 dB | 4.62 dB |

Soft shape weighting adds about 2.6 dB noise rejection at nearly unchanged clean
energy relative to occupancy. Its gain-independent clean correlation is 0.734
versus occupancy's 0.727; baseline remains higher at 0.877. On continuous FM at
3 kHz, SNR improvement rises from 2.31 to 4.54 dB (baseline 6.40 dB). The weak
chirp at 1.2 kHz is retained at −5.90 dB clean rather than the hard trial's
−44.21 dB. These are encouraging measured changes on controlled synthetic
fixtures, not a claim of baseline equivalence or natural-speech quality.

Reports: `docs/evidence/shape-soft-pairs.json` and
`docs/evidence/shape-pairs-gain-analysis.json`. The latter is reproducible with
`research/evaluation/shape_gain_analysis.py` and the saved pair reports.

## Remaining physical footprint difference

The native recursive smoother still averages three neighboring frequency bins
per frequency branch, while the time branch remains thirteen columns on the
restored 128-sample mask lattice. At 48 kHz, baseline FFT512 frequency spacing is
93.75 Hz and registered guide spacing is about 11.724 Hz. The same three-bin
frequency kernel is therefore about eight times narrower in hertz on the guide.
This documented difference is a concrete candidate for a later isolated test of
mask connectivity and signal retention; the current trial does not change the
native recurrence or assume that widening it will improve listening quality.

## Published demo verification

All three final renders completed with zero runtime faults and zero clipped
samples. Worst blocks were approximately 524–560 ms for 171 ms of audio, so these
remain listening renders rather than real-time-ready configurations.

The comfort-only comparison has byte-identical registered analysis, all floor
statistics, denoising mask, and protected lower-band magnitude publications to
the accepted occupancy/excitation version. The separate entity checks establish
exact protection of lower complex coefficients. Its WAV difference has RMS
about −73.74 dBFS across the 30-second recording; this is a difference level,
not a perceptual loudness score. All ten previous WAVs remain byte-identical.

Browser verification covers all thirteen comparison buttons, 71 spectral panel
references, diagnostic-field selection, playback switching, pause and cursor
preservation. At 2.7 seconds the soft shape result visually has more connected
tracks and fewer large isolated bright mask regions than occupancy alone. Fine
upper-band striations remain visible. That visual observation does not establish
that metallic coloration has been removed.

Evidence is in `docs/evidence/shape-demo-audit.json`,
`shape-local-assets.json`, `shape-comfort-browser.json`, `shape-soft-tests.log`
and the pair reports. Screenshots are retained in `build/listening-demo-v15/`.
