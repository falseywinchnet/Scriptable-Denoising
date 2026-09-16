# Low-reference occupancy contour and excitation continuation

Two independent experiments following the bounded mean/variance surface. They
retain prior listening comparisons and introduce no plugin sliders. All DSP
policy remains in the one strict-Numba `Filter.py`.

## 1. What occupancy means here

The previous candidate was `F = mu / (1 + v/mu²)`. Algebraically, this is
`mu * rho0`, with `rho0 = mu²/(mu²+v)`. This is a moment-based effective occupancy:
for a two-level field that is zero outside an occupied fraction rho and has one
constant amplitude inside it, rho0 equals that fraction. For general magnitudes,
it is a descriptive moment ratio, **not a noise/speech probability**. Lower rho0
lowers the previous decision threshold, which helps explain its permissiveness.

The new experiment obtains a low reference L from the mean of the lowest 10% of
the bounded observations in each ±150 Hz frequency neighborhood, with fractional
weight at the last order statistic. It then tracks L over time with the same
40 ms time constant and a bounded innovation. Thus it is a time/frequency
reference, not a search for silent or signal-free frames. A fixed *fraction* of
the neighborhood avoids defining the reference as the single minimum, whose
behavior changes with the number of available cells. The choice 10% is a
provisional experiment constant, not a derived universal optimum.

Let `d = mu-L`. The residual second moment and effective occupancy are

    q = v + d² = E[(A-L)²],
    rho = d²/q.

The identity for q is exact for the same moments and reference; our actual mu/v
are bounded tracked estimates, so they are not unbiased sample estimates of a
physical noise distribution. The two-level interpretation of rho is likewise a
surrogate for arbitrary data, not a claim that amplitudes literally have two
states. Values below L can occur.

Use the candidate decision contour

    T = L + sqrt(q) = L + d/sqrt(rho),

with the first expression used numerically so zero rho needs no division.
Baseline Cleanup's row threshold has the same **reference + RMS distance from
reference** form (`MAN + sqrt(mean((A-MAN)²))`, before its other weighting).
We replace that reference and its statistical domain; we do not claim equality
to the entire baseline mask or optimize a multiplier to this recording.

The final contour movement is bounded and checkpointed separately from L. Both
passes reuse this contour measured before mask insertion. The previous penalized
mean remains published as a comparison. The native diagnostic reports L, rho,
mu, variance, T and the final mask. Whole-frame total/contiguous interval rules
are unchanged; local amplitude occupancy is not the number of independent time
columns or a new admission rule.

Algorithm selection: `REGISTERED_FLOOR = "occupancy_surface"`.

## 2. Harmonics: measured spacing, excitation and envelope

Algorithm selection: `HARMONIC_MODEL = "excitation"`, with registered analysis
and the existing Harmonics toggle. The previous `"packet"` method remains the
reference. This addon is independent of denoising and has its own periodicity
evidence; it does not turn that evidence into a noise classifier for the floor.

### Shared period hypothesis

Search 70–360 Hz in 2 Hz steps on the registered guide. Several lower members
must be supported. Expected harmonic positions are compared with the gaps half
an order away. Those gaps penalize octave-doubled proposals, while averaging all
expected members penalizes subharmonic proposals supported only on every second
member. A modest continuity term discourages gratuitous jumps. The confidence is
an uncalibrated comb score, not a voiced-speech probability. Single isolated
lines do not suffice. This is not a general-purpose pitch estimator or proof
that every accepted periodic source is speech.

### Separate envelope from excitation

The lower complex STFT magnitudes provide a smooth log-amplitude envelope.
Measured partial amplitudes divided by that envelope estimate excitation strength.
A separately fitted high-band envelope continues the upper observed envelope
with a bounded log-frequency slope, plus decreasing harmonic-order weights.
The old donor's complete formant-shaped spectrum is not tiled upward. The
extrapolated envelope is an explicit hypothesis: new high formant resonances
cannot be recovered from absent observations by this fit alone.

### Shared evolving phase and fractional placement

Lower complex coefficients anchor a common evolving source phase and their
individual residual phases. The source phase advances from successive period
estimates, with bounded correction from measured lower components. Each upper
order uses `order * source_phase + continued_residual`. A common timing variation
therefore has a larger phase effect at higher orders. The residual continuation
uses a circular summary of observed residuals; it is not a measured missing
high-band phase or a reimplementation of Mowlaee's algorithm.

Components at noninteger STFT-bin frequencies are distributed using the exact
finite symmetric-Hann response, including its centered-frame phase. The
coefficient mixture is transported through the native E=1 DIP packet. E=1 allows
arbitrary source/target bins; it is not presented as vocal pitch. As with the
previous single-component construction, this transport is algebraically a
coefficient mapping, not additional recovered information. No Python FFT was
introduced; native bfft retains analysis and inversion.

The existing soft erosion between temporal energy peaks remains. Its energy is
now shared across the periodic family rather than assigning speech pitch to the
fixed 750 Hz residue lattice. A bounded aperiodic texture grows as measured
periodicity falls; it is deterministic in absolute time. With insufficient comb
support the addon abstains. This first version therefore does not claim to
reconstruct fricatives or fully unvoiced high-band content.

All lower protected coefficients remain exact. Existing upper content reduces
addition, and the original 12% addition-power budget remains. The upper-frequency
limit bounds component centers with a soft taper; each component's finite-window
spectral skirt extends a few synthesis bins around its center.

## 3. Evidence and comparisons

`tests/test_occupancy_excitation.py` checks the independent lower-tail calculation,
the moment/RMS identity, known two-level occupancy, amplitude scaling, exact
fractional Hann response, several known pitches, single-line rejection, disabled
identity, protected lower coefficients and the addition-power bound. Existing
packet/phase/erosion tests remain separate regression checks.

`research/evaluation/compare_occupancy.py` constructs known clean and noise pairs.
The native engine receives identical mixture analysis on two channels and applies
that same adaptive mask separately to the known clean and known noise content.
This measures retained clean energy and residual noise **without rerunning the
estimator on clean/noise separately**. Tests include uninterrupted synthetic
voiced harmonics, continuous FM and a weak short chirp at hidden physical LPF
cutoffs 1.2, 3 and 6 kHz. Baseline receives its known bandwidth; the new estimators
do not use that setting. These fixtures are not clean natural-speech references
or intelligibility tests, and their clean/noise labels are unavailable to DSP.

The listening demo adds three independent comparisons:

- **Occupancy contour:** new denoising contour, harmonics off.
- **Surface + excitation:** previous surface, new harmonic addon.
- **Occupancy + excitation:** both experiments together.

The previous surface and previous harmonic versions remain available unchanged.
Mean, low reference, effective occupancy, former penalized mean, decision contour,
final mask and output are inspectable. Pitch/comb diagnostics are block summaries,
not ground truth. More attenuation alone is not improvement; evaluate speech
loss, weak events and metallic coloration as well as background reduction.

## Relevant primary research

[Turan and Erzin, INTERSPEECH 2015](https://www.isca-archive.org/interspeech_2015/turan15_interspeech.pdf)
separates excitation and envelope in bandwidth extension and discusses overstrong
upper harmonics, joins and the increasing effect of pitch variation at higher
orders. That informs this experiment; their SOLA/HMM system is not implemented.

[Watanabe and Mowlaee, INTERSPEECH 2013](https://www.isca-archive.org/interspeech_2013/watanabe13_interspeech.pdf)
is relevant to sinusoidal anchors and partial phase reconstruction. The present
common-phase tracker is our bounded construction, not their published estimator.
See `harmonic-reconstruction-research.md` for the earlier research and proposal.

## Reproduction

```
/Users/ultimussecundai/.local/bin/m4build -- sh tools/verify-next-mini.sh
/Users/ultimussecundai/.local/bin/m4build -- sh tools/render-next-mini.sh
```

The latter retains the Mini's v6 demo, renders occupancy into v7, surface plus
excitation into v8, and both into v9. Retrieve evidence before another sync.

## First paired-signal result: useful but not an overall win

At the 3 kHz physical cutoff, synthetic voiced content yielded:

| Method | Wanted-signal energy change | Noise energy change |
|---|---:|---:|
| Baseline Cleanup | -3.86 dB | -9.60 dB |
| Previous variance surface | -5.46 dB | -5.29 dB |
| Occupancy contour | -8.02 dB | -10.10 dB |

The new contour gets close to baseline's noise reduction but loses more wanted
energy. Thus it is a comparison experiment, not a demonstrated overall upgrade.
On the weak chirp at the same cutoff, retained clean energy changes were -47.27,
-2.92 and -4.52 dB respectively: matching baseline wholesale would also reproduce
its failure on this short weak event. The effective-occupancy interpretation is
analytically useful, but these results leave the contour's signal-retention
tradeoff unresolved. No coefficient was tuned against these held-out scenes.

## Demo verification and limits

The three 30-second native renders completed with zero runtime faults and zero
clipped samples. Their worst blocks took about 455–469 ms for a 171 ms audio
budget: these full-band experiments are not yet ready for live processing.
The Windows native DLL also cross-builds; that is not a new SDR# playback test.

Browser checks cover all ten comparisons, all 42 panel references, field
selection, playback switching, pause, and preserved cursor position. The seven
previous WAVs are byte-identical. Five new mathematical/entity tests and seven
existing packet/phase/erosion regression tests pass. The final root script also
resets stale source phase when squelch has cleared the harmonic mix; the frozen
demo scripts precede that reset repair and run with squelch off.

Visual inspection at 2.7 seconds shows more background removal with occupancy,
but sharply separated mask islands and tracks remain. The new harmonic output
joins the occupied band without the previous wide gap and decays in the upper
band, while fine vertical striations remain visible. This is not evidence that
the metallic coloration has been eliminated. The addon reports some accepted
frames in 176 of 178 context blocks, with mean block comb scores around 0.259;
that score is uncalibrated, and block activity does not prove correct pitch or
voicing. Its behavior on this recording remains a listening experiment.
