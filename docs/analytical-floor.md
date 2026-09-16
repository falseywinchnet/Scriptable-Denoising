# Unknown-passband floor: analytical investigation

**Subsequent user direction:** implement a bounded time–frequency mean/variance
surface, with variance penalizing the mean. That approved experiment is now
implemented; see [variance-surface.md](variance-surface.md). The population-only
proposal below records the preceding discussion, not the final implementation.

Status: research helpers and controlled measurements, **not a new live filter**.
The accepted browser demo and FilterBank selection remain unchanged. Candidate
DSP is in `Filter.py`; `research/floor_analysis/run.py` constructs test inputs,
invokes the native registered guide, and records measurements and figures.

## Current direction after the user correction

The user explicitly rejected developing an intelligent noise/not-noise or
signal-presence classifier as the floor estimator. Continuous signals and FM
are first-class inputs. Do not promote the temporal quantile / noise-null tail
classifier below into FilterBank, even with the frequency-neighborhood patch.
Do not require speech pauses, quiet frames, or sparse occupancy.

The relevant distinction is **which spectral population supplies the statistics**,
not whether the source is currently transmitting. The immediate experiment should
repair the contamination of Cleanup's empirical distribution by the filtered-out
region, then examine its floor/peak fitting on the registered representation.
A persistent component remains part of the represented signal; persistence alone
must not turn it into an estimated noise floor.

### The population error can be written exactly

In the idealized flat passband / zero stopband case, let rho be the fraction of
frequency cells inside the passed region. If F_in is the in-band amplitude CDF,
then, for nonnegative amplitudes,

    F_all(a) = (1-rho) + rho F_in(a).

There is a point mass of 1-rho at zero. For quantile probability p:

    Q_all(p) = 0                         when p <= 1-rho
             = Q_in((p-1+rho)/rho)      otherwise.

For a 3 kHz occupied region in a 24 kHz available span, rho=1/8: 87.5% of the
population is stopband. A full-span median or low quantile measures that mass,
not the distribution that produced the audible background. Real taper, leakage
and registration replace the zero mass with many small nonzero numbers. Ignoring
exact zeros does not fix this.

Even if a reference m were held fixed, a pooled second moment changes to

    E_all[(A-m)^2] = rho E_in[(A-m)^2] + (1-rho)m^2.

Cleanup's MAN itself also changes, so both location/spread terms are affected.
This explains the dependence on bandwidth without making a speech/noise decision.

### Separate support estimation from the floor statistic

Estimate the observed spectral support from the frequency envelope and broad
rolloffs. Time aggregation may stabilize that envelope, but **use it only to
locate/weight the represented frequency domain**, not to declare persistent
energy noise or divide it away. A steady tone, a continuous FM signal and speech
all contribute to this support estimate.

A possible soft population statistic is

    F_w(a) = sum[t,f] w(f) * 1[A(t,f) <= a] / sum[t,f] w(f),

where w describes membership in the observed spectral domain, not probability of
noise or speech. From that empirical distribution, examine robust lower-envelope,
location and spread estimates, then Cleanup-like peaks, skirts and recursion.
Appending bins with w=0 leaves every statistic exactly unchanged. The weights
need not multiply the audio or hard-cut its spectrum; population selection and
audio gain are separate operations.

A shape fit could use broad frequency-envelope plateaus and rolloffs, with full
band permitted as a valid result. It must preserve internal valleys and narrow
ridges rather than equating every local dip with an excluded band. It must also
avoid using a fraction of the loudest peak as the definition of support, because
that would make a strong carrier suppress weaker reception. The fitting method
is **not yet implemented or validated**. A universal physical LPF estimate is
not identifiable from arbitrary content: source bandwidth and filter bandwidth
can produce the same observation. Observed statistical support is the narrower
quantity needed here.

### Next experiments, without the rejected classifier

1. Compare original known-band Cleanup statistics against the same data padded
   with increasingly large filtered-out spans. The in-band data are identical.
2. Compare hard oracle support (evaluation only) with an estimated soft support;
   report how each changes MAN, ATD and the full recursive mask.
3. Use uninterrupted tones, multi-tones, continuous FM and broadband content,
   not only intermittent speech-like fixtures. Known noise additions are for
   evaluation; the estimator does not get their labels or require absent signal.
4. Vary actual LPF width, transition slope and input level. Keep separate the
   support-fitting error and the registered distribution / peak-fitting error.
5. Examine the actual analysis, inferred support, fitted floor and final mask.
   Do not use a noise-only false-alarm target as the denoiser's governing rule.

The archived experiment shows why temporal normalization is a poor match to this
request. Its continuous-tone failure is evidence for rejecting that assumption,
not an invitation to build a more elaborate classifier around it.

## Archived experiment: local temporal noise normalization

The following records the tested and rejected direction. It remains useful as
a failure analysis, not an integration plan.

### 1. The nuisance variable is the receive-filter response

For noise upstream of a common linear receiver filter, the local Fourier model is

    Y(f,t) ≈ H(f) [S(f,t) + N(f,t)].

In a signal-free region, its power scale is |H(f)|² times the input noise power.
Pooling magnitudes across frequency confounds that response with noise statistics.
Adding more stopband bins then changes the population even though the useful
signal and its in-band noise did not change.

A useful invariant is instead a ratio at each frequency:

    R(f,t) = A(f,t) / Q(f),
    Q(f) = temporal quantile of A at frequency f.

For a positive frequency-dependent scaling h(f), Q(hA)=hQ(A), so R is exactly
unchanged. Appending zero frequency rows also leaves the existing ratios unchanged.
This is a mathematical property of the statistic, not proof that the registered
transform commutes with an arbitrary LPF. The latter contains finite windows,
registration, absolute values and a maximum, and mixes nearby frequencies. A
smooth local transfer approximation needs testing; sharp transitions particularly
do. The guarded spectral correction below additionally assumes local smoothness
and is not invariant to arbitrary pointwise frequency scaling.

This approach does not need the passband width as an input. It does need an
identifiable **local background**. A quantile is a scale reference; it is not
already a mean noise amplitude or a noise power spectral density. The actual
noise distribution determines that conversion. Noise introduced after the LPF
also breaks the simple common-H cancellation model, although a local post-filter
background estimator can still be useful.

## 2. Registration changes the null distribution

Ordinary complex Gaussian Fourier coefficients have a familiar magnitude/power
law under the usual assumptions. Our guide uses seven correlated absolute cosine
observations, data-dependent registration and a maximum. Neither a Rayleigh law
nor a maximum of seven independent samples is automatically its law.

The experiment calibrates the complete self-normalized statistic on the actual
native pipeline, including estimation of Q from the same bounded context. It
uses separate white-noise contexts for calibration and held-out evaluation.
Thresholding at the measured 99th percentile is an explicit experimental 1%
cell false-alarm target, not a theoretically optimal listening setting.

The provisional scale is the temporal 20th percentile over 42 complete native
columns in a 48-column context. The 16 center columns are scored. Time
interpolation happens only afterward. The optional evidence-pooling experiment
uses Cleanup's exact native padded, paired-branch recurrence (three rounds,
time kernel 13, frequency kernel 3), with its own separately calibrated threshold.
Pooling evidence before a decision is distinct from smoothing a binary decision
mask; this experiment compares evidence scores and does not replace the live mask.

## 3. Persistent signal poisons a temporal-only floor

Let the temporal amplitude distribution at one frequency be

    G(a) = pi0 F0(a/scale) + (1-pi0) F1(a),

where pi0 is the fraction without signal and F1 is the distribution during signal.
If F1 contributes little in the lower tail, then

    Qq(G) ≈ scale * inverse_F0(q/pi0).

Even the lower quantile rises as signal-free observations become scarce. If a
continuous ridge occupies the whole context, time alone cannot identify the
background beneath it. The test's continuous 900 Hz tone demonstrates this.
This is an identifiability issue, not something a different universal uplift
constant can repair. Stationary narrowband interference and a wanted steady tone
can likewise be statistically indistinguishable without further assumptions.

The second experiment obtains a reference from nearby frequencies:

    L = median temporal floor in a guarded left flank
    U = median temporal floor in a guarded right flank
    Qjoint = min(Qtemporal, max(L, U)).

The four-bin guard excludes the ridge neighborhood. Each flank contains 16 bins
(~188 Hz at this geometry). Taking the greater flank prevents a single dark
stopband flank from dragging the estimate down at an LPF edge. Missing or zero
reference flanks cause the spectral correction to abstain. This is inspired by
CFAR reference/guard-cell reasoning, not an implementation of a published CFAR
algorithm. The union with the temporal reference is calibrated as a whole.

It is a hypothesis with a clear assumption: background changes smoothly over the
reference span, and signal does not occupy both flanks. Dense harmonic combs,
wide sustained formants, narrow noise humps and notches need additional testing.
The guard/reference span is provisional and must track physical resolution when
the aperture changes; it is not an extra user slider.

## 4. A large ratio is not automatically usable evidence

Inspection of the actual Dave/Simon guide shows a dense background bed up to
approximately 3 kHz and a sharp rolloff. Local normalization reveals speech
ridges, but also creates large event patches above the rolloff, where tiny
magnitudes fluctuate relative to tiny baselines. These are visible in the
recording diagnostic. They must not silently count as credible detected signal.

Clamping a denominator with an arbitrary epsilon does not identify a passband.
Nor does a fixed fraction of the largest spectral peak: a loud carrier would
then make faint real reception disappear. A better next model needs an explicit
confidence in the local background reference, with separate components for:

- a persistent filtered-noise envelope;
- leakage or transform-induced spreading from adjacent strong content;
- a numerical/acquisition floor when that floor is measurable.

The effective reference must account for those components in the same guide
units. They are not necessarily additive in this nonlinear magnitude guide.
An ordinary-STFT window-sidelobe formula cannot simply be pasted onto registered
magnitudes; use actual native responses to known signals to establish or reject
an appropriate leakage envelope. Registration can be data-dependent, so even a
precomputed envelope needs validation over contexts.

An exact zero has no estimable local noise law. Extremely attenuated but nonzero
upstream noise, however, is not automatically invalid merely because it is small.
An ideal invariant should still work there. Confidence should express failure of
the assumed background model or measurement resolution, not decree an arbitrary
bandwidth. Regions with uncertain references should not independently open the
whole-frame squelch. Their audio fallback remains a separate decision.

## 5. Superseded integration proposal: do not implement

Before the user correction, the proposed direction was local, registered-noise-calibrated
**evidence**, followed by Cleanup's peak regions, soft skirts and recursive mask.
It is not to feed a binary ratio mask directly into synthesis. Binary tail tests
alone are sparse and do not solve musical noise or faint-signal recovery.

Any eventual integration should preserve diagnostics for local scale, reference
confidence, evidence, both mask passes, final guide gain and the gain projected
onto synthesis coefficients. Residual-stage statistics need calibration under
the first-stage selection: that residual is a censored population, not a fresh
unconditioned noise realization.

Whole-frame admission must preserve the user's two separate questions:

1. How many of the eligible intervals contained meaningful evidence?
2. What was the longest contiguous run of such intervals?

The historical >22 of 129 eligible intervals and >16 contiguous intervals came
from the original RFFT512 path. Changing resolution changes the number of
possible intervals and how an event occupies them. Registered columns overlap;
interpolating 48 columns to 192 does not create 192 independent observations.
Measure the joint distribution of total count and longest run under noise and
known events at each geometry, including short chirps and boundary crossings.
Do not substitute a sample-rate/hop ratio or an independent Bernoulli model.
The live reference admission path has not been changed by this investigation.

## 6. Experimental scope and remaining failures

Known additive pairs use the same upstream noise and clean waveform, both passed
through the hidden filter. The estimator never receives cutoff labels. Scoring
uses those labels to report in-band interiors and transitions separately; it
never divides an accepted-bin count by all 24 kHz and calls that noise rejection.
The fixtures include Butterworth cutoffs from 500 Hz through 12 kHz, full band,
a sharp 3 kHz cutoff, colored noise, impulses, an abrupt noise-level increase,
intermittent and continuous tones, a weak tone, and a brief chirp.

This is a fixed 48 kHz / guide aperture 2048 / hop 512 experiment. It does not yet
establish invariance across transform configurations, arbitrary noise laws,
streaming history, receiver AGC, all-pass phase responses or real speech mixtures.
It is bounded-context estimation with future context already available to this
engine, not a newly validated causal noise tracker. No denoised audio or speech
quality score is claimed. Signal-region seed coverage is a fraction of selected
cells in a prescribed region, not a probability of detection or intelligibility.

The lower-amplitude tone remains an important failed sensitivity case. Impulses
and rising broadband noise also exceed the stationary-noise calibration. They
need different evidence and adaptation rules, rather than raising every speech
threshold. A 1% cell test is not a 1% frame-open test; wide-band noise often has at
least one selected cell somewhere.

## 7. Relevant foundations

- [Martin, 2001: optimal smoothing and minimum statistics](https://www.csd.uoc.gr/~hy578/2005/projects/ieee_sp_tsap_2001009_05jul_0504mart.pdf): per-frequency noise estimation, bias and smoothing matter; minimum tracking relies on opportunities to observe the background.
- [Cohen and Berdugo, 2001: minima controlled recursive averaging](https://www.isca-archive.org/hsc_2001/cohen01_hsc.html): local presence evidence controls noise adaptation.
- [Gerkmann and Hendriks, 2012: MMSE noise PSD estimation](https://www.inf.uni-hamburg.de/en/inst/ab/sp/publications/taslp12noisepsd.html): a further reference for explicit speech-presence modeling. The author summary was reviewed, not the complete paper.
- [Rohling, 1983: CFAR in clutter and multiple targets](https://ieeexplore.ieee.org/document/4102829/): relevant reference-cell and nonhomogeneous-background reasoning. The present guarded estimator is our experimental construction.

These methods motivate assumptions and tests; their FFT distribution formulas
are not asserted to apply unchanged to the registered guide.

## Reproduction

From the authoritative local checkout:

```
/Users/ultimussecundai/.local/bin/m4build -- /Users/joshuahkuttenkuler/miniforge3/bin/python research/floor_analysis/run.py --out build/floor-analysis-v3
```

The native `build/libCleanupNative.dylib` must already exist on the selected Mini.
Retrieve results from the printed mirror path before another sync. All candidate
helpers have explicit Numba signatures; the driver runs the compiler's syntax
policy audit and verifies their nopython specializations. It also checks exact
frequency-scale invariance of the temporal ratio, appended-zero independence,
zero-input abstention and native global-gain equivariance. Output JSON records
Filter, driver and native-library hashes, geometry, calibration and held-out
measurements. Figures depict diagnostic seeds, not synthesized output.

## Retained negative-control measurements (v3)

Evidence: `docs/evidence/rejected-temporal-floor.json`; eight calibration contexts
and twelve held-out contexts per case, all using the actual native registered
transform. The measurement loop took 50.07 seconds on the Mini, excluding the
initial Numba compilation and plotting.

- Temporal ratio invariance error under positive frequency scaling: 7.11e-15.
  Appending zero bins changed existing ratios by exactly zero.
- Continuous tone: temporal floor bias at 900 Hz ranged from +29.50 to +32.39 dB.
  The frequency-flank patch reduced that range to -1.48 through +2.40 dB. This
  repairs one narrow-ridge fixture, not the general continuous-content assumption.
- Weak-tone target-region seed coverage was 0–5% temporally and 0–6.67% with the
  flank patch. These are coverage fractions, not detection probabilities.
- With the flank patch, impulses selected 5.77% of in-band cells and a rising
  broadband-noise step selected 18.23%, versus a nominal 1% stationary cell target.
- In the actual recording diagnostic, 34.52% of cells from 3.1–4 kHz were selected
  despite mean squared guide magnitude there being approximately 63.3 dB below
  that of 250–2800 Hz. This is squared **guide magnitude**, not an acoustic power
  measurement. The plotted stopband patches are real failures of that ratio test.

These outcomes justify retaining the experiment as a negative control and
following the corrected statistical-population direction above. No experimental
binary seed map was used to render or replace the accepted listening demo.
