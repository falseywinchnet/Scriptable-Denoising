# Cleanup: measured baseline, then optional harmonic extension

This is a research plan, not a claim that stage-two DSP has already shipped.
The original Cleanup algorithm remains the comparison baseline. Preserve its
recursive mask processing while changing and testing one component at a time.

## What the current displays mean

The offline `probe.npz` replay uses registered sub-hop inverse magnitudes as the
analysis panel. The lower panel is the magnitude of the ordinary complex STFT
multiplied by an interpolated provisional MAN/ATD gain, with the offline speech
mask providing a block gate. It is genuinely masked, but is not yet the complete
recursive Cleanup mask evaluated on the enhanced analysis field. The live
parent's default diagnostic guide remains its ordinary STFT unless Filter.py
publishes a custom guide matrix. Do not label that ordinary STFT superresolution.

The intended baseline is:

1. Registered sub-hop enhanced analysis; retain the full enhanced field as a
   comparison reference and cover the entire receive band.
2. Noise/structure evidence and Cleanup's two passes plus recursive mask smoothing
   in that analysis geometry, with explicit amplitude and window normalization.
3. Transfer the bounded gain to synthesis bin centers and frame times, preserving
   the noisy complex synthesis coefficients' phase. Compare point interpolation
   with weighted aggregation over each synthesis bin's actual analysis response;
   a coarse bin integrates a frequency neighborhood.
4. ODFT/DFT synthesis using bfft's tested analysis/canonical-dual window pair.
5. The whole-overlap gate retains total qualifying intervals and longest contiguous
   interval as separate statistics, with their own counting regions. Calibrate
   against available intervals and the representation of the same event at each
   geometry. More registered observations do not create independent evidence.

## What is principled in the current algorithm

These are useful structural priors even without a single fitted generative model:

- The sorted-spectrum correlation score distinguishes distribution shapes without
  depending on absolute level. It is a structure score, not Shannon entropy or
  an already calibrated probability of speech.
- Combining global and per-column evidence balances shared background estimates
  against local structure.
- Repeated peak selection and refinement impose support constraints.
- The repeated time/frequency smoothing spreads support continuously around peaks.
  This is a discrete regularization of the gain surface: the user's "tent."
- Sustained or sufficiently populated evidence controls a whole contextual block.
- Preserving complex phase and using a dual synthesis window gives a controlled
  reconstruction path. The window guarantees do not prove a modified STFT is
  itself perfectly consistent or that the denoising mask is optimal.

## First measurable weaknesses

1. **Amplitude units.** `exp(-0.5*abs(global-local))` in the existing peak blend
   has no explicit magnitude scale. Its weights change if identical audio is
   multiplied by a constant. Test gain equivariance before retuning thresholds.
2. **Population composition.** MAN/ATD depend on accepted bins, zero exclusion,
   bandwidth, speech occupancy, and window gain. Bright speech can raise the
   estimated background. Dense registered maxima also alter a noise-only
   distribution; calibrate after the actual analysis transform.
3. **Known legacy conventions.** Global and row statistics use different
   dispersion definitions. Row deviations use the first n original bins after
   counting nonzero values. The fallback VAD row index is also preserved from the
   legacy code. These are explicit ablations, not silent cleanup edits.
4. **Stationarity and impulses.** The current statistics are recomputed from
   overlapping context, not a separately tracked, slowly evolving per-band noise
   distribution. A single broad threshold should not represent both hiss and
   impulses. Impulses contaminate many FFT bins and can resemble broadband onsets.
5. **Geometry.** Fixed smoothing widths in bins/columns change their physical
   extent and decision behavior when FFT/hop changes. Keep that issue distinct
   from calibrating whole-frame count/run evidence.

## Floor estimation and the zero-crossing proposal

Keep separate state for slow diffuse noise power, transient/impulse confidence,
and signal-presence confidence. Compare the original estimator with per-band
robust quantiles and minima-controlled recursive averaging. Use speech confidence
to slow updates under likely speech; let noise-only evidence update faster after
an actual background change. Track uncertainty as well as the estimated floor.

Selecting samples where the *noisy observation* is close to zero creates a
selection bias. Even pure noise has small conditional amplitude after that
selection. A speech crossing does not establish that the additive noise is small,
and speech/noise cancellation can create a crossing. A better test of the user's
idea is to predict crossings from a coherent component using independent or
held-out context and examine local residuals, slopes, and prediction uncertainty.
Treat those measurements as another witness, not a universal noise estimator.

MCRA is a concrete comparison, not an instruction to discard Cleanup:
[Cohen and Berdugo, 2002](https://israelcohen.com/wp-content/uploads/2018/05/SPL_Jan2002.pdf).

## Paired experiments

Keep two clearly labeled datasets:

- **Proxy degradation:** Dave and Simon is the reference observation; add known
  noise and ask how well we preserve/recover that reference. This measures damage
  from extra corruption. Existing noise, fading, filtering, or distortion in the
  reference is not suddenly clean ground truth.
- **Controlled clean reference:** several clean speakers and utterances, then
  receive-band filtering and known added noise/channel impairments. Hold out
  speakers, utterances, and noise realizations from parameter selection.

Test flat/colored noise, slowly changing noise, pulsed bursts, sparse sharp
impulses, ringing impulses, and narrowband interferers separately and in mixtures.
Include consonants, weak speech, silence, short sweeps, and events at block edges.
For RF impairments, later add an SSB modulation/channel/demodulation simulation:
audio-additive noise alone is not a complete fading/AGC/adjacent-channel model.

Record clean/reference s, known corruption n, observed y=s+n, seed, active-speech
SNR convention, receiver passband, FFT/hop/window, and filter revision. Keep sample
alignment and latency explicit. Avoid independent output normalization. Do not
let a threshold search see held-out utterances or noise seeds.

For a scalar real gain on each noisy complex coefficient Y, a useful oracle is

    G_oracle = clip(Re(S * conj(Y)) / |Y|^2, 0, 1), with zero when |Y|=0.

This minimizes each coefficient's squared complex error under a [0,1] real-gain
constraint. It is a **per-coefficient oracle**, not a proof of globally optimal
waveform error after overlapping synthesis. Compare it with the noise/clean-power
ratio oracle, original Cleanup, the new guide-driven mask, and bypass.

Measure:

- Output error and SI-SDR together with gain-sensitive error/level preservation.
- Residual noise versus speech distortion, weak-consonant loss, onset/tail damage.
- Gate false opens, false closes, lost short events, and count/run statistics.
- For a frozen estimated mask, synthesize G*S and G*N separately to reveal speech
  attenuation and residual noise with the same gain. Do not rerun the adaptive
  filter separately on clean and noise and call that an equivalent decomposition.
- Musical-noise modulation, burst recovery time, peak/crest changes, and listening.
- CPU deadlines, memory, and latency. A prettier analysis image is insufficient.

## Stage two: DIP harmonic extension with controlled irregularity

DIP here means the local bfft diagonal-in-packets / finite-Zak intermediate
representation, not a neural Deep Image Prior. The proposal is to use existing
lower members of each harmonic comb to enrich progressively higher members of
that same comb. Their complex periodic components govern the continuation.

Candidate path:

1. Enter a selected DIP level and identify supported lower members in each comb.
   Use their measured complex periodic components as donors. Additional voice
   confidence can eventually restrict synthesis without replacing that mechanism
   with an independently generated harmonic oscillator bank.
2. Continue those periodic components to higher members, retaining coherent
   phase through successive frames. The initial implementation uses a fixed
   comb lattice; adaptive selection of a speech-relevant lattice remains an
   experiment, and fixed packet spacing must not be presented as inferred pitch.
3. Complete the transform and project modifications onto the allowed upper band.
   Restore protected measured lower-band coefficients after every iteration.
4. Shape added harmonics by an estimated speech envelope and smooth cutoff taper;
   bound added energy relative to confident lower-band signal energy.
5. Find temporal energy peaks supporting the same comb, then place soft erosion
   holes at deficit centroids BETWEEN those peaks. Their decay extends sideways
   in time and across nearby upper members. Flat sustained energy should not
   acquire arbitrary holes. Add coherent, softly increasing upper-band
   irregularity; neither blue weighting nor these ellipses are established
   speech statistics. Avoid independent per-frame random phases.
6. Enforce real-signal transform constraints, overlap continuity, and disabled-mode
   identity. Increased sample rate is required before generating above Nyquist;
   extending above the receive passband but below Nyquist does not require it.

A narrowband receiver has removed evidence. Upper-band continuation is a
perceptual reconstruction hypothesis, so evaluate against genuinely full-band
clean references and listening. It cannot undo speech that the denoiser discarded.
Develop independently of the first-stage comparison; leave this extension off
by default behind one additional plugin button, **Harmonics**.
No new tuning sliders. Compare plain harmonic continuation, continuation plus
noise, and the DIP-constrained version to determine whether DIP adds value.

Relevant phase-aware bandwidth-extension comparison:
[Lu et al., AP-BWE](https://arxiv.org/abs/2401.06387). Its learned generator is not
our proposed algorithm; it supports treating phase as an explicit evaluation axis.
