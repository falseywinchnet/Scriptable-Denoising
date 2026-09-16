# Beyond repeating a lower harmonic

Research notes, September 15, 2026. This is a design investigation, not a claim
that the cited algorithms have been implemented in our harmonic entity.

## What the current code still does

The DIP transport is mathematically tested, but its current single-member
projection produces a donor coefficient times a complex weight at the target.
At FFT512, E8, 48kHz, packet spacing is750Hz irrespective of the speaker's pitch.
Revision3 changed the join and amplitude envelope. It did not estimate vocal
excitation or remove the donor's vocal-tract envelope. The remaining repeated
structure reported by the user therefore matches what the implementation does.

## Mowlaee references

- Peharz, Kapeller, Mowlaee, Pernkopf, **Modeling Speech with Sum-Product Networks:
  Application to Bandwidth Extension**, ICASSP2014,3699–3703.
  [Author's comparison](https://www2.spsc.tugraz.at/people/pmowlaee/AWBE.html),
  [author-uploaded paper](https://www.researchgate.net/publication/260979733_Modeling_Speech_with_Sum-Product_Networks_Application_to_Bandwidth_Extension).
  Models log spectra using SPNs as HMM observations, then conditions missing
  high-band inference on observed bins and temporal state. A learned conditional
  spectral shape is a concrete alternative to copying a low-band envelope.
  This requires full-band training examples; it is not a plug-in physical law.
- Mowlaee's [Phase-Aware Processing for High Bandwidth Extension demo](https://www2.spsc.tugraz.at/people/pmowlaee/BWE.html)
  compares zero, random, Griffin–Lim and estimated phase for male/female examples.
  This page does not identify enough implementation detail to attribute a
  particular phase estimator to that demo without further source inspection.
- Mowlaee and Kulmer, **Harmonic Phase Estimation in Single-Channel Speech
  Enhancement Using Phase Decomposition and SNR Information**, TASLP2015.
  [Publisher](https://doi.org/10.1109/TASLP.2015.2439038),
  [author's phase-research index](https://www2.spsc.tugraz.at/people/pmowlaee/PhaseEval).
  Harmonic phase estimation in noisy observed speech is relevant to anchor
  reliability, but is distinct from predicting missing high-band amplitude.
- Maly and Mowlaee, **On the Importance of Harmonic Phase Modification for
  Improved Speech Signal Reconstruction**, ICASSP2016,584–588.
  [Paper](https://dihana.cps.unizar.es/proceedings/ICASSP/2016/pdfs/0000584.pdf).
  Separates linear phase from unwrapped harmonic phase and examines their roles
  in reconstruction. This supports tracking timing and residual phase separately.
- Watanabe and Mowlaee, **Iterative Sinusoidal-based Partial Phase Reconstruction
  in Single-channel Source Separation**, INTERSPEECH2013.
  [Paper](https://www.isca-archive.org/interspeech_2013/watanabe13_interspeech.pdf).
  Uses estimated sinusoidal components to define a signal-dependent confidence
  region for partial phase reconstruction. Transferable idea: retain trustworthy
  phase anchors and constrain uncertain components through reconstruction.

## A particularly close match to the reported artifact

Turan and Erzin, **Synchronous Overlap and Add of Spectra for Enhancement of
Excitation in Artificial Bandwidth Extension of Speech**, INTERSPEECH2015.
[Paper](https://www.isca-archive.org/interspeech_2015/turan15_interspeech.pdf).
They distinguish excitation from spectral envelope, and explicitly discuss
excessively strong upper harmonics and a discontinuity between original and
extended regions. Their method correlates and overlaps high-end excitation
spectra, with voicing-dependent attenuation. They also explain how small pitch
variation has a larger absolute effect at higher harmonics. Their algorithm
still reuses spectral material; the useful distinction is what is extended,
how it is aligned, and how its envelope is supplied.

## Proposed next harmonic experiment (our synthesis of these ideas)

1. Track a common fundamental/periodic timing hypothesis from several supported
   lower components, with confidence and octave-ambiguity checks. Keep the
   registered field for locating evidence and retain complex STFT coefficients
   for phase. Do not equate the fixed DIP residue spacing with vocal pitch.
2. Separate smooth resonant envelope from periodic excitation. Continue the
   normalized excitation comb, then apply a separately estimated high-band
   envelope. Do not translate the entire formant-shaped donor spectrum upward.
3. Advance every harmonic from shared evolving source timing plus its own
   residual phase. A source phase perturbation is multiplied by harmonic order;
   this naturally spreads higher members more strongly than lower ones.
   The higher-band aperiodic component must also depend on observed voicing,
   rather than a fixed amount of unrelated random phase.
4. Retain the user's soft erosion between plausible temporal energy peaks, the
   occupied-band join, lower-band preservation and addition-power budget.
5. Test against full-band clean speech, its band-limited version, and degraded
   versions. Separately score the join, upper-band envelope, harmonic alignment,
   phase continuity, consonants and listener-rated metallic coloration. The
   shortwave recording remains a listening case, not a missing-high-band oracle.

This is a bounded DSP direction compatible with the current FilterBank and
native transform boundary. It does not require immediately replacing the system
with a trained speech generator. A trained conditional envelope model is a later,
separately measurable option suggested by the2014 SPN work.
