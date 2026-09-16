# Population Cleanup: change the statistical domain, retain both passes

The new `CLEANUP_POPULATION="rolloff"` experiment uses the registered guide and
Cleanup's existing separation operator. `"receiver"` remains the default and
reference. `"oracle"` is a test control using supplied bandwidth to select the
statistical population while still allowing gain over the full spectrum.

## What changed

Statistical membership is now separate from the frequency cells eligible for
gain. A prefix of the guide supplies Cleanup's distribution statistic,
contextual MAN/ATD, row statistics, and the recomputed selected-field statistics.
The peak comparisons, in-place second pass, minimum combination, temporal
triangle, three native recursive smoothing rounds, and unsquelched residual gain
retain their existing rules. All guide cells remain eligible for these gains;
the population boundary is **not a hard audio cutoff**.

`population_peaks` is the common implementation. The original `peaks` wrapper
passes the same count for decision and statistical domains, preserving the
reference route. The alternative passes a separate statistical count. No second
implementation of the threshold formula was introduced.

The corresponding physical population also selects reference FFT bins for the
existing whole-block evidence. Its fixed 512/128 time lattice, 129-interval
counting region, total/contiguous admission rules, and run repairs are unchanged.
No new guide-time count calibration or duration-only replacement was added.

## First support estimator

The estimate uses the complete 2048-bin registered guide:

1. Average magnitude over all 48 actual guide time columns. This is an envelope
   observation, not a noise estimate or a classifier requiring pauses.
2. Sample a local frequency median at 96 logarithmically spaced frequencies.
   These suppress narrow peaks in the **boundary fit only**. Cleanup still sees
   the original magnitudes, including carriers and internal valleys.
3. Fit robust lines before and after candidate bends in log frequency/log
   magnitude. Huber residuals bound the influence of outliers. Compare the
   two-line loss against a single line with a complexity penalty.
4. Select the outermost distinct accepted downward bend. Reject sharp upward
   recoveries such as the flank of a deep notch. Use the line intersection as
   the population boundary; include everything below it, including internal
   valleys. Fall back to the full domain when the model finds no qualifying bend.

The constants define a **steep-rolloff hypothesis**, not calibrated statistical
confidence: 0.35 log-amplitude Huber transition; 8 prefix and 14 tail observations;
prefix slope at most 2, tail slope below -2, slope drop at least 2; penalized-loss
improvement above 10. Search and local-max comparisons use fixed storage. This
is an experimental model of the observed envelope, not proof of a physical
receiver transfer function. Spectral content can itself contain rolloffs.

The estimator is homogeneous in amplitude in the tested scale range (0.01×,
1×, 100×). The complete Cleanup operator deliberately retains its original
amplitude-dependent behavior. The fit uses the existing overlapping context;
there is no new delay or learned quiet-state history.

The first global regression was rejected: a strong carrier moved its inferred
boundary to about 821 Hz on a 3 kHz test, and a notch moved it to about 3904 Hz.
Local fits with bounded influence and explicit rejection of a notch recovery
addressed those cases. Their remaining errors are reported below.

## Measured controls

### Preserve the operator

All comparisons use the same full-width registered transform to avoid conflating
registration-domain changes with population selection. The oracle-population
control matches the reference's voiced and continuous-FM component metrics at
the reported precision. Unit tests also verify:

- Appended excluded bins do not alter population statistics or original-bin
  peak decisions, including the amplitude-valued in-place second pass.
- Oracle selection preserves the interior reference mask (boundary excluded
  by three frequency cells) to numerical precision.
- An above-population strong component can still receive gain.
- The inferred path gives byte-identical coefficient arrays when supplied
  receiver bandwidth changes among 1.2, 3.4, and 16 kHz with observations fixed.

The Downloads C++ baseline parity test passed all six bandwidth/squelch cases,
with maximum error 3.06e-16. The existing registered complete-mask/projection
parity test also passed. The three new population tests passed after the final
estimator change. Native renders reported zero faults.

### Known clean/noise pairs

Seed 71313, same corpus as the mechanism audit. Separate clean and noise channels
are filtered using identical observed-mixture analysis, so both receive the same
adaptive decision. These are signal-preservation measurements, not perceptual
quality scores.

| Fixture | Reference clean / noise gain | Inferred-population clean / noise gain | Reference / inferred coherence |
|---|---|---|---|
| Voiced, 3 kHz LPF | −7.546 / −14.577 dB | −7.578 / −14.692 dB | 0.751 / 0.747 |
| Continuous FM, 3 kHz LPF | −4.992 / −11.602 dB | −5.042 / −11.681 dB | 0.896 / 0.896 |
| Weak chirp, 3 kHz LPF | −17.927 / −43.725 dB | −17.934 / −43.736 dB | 0.894 / 0.894 |
| Weak chirp, 1.2 kHz LPF | −7.407 / −27.645 dB | −7.407 / −27.647 dB | 0.937 / 0.937 |

### Boundary estimates, eighth-order physical low-pass filters

| Input | Physical cutoff | Estimated statistical boundary |
|---|---:|---:|
| Broadband noise | 1200 Hz | 1196 Hz |
| Broadband noise | 3000 Hz | 2908 Hz |
| Broadband noise | 6000 Hz | 5933 Hz |
| Voiced harmonics plus noise | 3000 Hz | 3001 Hz |
| Continuous FM plus noise | 3000 Hz | 2919 Hz |
| Strong continuous carrier plus noise | 3000 Hz | 2720 Hz |
| Deep internal notch | 3000 Hz | 2767 Hz |
| Full-band noise | 24000 Hz available | Full domain |

**Held-out failures:** second-order 4 kHz and fourth-order 2 kHz rolloffs were
not resolved; both fell back to the full domain. That fallback can reintroduce
the original population-dilution problem. It is not a successful passband
estimate. Held-out twelfth-order 5 kHz, blue-spectrum 3 kHz, and impulse-contaminated
3 kHz cases yielded 4830, 2908, and 2931 Hz respectively. Full-band colored noise
and the tone-only case selected full domain. The latter has insufficient
broadband excitation to identify its physical LPF from this envelope.

Thus this is a useful first steep-rolloff experiment, not a completed arbitrary-
input solution. Softer transitions, high-pass edges, multiple disjoint bands,
and broader source-envelope ambiguity remain open. The next estimator revision
should distinguish small but well-measured bends from poorly measured ones,
instead of using one fixed residual scale. Keep the decision operator fixed
while evaluating that change.

## Listening and visual comparison

The accepted browser platform has one additional **Population Cleanup** button.
Existing rendered audio is retained byte-for-byte; the previous default selection
is unchanged. The new trial uses no harmonics or comfort noise. A dashed white
curve over its actual analysis spectrum shows the statistical boundary over time.
The lower panel remains actual final coefficients immediately before inversion.
Current population size is also displayed in the statistics area.

The 30-second trial has zero clipped samples and zero native faults. Its peak
callback was about 254 ms for a 171 ms audio block on the Mini: **this full-width
research path is not yet fast enough for real-time reception on that host**.
The browser plays the completed native render at the original sample rate with
the declared 16,768-sample latency removed. No SDR# deployment is claimed here.

On the first eight radio seconds, the inferred boundary ranged approximately
2.15–2.51 kHz over the interior audit blocks, versus the reference's supplied
3.4 kHz setting. The recording's true receiver cutoff is not an oracle here;
inspect the plotted boundary and listen rather than treating either as truth.

## Files and reproduction

- `Filter.py`: all estimator and separation policy, explicit Numba signatures,
  fixed scratch storage, reference default.
- `runtime/compile_filter.py`: validates population mode and restricts it to
  registered legacy Cleanup.
- `native/src/engine.cpp`: publishes population diagnostics; no new native DSP.
- `tests/test_population.py`: population/decision invariants.
- `research/population/probe.py`, `--holdout`: native-guide envelope controls.
- `research/mechanisms/audit.py --population`: known-pair and radio controls.
- `tools/verify-population-mini.sh`: native build and focused regressions.
- `docs/evidence/population-{probe,holdout,evaluation,render,browser}.json`:
  measurements, frozen-script hashes, and browser verification.

Run the scripts using the repository's `m4build` workflow and existing Mini
Python. Full native render artifacts and frozen filters reside in
`build/listening-demo-v16` and `build/population-evaluation` on the Mini. The
served local demo receives only the new trial and updated UI/manifest.
