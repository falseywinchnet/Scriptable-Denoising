# Optional registration; deployment defaults off

The user selected registration off as the deployment default. `Filter.py` now
sets `GUIDE_REGISTRATION = False`; the Windows packager explicitly emits False
even if a developer enables a local trial. It preserves the script's selected
population policy; packaging must not silently change the denoiser. The current
live script uses the receiver-aware statistical reference. Population Cleanup
(`CLEANUP_POPULATION="rolloff"`) remains the accepted listening-demo experiment.

`ANALYSIS_MODE="registered"` remains the historical name of the native enhanced
guide route for compatibility. The new boolean controls whether that route
actually registers observations. Status publishes `guide_registration` and the
actual `guide_threads` count. Old scripts without the new header retain their
registered behavior. Baseline ordinary-STFT analysis remains a separate mode.

With registration off:

- Compute only the central enhanced-magnitude observation, mathematically the
  absolute second inverse of the windowed frame, on the same guide grid.
- Retain aperture, hop, frequency coordinates and coherent amplitude conversion.
- Omit six offset views, percentile normalization across views, perceptual
  preparation, chart alignment, and registered maximum fusion.
- Keep Population inference, both Cleanup mask passes, recursive smoothing,
  fixed-reference whole-frame admission, interpolation to the synthesis grid,
  and synthesis unchanged. No replacement denoiser or parameter retuning.
- Allocate one unrolled Fourier plan and history cache, with no guide workers.
  Overlapping apertures are reused only after exact overlap validation; changed
  zero-padding edges and discontinuities are recomputed.

Set `GUIDE_REGISTRATION = True` and Reload to opt in again. Existing
`cleanup_guide_create` preserves its registered research ABI; the engine uses
the new `cleanup_guide_create_mode` selector. Registration remains available for
future mathematical optimization, subject to the four-lane ceiling.

## Thirty-second controlled comparison

`build/no-registration/demo` contains fresh same-build renders with identical
source, original gain, latency removal, Population settings, and harmonics off.
The newly rendered registration-on PCM is **byte-for-byte identical** to the
accepted `build/listening-demo/population.wav`. The served demo adds only
`Population · registration off` alongside the retained comparison.

Observed differences, not quality scores:

- Registration-off output RMS is 0.836 dB lower.
- Waveform correlation is 0.9866; this is not a speech-intelligibility measure.
- Median inferred statistical edge is unchanged at approximately 2298 Hz.
- Median first-pass selected fraction is 1.445% versus 1.464% with registration.
- Median second-pass selected fraction is 0.704% versus 0.846% (about 17% fewer
  selected cells before smoothing). Fractions cover the full analysis domain.
- Both renders have zero clipping and zero native faults.

The common-scale spectra show that the central analysis retains fine dark
nulls which multi-view maximum fusion fills. The final speech/formant shapes
remain broadly similar, with differences in weaker regions. There is no clean
ground truth for this recording: a lower level or sparser selection cannot by
itself establish that only noise was removed. Listen to both at the same word.

## Performance and verification

The first 64-block paced identical-stereo trial on the M4 measured 56.45 ms
median, 59.48 ms p95 and 62.06 ms maximum, against the preceding registered
trial's 77.62/80.86/81.33 ms. Registration and view-preparation counters are zero;
one guide lane is reported. All observed calls met the 170.67 ms interval.

These are separate runs. The unregistered paced run also measured a much slower
filter stage than the previous registered run (46.93 versus 20.71 ms). The
unpaced mono render's final callback measured 15.42 ms, of which 14.07 ms was
filtering, and its maximum was 24.01 ms. Different workload, pacing, CPU scheduling
and clock behavior prevent attributing the entire timing difference to DSP
arithmetic. No fixed speedup or one-core deadline guarantee is established.

All nine registered-guide regression tests passed, including the new independent
central-observation oracle, exact cached/fresh parity, discontinuity invalidation,
one-lane count and zero registration/view time. The Windows DLL cross-build also
succeeds. An eight-block Windows/Wine identical-stereo processing smoke test
reported registration disabled, one guide lane, zero faults, 42.38 ms median
and 47.99 ms maximum (unpaced, no audio device). These results do not qualify an
actual SDR# audio-device load.

With SDR# stopped, the matching Windows engine DLL and compiler were copied into
the local Downloads SDR# installation; prior copies were backed up. Its existing
Filter.py link resolves to the authoritative default-off script. See
`docs/evidence/no-registration/staging.json`. Actual host audio validation remains
separate; the public website and existing downloadable archives were not changed.

Evidence: `docs/evidence/no-registration/` contains `tests.log`, `callback.json`,
`comparison.json`, and `registration-comparison.png`. The renderer is
`tools/listening_demo.py --only population-unregistered`; the descriptive
comparison/plot is `tools/analyze_registration.py`. `tools/benchmark_filter.py`
accepts explicit `--registration on|off` for controlled comparisons; without
that override it follows the current Filter.py default.
