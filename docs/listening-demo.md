# Listening demo

The current `Filter.py` selects registered analysis and enables the native viewer
for this listening experiment. Reload reconstructs its analysis plans and state.
This is an experimental selection: the Windows/Wine host still has occasional
callback timings above its audio deadline. Baseline remains available by setting
`ANALYSIS_MODE = "baseline"` and reloading.

## Matched audio

`tools/listening_demo.py` renders the same source through the native engine with:

1. Original source, unchanged.
2. Baseline Cleanup.
3. Complete Cleanup on registered sub-hop analysis.
4. Registered Cleanup with optional DIP harmonics enabled.

The engine's declared latency is removed and all variants have identical lengths.
Squelch is off. There is no per-file normalization. Each generated script and its
hash, full native status, processing time, output RMS, peak and clipping count are
retained beside the audio in `manifest.json`.

The page switches between synchronized, looping audio buffers with a short gain
transition. Optional overall RMS matching uses baseline as its target, with one
shared headroom factor. Overall RMS includes noise; it is not a speech loudness
metric or a measure of intelligibility.

Generate on the Mini using a source file already present there:

```sh
/Users/ultimussecundai/.local/bin/m4build -- \
  /Users/joshuahkuttenkuler/miniforge3/bin/python tools/listening_demo.py \
  /tmp/cleanup-demo-source.wav --library build/libCleanupNative.dylib --seconds 30
```

Retrieve `build/listening-demo/` from the mirror before another sync. Serve locally:

```sh
.venv/bin/python -m http.server 8769 --bind 127.0.0.1 --directory build/listening-demo
```

Open `http://127.0.0.1:8769` and press Play. The WAVs are independently playable.

## First rendering measurements

The first 30 seconds of the local `daveandsimon.wav` (48 kHz mono) produced no
native faults and no clipped samples in any version. Registered Cleanup rendered
in 9.70 seconds on the M4 Mini; adding harmonics took 9.93 seconds.

Registered output RMS was approximately 6.4 dB below baseline Cleanup. This is
attenuation, not a measured improvement in noise rejection or intelligibility.
The different guide amplitude distribution and changed smoothing footprint still
need listening and paired-signal calibration.

## Second listening revision: units, time spacing and visuals

The registered mask now converts inverse-field magnitudes to Cleanup's reference
FFT units and interpolates its input onto the original hop128 time lattice before
all Cleanup stages. It keeps the13-tap recursive time kernel, its persistent
padding and three rounds. This corrects the fourfold temporal spread introduced
by evaluating that kernel directly on hop512 observations. Current-interval
evidence replaces the misindexed fallback. See `registered-cleanup.md`.

The30-second revised registered rendering has RMS0.0764565 (previous0.0450096;
baseline0.0939576), with zero native faults and no clipping. It is approximately
1.8dB below baseline overall RMS, compared with6.4dB in the first revision. That
describes level only, not an established quality improvement.

Native diagnostics are collected after every processed block from the same
rendering instance. Exact published float32 magnitudes and gate states remain in
`*-spectra.npz`; the page reads fixed-range dB display bytes. Frame centers include
the initial negative-time context and the registered guide's128-sample offset.
Top is analysis, bottom is the actual emitted complex-coefficient magnitude before
inverse synthesis. An additional slice overlays original, registered and optional
harmonic output at the audio cursor. Click a heatmap to seek; Pause freezes the
audio and cursor together. A2/6/30-second view and0–4/0–8kHz view aid inspection.

The harmonic packet inversion and default bin translation passed an independent
audit. A latent centered-phase bug for odd-bin shifts was corrected and tested
with actual native RFFT/ODFT coefficients; default E8 was unaffected by that bug.
The fixed750Hz comb remains an experimental policy, not an estimate of speech F0.

## Third listening revision: envelope experiment

The fifth button, Previous harmonics, retains revision2's audio and spectra at
the same playback position. The new +Harmonics uses the occupied-band join and
primary/exponential continuation described in `harmonics.md`. Reproduce with
`--previous build/listening-demo-v2` in addition to the renderer's other arguments;
the previous comparison must use byte-identical source audio.

The renderer also captures both mask passes' MAN, ATD, mean evaluated threshold,
and selected fraction for every contextual block. Open Floor and peak statistics
under the spectra. Selection fractions include the entire accepted time/frequency
rectangle before recursive smoothing, including rows skipped by local evidence.
MAN remains Cleanup's deviation-based proxy, not a separately measured noise PSD.

On this30-second clip, mean first/second selection fractions were15.55%/10.83%
for baseline and10.76%/6.05% for registered. The grids differ, so these figures
support the sparsity observation without establishing equivalent false-alarm rates.
Both denoising WAVs remain byte-identical to revision2.

The measured receiver-edge heuristic selected2812.5,2906.25 or3000Hz. Added-audio
Welch power shifted toward the join:24.90% between2.8–3.4kHz,70.44% between3.4–4.2kHz,
4.65% between4.2–5.5kHz and0.0044% between5.5–8kHz. The previous extension placed
25.88% between5.5–8kHz. These describe the synthesized difference signal, not
recovered speech accuracy. All three new renders had zero faults and no clipping.
See `docs/evidence/harmonics-v3-results.json` and `listening-v3-tests.log`.

For the next registered-floor experiment, measure the noise-only distribution
after the actual registered maximum (its shifted views are correlated), threshold
margins of faint known components, and survival through each mask pass. A finer
frequency grid changes peak occupancy and the physical width of the3-bin branch.
Faint coherent support and isolated noise maxima need different treatment. The
current scalar diagnostics expose the fitting behavior; they do not calibrate a
new admission threshold or replace the fixed-reference whole-frame squelch rule.

The separate native viewer can poll a live demo engine served by
`tools/experiment.py serve`; that command deliberately opens no audio device.
Its rolling display runs independently of the comparison page's playhead. The
page is the synchronized audio comparison; SDR# has its own viewer and controls.

## Fourth listening revision: crossing-neighbor floor

The same page adds **Crossing floor**, with harmonics off. It uses the mean of
absolute signed-input samples bounding zero crossings, converted explicitly to
cosine-inverse magnitude units. It removes MAN/ATD fitting from this experimental
mask branch; the baseline and previous registered algorithm remain comparisons.
See `crossing-floor.md` for the exact rule and assumptions. The statistics panel
shows the raw mean, event count, converted floor and both selected fractions.
The frequency slice adds a green crossing-floor trace.

The first internal crossing render still inherited the receiver-band mask cutoff.
Following the user's correction, the served experiment expands the registered
analysis through Nyquist and removes that cutoff from the mask, reference
statistics and synthesis. Its full-band selected fractions cannot be compared
directly with the legacy crop's fractions. The0–24kHz viewer option exposes the
entire frequency axis. These are experimental hypotheses, not established
choices of a correct floor estimator.

The harmonic branch is unchanged in this revision. Research addressing its
remaining repetition is recorded in `harmonic-reconstruction-research.md`.
