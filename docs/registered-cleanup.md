# Live registered-sub-hop Cleanup integration

This is an explicit `Filter.py` experiment, selected with
`ANALYSIS_MODE = "registered"` and Reload. The current listening-demo checkout
selects it; `"baseline"` remains the comparison route. There are no new tuning
sliders. Harmonics is a separate optional entity.

## Live data flow

For each channel and overlapping 24,576-sample context, the parent computes:

1. Seven exact absolute cosine observations from the double inverse construction,
   at offsets `(-379,-241,-64,0,83,214,427)`. Aperture 2048, periodic Hann, second
   inverse denominator 4094. All Fourier operations use preplanned bfft workspaces.
2. The frozen oracle's registered maximum: 99.5th-percentile amplitude alignment;
   positive 75th-percentile log compression and reflected Gaussian `(1,3)` only
   for registration; overlapping 48-by-48 charts at stride 16; seven smooth Fourier
   annuli; energy/reliability-weighted translations; residual/coherence chart
   weighting; reflected linear transport of the untouched amplitudes; maximum.
3. A time-major registered raster in reserved fixed state slot 2. Python is never
   dispatched on the audio thread. Plans, chirps, chart windows, and all workspaces
   are created at Reload, one independent instance per channel. Two persistent
   workers and the audio caller each register two lattices; the maximum merge
   follows a fixed order. Identical channel histories reuse pristine native guide
   and reference results, while each channel retains independent filter state.
4. `cleanup_registered` in the single strict-Numba `Filter.py` runs Cleanup's
   structure statistic, triangular smoothing, global/local MAN/ATD peak selection,
   first mask, masked-magnitude statistics, second peak pass, mask combination,
   triangular smoothing, and all three recursive time/frequency smoothing rounds.
   The non-squelch fallback remains explicit, with gains finally bounded to [0,1].
5. Bilinear interpolation transfers the resulting gain at actual sample centers
   and physical frequency centers onto the parent's complex coefficients. RFFT
   uses integer centers; ODFT uses half-bin centers. bfft's original synthesis and
   canonical dual window invert those coefficients. No phase is invented by the
   positive analysis image.

The selected bounded-context field is exactly the oracle applied to that context
and its specified centers. It is not identical to running a single registration
atlas over an entire recording: percentile and chart neighborhoods have bounded
context here, and observations beyond that context are zero-padded. Emitted
centers have substantial audio context on both sides.

## Geometry, gain and coverage

Default guide: 512 frequency rows and 48 time columns, centers every 512 samples.
At 48 kHz the last row is `511*48000/4094 = 5991.2066 Hz`. This expands the research
crop of 256 rows, which covered only 2989.74Hz. A Reload whose guide cannot cover
current receive bandwidth is rejected before replacing the active generation.
Changing bandwidth beyond coverage afterward fails the callback and invokes the
existing explicit fault/bypass behavior; enlarge `GUIDE_BINS` and Reload.

Outside the receiver's accepted synthesis bins, gain is zero as in the original
Cleanup. Outside guide coverage, the interpolation helper is neutral; live
registered filtering rejects insufficient receive-band coverage instead of
silently relying on that fallback. All interpolated gains are bounded [0,1].

The expensive registered observations remain 48 columns at hop512. Before mask
evaluation, their amplitudes are linearly interpolated onto a hop128,192-column
time lattice. This creates no additional observations; it restores the original
duration of the triangular15 and recursive time13 kernels. Applying those same
kernels directly to hop512 columns had spread decisions over four times as much
time. Frequency decisions remain on the512 guide rows, with the original3-bin
frequency branch; that frequency footprint is still narrower than baseline.

Guide amplitudes are converted to the fixed FFT512 reference's coherent-window
units before Cleanup's amplitude-dependent exponential weights and second-pass
residuals. The factor accounts for the second inverse's division by2047, the
periodic Hann2048 window and symmetric Hann512 reference window, including the
inverse's endpoint half-weights. A constant-frame double-inverse test checks it
against the reference FFT directly. This removes a normalization mismatch; it
does not establish equivalent distributions or quality calibration after the
registered maximum. The non-squelch fallback now uses the current mask interval's
detection, rather than indexing backward by the emitted-frame offset.

## Whole-context admission is deliberately separate

The native parent independently computes an actual **fixed RFFT512/HOP128**
reference analysis, regardless of synthesis FFT size, hop, or RFFT/ODFT choice.
This reference goes into reserved slot 3. Cleanup's reference structure detector
then admits the entire block when either:

- More than 22 of the 129 intervals in `[32,161)` qualify; or
- The longest contiguous qualifying run anywhere in `[0,192)` exceeds 16.

The binary evidence is measured before run-repair postprocessing. This preserves
both the total-interval and contiguous-event tests and the short-event intent.
It does not invent a duration-only guide threshold. The registered guide supplies
its own local mask evidence, but has no independently calibrated admission rule.
The reference verdict governs whole-block squelch and the guide's run repair.

Mask diagnostics map the original counting region onto interpolated mask centers:
`4096 <= center_sample < 20608`;129 possible intervals at hop128. The full mask
has192 intervals for longest-run measurement. These are correlated interpolated
measurements, not192 independent native guide observations. They remain diagnostics,
**not** proof that thresholds 22/16 should scale by a ratio. Calibrating guide
admission still requires measuring how many guide intervals represent the same
ionosound, vowel, gap, and boundary event. Sample-rate changes likewise do not
magically establish a new event calibration.

Current registered mode requires 8192-sample blocks, three context blocks, and
four state slots. Slots 0/1 hold Cleanup/Harmonics; slot 2 holds the native guide;
slot 3 holds native reference magnitudes plus reference scratch. Compiler checks
reject overlap or insufficient fixed storage. Other block/context geometries
remain available in baseline mode.

## Diagnostics and verification

Native status reports `analysis_mode`, guide aperture/rows/columns/hop and coverage.
With `VIEWER=True`, top is the exact registered raster consumed by Cleanup, via
`VIEWER_GUIDE`; bottom is the actual final emitted coefficient magnitude. The
usual snapshot includes separate reference and guide count/available/longest/run
available fields. The viewer does not display a provisional mask approximation.

Run `/Users/ultimussecundai/.local/bin/m4build -- sh tools/verify-guide-mini.sh`.
It checks registered oracle parity, every recursive-mask stage and final projected
coefficients, response to changed guide and independence from unrelated synthesis
analysis, gain centers/identity/bounds for RFFT and ODFT, native streaming,
viewer publication, transactional rejected Reload and geometry/state reset.
The existing legacy parity suite continues to check the baseline independently.

The native DLL/compiler/plugin were staged into actual SDR# and compiled a
registered generation successfully. The host has not processed a live audio
stream in this verification, and the Wine viewer launcher timed out. The accepted
listening platform is now the browser comparison, with spectra captured directly
from the same native render. Check full callback timing before using this route
for live reception; the core transform benchmark is not the whole pipeline budget.

## Measured implementation budget (M4 Mini)

The complete native callback, including registration, both Cleanup mask passes,
three recursive smoothing rounds and synthesis, was measured with default 512-row
analysis and 170.667ms of mono/stereo audio per block:

| Input | Measured peak callback |
|---|---:|
| Mono | 66.63 ms |
| Identical stereo analysis | 70.52 ms |
| Independent stereo analysis | 133.25 ms |

All streams had finite output and zero native faults. Declared algorithmic latency
is 16,768 samples, approximately 349.3 ms at 48 kHz. These measurements establish a
budget on that machine, not a guarantee on every host or under arbitrary system
load. Windows/Wine and actual SDR# have separate host execution gates.

The initial exact serial implementation took ~522 ms per block. The faster result
retains the same observations, charts and seven rings: it caches reference chart
spectra and separable spatial windows, uses exact radix 3×16 bfft chart transforms,
uses real/Hermitian boundaries, and distributes lattice registration among fixed
workers. No field approximation or reduced-resolution substitute was introduced.

Oracle maximum absolute errors on bounded 48-column/320-row test rasters were 0
for silence, 5.72e-16 for deterministic noise, 3.77e-11 for clean tones and 1.55e-13
for the radio fixture. Independent orchestration of the complete guide mask and
its projected final coefficients matched exactly in the deterministic mask test.
Legacy baseline mask error remained 5.56e-17 against the downloaded C++ oracle.
See `docs/evidence/registered-guide-results.json` and the verification log.
