# True superresolution: extraction and integration research

Status: **live selectable registered-guide integration implemented, pending host deployment and quality calibration**.
See [registered Cleanup](registered-cleanup.md) for the complete recursive mask,
fixed reference admission, exact native registration and measured deadline gates.
The default remains baseline until the deployment/performance gates are passed.

## Which sibling and which representation?

The source task is **Reconstruct 3D scene from image**, ID
`01a03b18-6686-79a3-87b1-ddee0b5a7b14`. It grew into the voice-to-text experiment
under `/Users/ultimussecundai/bfft/experiments/ostensibly_frontend`.
The user's description was that the double inverse absorbs phase into geometry,
and that the enhanced magnitude exposes signal detail when zoomed.

That task contains several materially different representations. Do not silently
replace the requested one with a similarly named superresolution experiment:

1. Literal `irfft(irfft(STFT))`, followed by absolute value and low-row crop.
2. The earlier positive ridge construction: compute the same second cosine
   inverse and `irfft(-1j * recovered_frame)`, then `hypot(cosine, quadrature)`.
3. `uncertainty_fusion.py`: register complementary sub-hop observations and take
   their maximum in the reference coordinates. It operates on the absolute
   cosine observation, not the quadrature envelope.
4. `fourier_unrolled_densified.py`: first complete that registration separately
   at apertures 1024, 2048, 4096, then register across apertures and take a maximum.
5. Its later `reassigned_texture_baseline`: a 3-by-3 average of that registered
   field plus the square root of reassigned power, scaled into the same amplitude
   units by the 99.5th percentile. This was the promoted phone-comparison field.

The user confirmed the final enhanced field (row 5 of the generated comparison)
was the intended reference, then selected registered sub-hop magnitude (row 4
of that comparison) as a potentially sufficient starting point. Those image row
numbers differ from the construction stages listed above. Keep the full field
as the comparison reference; start live integration with registered sub-hop.

The separate `notes/two_lattice_superresolution.md` and `viewer/superres.py`
describe other phase-retrieval experiments. They are not this extraction.

The snapshot in `research/true_superresolution/oracle` retains the actual source
with only relative imports and lazy imports for unused Meyer audit functions.
`provenance.json` records original and adapted hashes, source commit, and task ID.
The source tree had local changes; hashes identify what was actually copied.
The first-second WAV fixture comes from the user's local PyITD `daveandsimon.wav`.

## Exact inverse and a bfft implementation

For a real N-point STFT, the first inverse recovers the windowed frame. The
second default-length inverse has length **L = 2(N-1)**: 1022 for N=512, 4094
for N=2048. NumPy documents that default in
[the irfft reference](https://numpy.org/doc/stable/reference/generated/numpy.fft.irfft.html).

For the recovered real vector v, its signed second inverse is

    C[r] = (v[0] + (-1)^r v[N-1]
            + 2 sum(j=1..N-2) v[j] cos(2πjr/L)) / L.

The quadrature inverse is

    S[r] = 2 sum(j=1..N-2) v[j] sin(2πjr/L) / L.

Therefore the original observation is abs(C); the positive-envelope variant is
hypot(C,S). They must remain distinguishable. Doubling rows alone is not a
measurement that two nearby sources have become resolvable.

`research/true_superresolution/unrolled.cpp` evaluates both exactly, with a
preplanned Bluestein convolution implemented using the vendored bfft transforms.
The first STFT forward/inverse cancels when there is no intervening gain, so the
native routine starts directly from the raw, periodically Hann-windowed frame.
If a guide gain is inserted before the first inverse, that cancellation no
longer applies; the API must then accept the recovered frame explicitly.

All chirps, FFT workspaces, and buffers are allocated at plan construction.
Calls reuse them. One plan per concurrent channel is required. The same native source is now linked into the live parent for the selectable
registered analysis route, with one independent plan per channel.

Measured on the Mini: maximum absolute difference **6.213e-15** across sizes
512, 1024, 2048, 4096 and silence, DC, alternating-sign, and random frames, testing
both cosine magnitude and quadrature envelope. N=2048/crop=256 took about
**66 microseconds per frame**, including ctypes call overhead in the loop.
See `docs/evidence/unrolled-parity.json`.

## ODFT reconstruction

bfft's ODFT uses frequencies `(k+1/2) * sample_rate / N`, k=0..N/2-1.
It has N/2 complex bins, with no standalone real DC or Nyquist endpoint. It does
not change the sampling Nyquist limit. The native inverse and STFT overlap-add
already implement the corresponding antiperiodic centering convention.

The parent and compiler now accept a reload-time `TRANSFORM="odft"` header,
with `BINS=FFT_SIZE//2`. Absent that header they retain RFFT compatibility.
Reload reconstructs both plans and all state. The parent checks actual bfft bin
counts against the script metadata and exposes transform/bins in its status.

ODFT passed native stereo identity, irregular and in-place buffers, passthrough,
skip/taper/zeros, runtime errors, reload, and a real Cleanup smoke test. Identity
was exact at float32 output in this fixture. These were macOS native tests;
this new parent build has **not** been deployed or checked in Windows SDR#.

ODFT is the intended reconstruction default after deployment and bandwidth-bin
mapping are updated together. A 512-point ODFT at 48 kHz has centers from
46.875 through 23953.125 Hz. The current integer-bin bandwidth formula must be
audited against those half-bin centers when promoting it.

Keep the exact inverse guide's periodic Hann convention separate from the
symmetric Hann and canonical dual used for synthesis. Replacing the 4094-row
guide by a 4096-row IRODFT is an additional mathematical change; it cannot be
called exact equivalence without a new comparison.

## Guide decisions onto the smaller synthesis grid

The exact unrolled row coordinate is `r * sample_rate / (2*(N-1))`.
For N=2048 a 256-row crop reaches 2989.741 Hz at 48 kHz. Synthesis bin k maps to

    r = (k+1/2) * 2*(N_guide-1) / N_synthesis.

`bridge.project_decisions` interpolates bounded real gains using physical Hz and
actual frame-center times. Uncovered frequencies or times receive neutral gain
1; the experiment does not invent guide evidence beyond the crop. The requested
integration must enlarge the crop for a wider demodulated band.

The older offline probe applies a provisional local/global MAN/ATD blend to the
enhanced field and projects those decisions onto a 512-point ODFT, hop 128.
It stores the field, high-resolution decision mask, projected mask, and VAD
evidence separately. This proves the coordinate bridge; it does **not** yet
replace the two-pass recursive Cleanup mask or establish denoising improvement.
Synthesis will retain its complex coefficients and phase, multiplied by the
projected decision. An enhanced positive image has no synthesis phase of its own.

## Whole-overlap squelch counting: correction and calibration

The user's rule is a decision for the **entire overlapping processing frame**:
total qualifying intervals and one longest contiguous qualifying interval are
separate evidence tests. The thresholds were calibrated for the 512-point
analysis geometry. This is not merely a duration-based per-column speech gate.

The downloaded C++ baseline has 192 contextual columns and 64 emitted columns:

- total qualifying count in `[32,161)` (129 possible intervals), strict `>22`;
- longest contiguous qualifying run across `[0,192)`, strict `>16`;
- either condition admits the whole processing block.

`realtime.py` instead uses `[32,160)` for both tests. The C++ comment says its
bounds are equivalent to that Python slice, but they are not. Do not conceal
this difference when calibrating the new behavior.

`wideband_fileIO.py` lines 249–260 explicitly discusses short vowels, fragmented
evidence, an ionosound sweep around 24 intervals, and hop-dependent counts. Its
current wideband geometry is also different from the 512-point audio baseline;
its comments are design intent, not an internally consistent universal formula.

Changing FFT length, overlap, context size, or the detector changes both the
number of available intervals and how many qualifying intervals represent an
event. A longer aperture can blur an event across neighboring columns while the
larger hop reduces the number of columns. Smoothing and phase-lattice fusion
also change the binary run. Thus `threshold *= old_hop/new_hop` alone does not
establish equivalent detection.

`bridge.OverlapEvidence` preserves both measurements and their separate counting
regions. It includes a **candidate**, not promoted, scaling interface: total
threshold tracks available intervals; run threshold takes an explicitly measured
mapping from reference-event intervals to target-event intervals. It does not
guess that mapping from elapsed time. Tests distinguish 23 scattered hits from
17 contiguous hits and preserve the exact strict baseline boundaries.

The next calibration needs noise, ionosound-like sweeps, brief vowels, interrupted
speech, and events straddling the emitted block. Measure the original 512-point
verdict, both counts, and available intervals; repeat at proposed geometries.
Fit the event mapping while tracking false admissions and lost short events.
Run evidence must persist appropriately across block boundaries. Counting
multiple shifted observations of one event must not multiply its evidence.

## Enhanced VAD and intelligent squelch

The extracted `cleanup_shark_vad.py` provides:

- sorted three-frame Cleanup logit correlation;
- quiet-population center tracking with a non-collapsing deviation floor;
- 25 ms normalized pitch autocorrelation, roughly 70–300 Hz;
- spectral prominence measured **before** denoising;
- calibrated Cleanup certification of unvoiced structure, leaving the pitch
  anchor independent (voice strength 0, structure strength 0.125);
- confirmed voice, bounded unvoiced onset/continuation, and trimmed tails.

Nine original VAD tests pass in the copied oracle. On the one-second radio crop,
the offline detector retained approximately 0.341–0.971 seconds and ran in about
7.6 ms. This crop result uses its own quiet-population estimate; it is not a
reproduction of the full-recording calibration or a streaming quality claim.

The original implementation includes full-recording quantiles, zero-phase
`sosfiltfilt`, and retrospective interval commits. Porting it verbatim into a
callback would change causality and violate our Numba contract. The streaming
port needs bounded fixed history, a defined noise bootstrap, sample-time tracker
updates, and delayed commits within the parent's available context. Calibrate
confirmation counts when changing the VAD hop too.

Voice support should strengthen intelligent squelch without erasing meaningful
nonvoice radio events. Preserve Cleanup's whole-overlap evidence path for those
events instead of requiring every accepted event to contain a voiced nucleus.

## Reproduction and current limits

Run from the authoritative repository:

    /Users/ultimussecundai/.local/bin/m4build -- sh tools/research-mini.sh

Retrieve the mirror's `build/true-superresolution` outputs before another sync.
Render `probe.npz` with `python -m research.true_superresolution.render_probe`
using an environment with NumPy and Matplotlib. The research oracle additionally
uses SciPy; this is not a new production Filter.py dependency.

The suite currently contains **20 passing tests**, including native ODFT,
exact bfft inverse fields, decision coordinates, overlap counts, and copied VAD.
The complete later Python field takes about **2.02 seconds per second of mono
audio** on the Mini at its original geometry. Registration still needs a native
or compiled implementation and a measured real-time budget. The core transform
alone is much cheaper; its benchmark must not be reported as the full pipeline.

The exact registered construction, complete recursive mask, expanded guide crop,
physical-coordinate gain transfer and fixed reference evidence now have a live
implementation; see the linked integration document. Remaining work includes
streaming enhanced VAD, guide-lattice event calibration, paired quality evaluation,
and deployment/Reload/Bypass/Squelch checks in actual SDR# with WASAPI.
