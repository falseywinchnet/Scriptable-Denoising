# Mono analysis hook correction — 2026-09-16

The live 2.5–3.5 kHz gap was reproduced without running Cleanup's mask. The SDR#
wrapper had attached a `RealtimeResampler` to `DemodulatorOutput`, a mono float
stream. In the installed SDR# SDK, that resampler's float overload casts the
buffer to complex pairs and divides its length by two. Its sample-rate setting
does not account for consecutive mono samples being paired this way. It also
resamples back and writes its result into the host buffer.

Thus the supposed analysis tap modified the main audio, even when its consumer
was an identity operation. Each component of the falsely paired stream runs at
half the declared mono rate. At a 12 kHz hook rate the round-trip filtering
produces a symmetric notch centered at 3 kHz, inside the 3.8 kHz receiver band.
The wrong analysis coordinates also invalidate decisions projected from that
analysis onto the correctly sampled synthesis grid.

## Direct reproduction on Windows/Wine

`tests/AnalysisTapProbe` calls the installed SDK with unity-amplitude tones and a
no-op processor. It compares the old round-trip with the new production tap.

| Tone, 12 kHz mono source | Old host output amplitude | New analysis amplitude |
| --- | ---: | ---: |
| 1500 Hz | 0.99930 | 1.00001 |
| 2500 Hz | 0.66636 | 0.99999 |
| 2750 Hz | 0.14315 | 1.00000 |
| 3000 Hz | 0.0002783 | 1.00001 |
| 3250 Hz | 0.14316 | 1.00000 |
| 3500 Hz | 0.66635 | 1.00000 |
| 3750 Hz | 0.85892 | 1.00001 |

The old 3 kHz attenuation is approximately 71 dB. All seven 48 kHz controls also
pass. The new tap leaves every input sample bit-for-bit unchanged, and a stereo
test preserves separate channels and exact frame count. Evidence:
`evidence/scratch-workspace/analysis-tap-windows.json`.

## Corrected path

`plugin/AnalysisTap.cs` implements `IRealProcessor` directly. It duplicates each
mono sample into two temporary components at the actual mono sample rate, uses
the SDK's forward-only IQ resampler to obtain 48 kHz analysis, and queues that
analysis. It never resamples back or writes to the radio's stream. At 48 kHz the
resampler is skipped. The FM filtered-audio hook is explicitly interleaved stereo
and preserves both channels. The late content/output hook remains stereo.

Input and output scratch arrays are allocated with the tap. A sample-rate change
reconstructs its resampler and clears queued analysis; there is no per-chunk array
allocation. Chunks are bounded to 1024 frames, and output capacity covers the
supported minimum 1 kHz input rate resampled to 48 kHz with margin.

The correction does not retune Cleanup's noise floor, thresholds, total-count
rule, contiguous-run rule, FFT window, or overlap-add. The user confirmed that the gap was gone and squelch worked with the corrected
spectrum. The former live sample had 102/129 admitted columns and a longest run
of 15, so its total-count condition held it open. A later report of excessive
mask attenuation led to the separate [After routing correction](mask-source.md).

Build the probe with the existing local .NET SDK, then run under Windows with
arguments: SDR# directory containing shark.dll, and output JSON path. SDK
assemblies and native dependencies are used from the installed host and are not
part of the public wrapper-source distribution.
