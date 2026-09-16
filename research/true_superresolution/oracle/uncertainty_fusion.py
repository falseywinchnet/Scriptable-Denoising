"""Registered phase-lattice fusion for the double-IRFFT speech field.

The second IRFFT is a cosine-only observation of each recovered STFT frame.
Moving the analysis centre by a fraction of one hop changes the cosine phase:
one lattice can place a void on a ridge where another lattice places a local
maximum.  This module treats those lattices as complementary observations,
registers them in one overlapping Fourier-circle atlas, and takes their
per-pixel maximum only after transport into that common gauge.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi
from scipy.signal import get_window

from .circles import phase_circle_flow


@dataclass(frozen=True)
class RegisteredMaximum:
    offsets: tuple[int, ...]
    observations: np.ndarray
    aligned: np.ndarray
    fused: np.ndarray
    flows: np.ndarray
    observability: np.ndarray
    diagnostics: tuple[dict[str, object], ...]


@dataclass(frozen=True)
class TextureCartoonCascade:
    input_scaled: np.ndarray
    first_cartoon: np.ndarray
    first_texture: np.ndarray
    texture_cartoon: np.ndarray
    second_texture: np.ndarray
    magnitude_texture_cartoon: np.ndarray
    magnitude_second_texture: np.ndarray
    promoted_trace: np.ndarray
    input_peak: float
    magnitude_texture_peak: float
    texture_offset: float
    texture_scale: float


def _as_float_audio(samples: np.ndarray) -> np.ndarray:
    value = np.asarray(samples)
    if np.issubdtype(value.dtype, np.integer):
        info = np.iinfo(value.dtype)
        scale = float(max(abs(info.min), info.max))
        value = value.astype(np.float64) / scale
    else:
        value = value.astype(np.float64, copy=False)
    if value.ndim == 2:
        value = np.mean(value, axis=1)
    if value.ndim != 1:
        raise ValueError("audio must be mono or samples-by-channels")
    return np.ascontiguousarray(value)


def centered_stft_lattice(
    samples: np.ndarray,
    *,
    n_fft: int,
    hop_length: int,
    center_offset: int = 0,
) -> np.ndarray:
    """Return a librosa-compatible centered Hann STFT at one offset lattice."""
    audio = _as_float_audio(samples)
    if n_fft < 4 or hop_length < 1:
        raise ValueError("invalid STFT geometry")
    frame_count = 1 + audio.size // hop_length
    centers = hop_length * np.arange(frame_count, dtype=np.int64)
    return stft_lattice_at_centers(
        audio,
        centers,
        n_fft=n_fft,
        center_offset=center_offset,
    )


def stft_lattice_at_centers(
    samples: np.ndarray,
    centers: np.ndarray,
    *,
    n_fft: int,
    center_offset: int = 0,
) -> np.ndarray:
    """Return centred Hann spectra for explicitly named sample centres."""
    audio = _as_float_audio(samples)
    center_array = np.asarray(centers, dtype=np.int64)
    if center_array.ndim != 1 or center_array.size < 1:
        raise ValueError("centers must be one nonempty vector")
    extra = abs(int(center_offset))
    pad = n_fft // 2 + extra
    padded = np.pad(audio, (pad, pad), mode="constant")
    starts = (
        pad - n_fft // 2 + int(center_offset) + center_array
    )
    indices = starts[:, None] + np.arange(n_fft, dtype=np.int64)[None, :]
    frames = padded[indices]
    window = get_window("hann", n_fft, fftbins=True).astype(np.float64)
    return np.fft.rfft(frames * window[None, :], axis=1).T


def double_irfft_low_rows(
    spectrum: np.ndarray,
    *,
    n_fft: int,
    crop_rows: int,
) -> np.ndarray:
    """Literal IRFFT followed by default-length IRFFT, then abs/low crop."""
    recovered = np.fft.irfft(np.asarray(spectrum), n=n_fft, axis=0)
    signed = np.fft.irfft(recovered, axis=0)
    expected = 2 * (n_fft - 1)
    if signed.shape[0] != expected:
        raise RuntimeError("unexpected second-IRFFT length")
    if not 0 < crop_rows <= expected:
        raise ValueError("crop_rows lies outside the second-IRFFT field")
    return np.abs(signed[:crop_rows])


def resample_rfft_gain(spectral_gain: np.ndarray, n_fft: int) -> np.ndarray:
    """Interpolate a real nonnegative RFFT gain onto another aperture grid."""

    gain = np.asarray(spectral_gain, dtype=np.float64)
    if (
        gain.ndim != 2
        or gain.shape[1] < 2
        or n_fft < 4
        or not np.all(np.isfinite(gain))
        or np.any(gain < 0.0)
    ):
        raise ValueError("spectral gain has invalid RFFT geometry")
    source = np.linspace(0.0, 1.0, gain.shape[1])
    target = np.linspace(0.0, 1.0, n_fft // 2 + 1)
    return np.stack(
        [np.interp(target, source, row) for row in gain],
        axis=0,
    )


def phase_lattice_observations(
    samples: np.ndarray,
    *,
    n_fft: int = 2048,
    hop_length: int = 512,
    crop_rows: int = 256,
    offsets: tuple[int, ...] = (-379, -241, -64, 0, 83, 214, 427),
) -> np.ndarray:
    """Build complementary double-IRFFT views at sub-hop centre offsets."""
    if 0 not in offsets:
        raise ValueError("offset lattice must include the zero-centred view")
    fields = []
    for offset in offsets:
        spectrum = centered_stft_lattice(
            samples,
            n_fft=n_fft,
            hop_length=hop_length,
            center_offset=offset,
        )
        fields.append(double_irfft_low_rows(
            spectrum, n_fft=n_fft, crop_rows=crop_rows))
    return np.stack(fields, axis=0)


def phase_lattice_observations_at_centers(
    samples: np.ndarray,
    centers: np.ndarray,
    *,
    n_fft: int = 2048,
    crop_rows: int = 256,
    offsets: tuple[int, ...] = (-379, -241, -64, 0, 83, 214, 427),
    spectral_gain: np.ndarray | None = None,
) -> np.ndarray:
    """Build complementary views for a bounded set of global frame centres."""
    if 0 not in offsets:
        raise ValueError("offset lattice must include the zero-centred view")
    gain = None
    if spectral_gain is not None:
        gain = resample_rfft_gain(spectral_gain, n_fft)
        if gain.shape[0] != np.asarray(centers).size:
            raise ValueError("spectral gain and frame centres are misaligned")
    return np.stack([
        double_irfft_low_rows(
            stft_lattice_at_centers(
                samples,
                centers,
                n_fft=n_fft,
                center_offset=offset,
            ) * (1.0 if gain is None else gain.T),
            n_fft=n_fft,
            crop_rows=crop_rows,
        )
        for offset in offsets
    ], axis=0)


def _perceptual_registration_view(field: np.ndarray) -> np.ndarray:
    value = np.maximum(np.asarray(field, dtype=np.float64), 0.0)
    positive = value[value > 0.0]
    scale = (
        float(np.quantile(positive, 0.75))
        if positive.size else 1.0
    )
    compressed = np.log1p(value / max(scale, np.finfo(float).tiny))
    # The rapid cosine nulls are precisely the uncertainty being fused.  If
    # they enter phase correlation they become false landmarks and the atlas
    # aligns every lattice's voids.  Register only the slower ridge envelope;
    # the resulting flow is still applied to the untouched linear field.
    return ndi.gaussian_filter(
        compressed, sigma=(1.0, 3.0), mode="reflect")


def _sample_flow(field: np.ndarray, flow_xy: np.ndarray) -> np.ndarray:
    height, width = field.shape
    yy, xx = np.mgrid[:height, :width]
    return ndi.map_coordinates(
        np.asarray(field, dtype=np.float64),
        (yy + flow_xy[..., 1], xx + flow_xy[..., 0]),
        order=1,
        mode="reflect",
        prefilter=False,
    )


def registered_perceptual_maximum(
    observations: np.ndarray,
    offsets: tuple[int, ...],
    *,
    patch_size: int = 48,
    stride: int = 16,
    ring_count: int = 7,
) -> RegisteredMaximum:
    """Transport all views to the zero lattice and take the literal maximum.

    A robust upper quantile puts every lattice in one amplitude gauge before
    registration.  The logarithm is used only to estimate geometric flow;
    the flow is applied to linear amplitude and the terminal fusion is a
    per-pixel maximum, not an average or a classifier.
    """
    stack = np.asarray(observations, dtype=np.float64)
    if stack.ndim != 3 or stack.shape[0] != len(offsets):
        raise ValueError("observations must have shape lattice-by-row-by-time")
    reference_index = offsets.index(0)
    reference = stack[reference_index]
    reference_q = max(float(np.quantile(reference, 0.995)), 1e-30)
    normalized = []
    for field in stack:
        q = max(float(np.quantile(field, 0.995)), 1e-30)
        normalized.append(field * (reference_q / q))
    normalized_stack = np.stack(normalized, axis=0)
    reference_registration = _perceptual_registration_view(
        normalized_stack[reference_index])

    aligned = []
    flows = []
    observability = []
    records: list[dict[str, object]] = []
    for index, (offset, field) in enumerate(zip(offsets, normalized_stack)):
        if index == reference_index:
            flow = np.zeros((*field.shape, 2), dtype=np.float64)
            transported = field.copy()
            observed = np.ones(field.shape, dtype=np.float64)
            record: dict[str, object] = {
                "method": "identity_reference_chart",
                "center_offset_samples": int(offset),
                "flow_rms_pixels": 0.0,
                "observability_mean": 1.0,
            }
        else:
            flow, record = phase_circle_flow(
                reference_registration,
                _perceptual_registration_view(field),
                patch_size=patch_size,
                stride=stride,
                ring_count=ring_count,
            )
            transported = _sample_flow(field, flow)
            observed = np.asarray(
                record["observability_field"], dtype=np.float64)
            record = {
                key: value
                for key, value in record.items()
                if not isinstance(value, np.ndarray) and key != "chart_records"
            }
            record["center_offset_samples"] = int(offset)
        aligned.append(transported)
        flows.append(flow)
        observability.append(observed)
        records.append(record)
    aligned_stack = np.stack(aligned, axis=0)
    return RegisteredMaximum(
        offsets=offsets,
        observations=normalized_stack,
        aligned=aligned_stack,
        fused=np.max(aligned_stack, axis=0),
        flows=np.stack(flows, axis=0),
        observability=np.stack(observability, axis=0),
        diagnostics=tuple(records),
    )


def texture_cartoon_cascade(
    fused: np.ndarray,
    *,
    threads: int = 4,
    input_peak: float | None = None,
    magnitude_texture_peak: float | None = None,
) -> TextureCartoonCascade:
    """Keep first-pass texture, then keep the cartoon of that texture."""
    value = np.maximum(np.asarray(fused, dtype=np.float64), 0.0)
    peak = max(
        float(np.max(value)) if input_peak is None else float(input_peak),
        1e-30,
    )
    work, first_cartoon, first_texture = first_texture_split(
        value, input_peak=peak, threads=threads)

    low = float(np.min(first_texture))
    span = float(np.max(first_texture) - low)
    scale = 255.0 / span if span > 1e-30 else 1.0
    texture_work = np.ascontiguousarray((first_texture - low) * scale)
    import bfft  # Optional audit only; enhanced-field path does not use Meyer.
    second_cartoon_work, second_texture_work = bfft.meyer_split(
        texture_work, threads=threads)
    texture_cartoon = second_cartoon_work / scale + low
    second_texture = second_texture_work / scale

    magnitude_peak = max(
        float(np.max(np.abs(first_texture)))
        if magnitude_texture_peak is None
        else float(magnitude_texture_peak),
        1e-30,
    )
    magnitude_texture_cartoon, magnitude_second_texture = (
        magnitude_texture_cartoon_split(
            first_texture,
            magnitude_texture_peak=magnitude_peak,
            threads=threads,
        )
    )
    promoted_trace = magnitude_texture_cartoon * (peak / 255.0)
    return TextureCartoonCascade(
        input_scaled=work,
        first_cartoon=first_cartoon,
        first_texture=first_texture,
        texture_cartoon=texture_cartoon,
        second_texture=second_texture,
        magnitude_texture_cartoon=magnitude_texture_cartoon,
        magnitude_second_texture=magnitude_second_texture,
        promoted_trace=promoted_trace,
        input_peak=peak,
        magnitude_texture_peak=magnitude_peak,
        texture_offset=low,
        texture_scale=scale,
    )


def first_texture_split(
    fused: np.ndarray,
    *,
    input_peak: float,
    threads: int = 4,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run the first split in an explicitly shared linear-amplitude gauge."""
    peak = max(float(input_peak), 1e-30)
    work = np.ascontiguousarray(
        np.maximum(np.asarray(fused, dtype=np.float64), 0.0) * (255.0 / peak))
    import bfft
    cartoon, texture = bfft.meyer_split(work, threads=threads)
    return work, cartoon, texture


def magnitude_texture_cartoon_split(
    first_texture: np.ndarray,
    *,
    magnitude_texture_peak: float,
    threads: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """Split abs(first texture) in an explicitly shared texture gauge."""
    peak = max(float(magnitude_texture_peak), 1e-30)
    work = np.ascontiguousarray(
        np.abs(np.asarray(first_texture, dtype=np.float64)) * (255.0 / peak))
    cartoon_work, texture_work = bfft.meyer_split(work, threads=threads)
    return cartoon_work * (peak / 255.0), texture_work * (peak / 255.0)


def strongest_ridge_audit(
    field: np.ndarray,
    *,
    row_limit: int = 128,
) -> dict[str, object]:
    """Measure gaps on one smooth maximum-energy path through low rows."""
    value = np.maximum(np.asarray(field, dtype=np.float64), 0.0)
    value = value[: min(row_limit, value.shape[0])]
    smooth = ndi.gaussian_filter(value, sigma=(1.0, 0.65), mode="nearest")
    rows, columns = smooth.shape
    scale = max(float(np.quantile(smooth, 0.995)), 1e-30)
    penalty = 0.07 * scale
    score = np.empty_like(smooth)
    parent = np.zeros((rows, columns), dtype=np.int16)
    score[:, 0] = smooth[:, 0]
    row_index = np.arange(rows)
    for column in range(1, columns):
        previous = score[:, column - 1]
        for row in range(rows):
            candidates = previous - penalty * np.abs(row_index - row)
            best = int(np.argmax(candidates))
            score[row, column] = smooth[row, column] + candidates[best]
            parent[row, column] = best
    path = np.empty(columns, dtype=np.int64)
    path[-1] = int(np.argmax(score[:, -1]))
    for column in range(columns - 1, 0, -1):
        path[column - 1] = parent[path[column], column]
    values = value[path, np.arange(columns)]
    column_peak = np.max(smooth, axis=0)
    active_threshold = 0.20 * max(float(np.quantile(column_peak, 0.95)), 1e-30)
    active = column_peak >= active_threshold
    active_values = values[active]
    ridge_level = max(
        float(np.quantile(active_values, 0.80)) if active_values.size else 0.0,
        1e-30,
    )
    void = active_values < 0.20 * ridge_level
    return {
        "path_rows": path,
        "path_values": values,
        "active_columns": active,
        "active_column_count": int(np.count_nonzero(active)),
        "ridge_level": ridge_level,
        "void_fraction": float(np.mean(void)) if void.size else 1.0,
        "path_median": float(np.median(active_values)) if active_values.size else 0.0,
        "path_q10": float(np.quantile(active_values, 0.10)) if active_values.size else 0.0,
    }
