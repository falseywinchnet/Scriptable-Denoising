"""Multiscale densification of the registered double-IRFFT speech field.

This module deliberately keeps two objects distinct:

* ``unrolled`` is the real, positive double-IRFFT field used by the phone
  comparator;
* ``reassigned_support`` is an exact three-FFT power reassignment raster used
  only to make cross-aperture registration observable.

Reassignment relocates power but does not provide a phase-consistent spectrum,
so it is not inverted into a fabricated waveform.  The terminal field remains
a literal maximum of transported linear-amplitude observations.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi

from .circles import phase_circle_flow

from .uncertainty_fusion import (
    RegisteredMaximum,
    _as_float_audio,
    _perceptual_registration_view,
    _sample_flow,
    phase_lattice_observations_at_centers,
    registered_perceptual_maximum,
    resample_rfft_gain,
)


DEFAULT_APERTURES = (1024, 2048, 4096)
DEFAULT_OFFSETS = (-379, -241, -64, 0, 83, 214, 427)


@dataclass(frozen=True)
class ReassignedSupport:
    power: np.ndarray
    accepted_power: float
    deposited_power: float
    accepted_coefficients: int


@dataclass(frozen=True)
class FourierUnrolledDensified:
    apertures: tuple[int, ...]
    target_n_fft: int
    target_crop_rows: int
    centers: np.ndarray
    source_crop_rows: tuple[int, ...]
    aperture_fields: np.ndarray
    reassigned_support: np.ndarray
    aligned_fields: np.ndarray
    aligned_support: np.ndarray
    flows: np.ndarray
    fused: np.ndarray
    support_union: np.ndarray
    phase_fusions: tuple[RegisteredMaximum, ...]
    diagnostics: tuple[dict[str, object], ...]


@dataclass(frozen=True)
class ReassignedTextureBaseline:
    cartoon: np.ndarray
    texture: np.ndarray
    merged: np.ndarray
    texture_amplitude_scale: float


def reassigned_texture_baseline(
    fusion: np.ndarray,
    reassigned_support_power: np.ndarray,
    *,
    average_size: int = 3,
    gauge_quantile: float = 0.995,
) -> ReassignedTextureBaseline:
    """Promoted positive field: local average plus reassigned support texture.

    Reassigned support is a power observation, so its amplitude coordinate is
    the positive square root.  A shared robust quantile places that amplitude
    in the fusion's linear gauge before exact additive recomposition.
    """
    source = np.maximum(np.asarray(fusion, dtype=np.float64), 0.0)
    power = np.maximum(
        np.asarray(reassigned_support_power, dtype=np.float64), 0.0)
    if source.shape != power.shape or source.ndim != 2:
        raise ValueError("fusion and reassigned support must be equal 2-D fields")
    if average_size < 1 or average_size % 2 != 1:
        raise ValueError("average_size must be a positive odd integer")
    if not 0.0 < gauge_quantile < 1.0:
        raise ValueError("gauge_quantile must lie strictly between zero and one")
    cartoon = ndi.uniform_filter(
        source, size=(average_size, average_size), mode="reflect")
    texture = np.sqrt(power)
    scale = (
        float(np.quantile(source, gauge_quantile))
        / max(float(np.quantile(texture, gauge_quantile)), 1e-30)
    )
    texture = texture * scale
    return ReassignedTextureBaseline(
        cartoon=cartoon,
        texture=texture,
        merged=cartoon + texture,
        texture_amplitude_scale=float(scale),
    )


def source_crop_for_target(
    n_fft: int,
    *,
    target_n_fft: int,
    target_crop_rows: int,
) -> int:
    """Rows needed before mapping an unrolled aperture to the target gauge."""
    if min(n_fft, target_n_fft) < 4 or target_crop_rows < 1:
        raise ValueError("invalid unrolled geometry")
    final_coordinate = (target_crop_rows - 1) * (n_fft - 1) / (
        target_n_fft - 1)
    return min(2 * (n_fft - 1), int(np.ceil(final_coordinate)) + 1)


def resample_unrolled_rows(
    field: np.ndarray,
    *,
    n_fft: int,
    target_n_fft: int,
    target_crop_rows: int,
) -> np.ndarray:
    """Map double-inverse rows by their exact physical-frequency coordinate.

    The second IRFFT has length ``2*(N-1)``.  Therefore its row coordinate is
    proportional to ``N-1``, not to the ordinary STFT bin count.
    """
    value = np.asarray(field, dtype=np.float64)
    if value.ndim != 2:
        raise ValueError("field must be row-by-time")
    source_y = np.arange(target_crop_rows, dtype=np.float64) * (
        (n_fft - 1) / (target_n_fft - 1))
    if source_y[-1] > value.shape[0] - 1 + 1e-9:
        raise ValueError("source field does not cover the target crop")
    source_y = np.minimum(source_y, value.shape[0] - 1)
    return np.stack([
        np.interp(source_y, np.arange(value.shape[0]), value[:, column])
        for column in range(value.shape[1])
    ], axis=1)


def reassigned_power_support_at_centers(
    samples: np.ndarray,
    centers: np.ndarray,
    *,
    n_fft: int,
    hop_length: int,
    target_n_fft: int,
    target_crop_rows: int,
    spectral_gain: np.ndarray | None = None,
) -> ReassignedSupport:
    """Exact symmetric-Hann three-FFT reassignment on a target unrolled grid.

    A reassigned FFT bin ``k_hat`` maps to a target double-IRFFT row as

        row = k_hat * 2*(target_n_fft - 1)/n_fft.

    Bilinear deposition conserves every accepted coefficient whose complete
    four-cell footprint lies in the retained positive-frequency crop.  Power
    is divided by squared coherent gain so apertures share one gauge.
    """
    audio = _as_float_audio(samples)
    center_array = np.asarray(centers, dtype=np.int64)
    if center_array.ndim != 1 or center_array.size < 1:
        raise ValueError("centers must be one nonempty vector")
    if n_fft < 4 or hop_length < 1:
        raise ValueError("invalid reassignment geometry")

    half = n_fft // 2
    pad = half
    padded = np.pad(audio, (pad, pad), mode="constant")
    starts = pad - half + center_array
    indices = starts[:, None] + np.arange(n_fft, dtype=np.int64)[None, :]
    frames = padded[indices]

    sample_index = np.arange(n_fft, dtype=np.float64)
    window = 0.5 - 0.5 * np.cos(2.0 * np.pi * sample_index / (n_fft - 1))
    tau = sample_index - (n_fft - 1) / 2.0
    derivative = np.gradient(window)
    spectrum = np.fft.fft(frames * window[None, :], axis=1)
    time_moment = np.fft.fft(frames * (tau * window)[None, :], axis=1)
    derivative_spectrum = np.fft.fft(
        frames * derivative[None, :], axis=1)

    energy = np.abs(spectrum) ** 2
    threshold = 1e-8 * (np.max(energy, axis=1, keepdims=True) + 1e-30)
    accepted = energy > threshold
    inverse = 1.0 / (energy + 1e-30)
    time_delta = np.real(time_moment * np.conjugate(spectrum)) * inverse
    frequency_cross = np.imag(
        derivative_spectrum * np.conjugate(spectrum))
    nominal_bin = np.arange(n_fft, dtype=np.float64)[None, :]
    reassigned_bin = nominal_bin - frequency_cross * inverse * n_fft / (
        2.0 * np.pi)

    # The equations use the frame centre as the time origin.  Since all
    # apertures share these centres, the destination is expressed directly in
    # target frame columns.
    time_coordinate = (
        np.arange(center_array.size, dtype=np.float64)[:, None]
        + time_delta / hop_length
    )
    row_coordinate = reassigned_bin * (
        2.0 * (target_n_fft - 1) / n_fft)

    t0 = np.floor(time_coordinate).astype(np.int64)
    r0 = np.floor(row_coordinate).astype(np.int64)
    ft = time_coordinate - t0
    fr = row_coordinate - r0
    # Keep positive-frequency support whose bilinear footprint intersects the
    # retained crop.  Boundary time mass is clamped exactly like the native
    # viewer engine.
    valid = accepted & (r0 >= 0) & (r0 < target_crop_rows)
    coherent_gain_squared = max(float(np.sum(window)) ** 2, 1e-30)
    if spectral_gain is not None:
        positive_gain = resample_rfft_gain(spectral_gain, n_fft)
        if positive_gain.shape[0] != center_array.size:
            raise ValueError("spectral gain and reassignment centres are misaligned")
        bins = np.arange(n_fft, dtype=np.int64)
        folded_bins = np.minimum(bins, n_fft - bins)
        full_gain = positive_gain[:, folded_bins]
    else:
        full_gain = 1.0
    normalized_energy = energy * np.square(full_gain) / coherent_gain_squared
    output = np.zeros((target_crop_rows, center_array.size), dtype=np.float64)
    deposited = 0.0
    for dt, wt in ((0, 1.0 - ft), (1, ft)):
        columns = np.clip(t0 + dt, 0, center_array.size - 1)
        for dr, wr in ((0, 1.0 - fr), (1, fr)):
            rows = r0 + dr
            here = valid & (rows >= 0) & (rows < target_crop_rows)
            weights = normalized_energy * wt * wr
            np.add.at(output, (rows[here], columns[here]), weights[here])
            deposited += float(np.sum(weights[here]))
    return ReassignedSupport(
        power=output,
        accepted_power=float(np.sum(normalized_energy[accepted])),
        deposited_power=deposited,
        accepted_coefficients=int(np.count_nonzero(accepted)),
    )


def _shared_scale(value: np.ndarray, reference: np.ndarray) -> np.ndarray:
    source_q = max(float(np.quantile(value, 0.995)), 1e-30)
    reference_q = max(float(np.quantile(reference, 0.995)), 1e-30)
    return np.asarray(value, dtype=np.float64) * (reference_q / source_q)


def _support_registration_view(
    field: np.ndarray,
    support: np.ndarray,
) -> np.ndarray:
    ordinary = _perceptual_registration_view(field)
    positive = support[support > 0.0]
    scale = float(np.quantile(positive, 0.90)) if positive.size else 1.0
    reassigned = np.log1p(support / max(scale, 1e-30))
    reassigned = ndi.gaussian_filter(reassigned, sigma=(1.0, 1.5), mode="reflect")
    ordinary_norm = ordinary / max(float(np.quantile(ordinary, 0.995)), 1e-30)
    reassigned_norm = reassigned / max(
        float(np.quantile(reassigned, 0.995)), 1e-30)
    # Equal positive coordinate witnesses.  Neither term enters the terminal
    # amplitude raster; they only determine transport.
    return ordinary_norm + reassigned_norm


def fourier_unrolled_densified_at_centers(
    samples: np.ndarray,
    centers: np.ndarray,
    *,
    apertures: tuple[int, ...] = DEFAULT_APERTURES,
    target_n_fft: int = 2048,
    target_crop_rows: int = 256,
    hop_length: int = 512,
    offsets: tuple[int, ...] = DEFAULT_OFFSETS,
    patch_size: int = 48,
    stride: int = 16,
    ring_count: int = 7,
    spectral_gain: np.ndarray | None = None,
) -> FourierUnrolledDensified:
    """Build and align the 1024/2048/4096 unrolled aperture family."""
    aperture_tuple = tuple(int(value) for value in apertures)
    if target_n_fft not in aperture_tuple:
        raise ValueError("target_n_fft must be one of the apertures")
    if len(set(aperture_tuple)) != len(aperture_tuple):
        raise ValueError("apertures must be unique")
    center_array = np.asarray(centers, dtype=np.int64)
    if spectral_gain is not None:
        gain = np.asarray(spectral_gain, dtype=np.float64)
        if gain.ndim != 2 or gain.shape[0] != center_array.size:
            raise ValueError("spectral gain and densified centres are misaligned")
    else:
        gain = None

    crops = tuple(source_crop_for_target(
        n_fft,
        target_n_fft=target_n_fft,
        target_crop_rows=target_crop_rows,
    ) for n_fft in aperture_tuple)
    phase_fusions = []
    fields = []
    supports = []
    support_records = []
    for n_fft, crop_rows in zip(aperture_tuple, crops):
        observations = phase_lattice_observations_at_centers(
            samples,
            center_array,
            n_fft=n_fft,
            crop_rows=crop_rows,
            offsets=offsets,
            spectral_gain=gain,
        )
        phase_fusion = registered_perceptual_maximum(
            observations,
            offsets,
            patch_size=patch_size,
            stride=stride,
            ring_count=ring_count,
        )
        phase_fusions.append(phase_fusion)
        fields.append(resample_unrolled_rows(
            phase_fusion.fused,
            n_fft=n_fft,
            target_n_fft=target_n_fft,
            target_crop_rows=target_crop_rows,
        ))
        reassigned = reassigned_power_support_at_centers(
            samples,
            center_array,
            n_fft=n_fft,
            hop_length=hop_length,
            target_n_fft=target_n_fft,
            target_crop_rows=target_crop_rows,
            spectral_gain=gain,
        )
        supports.append(reassigned.power)
        support_records.append(reassigned)

    reference_index = aperture_tuple.index(target_n_fft)
    reference_field = fields[reference_index]
    reference_support = supports[reference_index]
    scaled_fields = np.stack([
        _shared_scale(value, reference_field) for value in fields
    ])
    scaled_support = np.stack([
        _shared_scale(value, reference_support) for value in supports
    ])
    reference_view = _support_registration_view(
        scaled_fields[reference_index], scaled_support[reference_index])

    aligned_fields = []
    aligned_support = []
    flows = []
    records = []
    for index, n_fft in enumerate(aperture_tuple):
        if index == reference_index:
            flow = np.zeros((*reference_field.shape, 2), dtype=np.float64)
            field = scaled_fields[index].copy()
            support = scaled_support[index].copy()
            record = {
                "method": "identity_target_aperture",
                "n_fft": n_fft,
                "flow_rms_pixels": 0.0,
            }
        else:
            flow, detail = phase_circle_flow(
                reference_view,
                _support_registration_view(
                    scaled_fields[index], scaled_support[index]),
                patch_size=patch_size,
                stride=stride,
                ring_count=ring_count,
            )
            field = _sample_flow(scaled_fields[index], flow)
            support = _sample_flow(scaled_support[index], flow)
            record = {
                key: value for key, value in detail.items()
                if not isinstance(value, np.ndarray) and key != "chart_records"
            }
            record["n_fft"] = n_fft
        record.update({
            "source_crop_rows": crops[index],
            "reassigned_accepted_coefficients": (
                support_records[index].accepted_coefficients),
            "reassigned_accepted_power": support_records[index].accepted_power,
            "reassigned_deposited_crop_power": (
                support_records[index].deposited_power),
            "spectral_gain_applied": gain is not None,
        })
        aligned_fields.append(field)
        aligned_support.append(support)
        flows.append(flow)
        records.append(record)

    aligned_field_stack = np.stack(aligned_fields)
    aligned_support_stack = np.stack(aligned_support)
    return FourierUnrolledDensified(
        apertures=aperture_tuple,
        target_n_fft=target_n_fft,
        target_crop_rows=target_crop_rows,
        centers=center_array,
        source_crop_rows=crops,
        aperture_fields=scaled_fields,
        reassigned_support=scaled_support,
        aligned_fields=aligned_field_stack,
        aligned_support=aligned_support_stack,
        flows=np.stack(flows),
        fused=np.max(aligned_field_stack, axis=0),
        support_union=np.max(aligned_support_stack, axis=0),
        phase_fusions=tuple(phase_fusions),
        diagnostics=tuple(records),
    )


def fourier_unrolled_densified(
    samples: np.ndarray,
    *,
    apertures: tuple[int, ...] = DEFAULT_APERTURES,
    target_n_fft: int = 2048,
    target_crop_rows: int = 256,
    hop_length: int = 512,
    offsets: tuple[int, ...] = DEFAULT_OFFSETS,
    spectral_gain: np.ndarray | None = None,
) -> FourierUnrolledDensified:
    audio = _as_float_audio(samples)
    centers = hop_length * np.arange(
        1 + audio.size // hop_length, dtype=np.int64)
    return fourier_unrolled_densified_at_centers(
        audio,
        centers,
        apertures=apertures,
        target_n_fft=target_n_fft,
        target_crop_rows=target_crop_rows,
        hop_length=hop_length,
        offsets=offsets,
        spectral_gain=spectral_gain,
    )
