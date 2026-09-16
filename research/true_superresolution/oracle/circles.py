"""Fourier-circle pooling and coverage diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PhaseCircleSpectrum:
    spectrum: np.ndarray
    shape: tuple[int, int]


def prepare_phase_circle_spectrum(image: np.ndarray) -> PhaseCircleSpectrum:
    """Cache the observation FFT shared by every pairwise circle edge."""
    value = np.asarray(image, dtype=np.float64)
    if value.ndim != 2:
        raise ValueError("phase-circle spectrum needs one 2-D image")
    height, width = value.shape
    window = np.outer(np.hanning(height), np.hanning(width))
    return PhaseCircleSpectrum(
        np.fft.fft2((value - np.mean(value)) * window),
        (height, width),
    )


def _sample_translation(image: np.ndarray, translation_xy: np.ndarray) -> np.ndarray:
    from scipy.ndimage import map_coordinates

    value = np.asarray(image, dtype=np.float64)
    height, width = value.shape
    yy, xx = np.mgrid[:height, :width]
    return map_coordinates(
        value,
        (yy + float(translation_xy[1]), xx + float(translation_xy[0])),
        order=1,
        mode="reflect",
        prefilter=False,
    )


def radial_bins(shape: tuple[int, int], width: float = 1.0) -> np.ndarray:
    """Integer Fourier-circle index at every unshifted FFT coefficient."""
    height, width_px = map(int, shape)
    fy = np.fft.fftfreq(height) * height
    fx = np.fft.fftfreq(width_px) * width_px
    radius = np.hypot(fy[:, None], fx[None, :])
    return np.floor(radius / max(float(width), 1e-12)).astype(np.int32)


def circle_pool(
    values: np.ndarray,
    *,
    weights: np.ndarray | None = None,
    width: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return weighted circle means and their literal support mass."""
    value = np.asarray(values, dtype=np.float64)
    if value.ndim != 2:
        raise ValueError("circle pooling expects one 2-D Fourier field")
    rings = radial_bins(value.shape, width)
    count = int(rings.max(initial=0)) + 1
    weight = (
        np.ones(value.shape, dtype=np.float64)
        if weights is None
        else np.asarray(weights, dtype=np.float64)
    )
    if weight.shape != value.shape:
        raise ValueError("circle weights must share the Fourier field shape")
    mass = np.bincount(rings.ravel(), weights=weight.ravel(), minlength=count)
    total = np.bincount(
        rings.ravel(), weights=(weight * value).ravel(), minlength=count)
    pooled = total / np.maximum(mass, np.finfo(float).tiny)
    return pooled, mass


def coverage_field(
    transfers: list[np.ndarray],
    precisions: np.ndarray | None = None,
) -> np.ndarray:
    """Fisher-like spectral coverage supplied by all observations."""
    if not transfers:
        raise ValueError("coverage needs at least one transfer function")
    weight = (
        np.ones(len(transfers), dtype=np.float64)
        if precisions is None
        else np.asarray(precisions, dtype=np.float64)
    )
    if weight.shape != (len(transfers),) or np.any(weight <= 0.0):
        raise ValueError("one positive precision is required per observation")
    result = np.zeros_like(np.asarray(transfers[0]), dtype=np.float64)
    for precision, transfer in zip(weight, transfers):
        result += precision * np.abs(transfer) ** 2
    return result


def coverage_report(
    transfers: list[np.ndarray],
    precisions: np.ndarray | None = None,
    *,
    dead_relative: float = 1e-3,
) -> dict[str, object]:
    coverage = coverage_field(transfers, precisions)
    peak = max(float(coverage[0, 0]), np.finfo(float).tiny)
    normalized = coverage / peak
    rings, ring_mass = circle_pool(normalized)
    threshold = max(float(dead_relative), 0.0)
    return {
        "normalized": normalized,
        "circle_mean": rings,
        "circle_mass": ring_mass,
        "dead_relative": threshold,
        "dead_fraction": float(np.mean(normalized < threshold)),
        "minimum": float(np.min(normalized)),
        "median": float(np.median(normalized)),
    }


def _phase_peak(
    correlation: np.ndarray,
) -> tuple[np.ndarray, float, float]:
    """Return wrapped translation and its strongest separated competitor."""
    height, width = correlation.shape
    peak_y, peak_x = np.unravel_index(np.argmax(correlation), correlation.shape)
    peak = float(correlation[peak_y, peak_x])
    excluded = correlation.copy()
    for dy in range(-3, 4):
        for dx in range(-3, 4):
            excluded[(peak_y + dy) % height, (peak_x + dx) % width] = -np.inf
    competitor = float(np.max(excluded))
    offset_x = peak_x if peak_x <= width // 2 else peak_x - width
    offset_y = peak_y if peak_y <= height // 2 else peak_y - height
    return np.asarray((offset_x, offset_y), dtype=np.float64), peak, competitor


def phase_circle_translation(
    first: np.ndarray,
    second: np.ndarray,
    *,
    ring_count: int = 7,
) -> tuple[np.ndarray, dict[str, object]]:
    """Estimate translation as one energy-weighted Fourier-circle path.

    Each smooth annulus supplies a phase-correlation displacement. Their
    positive joint spectral energy induces one barycenter; no ring or peak is
    chosen as a motion class.
    """
    a = np.asarray(first, dtype=np.float64)
    b = np.asarray(second, dtype=np.float64)
    if a.ndim != 2 or b.shape != a.shape:
        raise ValueError("phase-circle translation needs matching 2-D images")
    return phase_circle_translation_from_spectra(
        prepare_phase_circle_spectrum(a),
        prepare_phase_circle_spectrum(b),
        ring_count=ring_count,
    )


def phase_circle_translation_from_spectra(
    first: PhaseCircleSpectrum,
    second: PhaseCircleSpectrum,
    *,
    ring_count: int = 7,
) -> tuple[np.ndarray, dict[str, object]]:
    """Estimate one pair edge while reusing observation-only Fourier state."""
    if first.shape != second.shape:
        raise ValueError("phase-circle spectra must share one raster")
    height, width = first.shape
    spectrum_a = first.spectrum
    spectrum_b = second.spectrum
    cross = spectrum_b * np.conjugate(spectrum_a)
    unit_cross = cross / np.maximum(np.abs(cross), 1e-12)
    frequency_y = np.fft.fftfreq(height)[:, None]
    frequency_x = np.fft.fftfreq(width)[None, :]
    radius = np.sqrt(frequency_x * frequency_x + frequency_y * frequency_y)
    centers = np.linspace(0.035, 0.46, max(int(ring_count), 3))
    bandwidth = 0.055
    joint_amplitude = np.sqrt(
        np.abs(spectrum_a) * np.abs(spectrum_b))
    records = []
    vectors = []
    weights = []
    ambiguities = []
    for center in centers:
        mask = np.exp(-0.5 * ((radius - center) / bandwidth) ** 2)
        correlation = np.fft.ifft2(unit_cross * mask).real
        vector, peak, competitor = _phase_peak(correlation)
        energy = float(np.sum(mask * joint_amplitude))
        ambiguity = float(np.clip(
            competitor / max(peak, 1e-12), 0.0, 1.0))
        records.append({
            "radius_cycles_per_pixel": float(center),
            "translation_xy": vector.tolist(),
            "peak_ambiguity": ambiguity,
            "joint_energy": energy,
            "phase_reliability": float((1.0 - ambiguity) ** 2),
        })
        vectors.append(vector)
        weights.append(energy * max((1.0 - ambiguity) ** 2, 1e-4))
        ambiguities.append(ambiguity)
    vector_array = np.stack(vectors, axis=0)
    weight_array = np.asarray(weights, dtype=np.float64)
    weight_array /= max(float(np.sum(weight_array)), 1e-12)
    mean = np.sum(weight_array[:, None] * vector_array, axis=0)
    dispersion = np.sqrt(np.sum(
        weight_array * np.sum((vector_array - mean[None, :]) ** 2, axis=1)))
    path_length = float(np.sum(np.sqrt(np.sum(
        np.diff(vector_array, axis=0) ** 2, axis=1))))
    return mean, {
        "method": "positive_energy_fourier_circle_phase_transport",
        "ring_records": records,
        "weighted_translation_xy": mean.tolist(),
        "translation_dispersion_pixels": float(dispersion),
        "circle_path_length_pixels": path_length,
        "weighted_peak_ambiguity": float(np.sum(
            weight_array * np.asarray(ambiguities))),
        "maximum_peak_ambiguity": float(np.max(ambiguities)),
    }


def phase_circle_flow(
    first: np.ndarray,
    second: np.ndarray,
    *,
    patch_size: int | None = None,
    stride: int | None = None,
    ring_count: int = 5,
) -> tuple[np.ndarray, dict[str, object]]:
    """Transport overlapping local Fourier circles into one smooth flow field.

    Every chart contributes through a positive spatial window.  Static charts
    remain zero-vector anchors, while residual improvement raises the mass of
    charts carrying observable motion.  The result is a vector barycenter and
    dispersion field, never a winning patch or motion label.
    """
    a = np.asarray(first, dtype=np.float64)
    b = np.asarray(second, dtype=np.float64)
    if a.ndim != 2 or b.shape != a.shape:
        raise ValueError("phase-circle flow needs matching 2-D images")
    height, width = a.shape
    extent = int(patch_size or max(24, round(0.30 * min(a.shape))))
    extent = min(extent, height, width)
    extent = max(extent, 12)
    step = int(stride or max(8, round(0.42 * extent)))
    radius = extent // 2
    pad_before = radius
    pad_after = extent - radius
    padded_a = np.pad(
        a, ((pad_before, pad_after), (pad_before, pad_after)), mode="reflect")
    padded_b = np.pad(
        b, ((pad_before, pad_after), (pad_before, pad_after)), mode="reflect")
    centers_y = np.arange(step // 2, height, step, dtype=np.int64)
    centers_x = np.arange(step // 2, width, step, dtype=np.int64)
    if centers_y[-1] < height - 1 - step // 2:
        centers_y = np.append(centers_y, height - 1 - step // 2)
    if centers_x[-1] < width - 1 - step // 2:
        centers_x = np.append(centers_x, width - 1 - step // 2)

    yy, xx = np.mgrid[:height, :width]
    spatial_sigma = max(0.42 * extent, 1.0)
    numerator = np.zeros((height, width, 2), dtype=np.float64)
    second_moment = np.zeros((height, width), dtype=np.float64)
    spectral_dispersion_numerator = np.zeros(
        (height, width), dtype=np.float64)
    mass = np.zeros((height, width), dtype=np.float64)
    observable_mass = np.zeros((height, width), dtype=np.float64)
    records = []
    translations = []
    for center_y in centers_y:
        for center_x in centers_x:
            patch_a = padded_a[
                center_y:center_y + extent,
                center_x:center_x + extent,
            ]
            patch_b = padded_b[
                center_y:center_y + extent,
                center_x:center_x + extent,
            ]
            translation, circle_record = phase_circle_translation(
                patch_a, patch_b, ring_count=ring_count)
            margin = max(2, extent // 10)
            interior = (slice(margin, -margin), slice(margin, -margin))
            zero_residual = patch_b[interior] - patch_a[interior]
            aligned_residual = (
                _sample_translation(patch_b, translation)[interior]
                - patch_a[interior]
            )
            zero_error = float(np.mean(zero_residual * zero_residual))
            aligned_error = float(np.mean(aligned_residual * aligned_residual))
            improvement = float(np.clip(
                (zero_error - aligned_error) / max(zero_error, 1e-8),
                0.0,
                1.0,
            ))
            magnitude = float(np.linalg.norm(translation))
            dispersion = float(
                circle_record["translation_dispersion_pixels"])
            coherence = float(np.exp(-(
                dispersion / (0.5 + magnitude)) ** 2))
            chart_mass = coherence * (0.10 + 0.90 * np.sqrt(improvement))
            window = np.exp(-0.5 * (
                ((xx - center_x) / spatial_sigma) ** 2
                + ((yy - center_y) / spatial_sigma) ** 2
            ))
            weight = chart_mass * window
            numerator += weight[..., None] * translation[None, None, :]
            second_moment += weight * magnitude * magnitude
            spectral_dispersion_numerator += weight * dispersion
            mass += weight
            observable_mass += weight * improvement
            translations.append(translation)
            records.append({
                "center_xy": [int(center_x), int(center_y)],
                "translation_xy": translation.tolist(),
                "translation_dispersion_pixels": dispersion,
                "zero_residual_mse": zero_error,
                "aligned_residual_mse": aligned_error,
                "residual_improvement": improvement,
                "coherence": coherence,
                "positive_chart_mass": chart_mass,
            })
    safe_mass = np.maximum(mass, np.finfo(float).tiny)
    flow = numerator / safe_mass[..., None]
    variance = np.maximum(
        second_moment / safe_mass - np.sum(flow * flow, axis=2), 0.0)
    dispersion_field = np.sqrt(variance)
    observability = observable_mass / safe_mass
    spectral_dispersion = spectral_dispersion_numerator / safe_mass
    translation_array = np.stack(translations, axis=0)
    return flow, {
        "method": "positive_overlapping_fourier_circle_atlas_transport",
        "patch_size": extent,
        "stride": step,
        "chart_count": len(records),
        "chart_records": records,
        "flow_rms_pixels": float(np.sqrt(np.mean(np.sum(flow * flow, axis=2)))),
        "flow_max_pixels": float(np.max(np.sqrt(np.sum(flow * flow, axis=2)))),
        "dispersion_mean_pixels": float(np.mean(dispersion_field)),
        "dispersion_max_pixels": float(np.max(dispersion_field)),
        "spectral_dispersion_mean_pixels": float(np.mean(
            spectral_dispersion)),
        "spectral_dispersion_max_pixels": float(np.max(
            spectral_dispersion)),
        "observability_mean": float(np.mean(observability)),
        "observability_max": float(np.max(observability)),
        "chart_translation_mean_xy": np.mean(
            translation_array, axis=0).tolist(),
        "dispersion_field": dispersion_field,
        "spectral_dispersion_field": spectral_dispersion,
        "observability_field": observability,
    }
