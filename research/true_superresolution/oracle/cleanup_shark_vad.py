"""Cleanup-statistic + SHARK-style voice anchoring for speech crops.

This is a deterministic front-end experiment.  It borrows Cleanup's sorted
three-frame logit-correlation statistic and local/global MAN/ATD blending, then
uses SHARK-style normalized pitch periodicity and spectral prominence.  A
delayed-commit state machine anchors unvoiced structure to nearby voiced
evidence instead of treating VAD as a destructive per-frame mute.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import numpy as np
from scipy.signal import butter, sosfiltfilt
from scipy.special import expit


class SpeechState(IntEnum):
    QUIET = 0
    CANDIDATE = 1
    VOICED = 2
    UNVOICED = 3
    HANGOVER = 4


@dataclass(frozen=True)
class CleanupSharkConfig:
    n_fft: int = 512
    hop_length: int = 512
    cleanup_bins: int = 37
    cleanup_smoothing_frames: int = 15
    pitch_window_seconds: float = 0.025
    pitch_low_hz: float = 70.0
    pitch_high_hz: float = 300.0
    pitch_filter_high_hz: float = 1000.0
    pitch_threshold: float = 0.4
    prominence_low_hz: float = 200.0
    prominence_high_hz: float = 2000.0
    prominence_threshold_db: float = 6.5
    spectral_bonus_cap: float = 0.25
    voice_threshold: float = 0.95
    cleanup_z_center: float = 1.5
    cleanup_z_slope: float = 1.5
    cleanup_voice_certifier_strength: float = 0.0
    cleanup_structure_certifier_strength: float = 0.125
    voice_confirm_frames: int = 2
    voice_confirm_window: int = 3
    hangover_seconds: float = 0.20
    preroll_seconds: float = 0.12
    tail_seconds: float = 0.025
    minimum_speech_seconds: float = 0.08
    unvoiced_threshold: float = 0.58
    tracker_alpha: float = 0.02
    vocalization_lead_seconds: float = 0.192
    vocalization_tail_seconds: float = 0.075
    vocalization_gap_seconds: float = 0.025
    minimum_voiced_core_seconds: float = 0.020

    def __post_init__(self) -> None:
        if (
            self.n_fft < 64
            or self.hop_length < 1
            or self.cleanup_bins < 4
            or self.cleanup_bins > self.n_fft // 2 + 1
            or self.cleanup_smoothing_frames < 1
            or self.voice_confirm_frames < 1
            or self.voice_confirm_window < self.voice_confirm_frames
            or min(
                self.pitch_window_seconds,
                self.pitch_low_hz,
                self.pitch_high_hz,
                self.prominence_low_hz,
                self.prominence_high_hz,
                self.hangover_seconds,
                self.minimum_speech_seconds,
                self.vocalization_lead_seconds,
                self.vocalization_tail_seconds,
                self.vocalization_gap_seconds,
                self.minimum_voiced_core_seconds,
            )
            <= 0.0
            or self.pitch_low_hz >= self.pitch_high_hz
            or self.prominence_low_hz >= self.prominence_high_hz
            or not 0.0 <= self.cleanup_voice_certifier_strength <= 0.25
            or not 0.0 <= self.cleanup_structure_certifier_strength <= 0.25
        ):
            raise ValueError("invalid Cleanup/SHARK VAD configuration")


@dataclass(frozen=True)
class SpeechInterval:
    frame0: int
    frame1: int
    seconds0: float
    seconds1: float
    voiced_frames: int
    unvoiced_frames: int
    mean_fused_score: float
    peak_fused_score: float


@dataclass(frozen=True)
class CleanupSharkAnalysis:
    cleanup_similarity: np.ndarray
    cleanup_center: np.ndarray
    cleanup_atd: np.ndarray
    cleanup_probability: np.ndarray
    pitch_periodicity: np.ndarray
    pitch_hz: np.ndarray
    spectral_prominence_db: np.ndarray
    energy_snr_db: np.ndarray
    shark_score: np.ndarray
    fused_score: np.ndarray
    unvoiced_score: np.ndarray
    spectral_gain: np.ndarray
    denoised_magnitude: np.ndarray
    state: np.ndarray
    speech_mask: np.ndarray
    voiced_mask: np.ndarray
    unvoiced_mask: np.ndarray
    vocalization_mask: np.ndarray
    intervals: tuple[SpeechInterval, ...]
    vocalizations: tuple[SpeechInterval, ...]
    noise_frame_count: int


def true_logistic_reference(count: int) -> np.ndarray:
    """Reproduce Cleanup's finite normalized logit reference."""

    if count < 4:
        raise ValueError("logistic reference needs at least four points")
    probability = np.arange(count, dtype=np.float64) / (count - 1)
    result = np.empty(count, dtype=np.float64)
    result[1:-1] = np.log(probability[1:-1] / (1.0 - probability[1:-1]))
    result[-1] = 2.0 * result[-2] - result[-3]
    result[0] = -result[-1]
    result -= result[0]
    result /= result[-1]
    return result


def cleanup_similarity_from_magnitude(
    magnitude: np.ndarray,
    *,
    bins: int = 37,
    smoothing_frames: int = 15,
) -> np.ndarray:
    """Cleanup's normalized three-timeframe logit-correlation statistic."""

    values = np.asarray(magnitude, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] < bins or bins < 4:
        raise ValueError("Cleanup similarity has incompatible magnitudes")
    reference = true_logistic_reference(3 * bins)
    spike = np.zeros(3 * bins, dtype=np.float64)
    spike[-1] = 1.0
    maximum = 1.0 - float(np.corrcoef(spike, reference)[0, 1])
    padded = np.pad(values[:, :bins], ((1, 1), (0, 0)), mode="edge")
    output = np.zeros(values.shape[0], dtype=np.float64)
    for index in range(values.shape[0]):
        ordered = np.sort(padded[index : index + 3].reshape(-1))
        extent = float(ordered[-1] - ordered[0])
        if extent <= 1e-30:
            continue
        normalized = (ordered - ordered[0]) / extent
        correlation = float(np.corrcoef(normalized, reference)[0, 1])
        if np.isfinite(correlation):
            output[index] = (1.0 - correlation) / max(maximum, 1e-30)
    length = max(1, int(smoothing_frames))
    left = np.arange(1, (length + 1) // 2 + 1, dtype=np.float64)
    right = np.arange(length // 2, 0, -1, dtype=np.float64)
    kernel = np.concatenate((left, right))
    kernel /= np.sum(kernel)
    full = np.convolve(output, kernel, mode="full")
    start = (kernel.size - 1) // 2
    return full[start : start + output.size]


def cleanup_man_atd(values: np.ndarray) -> tuple[float, float]:
    """Return Cleanup's MAN and ATD statistics over finite nonzero values."""

    data = np.asarray(values, dtype=np.float64)
    data = data[np.isfinite(data) & (data != 0.0)]
    if not data.size:
        return 0.0, 0.0
    median = float(np.median(data))
    man = float(np.median(np.abs(data - median)))
    atd = float(np.sqrt(np.mean((data - man) ** 2)))
    return man, atd


def _centered_frames(
    samples: np.ndarray, centers: np.ndarray, length: int
) -> np.ndarray:
    left = length // 2
    right = length - left
    padded = np.pad(samples, (left, right), mode="constant")
    windows = np.lib.stride_tricks.sliding_window_view(padded, length)
    return np.asarray(windows[centers], dtype=np.float64)


def normalized_pitch_periodicity(
    samples: np.ndarray,
    centers: np.ndarray,
    *,
    sample_rate: int,
    window_seconds: float = 0.025,
    low_hz: float = 70.0,
    high_hz: float = 300.0,
    filter_high_hz: float = 1000.0,
    block_size: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    """Exact overlap-normalized autocorrelation over the SHARK pitch range."""

    waveform = np.asarray(samples, dtype=np.float64)
    if waveform.ndim != 1 or sample_rate < 1:
        raise ValueError("pitch periodicity expects mono audio")
    nyquist = sample_rate / 2.0
    high_cut = min(filter_high_hz, 0.95 * nyquist)
    sos = butter(
        4,
        (max(30.0, low_hz * 0.70), high_cut),
        btype="bandpass",
        fs=sample_rate,
        output="sos",
    )
    filtered = sosfiltfilt(sos, waveform)
    length = max(64, int(round(window_seconds * sample_rate)))
    minimum_lag = max(1, int(np.floor(sample_rate / high_hz)))
    maximum_lag = min(length - 2, int(np.ceil(sample_rate / low_hz)))
    lags = np.arange(minimum_lag, maximum_lag + 1, dtype=np.int64)
    n_fft = 1 << int(np.ceil(np.log2(2 * length - 1)))
    periodicity = np.zeros(centers.size, dtype=np.float64)
    pitch_hz = np.zeros(centers.size, dtype=np.float64)
    for start in range(0, centers.size, block_size):
        stop = min(centers.size, start + block_size)
        frames = _centered_frames(filtered, centers[start:stop], length)
        frames -= np.mean(frames, axis=1, keepdims=True)
        spectrum = np.fft.rfft(frames, n=n_fft, axis=1)
        correlation = np.fft.irfft(
            spectrum * np.conjugate(spectrum), n=n_fft, axis=1
        )[:, lags]
        cumulative = np.concatenate(
            (
                np.zeros((frames.shape[0], 1), dtype=np.float64),
                np.cumsum(frames * frames, axis=1),
            ),
            axis=1,
        )
        left_energy = cumulative[:, length - lags]
        right_energy = cumulative[:, length, None] - cumulative[:, lags]
        normalized = correlation / np.sqrt(
            np.maximum(left_energy * right_energy, 1e-30)
        )
        best = np.argmax(normalized, axis=1)
        periodicity[start:stop] = normalized[np.arange(stop - start), best]
        pitch_hz[start:stop] = sample_rate / lags[best]
    return np.clip(periodicity, -1.0, 1.0), pitch_hz


def _robust_center_atd(values: np.ndarray) -> tuple[float, float]:
    center = float(np.median(values))
    atd = float(np.sqrt(np.mean((values - center) ** 2)))
    return center, max(atd, 1e-6)


def _track_cleanup_probability(
    similarity: np.ndarray,
    pitch: np.ndarray,
    energy_snr_db: np.ndarray,
    noise_mask: np.ndarray,
    config: CleanupSharkConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    center, scale = _robust_center_atd(similarity[noise_mask])
    # Cleanup blends each local estimate back toward its global statistic.
    # Retain that global uncertainty as a floor: otherwise a long quiet run
    # collapses the adaptive deviation and makes the next harmless fluctuation
    # look infinitely significant.
    global_center = center
    global_scale = scale
    centers = np.empty_like(similarity)
    scales = np.empty_like(similarity)
    probability = np.empty_like(similarity)
    for index, value in enumerate(similarity):
        centers[index] = center
        scales[index] = scale
        z_score = (value - center) / max(scale, 1e-6)
        probability[index] = expit(
            config.cleanup_z_slope * (z_score - config.cleanup_z_center)
        )
        quiet_update = (
            pitch[index] < config.pitch_threshold
            and energy_snr_db[index] < 3.0
            and z_score < 1.0
        )
        if quiet_update:
            alpha = config.tracker_alpha
            center += alpha * (value - center)
            deviation = abs(value - center)
            scale += alpha * (deviation - scale)
            center = float(
                np.clip(
                    center,
                    global_center - global_scale,
                    global_center + global_scale,
                )
            )
            scale = max(scale, 0.5 * global_scale, 1e-6)
    return centers, scales, probability


def fuse_cleanup_shark_evidence(
    shark_score: np.ndarray,
    cleanup_probability: np.ndarray,
    structure: np.ndarray,
    *,
    voice_certifier_strength: float,
    structure_certifier_strength: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Continuously certify SHARK voice and structure with Cleanup statistics."""

    shark = np.asarray(shark_score, dtype=np.float64)
    cleanup = np.asarray(cleanup_probability, dtype=np.float64)
    support = np.asarray(structure, dtype=np.float64)
    if (
        shark.ndim != 1
        or cleanup.shape != shark.shape
        or support.shape != shark.shape
        or not np.all(np.isfinite(shark))
        or not np.all(np.isfinite(cleanup))
        or not np.all(np.isfinite(support))
        or not 0.0 <= voice_certifier_strength <= 0.25
        or not 0.0 <= structure_certifier_strength <= 0.25
    ):
        raise ValueError("Cleanup/SHARK evidence has incompatible geometry")
    cleanup = np.clip(cleanup, 0.0, 1.0)
    support = np.clip(support, 0.0, 1.0)
    fused = shark * (1.0 - voice_certifier_strength * (1.0 - cleanup))
    unvoiced = np.sqrt(support) * cleanup ** (2.0 * structure_certifier_strength)
    return fused, unvoiced


def speech_state_machine(
    fused_score: np.ndarray,
    unvoiced_score: np.ndarray,
    *,
    sample_rate: int,
    config: CleanupSharkConfig,
) -> tuple[np.ndarray, np.ndarray, tuple[SpeechInterval, ...]]:
    """Anchor unvoiced spans to debounced voiced evidence with bounded memory."""

    fused = np.asarray(fused_score, dtype=np.float64)
    unvoiced = np.asarray(unvoiced_score, dtype=np.float64)
    if fused.ndim != 1 or unvoiced.shape != fused.shape:
        raise ValueError("state-machine evidence has incompatible shapes")
    count = fused.size
    state = np.full(count, SpeechState.QUIET, dtype=np.int8)
    voice_hit = fused >= config.voice_threshold
    structured = unvoiced >= config.unvoiced_threshold
    hangover = max(
        1, int(round(config.hangover_seconds * sample_rate / config.hop_length))
    )
    preroll = max(
        0, int(round(config.preroll_seconds * sample_rate / config.hop_length))
    )
    tail = max(0, int(round(config.tail_seconds * sample_rate / config.hop_length)))
    minimum = max(
        1,
        int(round(config.minimum_speech_seconds * sample_rate / config.hop_length)),
    )
    intervals: list[SpeechInterval] = []
    pending_start: int | None = None
    active_start: int | None = None
    last_voice = -10 * hangover
    last_content = -1

    def confirm(index: int) -> bool:
        start = max(0, index - config.voice_confirm_window + 1)
        return int(np.sum(voice_hit[start : index + 1])) >= config.voice_confirm_frames

    def commit(stop_hint: int) -> None:
        nonlocal active_start, pending_start, last_voice, last_content
        if active_start is None:
            return
        content_stop = min(
            count,
            stop_hint,
            max(active_start, last_content + 1) + tail,
        )
        state[max(last_content + 1, active_start) : content_stop] = SpeechState.HANGOVER
        if content_stop - active_start >= minimum:
            local = state[active_start:content_stop]
            voiced_frames = int(np.sum(local == SpeechState.VOICED))
            unvoiced_frames = int(np.sum(local == SpeechState.UNVOICED))
            if voiced_frames >= config.voice_confirm_frames:
                intervals.append(
                    SpeechInterval(
                        frame0=active_start,
                        frame1=content_stop,
                        seconds0=active_start * config.hop_length / sample_rate,
                        seconds1=content_stop * config.hop_length / sample_rate,
                        voiced_frames=voiced_frames,
                        unvoiced_frames=unvoiced_frames,
                        mean_fused_score=float(
                            np.mean(fused[active_start:content_stop])
                        ),
                        peak_fused_score=float(
                            np.max(fused[active_start:content_stop])
                        ),
                    )
                )
            else:
                state[active_start:content_stop] = SpeechState.QUIET
        else:
            state[active_start:content_stop] = SpeechState.QUIET
        if stop_hint > content_stop:
            state[content_stop:stop_hint] = SpeechState.QUIET
        active_start = None
        pending_start = None
        last_voice = -10 * hangover
        last_content = -1

    for index in range(count):
        if active_start is None:
            if structured[index]:
                if pending_start is None:
                    pending_start = index
                pending_start = max(pending_start, index - preroll)
                state[index] = SpeechState.CANDIDATE
            elif pending_start is not None and index - pending_start > preroll:
                state[pending_start:index] = SpeechState.QUIET
                pending_start = None
            if voice_hit[index] and confirm(index):
                onset = max(0, index - config.voice_confirm_window + 1)
                active_start = (
                    max(pending_start, index - preroll)
                    if pending_start is not None
                    else onset
                )
                for frame in range(active_start, index + 1):
                    state[frame] = (
                        SpeechState.VOICED
                        if voice_hit[frame]
                        else SpeechState.UNVOICED
                    )
                last_voice = index
                last_content = index
                pending_start = None
            continue

        if voice_hit[index]:
            state[index] = SpeechState.VOICED
            last_voice = index
            last_content = index
        elif structured[index] and index - last_voice <= hangover:
            state[index] = SpeechState.UNVOICED
            last_content = index
        else:
            state[index] = SpeechState.HANGOVER
        if index - last_voice > hangover:
            # A newly structured frame belongs to the next candidate, so the
            # previous crop's cosmetic tail must stop before it.
            commit(index if structured[index] else index + 1)
            if structured[index]:
                pending_start = index
                state[index] = SpeechState.CANDIDATE
            else:
                state[index] = SpeechState.QUIET
    if active_start is not None:
        commit(count)
    elif pending_start is not None:
        state[pending_start:] = SpeechState.QUIET
    speech_mask = np.zeros(count, dtype=bool)
    for interval in intervals:
        speech_mask[interval.frame0 : interval.frame1] = True
    return state, speech_mask, tuple(intervals)


def voiced_vocalization_intervals(
    voiced_mask: np.ndarray,
    fused_score: np.ndarray,
    *,
    sample_rate: int,
    config: CleanupSharkConfig,
) -> tuple[np.ndarray, tuple[SpeechInterval, ...]]:
    """Bound vocalizations around persistent voiced nuclei.

    The ordinary speech mask deliberately retains spectrally supported
    unvoiced material.  That is useful for speech gating, but it is the wrong
    object for word-sized vocalization crops: an energy-only noise component
    can be attached to the next voiced interval.  Here isolated voice hits are
    first rejected, only very short pitch dropouts inside a stable nucleus are
    bridged, and the surviving nuclei receive finite asymmetric margins.
    """

    voiced = np.asarray(voiced_mask, dtype=bool)
    fused = np.asarray(fused_score, dtype=np.float64)
    if voiced.ndim != 1 or fused.shape != voiced.shape or sample_rate < 1:
        raise ValueError("vocalization evidence has incompatible geometry")

    def runs(mask: np.ndarray) -> list[tuple[int, int]]:
        edges = np.diff(np.r_[False, mask, False].astype(np.int8))
        return list(
            zip(
                np.flatnonzero(edges == 1).tolist(),
                np.flatnonzero(edges == -1).tolist(),
                strict=True,
            )
        )

    frames_per_second = sample_rate / config.hop_length
    minimum_core = max(
        config.voice_confirm_frames,
        int(round(config.minimum_voiced_core_seconds * frames_per_second)),
    )
    maximum_gap = max(
        0, int(round(config.vocalization_gap_seconds * frames_per_second))
    )
    stable = voiced.copy()
    for frame0, frame1 in runs(stable):
        if frame1 - frame0 < minimum_core:
            stable[frame0:frame1] = False

    nuclei = runs(stable)
    for (_, left1), (right0, _) in zip(nuclei[:-1], nuclei[1:], strict=True):
        if right0 - left1 <= maximum_gap:
            stable[left1:right0] = True
    nuclei = runs(stable)

    lead = max(0, int(round(config.vocalization_lead_seconds * frames_per_second)))
    tail = max(0, int(round(config.vocalization_tail_seconds * frames_per_second)))
    proposed = [
        [max(0, core0 - lead), min(voiced.size, core1 + tail), core0, core1]
        for core0, core1 in nuclei
    ]
    # Margins may approach each other, but separate voiced nuclei must remain
    # separate vocalizations.  Divide any overlap at the midpoint of the
    # measured inter-nucleus gap instead of merging on padding alone.
    for left, right in zip(proposed[:-1], proposed[1:], strict=True):
        if left[1] > right[0]:
            split = (left[3] + right[2]) // 2
            left[1] = max(left[3], split)
            right[0] = min(right[2], split)

    mask = np.zeros(voiced.size, dtype=bool)
    intervals: list[SpeechInterval] = []
    for frame0, frame1, _, _ in proposed:
        if frame1 <= frame0:
            continue
        mask[frame0:frame1] = True
        local_voiced = int(np.sum(stable[frame0:frame1]))
        intervals.append(
            SpeechInterval(
                frame0=int(frame0),
                frame1=int(frame1),
                seconds0=frame0 * config.hop_length / sample_rate,
                seconds1=frame1 * config.hop_length / sample_rate,
                voiced_frames=local_voiced,
                unvoiced_frames=int(frame1 - frame0 - local_voiced),
                mean_fused_score=float(np.mean(fused[frame0:frame1])),
                peak_fused_score=float(np.max(fused[frame0:frame1])),
            )
        )
    return mask, tuple(intervals)


def analyze_cleanup_shark(
    samples: np.ndarray,
    sample_rate: int,
    *,
    frame_count: int | None = None,
    config: CleanupSharkConfig = CleanupSharkConfig(),
) -> CleanupSharkAnalysis:
    """Measure, conservatively denoise, and bound speech in one mono recording."""

    waveform = np.asarray(samples, dtype=np.float64)
    if waveform.ndim != 1 or not waveform.size or sample_rate < 1:
        raise ValueError("Cleanup/SHARK analysis expects nonempty mono audio")
    expected = 1 + waveform.size // config.hop_length
    count = expected if frame_count is None else int(frame_count)
    if count < 1 or count > expected:
        raise ValueError("requested frame count exceeds the waveform geometry")
    centers = np.arange(count, dtype=np.int64) * config.hop_length
    frames = _centered_frames(waveform, centers, config.n_fft)
    window = np.hanning(config.n_fft)
    spectrum = np.fft.rfft(frames * window[None, :], axis=1)
    magnitude = np.abs(spectrum)
    power = magnitude * magnitude
    frequencies = np.fft.rfftfreq(config.n_fft, 1.0 / sample_rate)

    cleanup_similarity = cleanup_similarity_from_magnitude(
        magnitude,
        bins=config.cleanup_bins,
        smoothing_frames=config.cleanup_smoothing_frames,
    )
    pitch, pitch_hz = normalized_pitch_periodicity(
        waveform,
        centers,
        sample_rate=sample_rate,
        window_seconds=config.pitch_window_seconds,
        low_hz=config.pitch_low_hz,
        high_hz=config.pitch_high_hz,
        filter_high_hz=config.pitch_filter_high_hz,
    )
    broad = (frequencies >= config.prominence_low_hz) & (
        frequencies <= 4000.0
    )
    log_energy = 10.0 * np.log10(np.maximum(np.sum(power[:, broad], axis=1), 1e-30))
    quiet_energy = log_energy <= np.quantile(log_energy, 0.25)
    noise_mask = quiet_energy & (pitch < config.pitch_threshold)
    if int(np.sum(noise_mask)) < max(8, count // 100):
        noise_mask = quiet_energy
    noise_power = np.quantile(power[noise_mask], 0.50, axis=0)
    noise_magnitude = np.sqrt(np.maximum(noise_power, 1e-30))
    energy_snr_db = 10.0 * np.log10(
        np.maximum(np.sum(power[:, broad], axis=1), 1e-30)
        / max(float(np.sum(noise_power[broad])), 1e-30)
    )

    speech_band = (frequencies >= config.prominence_low_hz) & (
        frequencies <= config.prominence_high_hz
    )
    noise_values = magnitude[noise_mask][:, speech_band]
    global_man, global_atd = cleanup_man_atd(noise_values)
    global_atd = max(global_atd, 1e-12)
    local_man = np.empty(count, dtype=np.float64)
    local_atd = np.empty(count, dtype=np.float64)
    for index in range(count):
        local_man[index], local_atd[index] = cleanup_man_atd(
            magnitude[index, speech_band]
        )
    man_weight = np.exp(-0.5 * np.abs(local_man - global_man) / global_atd)
    atd_weight = np.exp(-0.5 * np.abs(local_atd - global_atd) / global_atd)
    fixed_man = local_man * man_weight + global_man * (1.0 - man_weight)
    fixed_atd = local_atd * atd_weight + global_atd * (1.0 - atd_weight)
    threshold = fixed_man + 0.5 * fixed_atd
    statistical_gain = expit(
        (magnitude - threshold[:, None]) / np.maximum(fixed_atd[:, None], 1e-12)
    )
    noise_gain = np.clip(
        1.0 - noise_magnitude[None, :] / np.maximum(magnitude, 1e-30),
        0.0,
        1.0,
    )
    spectral_gain = statistical_gain * noise_gain
    denoised = magnitude * spectral_gain

    # Prominence is evidence used to decide whether to gate.  Measuring it
    # after subtractive gating is circular: bins driven to zero manufacture a
    # fantastically low background and false 100+ dB "peaks".  SHARK's
    # quieter-population background therefore belongs on the ungated spectrum;
    # ``denoised`` remains the extraction product returned to downstream code.
    band_values = magnitude[:, speech_band]
    band_db = 20.0 * np.log10(np.maximum(band_values, 1e-12))
    if band_db.shape[1] >= 3:
        band_db = (
            0.25 * band_db[:, :-2]
            + 0.50 * band_db[:, 1:-1]
            + 0.25 * band_db[:, 2:]
        )
    quieter_limit = np.quantile(band_db, 0.60, axis=1)
    background = np.asarray(
        [
            np.mean(row[row <= limit])
            for row, limit in zip(band_db, quieter_limit, strict=True)
        ]
    )
    prominence = np.max(band_db, axis=1) - background

    centers_track, scales_track, cleanup_probability = _track_cleanup_probability(
        cleanup_similarity, pitch, energy_snr_db, noise_mask, config
    )
    pitch_term = np.clip(
        (pitch - config.pitch_threshold) / max(0.85 - config.pitch_threshold, 1e-6),
        0.0,
        1.15,
    )
    spectral_bonus = config.spectral_bonus_cap * np.clip(
        (prominence - config.prominence_threshold_db) / 12.0, 0.0, 1.0
    )
    shark_score = pitch_term + spectral_bonus
    # Cleanup is a certifier, not an alternate voiced detector.  Its bounded
    # strength is calibrated by utterance-held-out precision-constrained recall.
    spectral_structure = expit(
        (prominence - config.prominence_threshold_db) / 2.0
    )
    energy_structure = expit((energy_snr_db - 3.0) / 2.0)
    fused_score, unvoiced_score = fuse_cleanup_shark_evidence(
        shark_score,
        cleanup_probability,
        np.maximum(spectral_structure, energy_structure),
        voice_certifier_strength=config.cleanup_voice_certifier_strength,
        structure_certifier_strength=config.cleanup_structure_certifier_strength,
    )
    state, speech_mask, intervals = speech_state_machine(
        fused_score,
        unvoiced_score,
        sample_rate=sample_rate,
        config=config,
    )
    voiced_mask = state == SpeechState.VOICED
    unvoiced_mask = state == SpeechState.UNVOICED
    vocalization_mask, vocalizations = voiced_vocalization_intervals(
        voiced_mask,
        fused_score,
        sample_rate=sample_rate,
        config=config,
    )
    return CleanupSharkAnalysis(
        cleanup_similarity=cleanup_similarity,
        cleanup_center=centers_track,
        cleanup_atd=scales_track,
        cleanup_probability=cleanup_probability,
        pitch_periodicity=pitch,
        pitch_hz=pitch_hz,
        spectral_prominence_db=prominence,
        energy_snr_db=energy_snr_db,
        shark_score=shark_score,
        fused_score=fused_score,
        unvoiced_score=unvoiced_score,
        spectral_gain=spectral_gain,
        denoised_magnitude=denoised,
        state=state,
        speech_mask=speech_mask,
        voiced_mask=voiced_mask,
        unvoiced_mask=unvoiced_mask,
        vocalization_mask=vocalization_mask,
        intervals=intervals,
        vocalizations=vocalizations,
        noise_frame_count=int(np.sum(noise_mask)),
    )
