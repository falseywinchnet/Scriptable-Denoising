"""Offline integration experiment. Production DSP will remain in Filter.py.

Keep the oracle's row coordinate separate from the synthesis FFT coordinate.
Nothing in this file is installed into SDR# or called by its audio thread.
"""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class OverlapEvidence:
    """Two statistics used to decide an entire overlapping Cleanup block.

    The downloaded C++ counts columns [32,161) but measures its longest run
    across all 192 columns. Older realtime.py uses [32,160) for BOTH. Preserve
    and expose this distinction; do not silently pretend they are identical.
    """
    occupied: int
    longest: int
    count_available: int
    run_available: int

    @classmethod
    def measure(cls, detected, count_region, run_region):
        values = np.asarray(detected, dtype=bool)
        if values.ndim != 1:
            raise ValueError("one activity verdict per analysis interval required")
        for lo, hi in (count_region, run_region):
            if not 0 <= lo < hi <= values.size:
                raise ValueError("invalid counting region")
        occupied = int(np.count_nonzero(values[slice(*count_region)]))
        streak = longest = 0
        for hit in values[slice(*run_region)]:
            streak = streak + 1 if hit else 0
            longest = max(longest, streak)
        return cls(occupied, longest, count_region[1]-count_region[0],
                   run_region[1]-run_region[0])

    def baseline_512(self):
        if (self.count_available, self.run_available) != (129, 192):
            raise ValueError("512-point downloaded-C++ baseline requires 129/192 intervals")
        return self.occupied > 22 or self.longest > 16

    def calibrated(self, *, reference_event_intervals, target_event_intervals):
        """Candidate dynamic rule, requiring an explicit measured event mapping.

        Total occupancy scales with available intervals (129 at baseline).
        Contiguous evidence scales with intervals representing the same event.
        That event mapping must be measured for the chosen aperture/overlap and
        activity detector; hop/sample-rate scaling alone is not calibration.
        """
        if min(reference_event_intervals, target_event_intervals) <= 0:
            raise ValueError("positive calibrated event interval counts required")
        total_threshold = 22 * self.count_available / 129
        run_threshold = 16 * target_event_intervals / reference_event_intervals
        return dict(occupied=self.occupied, longest=self.longest,
                    count_available=self.count_available, run_available=self.run_available,
                    total_threshold=total_threshold, run_threshold=run_threshold,
                    open=self.occupied > total_threshold or self.longest > run_threshold)


def odft_frequencies(n_fft: int, sample_rate: int) -> np.ndarray:
    return (np.arange(n_fft // 2) + 0.5) * sample_rate / n_fft


def unrolled_frequencies(n_fft: int, rows: int, sample_rate: int) -> np.ndarray:
    return np.arange(rows) * sample_rate / (2 * (n_fft - 1))


def project_decisions(gain: np.ndarray, source_hz: np.ndarray, source_seconds: np.ndarray,
                      target_hz: np.ndarray, target_seconds: np.ndarray) -> np.ndarray:
    """Linear interpolation in physical frequency/time; output is time-by-bin.

    Uncovered frequency/time is neutral gain 1, not invented high-resolution
    support. Project decisions only; retain synthesis coefficient phase.
    Reject invalid values instead of quietly propagating them to audio.
    """
    g = np.asarray(gain, dtype=np.float64)
    axes = [np.asarray(x, dtype=np.float64) for x in
            (source_hz, source_seconds, target_hz, target_seconds)]
    if (g.shape != (axes[0].size, axes[1].size) or not np.all(np.isfinite(g))
            or np.any(g < 0) or np.any(g > 1)):
        raise ValueError("gain must be finite row-by-time in [0,1]")
    for axis in axes:
        if axis.ndim != 1 or not axis.size or not np.all(np.isfinite(axis)) or np.any(np.diff(axis) <= 0):
            raise ValueError("coordinates must be finite, nonempty, strictly increasing")
    sf, st, tf, tt = axes
    frequency = np.stack([np.interp(tf, sf, column, left=1., right=1.) for column in g.T])
    return np.stack([np.interp(tt, st, column, left=1., right=1.) for column in frequency.T], axis=1)


def guide_gain(field: np.ndarray) -> np.ndarray:
    """Provisional MAN/ATD guidance, deliberately named as an experiment.

    Uses the lifted Cleanup/SHARK per-frame/local-global threshold blend on
    enhanced magnitudes. Its calibration is not assumed to transfer from raw
    STFT magnitudes. Full recursive Cleanup integration is a subsequent gate.
    """
    from .oracle.cleanup_shark_vad import cleanup_man_atd
    from scipy.special import expit
    value = np.asarray(field, dtype=np.float64)
    man, atd = cleanup_man_atd(value)
    atd = max(atd, 1e-12)
    out = np.empty_like(value)
    for t in range(value.shape[1]):
        lm, la = cleanup_man_atd(value[:, t])
        wm, wa = np.exp(-.5*abs(lm-man)/atd), np.exp(-.5*abs(la-atd)/atd)
        fm, fa = lm*wm+man*(1-wm), la*wa+atd*(1-wa)
        out[:, t] = expit((value[:, t] - fm - .5*fa) / max(fa, 1e-12))
    return out
