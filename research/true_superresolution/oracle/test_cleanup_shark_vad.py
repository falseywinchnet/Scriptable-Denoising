from __future__ import annotations

import unittest

import numpy as np

from .cleanup_shark_vad import (
    CleanupSharkConfig,
    SpeechState,
    analyze_cleanup_shark,
    cleanup_similarity_from_magnitude,
    fuse_cleanup_shark_evidence,
    normalized_pitch_periodicity,
    speech_state_machine,
    true_logistic_reference,
    voiced_vocalization_intervals,
)


class CleanupSharkVadTests(unittest.TestCase):
    def test_logistic_reference_is_finite_symmetric_and_normalized(self) -> None:
        reference = true_logistic_reference(111)
        self.assertTrue(np.all(np.isfinite(reference)))
        self.assertAlmostEqual(float(reference[0]), 0.0)
        self.assertAlmostEqual(float(reference[-1]), 1.0)
        np.testing.assert_allclose(reference + reference[::-1], 1.0)

    def test_cleanup_similarity_is_amplitude_scale_invariant(self) -> None:
        rng = np.random.default_rng(7)
        magnitude = np.abs(rng.normal(size=(24, 64)))
        first = cleanup_similarity_from_magnitude(
            magnitude, bins=20, smoothing_frames=1
        )
        second = cleanup_similarity_from_magnitude(
            17.0 * magnitude + 3.0, bins=20, smoothing_frames=1
        )
        np.testing.assert_allclose(first, second, atol=1e-12)

    def test_pitch_periodicity_separates_tone_from_seeded_noise(self) -> None:
        sample_rate = 8000
        count = sample_rate
        time = np.arange(count) / sample_rate
        tone = np.sin(2.0 * np.pi * 125.0 * time)
        noise = np.random.default_rng(3).normal(size=count)
        centers = np.arange(1000, 7000, 400, dtype=np.int64)
        tone_score, tone_hz = normalized_pitch_periodicity(
            tone, centers, sample_rate=sample_rate
        )
        noise_score, _ = normalized_pitch_periodicity(
            noise, centers, sample_rate=sample_rate
        )
        self.assertGreater(float(np.median(tone_score)), 0.85)
        self.assertLess(float(np.median(noise_score)), 0.45)
        self.assertAlmostEqual(float(np.median(tone_hz)), 125.0, delta=5.0)

    def test_state_machine_backfills_unvoiced_lead_and_trims_hangover(self) -> None:
        config = CleanupSharkConfig(
            hop_length=10,
            voice_confirm_frames=2,
            voice_confirm_window=3,
            preroll_seconds=0.04,
            hangover_seconds=0.05,
            tail_seconds=0.01,
            minimum_speech_seconds=0.04,
        )
        fused = np.zeros(30)
        unvoiced = np.zeros(30)
        unvoiced[4:7] = 0.9
        fused[7:11] = 1.1
        unvoiced[11:14] = 0.9
        state, mask, intervals = speech_state_machine(
            fused, unvoiced, sample_rate=1000, config=config
        )
        self.assertEqual(len(intervals), 1)
        self.assertEqual((intervals[0].frame0, intervals[0].frame1), (4, 15))
        self.assertTrue(np.all(state[4:7] == SpeechState.UNVOICED))
        self.assertTrue(np.all(state[7:11] == SpeechState.VOICED))
        self.assertTrue(np.all(state[11:14] == SpeechState.UNVOICED))
        self.assertFalse(np.any(mask[15:]))

    def test_state_machine_rejects_single_spike_and_unanchored_structure(self) -> None:
        config = CleanupSharkConfig(hop_length=10)
        fused = np.zeros(40)
        unvoiced = np.zeros(40)
        fused[9] = 1.2
        unvoiced[15:25] = 0.95
        state, mask, intervals = speech_state_machine(
            fused, unvoiced, sample_rate=1000, config=config
        )
        self.assertEqual(intervals, ())
        self.assertFalse(np.any(mask))
        self.assertTrue(np.all(state == SpeechState.QUIET))

    def test_tail_never_overlaps_a_new_candidate(self) -> None:
        config = CleanupSharkConfig(
            hop_length=10,
            voice_confirm_frames=2,
            voice_confirm_window=2,
            hangover_seconds=0.03,
            tail_seconds=0.05,
            minimum_speech_seconds=0.02,
        )
        fused = np.zeros(30)
        unvoiced = np.zeros(30)
        fused[2:5] = 1.2
        unvoiced[8] = 0.9
        fused[9:12] = 1.2
        _, _, intervals = speech_state_machine(
            fused, unvoiced, sample_rate=1000, config=config
        )
        self.assertEqual(len(intervals), 2)
        self.assertLessEqual(intervals[0].frame1, intervals[1].frame0)

    def test_vocalization_rejects_noise_hit_and_bounds_persistent_core(self) -> None:
        config = CleanupSharkConfig(
            hop_length=512,
            voice_confirm_frames=2,
            vocalization_lead_seconds=0.192,
            vocalization_tail_seconds=0.075,
            vocalization_gap_seconds=0.025,
            minimum_voiced_core_seconds=0.020,
        )
        voiced = np.zeros(100, dtype=bool)
        voiced[3] = True  # An isolated periodic noise hit must not seed a crop.
        voiced[38] = True
        voiced[40:64] = True
        voiced[65:75] = True
        voiced[76:83] = True
        fused = np.where(voiced, 1.2, 0.1)
        mask, intervals = voiced_vocalization_intervals(
            voiced, fused, sample_rate=48_000, config=config
        )
        self.assertEqual(len(intervals), 1)
        self.assertEqual((intervals[0].frame0, intervals[0].frame1), (22, 90))
        self.assertFalse(mask[3])
        self.assertTrue(np.all(mask[22:90]))

    def test_cleanup_certifier_has_declared_endpoints(self) -> None:
        shark = np.asarray((0.8, 1.0))
        cleanup = np.asarray((0.0, 0.5))
        structure = np.asarray((0.25, 1.0))
        raw_voice, raw_unvoiced = fuse_cleanup_shark_evidence(
            shark,
            cleanup,
            structure,
            voice_certifier_strength=0.0,
            structure_certifier_strength=0.0,
        )
        np.testing.assert_array_equal(raw_voice, shark)
        np.testing.assert_allclose(raw_unvoiced, np.sqrt(structure))
        voice, unvoiced = fuse_cleanup_shark_evidence(
            shark,
            cleanup,
            structure,
            voice_certifier_strength=0.25,
            structure_certifier_strength=0.25,
        )
        np.testing.assert_allclose(voice, shark * (0.75 + 0.25 * cleanup))
        np.testing.assert_allclose(unvoiced, np.sqrt(cleanup * structure))

    def test_denoised_magnitude_is_the_exposed_spectral_gain_product(self) -> None:
        sample_rate = 8000
        samples = np.random.default_rng(11).normal(size=4096)
        analysis = analyze_cleanup_shark(samples, sample_rate)
        config = CleanupSharkConfig()
        centers = np.arange(analysis.spectral_gain.shape[0]) * config.hop_length
        left = config.n_fft // 2
        right = config.n_fft - left
        padded = np.pad(samples, (left, right), mode="constant")
        frames = np.lib.stride_tricks.sliding_window_view(
            padded, config.n_fft
        )[centers]
        magnitude = np.abs(
            np.fft.rfft(frames * np.hanning(config.n_fft)[None, :], axis=1)
        )
        self.assertTrue(np.all((0.0 <= analysis.spectral_gain) & (analysis.spectral_gain <= 1.0)))
        np.testing.assert_allclose(
            analysis.denoised_magnitude,
            magnitude * analysis.spectral_gain,
            rtol=1e-12,
            atol=1e-12,
        )


if __name__ == "__main__":
    unittest.main()
