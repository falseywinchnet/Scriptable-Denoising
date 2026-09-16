"""DIP transport and temporal erosion invariants; no speech-quality claims."""
import ast
import ctypes as C
import importlib.util
import os
from pathlib import Path
import sys
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'runtime'))
from compile_filter import bind_native_dsp


class PacketTransport(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lib = C.CDLL(os.environ.get('CLEANUP_TEST_LIBRARY', str(ROOT / 'build/libCleanupNative.dylib')))
        cls.call = cls.lib.cleanup_dip_transport
        cls.call.argtypes = [C.c_void_p] * 7 + [C.c_int64] * 3
        cls.call.restype = C.c_int32

    def transport(self, n, e, half_bin, bad_donor=None):
        rng = np.random.default_rng(n + e + half_bin)
        x = rng.normal(size=n)
        full = np.fft.fft(x * np.exp(-1j * np.pi * half_bin * np.arange(n) / n))
        bins = n // 2 + 1 - half_bin
        source = np.ascontiguousarray(full[:bins]).view(np.float64)
        donors = np.full(bins, -1.)
        weights = np.zeros(bins, np.complex128)
        for k in range(e + 1, bins - (1 - half_bin)):
            donors[k] = k % e
            weights[k] = .3 * np.exp(1j * rng.uniform(-np.pi, np.pi))
        if bad_donor is not None:
            donors[e + 1] = bad_donor
        q = n // e
        angle = 2 * np.pi * np.arange(n)[:, None] * np.arange(q)[None, :] / n
        cosine, sine = np.cos(angle), np.sin(angle)
        output = np.zeros(bins, np.complex128)
        scratch = np.zeros(4 * n)
        result = self.call(*(a.ctypes.data for a in (source, donors, weights, output, scratch, cosine, sine)), n, e, half_bin)
        return result, full, donors, weights, output

    def test_packet_projection_matches_finite_zak_definition(self):
        # Independent oracle: direct sums defining the intermediate packets,
        # followed by orthogonal projection; no production stage equations.
        for n in (32, 128, 512):
            for e in (1, 4, n // 2):
                for half_bin in (0, 1):
                    with self.subTest(n=n, e=e, half_bin=half_bin):
                        code, full, donors, weights, actual = self.transport(n, e, half_bin)
                        self.assertEqual(code, 0)
                        q = n // e
                        j = np.arange(q)
                        packet = np.zeros((e, q), np.complex128)
                        for delta in range(e):
                            members = np.arange(delta, n, e)
                            packet[delta] = np.sum(full[members, None] * np.exp(2j*np.pi*members[:, None]*j/n), axis=0)/q
                        expected = np.zeros_like(actual)
                        for k, donor in enumerate(donors.astype(int)):
                            if donor >= 0:
                                coefficient = np.sum(packet[donor % e] * np.exp(-2j*np.pi*donor*j/n))
                                expected[k] = coefficient * weights[k]
                        np.testing.assert_allclose(actual, expected, atol=2e-11, rtol=2e-11)

    def test_invalid_donors_rejected(self):
        for value in (float('nan'), float('inf'), 1.5, 6., 2.):
            with self.subTest(value=value):
                self.assertEqual(self.transport(32, 4, 0, value)[0], -2)


class HarmonicEntity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location('harmonic_fixture', ROOT / 'research/harmonics/Filter-retired.py')
        cls.m = importlib.util.module_from_spec(spec)
        bind_native_dsp(cls.m)
        spec.loader.exec_module(cls.m)

    def arrays(self):
        m = self.m
        analysis = np.zeros((m.FRAMES, m.BINS, 2))
        analysis[:, 11, 0] = 1.
        analysis[:, 19, 0] = .4
        content = analysis.copy()
        state = np.zeros(m.ENTITY_STORAGE)
        config = np.array([48000., 3400., 0., 1., -2.*m.BLOCK_SIZE])
        return analysis, content, state, config

    def test_centered_stft_carrier_phase_for_odd_comb_shift(self):
        # Compile the archived harmonic routines at E=1, where an
        # odd-bin shift exposes the leading-edge versus frame-center sign error.
        # Do not copy the carrier expression into this test's implementation.
        names = {'softstep', 'lattice_noise', 'coherent_noise',
                 'centroid_erosion', 'occupied_edge', 'harmonics'}
        tree = ast.parse((ROOT / 'research/harmonics/Filter-retired.py').read_text())
        routines = ast.Module(body=[node for node in tree.body
                                  if isinstance(node, ast.FunctionDef) and node.name in names],
                              type_ignores=[])
        self.assertEqual(len(routines.body), len(names))
        lib = C.CDLL(os.environ.get('CLEANUP_TEST_LIBRARY', str(ROOT / 'build/libCleanupNative.dylib')))
        create = lib.bfft_stft_plan_create
        create.argtypes = [C.c_size_t] * 3 + [C.c_void_p, C.c_int, C.POINTER(C.c_void_p)]
        create.restype = C.c_int
        forward = lib.bfft_stft_forward
        forward.argtypes = [C.c_void_p] * 3
        forward.restype = C.c_int
        lib.bfft_stft_plan_destroy.argtypes = [C.c_void_p]
        n, hop, frames, first, emit = 64, 16, 32, 3, 16
        rate, protected, history_origin = 48000., 3400., 12345.
        donor, target = 3, 6  # Odd shift; same comb because E=1.
        for half_bin in (0, 1):
            with self.subTest(transform='odft' if half_bin else 'rfft'):
                bins = n // 2 + 1 - half_bin
                namespace = dict(self.m.__dict__)
                angles = 2 * np.pi * np.arange(n)[:, None] * np.arange(n)[None, :] / n
                namespace.update(FFT_SIZE=n, HOP=hop, FRAMES=frames, FIRST=first,
                                 EMIT=emit, BINS=bins, PLANE=frames*bins,
                                 DIP_E=1, DIP_Q=n, DIP_HALF_BIN=half_bin,
                                 SORT_STACK_OFFSET=64+2*frames*bins+2*frames+5*bins+4*n,
                                 DIP_COS=np.ascontiguousarray(np.cos(angles)),
                                 DIP_SIN=np.ascontiguousarray(np.sin(angles)))
                exec(compile(routines, str(ROOT / 'research/harmonics/Filter-retired.py'), 'exec'), namespace)
                samples = np.arange(frames * hop) + history_origin
                signal = np.cos(2*np.pi*(donor+.5*half_bin)*samples/n + .34)
                packed = np.zeros((bins, frames), np.complex128)
                plan = C.c_void_p()
                self.assertEqual(create(len(signal), n, hop, None, half_bin, C.byref(plan)), 0)
                try:
                    self.assertEqual(forward(plan, signal.ctypes.data, packed.ctypes.data), 0)
                finally:
                    lib.bfft_stft_plan_destroy(plan)
                t = first + emit - 1
                relative = np.arange(n) - n // 2
                # Independent centered physical-time transform, including the
                # real tone's negative-frequency contribution and Hann window.
                expected_source = np.sum(signal[t*hop+relative] * np.hanning(n)
                                         * np.exp(-2j*np.pi*(donor+.5*half_bin)*relative/n))
                np.testing.assert_allclose(packed[donor, t], expected_source, atol=2e-12, rtol=2e-12)
                analysis = np.ascontiguousarray(packed.T).view(np.float64).reshape(frames, bins, 2)
                content = analysis.copy()
                storage = 64 + 2*frames*bins + 2*frames + 5*bins + 4*n
                state = np.zeros(storage + self.m.SORT_STACK_SIZE)
                config = np.array([rate, protected, 0., 1., history_origin])
                self.assertEqual(namespace['harmonics'](analysis, content, state, config), self.m.PROCESS)
                donor_offset = 64 + 2*frames*bins + 2*frames
                self.assertEqual(state[donor_offset + target], donor)
                added = complex(*(content[t, target] - analysis[t, target]))
                center = history_origin + t*hop
                leading_edge = center - n/2
                frequency = (target+.5*half_bin)*rate/n
                upper = min(namespace['HARMONIC_LIMIT_HZ'], rate*.47)
                texture = namespace['HARMONIC_BACKFILL'] * np.sqrt((frequency-protected)/(upper-protected))
                nr = namespace['coherent_noise'](leading_edge/(rate*.035), target//2, 11)
                ni = namespace['coherent_noise'](leading_edge/(rate*.043), target//2, 29)
                noise = complex(1.-texture+texture*nr, texture*ni)
                expected_carrier = np.exp(2j*np.pi*(target-donor)*(center % n)/n)
                # All remaining attenuation and power-limit terms are positive
                # real scalars. The old leading-edge carrier makes this negative.
                ratio = added / (expected_source * noise * expected_carrier)
                self.assertGreater(ratio.real, 1e-8)
                self.assertLess(abs(ratio.imag), 2e-11 * abs(ratio))

    def test_off_is_exact_identity_and_on_protects_measured_band(self):
        m = self.m
        analysis, content, state, config = self.arrays()
        config[3] = 0.
        self.assertEqual(m.harmonics(analysis, content, state, config), m.PASSTHROUGH)
        np.testing.assert_array_equal(analysis, content)
        config[3] = 1.
        self.assertEqual(m.harmonics(analysis, content, state, config), m.PROCESS)
        last = int(np.floor(config[1]*m.FFT_SIZE/config[0] - .5*m.DIP_HALF_BIN))
        np.testing.assert_array_equal(content[:, :last+1], analysis[:, :last+1])
        np.testing.assert_array_equal(content[:m.FIRST], analysis[:m.FIRST])
        self.assertTrue(np.isfinite(content).all())
        self.assertGreater(state[2], 0.)
        self.assertLessEqual(state[2], m.HARMONIC_MAX_POWER_RATIO*state[3]*(1+1e-12))
        config[3] = 0.
        m.harmonics(analysis, content, state, config)
        self.assertEqual(state[0], 0.)
        before = content.copy()
        self.assertEqual(m.harmonics(analysis, content, state, config), m.PASSTHROUGH)
        np.testing.assert_array_equal(content, before)

    def test_empty_receiver_tail_joins_lower_and_members_decay(self):
        m=self.m
        analysis, content, state, config=self.arrays()
        # Broadband occupied band ends at bin29, below the nominal3.4kHz.
        # Measured tones remain donors; the receiver tail is genuinely empty.
        analysis[:,2:30,0]+=.03
        content[:]=analysis
        self.assertEqual(m.harmonics(analysis,content,state,config),m.PROCESS)
        self.assertEqual(state[4],29*48000/m.FFT_SIZE)
        np.testing.assert_array_equal(content[:,:30],analysis[:,:30])
        delta=content-analysis
        self.assertGreater(np.max(abs(delta[m.FIRST:m.FIRST+m.EMIT,30:37])),0)
        # Same packet, same donor: successive continuation members must fade
        # substantially faster than the previous inverse-power tiled envelope.
        t=m.FIRST+m.EMIT-1
        primary=np.linalg.norm(delta[t,35]);second=np.linalg.norm(delta[t,43]);third=np.linalg.norm(delta[t,51])
        self.assertGreater(primary,0)
        self.assertLess(second/primary,.4)
        self.assertLess(third/primary,.09)
        self.assertLessEqual(state[2],m.HARMONIC_MAX_POWER_RATIO*state[3]*(1+1e-12))

    def test_formant_valley_and_sparse_tones_do_not_define_receiver_edge(self):
        m=self.m
        mag=np.full((m.FRAMES,m.BINS),.02)
        mag[:,25:30]=0.  # Interior valley; meaningful energy resumes above it.
        self.assertEqual(m.occupied_edge(mag,2,36,np.zeros(4*m.FFT_SIZE), np.zeros(192)),36)
        mag[:]=0.;mag[:,11]=1.;mag[:,19]=.4
        self.assertEqual(m.occupied_edge(mag,2,36,np.zeros(4*m.FFT_SIZE), np.zeros(192)),36)

    def test_erosion_is_soft_and_between_time_peaks(self):
        m = self.m
        energy = np.zeros((m.FRAMES, m.DIP_E))
        t = np.arange(m.FRAMES)
        left, right = m.FIRST + 5, m.FIRST + 25
        row = min(3, m.DIP_E-1)
        energy[:, row] = np.exp(-.5*((t-left)/2.)**2)+np.exp(-.5*((t-right)/2.)**2)
        smoothed = np.zeros_like(energy)
        holes = np.zeros((m.FRAMES, m.BINS))
        m.centroid_erosion(energy, smoothed, holes, 48000., 3400.)
        self.assertGreater(holes.max(), .2)
        self.assertLessEqual(holes.max(), m.HARMONIC_HOLE_DEPTH)
        self.assertEqual(np.count_nonzero(holes[:left+1]), 0)
        self.assertEqual(np.count_nonzero(holes[right:]), 0)
        time, frequency = np.unravel_index(np.argmax(holes), holes.shape)
        self.assertLessEqual(abs(time - (left+right)/2.), 1.)
        self.assertGreater(holes[time, frequency+1], 0.)
        self.assertLess(holes[time+4, frequency], holes[time, frequency])
        energy[:] = 1.
        m.centroid_erosion(energy, smoothed, holes, 48000., 3400.)
        self.assertEqual(np.count_nonzero(holes), 0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
