import importlib.util
import json
import os
from pathlib import Path
import socket
import sys
import tempfile
import time
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'tools'), str(ROOT / 'runtime')]
from native_client import Engine
from compile_filter import compile_candidate, release
LIB = os.environ.get('CLEANUP_TEST_LIBRARY', str(ROOT / 'build/libCleanupNative.dylib'))
HEADERS = (ROOT / 'Filter.py').read_text().split('# COMPILED METHODS')[0]


def fixture(action='return PROCESS', fft=512, hop=128, gain=1.):
    return HEADERS.replace('FFT_SIZE = 512', f'FFT_SIZE = {fft}').replace('HOP = 128', f'HOP = {hop}') + f'''
class FilterBank:
    @staticmethod
    @njit(FILTER_SIGNATURE, nogil=True)
    def run(analysis, content, state, config):
        state[0, 0] += 1.
        content[:, :, :] *= {gain}
        {action}
_run_bank = FilterBank.run
@njit(FILTER_SIGNATURE, nogil=True)
def Filter(analysis, content, state, config):
    return _run_bank(analysis, content, state, config)
'''


class CompilerTests(unittest.TestCase):
    def test_reject_undecorated_and_unused_and_bad_geometry(self):
        cases = [
            (fixture() + '\ndef forgotten():\n    return 1\n', 'every method'),
            (fixture() + '\n@njit\ndef unused(x):\n    return x\n', 'explicit Numba signature'),
            (fixture().replace('FFT_SIZE = 512', 'FFT_SIZE = 513'), 'power of two'),
            (fixture() + '\nTRANSFORM = "fct"\n', 'TRANSFORM must'),
            (fixture() + '\nTRANSFORM = "odft"\n', 'FRAMES/BINS/EMIT'),
            (fixture().replace('def run(analysis', 'def run(analysis', 1).replace('@njit(FILTER_SIGNATURE, nogil=True)\n    def run', 'def run'), 'every method'),
            (fixture(action='return 42'), 'invalid disposition'),
            (fixture(action='content[0, 0, 0] = np.nan\n        return PROCESS'), 'nonfinite')]
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / 'Filter.py'
            for source, text in cases:
                with self.subTest(text=text):
                    path.write_text(source)
                    with self.assertRaisesRegex(Exception, text):
                        compile_candidate(path)


class NativeTests(unittest.TestCase):
    def test_shared_channels_materialize_state_on_divergence(self):
        # Equality must survive a transition from duplicated SSB to distinct
        # stereo input, subsequent equal input, bypass, and independent OLA tails.
        source=fixture().replace('ANALYSIS_MODE = "registered"','ANALYSIS_MODE = "baseline"')
        source=source.replace('state[0, 0] += 1.', 'state[0, 0] = .87 * state[0, 0] + .001 * analysis[10, 5, 0]')
        source=source.replace('content[:, :, :] *= 1.0', 'content[:, :, :] *= .7 + .01 * state[0, 0]')
        other=self.path.with_name('Independent.py')
        other.write_text(source.replace('SHARE_IDENTICAL_CHANNELS = True','SHARE_IDENTICAL_CHANNELS = False'))
        self.path.write_text(source)
        rng=np.random.default_rng(55013)
        with Engine(LIB,self.path,channels=2) as shared, Engine(LIB,other,channels=2) as independent:
            self.assertTrue(shared.wait()['success']);self.assertTrue(independent.wait()['success'])
            block=shared.status()['block']
            for index in range(14):
                a=rng.normal(0,.05,(block,2)).astype('float32')
                if index!=5:a[:,1]=a[:,0]
                for engine in (shared,independent):engine.lib.cleanup_set_bypass(engine.handle,int(index in (3,9)))
                x,xc=shared.process(a);y,yc=independent.process(a)
                self.assertEqual((xc,yc),(0,0));np.testing.assert_array_equal(x,y)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / 'Filter.py'

    def tearDown(self):
        self.temp.cleanup()

    def engine(self, source, channels=1):
        self.path.write_text(source)
        e = Engine(LIB, self.path, channels=channels)
        status = e.wait()
        self.assertTrue(status['success'], status)
        return e

    def stream(self, engine, x, sizes=(1, 137, 4096, 503, 8193), inplace=False):
        out, index, k = [], 0, 0
        while index < len(x):
            size = min(sizes[k % len(sizes)], len(x) - index)
            y, code = engine.process(x[index:index + size].copy(), inplace=inplace)
            self.assertEqual(code, 0)
            out.append(y)
            index += size
            k += 1
        return np.concatenate(out)

    def test_bfft_identity_arbitrary_buffers_stereo_and_inplace(self):
        with self.engine(fixture(), channels=2) as e:
            x = np.random.default_rng(12).normal(0, .1, (100000, 2)).astype('float32')
            y = self.stream(e, x, inplace=True)
            latency = e.status()['latency_samples']
            error = np.max(np.abs(y[latency:] - x[:-latency]))
            self.assertLess(error, 2e-6)
            self.assertLess(np.max(np.abs(y[:latency])), 2e-6)
            print('identity', json.dumps(dict(latency=latency, max_error=float(error))))

    def test_passthrough_exact_and_controls_use_dll(self):
        with self.engine(fixture(action='return PASSTHROUGH', gain=.123)) as e:
            x = np.random.default_rng(3).normal(0, .1, (80000, 1)).astype('float32')
            y = self.stream(e, x)
            delay = e.status()['latency_samples']
            np.testing.assert_array_equal(y[delay:], x[:-delay])
            for command, field, expected in [('bypass on', 'bypass', True), ('squelch toggle', 'squelch', True), ('bypass off', 'bypass', False)]:
                with socket.create_connection(('127.0.0.1', e.status()['port'])) as s:
                    s.sendall((command + '\n').encode())
                    payload = s.makefile().readline()
                    self.assertEqual(json.loads(payload)[field], expected)

    def test_skip_drains_overlap_then_exact_zeros(self):
        action = 'return PROCESS if state[0, 0] < 6. else SKIP'
        with self.engine(fixture(action=action)) as e:
            x = np.ones((8192 * 9, 1), dtype='float32') * .2
            y = self.stream(e, x, sizes=(8192,))[:, 0]
            start = 6 * 8192
            tail = y[start:start + 384]
            self.assertGreater(tail[0], .19)
            self.assertLess(tail[-1], .0001)
            self.assertTrue(np.all(np.diff(tail) <= 1e-6))
            np.testing.assert_array_equal(y[start + 384:], 0.)

    def test_transactional_reload_geometry_reset_and_failure_retention(self):
        with self.engine(fixture()) as e:
            before = e.status()
            e.process(np.ones((40000, 1), 'float32'))
            self.path.write_text(fixture() + '\ndef bad():\n    return 1\n')
            self.assertEqual(e.reload(), 1)
            failed = e.wait()
            self.assertFalse(failed['success'])
            self.assertEqual(failed['generation'], before['generation'])
            self.assertIn('every method', failed['error'])
            self.path.write_text(fixture(fft=1024, hop=256))
            self.assertEqual(e.reload(), 1)
            after = e.wait()
            self.assertTrue(after['success'], after)
            self.assertEqual(after['generation'], before['generation'] + 1)
            self.assertEqual((after['fft'], after['hop']), (1024, 256))
            y = self.stream(e, np.zeros((40000, 1), 'float32'))
            np.testing.assert_array_equal(y, 0.)
            x = np.random.default_rng(13).normal(0, .1, (80000, 1)).astype('float32')
            y = self.stream(e, x)
            delay = after['latency_samples']
            self.assertLess(np.max(np.abs(y[delay:] - x[:-delay])), 2e-6)

    def test_runtime_failure_and_two_independent_entities(self):
        source = fixture(action='return PROCESS if state[0, 0] < 7. else ERROR')
        with self.engine(source) as e:
            x = np.ones((8192 * 9, 1), 'float32') * .1
            y, result = e.process(x)
            self.assertEqual(result, -2)
            self.assertTrue(np.isfinite(y).all())
            self.assertTrue(e.status()['runtime_fault'])
            self.assertGreater(e.status()['faults'], 0)
            self.path.write_text(fixture())
            e.reload()
            self.assertTrue(e.wait()['success'])
            self.assertFalse(e.status()['runtime_fault'])
        source = fixture().replace('state[0, 0] += 1.', 'state[0, 0] += 1.\n        state[1, 0] += 2.\n        if state[1, 0] != 2. * state[0, 0]:\n            return ERROR')
        with self.engine(source) as e:
            self.stream(e, np.ones((60000, 1), 'float32'))
            self.assertFalse(e.status()['runtime_fault'])

    def test_reload_while_audio_runs_and_real_filter_geometry(self):
        import threading
        with self.engine(fixture()) as e:
            quit_event = threading.Event()
            failures = []
            def audio():
                block = np.zeros((257, 1), 'float32')
                while not quit_event.is_set():
                    output, code = e.process(block)
                    if code != 0 or not np.isfinite(output).all():
                        failures.append(code)
                    time.sleep(.001)
            thread = threading.Thread(target=audio)
            thread.start()
            try:
                self.path.write_text((ROOT/'Filter.py').read_text().replace('FFT_SIZE = 512','FFT_SIZE = 1024').replace('HOP = 128','HOP = 256'))
                e.reload()
                status = e.wait()
                self.assertTrue(status['success'], status)
                self.assertEqual(status['fft'], 1024)
            finally:
                quit_event.set()
                thread.join(timeout=5)
            self.assertFalse(thread.is_alive())
            self.assertEqual(failures, [])

    def test_actual_cleanup_finite_and_denoising(self):
        with self.engine((ROOT / 'Filter.py').read_text()) as e:
            x = np.random.default_rng(33).normal(0, .03, (8192 * 12, 1)).astype('float32')
            t = np.arange(len(x)) / 48000.
            x[:, 0] += (.1 * np.sin(2*np.pi*600*t)).astype('float32')
            y = self.stream(e, x, sizes=(8192,))
            self.assertTrue(np.isfinite(y).all())
            delay = e.status()['latency_samples']
            rms_in = float(np.sqrt(np.mean(x[:-delay] ** 2)))
            rms_out = float(np.sqrt(np.mean(y[delay:] ** 2)))
            self.assertGreater(rms_out, 1e-5)
            self.assertLess(rms_out, rms_in)
            status = e.status()
            print('cleanup-smoke', json.dumps(dict(rms_in=rms_in, rms_out=rms_out, status=status)))
            (ROOT / 'build/native-evidence.json').write_text(json.dumps(status, indent=2))


if __name__ == '__main__':
    unittest.main(verbosity=2)
