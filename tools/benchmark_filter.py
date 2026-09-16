"""Measure the complete native callback, including guide/mask/synthesis.

No audio device is opened. A temporary script selects the requested experiment;
the user's Filter.py and running host generation are never edited.
"""
import argparse
import json
from pathlib import Path
import platform
import os
import re
import tempfile
import time
import wave
import numpy as np
from native_client import Engine

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--library', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--registered', action='store_true')
    parser.add_argument('--population', action='store_true', help='Full-width rolloff population; identical DSP to the listening trial')
    parser.add_argument('--no-registration', action='store_true', help='Central enhanced magnitude only; preserve the guided Cleanup operator')
    parser.add_argument('--registration', choices=('on','off'), help='Override the Filter.py registration default for a controlled comparison')
    parser.add_argument('--save-output', type=Path, help='Save callback outputs for numerical before/after comparison')
    parser.add_argument('--profile-marker', type=Path, help='Write PID after compilation, immediately before processing')
    parser.add_argument('--channels', type=int, choices=(1, 2), default=2)
    parser.add_argument('--identical', action='store_true', help='Duplicate mono input in stereo, as the SSB plugin does')
    parser.add_argument('--blocks', type=int, default=16)
    parser.add_argument('--harmonics', action='store_true')
    parser.add_argument('--guide-bins', type=int, help='Explicit alternative guide crop, keeping aperture/hop/resolution')
    parser.add_argument('--paced', action='store_true', help='Feed one block per audio interval when processing keeps up')
    args = parser.parse_args()
    if args.blocks < 4:
        parser.error('Use at least four blocks to populate contextual history')
    source = (ROOT/'Filter.py').read_text()
    if args.registered or args.population:
        source = source.replace('ANALYSIS_MODE = "baseline"', 'ANALYSIS_MODE = "registered"')
    if args.population:
        source = source.replace('CLEANUP_POPULATION = "receiver"', 'CLEANUP_POPULATION = "rolloff"')
    if args.no_registration or args.registration is not None:
        if not (args.registered or args.population):
            parser.error('--no-registration requires --registered or --population')
        if args.no_registration and args.registration=='on':
            parser.error('--no-registration conflicts with --registration on')
        source = re.sub(r'(?m)^GUIDE_REGISTRATION = .*$',
                        'GUIDE_REGISTRATION = '+str(args.registration=='on'), source)
    if args.guide_bins is not None:
        if not args.registered:
            parser.error('--guide-bins requires --registered')
        source = re.sub(r'(?m)^GUIDE_BINS = \d+', f'GUIDE_BINS = {args.guide_bins}', source)
    source = source.replace('VIEWER = True', 'VIEWER = False')
    with wave.open(str(ROOT/'research/true_superresolution/daveandsimon-first-second.wav')) as wav:
        sample_rate = wav.getframerate()
        audio = np.frombuffer(wav.readframes(wav.getnframes()), '<i2').astype('float32')/32768.
    with tempfile.TemporaryDirectory(prefix='cleanup-benchmark-') as folder:
        script = Path(folder)/'Filter.py'; script.write_text(source)
        with Engine(args.library, script, channels=args.channels, sample_rate=sample_rate) as engine:
            initial = engine.wait()
            if not initial['success']:
                raise RuntimeError(initial)
            engine.lib.cleanup_set_harmonics(engine.handle, int(args.harmonics))
            block = initial['block']; interval = block/sample_rate
            times = []
            outputs = []
            stages = []
            if args.profile_marker:args.profile_marker.write_text(str(os.getpid()))
            for segment in range(args.blocks):
                positions = np.arange(segment*block, (segment+1)*block) % len(audio)
                data = np.empty((block, args.channels), 'float32')
                data[:, 0] = audio[positions]
                if args.channels == 2:
                    data[:, 1] = data[:, 0] if args.identical else audio[(positions+7919) % len(audio)]*.9
                begin = time.perf_counter()
                output, code = engine.process(data)
                times.append(time.perf_counter()-begin)
                if args.save_output:outputs.append(output.copy())
                stages.append(engine.status().get('stage_ns', {}))
                if code or not np.isfinite(output).all():
                    raise RuntimeError(engine.status())
                if args.paced:
                    time.sleep(max(0., interval-times[-1]))
            result = dict(platform=platform.platform(), machine=platform.machine(),
                          identical_channels=args.identical, channels=args.channels,
                          paced=args.paced,
                          harmonic_extension=args.harmonics, block_seconds=interval,
                          callback_seconds=times, median_seconds=float(np.median(times)),
                          stage_ns=stages, population=args.population,
                          p95_seconds=float(np.quantile(times, .95)), peak_seconds=max(times),
                          observed_deadline_pass=bool(max(times) < interval),
                          final=engine.status())
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(result, indent=2)+'\n')
            if args.save_output:
                args.save_output.parent.mkdir(parents=True, exist_ok=True)
                np.save(args.save_output, np.concatenate(outputs))
            print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
