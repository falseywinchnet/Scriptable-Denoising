"""Live native diagnostic smoke; no audio device is opened.

Launches the optional GUI through Filter.py's reload lifecycle, feeds a timed
radio/quiet sequence, then reloads once to verify a fresh viewer generation.
"""
import argparse
import json
from pathlib import Path
import tempfile
import time
import wave
import numpy as np
from native_client import Engine

ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--library',type=Path,default=ROOT/'build/libCleanupNative.dylib')
    p.add_argument('--duration',type=float,default=22.)
    p.add_argument('--report',type=Path,default=ROOT/'build/viewer-smoke.json')
    args=p.parse_args()
    with wave.open(str(ROOT/'research/true_superresolution/daveandsimon-first-second.wav'),'rb') as w:
        audio=np.frombuffer(w.readframes(w.getnframes()),dtype='<i2').astype('float32')/32768.
    cycle=np.concatenate([np.zeros(48000,'float32'),np.tile(audio,2)])
    with tempfile.TemporaryDirectory(prefix='cleanup-viewer-') as td:
        script=Path(td)/'Filter.py'
        script.write_text((ROOT/'Filter.py').read_text().replace('VIEWER = False','VIEWER = True'))
        with Engine(args.library,script) as e:
            first=e.wait()
            if not first['success'] or first.get('viewer_error'):raise RuntimeError(first)
            e.lib.cleanup_set_squelch(e.handle,1)
            start=time.monotonic();index=0;reloaded=False
            while time.monotonic()-start<args.duration:
                frames=8192
                samples=cycle[np.arange(index,index+frames)%len(cycle)]
                _,code=e.process(samples)
                if code:raise RuntimeError(e.status())
                index+=frames
                if not reloaded and time.monotonic()-start>args.duration/2:
                    reload_start=time.monotonic()
                    e.reload();after=e.wait()
                    start+=time.monotonic()-reload_start # exercise audio on both generations
                    if not after['success'] or after.get('viewer_error'):raise RuntimeError(after)
                    reloaded=True
                time.sleep(frames/48000)
            report=dict(initial=first,final=e.status(),reloaded=reloaded)
            args.report.write_text(json.dumps(report,indent=2)+'\n')
            print(json.dumps(report,indent=2))

if __name__=='__main__':main()
