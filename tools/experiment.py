"""Run the same DLL offline, or keep it serving timed audio and CLI commands.

Examples:
  python tools/experiment.py render input.wav output.wav --library build/libCleanupNative.dylib
  python tools/experiment.py serve input.wav --library build/libCleanupNative.dylib
The serve command plays no sound; it feeds a repeating WAV through the real engine
at its sample rate, keeping native command controls available for experiments.
"""
import argparse
import json
from pathlib import Path
import sys
import time
import wave
import numpy as np
from native_client import Engine


def read_wav(path):
    with wave.open(str(path),'rb') as w:
        if w.getsampwidth()!=2 or w.getnchannels() not in (1,2):
            raise ValueError('Use a mono/stereo 16-bit PCM WAV')
        return np.frombuffer(w.readframes(w.getnframes()),'<i2').reshape(-1,w.getnchannels()).astype('float32')/32768.,w.getframerate()


def write_wav(path,data,rate):
    with wave.open(str(path),'wb') as w:
        w.setnchannels(data.shape[1]);w.setsampwidth(2);w.setframerate(rate)
        w.writeframes((np.clip(data,-1.,32767./32768.)*32768.).astype('<i2').tobytes())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['render','serve'])
    parser.add_argument('input',type=Path)
    parser.add_argument('output',type=Path,nargs='?')
    parser.add_argument('--library',required=True,type=Path)
    parser.add_argument('--filter',type=Path,default=Path(__file__).resolve().parents[1]/'Filter.py')
    parser.add_argument('--squelch',action='store_true')
    parser.add_argument('--harmonics',action='store_true',help='Enable experimental DIP comb extension')
    args=parser.parse_args()
    x,rate=read_wav(args.input)
    if not len(x):raise ValueError('Empty WAV')
    with Engine(args.library,args.filter,channels=x.shape[1],sample_rate=rate) as engine:
        state=engine.wait()
        if not state['success']:raise RuntimeError(state['error'])
        engine.lib.cleanup_set_squelch(engine.handle,int(args.squelch))
        engine.lib.cleanup_set_harmonics(engine.handle,int(args.harmonics))
        print(json.dumps(engine.status()),flush=True)
        if args.mode=='render':
            if args.output is None:parser.error('render needs an output WAV')
            delay=state['latency_samples']
            source=np.concatenate((x,np.zeros((delay+state['block'],x.shape[1]),'float32')))
            chunks=[];start=time.perf_counter()
            for i in range(0,len(source),1024):
                y,code=engine.process(source[i:i+1024])
                if code:raise RuntimeError(engine.status())
                chunks.append(y)
            elapsed=time.perf_counter()-start
            output=np.concatenate(chunks)[delay:delay+len(x)]
            write_wav(args.output,output,rate)
            report=dict(engine.status(),seconds_processing=elapsed,seconds_audio=len(x)/rate,
                        input_rms=float(np.sqrt(np.mean(x*x))),output_rms=float(np.sqrt(np.mean(output*output))),
                        input=str(args.input.resolve()),output=str(args.output.resolve()))
            args.output.with_suffix('.json').write_text(json.dumps(report,indent=2))
            print(json.dumps(report,indent=2))
        else:
            position=0;deadline=time.perf_counter()
            try:
                while True:
                    chunk=x[position:min(position+512,len(x))]
                    _,code=engine.process(chunk)
                    if code:print(json.dumps(engine.status()),file=sys.stderr)
                    position=(position+len(chunk))%len(x)
                    deadline+=len(chunk)/rate
                    time.sleep(max(0.,deadline-time.perf_counter()))
            except KeyboardInterrupt:pass


if __name__=='__main__':main()
