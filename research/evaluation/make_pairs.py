"""Deterministic additive-noise pairs; proxy references are labeled explicitly.

No denoising score is claimed by this generator. NPZ retains unclipped float64
samples and the exact added corruption so downstream evaluations stay aligned.
"""
import argparse
import hashlib
import json
from pathlib import Path
import wave
import numpy as np


def corruptions(size, sample_rate, seed):
    rng=np.random.default_rng(seed);t=np.arange(size)/sample_rate
    white=rng.standard_normal(size)
    yield 'white',white
    colored=np.convolve(rng.standard_normal(size),np.array([1.,2.,3.,2.,1.])/9.,mode='same')
    yield 'colored',colored
    yield 'evolving_floor',rng.standard_normal(size)*(.15+.85*(.5+.5*np.sin(2*np.pi*.8*t)))
    bursts=np.zeros(size)
    for center in rng.uniform(0,max(size-1,1),max(2,int(4*size/sample_rate))):
        bursts+=np.exp(-.5*((np.arange(size)-center)/max(1,.015*sample_rate))**2)
    yield 'bursts',rng.standard_normal(size)*bursts
    impulses=np.zeros(size)
    for center in rng.integers(0,size,max(3,int(12*size/sample_rate))):
        length=min(size-center,max(2,int(.003*sample_rate)))
        phase=np.arange(length)/sample_rate
        impulses[center:center+length]+=rng.choice([-1.,1.])*np.exp(-phase/.0005)*np.cos(2*np.pi*2200*phase)
    yield 'ringing_impulses',impulses
    yield 'drifting_tone',np.sin(2*np.pi*(900*t+80*t*t))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('reference',type=Path)
    parser.add_argument('--reference-kind',choices=['proxy','clean'],required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--seed',type=int,default=20260915)
    parser.add_argument('--snr',type=float,nargs='+',default=[0.,10.,20.])
    args=parser.parse_args()
    with wave.open(str(args.reference),'rb') as f:
        if f.getsampwidth()!=2 or f.getnchannels()!=1:parser.error('Use a mono PCM16 reference WAV; no implicit resampling or channel mixing')
        fs=f.getframerate();reference=np.frombuffer(f.readframes(f.getnframes()),dtype='<i2').astype(np.float64)/32768.
    power=np.mean(reference**2)
    if len(reference)<5 or power<=0:parser.error('Reference must contain nonzero audio')
    if not all(np.isfinite(args.snr)):parser.error('SNR must be finite')
    args.out.mkdir(parents=True,exist_ok=True)
    manifest=dict(reference=str(args.reference.resolve()),reference_kind=args.reference_kind,
                  reference_sha256=hashlib.sha256(args.reference.read_bytes()).hexdigest(),
                  generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  sample_rate=fs,seed=args.seed,snr_convention='whole-reference mean-square power, including any pauses',
                  channel_model='additive audio-domain corruption; no RF/AGC/fading simulation',pairs=[])
    for kind,base in corruptions(len(reference),fs,args.seed):
        for snr in args.snr:
            noise=base*np.sqrt(power/(np.mean(base**2)*10**(snr/10)))
            observed=reference+noise
            measured=float(10*np.log10(power/np.mean(noise**2)))
            if not np.isclose(measured,snr,atol=1e-10):raise AssertionError('SNR mismatch')
            name=f'{kind}_{snr:g}dB.npz'
            np.savez_compressed(args.out/name,reference=reference,noise=noise,observed=observed,
                                sample_rate=fs,reference_kind=args.reference_kind)
            manifest['pairs'].append(dict(file=name,corruption=kind,snr_db=measured,
                                          peak=float(np.max(np.abs(observed))),
                                          would_clip_pcm16=int(np.count_nonzero(np.abs(observed)>1))))
    (args.out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(dict(pairs=len(manifest['pairs']),reference_kind=args.reference_kind,manifest=str(args.out/'manifest.json'))))


if __name__=='__main__':main()
