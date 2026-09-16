"""Compare paired registration on/off native renders without loudness matching.

Descriptive differences only: the recording has no clean ground-truth target.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from experiment import read_wav

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('directory',type=Path)
a=p.parse_args()
names=['population','population-unregistered']
audio=[];fields=[];metadata=[]
for name in names:
    x,rate=read_wav(a.directory/(name+'.wav'));audio.append(x.astype(float))
    with np.load(a.directory/(name+'-spectra.npz')) as z:
        fields.append({key:z[key] for key in ('guide','final')})
    metadata.append(json.loads((a.directory/(name+'-spectra.json')).read_text()))
assert audio[0].shape==audio[1].shape
report=dict(samples=audio[0].size,comparison='Same native build, source gain, timing, Population Cleanup and no harmonics; only registration changes.',
    audio_correlation=float(np.corrcoef(audio[0].ravel(),audio[1].ravel())[0,1]),
    output_rms_db_change=float(20*np.log10(np.linalg.norm(audio[1])/np.linalg.norm(audio[0]))),
    variants={})
for name,x,meta in zip(names,audio,metadata):
    interior=meta['statistics'][3:-3]
    masks=np.array([r['mask'] for r in interior])
    populations=np.array([r['population'][0] for r in interior])
    report['variants'][name]=dict(rms=float(np.sqrt(np.mean(x*x))),peak=float(abs(x).max()),
        clipped_samples=int(np.count_nonzero(abs(x)>=1)),
        median_selected_fraction_pass1=float(np.median(masks[:,3])),
        median_selected_fraction_pass2=float(np.median(masks[:,7])),
        median_population_edge_hz=float(np.median(populations-1)*meta['panels'][0]['frequency_step']))

# A common-scale image is more useful than independently auto-contrasted panels.
fig,axes=plt.subplots(2,2,figsize=(14,7),sharex=True,sharey=True,layout='constrained')
for col,(name,data,meta) in enumerate(zip(names,fields,metadata)):
    for row,key in enumerate(('guide','final')):
        panel=meta['panels'][row];values=data[key]
        time=(panel['first_center']+np.arange(len(values))*panel['hop'])/rate
        frequency=(np.arange(values.shape[1])+panel['half_bin'])*panel['frequency_step']
        ts=(time>=1)&(time<=5);fs=frequency<=4000
        db=20*np.log10(np.maximum(values[np.ix_(ts,fs)]/panel['normalizer'],1e-12))
        image=axes[row,col].pcolormesh(time[ts],frequency[fs],db.T,shading='nearest',cmap='magma',vmin=-90,vmax=0,rasterized=True)
        axes[row,col].set_title(('Registration on' if col==0 else 'Registration off')+' · '+('analysis' if row==0 else 'final before inversion'))
        axes[row,col].set_ylabel('Hz');axes[row,col].set_xlabel('Seconds')
fig.colorbar(image,ax=axes,label='dB, shared coherent-window amplitude scale')
fig.savefig(a.directory/'registration-comparison.png',dpi=140)
(a.directory/'comparison.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
