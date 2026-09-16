"""Scientific evidence: observed envelopes and statistical-domain boundaries."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2]
data=np.load(ROOT/'build/population/probe.npz')
report=json.loads((ROOT/'build/population/probe.json').read_text())
cases={r['name']:r for r in report['cases']}
fig,axes=plt.subplots(2,2,figsize=(12,7),layout='constrained')
freq=np.arange(2048)*48000/4094
for ax,key,title in zip(axes.ravel()[:3],['noise3000','carrier3000','notch3000'],['Band-limited broadband noise','Same LPF, strong continuous carrier','Same LPF, deep internal notch']):
    row=cases[key];envelope=data[key+'_envelope']
    ax.plot(freq,20*np.log10(np.maximum(envelope,1e-20)),lw=1,color='#335b7d',label='Mean registered magnitude')
    ax.axvline(row['physical_cutoff_hz'],color='#777',ls='--',label='Physical LPF cutoff (test only)')
    ax.axvline(row['population_hz'],color='#198870',label='Inferred statistical boundary')
    ax.set(title=title,xlabel='Frequency (Hz)',ylabel='Magnitude (dB, raw guide units)',xlim=(0,7000))
axes[0,0].legend(fontsize=8)
ax=axes[1,1]
r=json.loads((ROOT/'build/population-evaluation/results.json').read_text())
v=next(v for v in r['variants'] if v['name']=='rolloff-population')
p=[p for p in v['population_trace'] if .6*48000<p['input_sample']<7.5*48000]
ax.plot([p['input_sample']/48000 for p in p],[(p['values'][0]-1)*48000/4094 for p in p],color='#198870')
ax.axhline(3400,color='#777',ls='--',label='Reference receiver setting, not measured truth')
ax.set(title='Radio: population follows the observed envelope',xlabel='Processed input position (s)',ylabel='Statistical boundary (Hz)',ylim=(1800,3700))
ax.legend(fontsize=8)
fig.suptitle('Population Cleanup — the two-pass decision operator stays intact',fontsize=14)
fig.savefig(ROOT/'build/population/population.png',dpi=160)
