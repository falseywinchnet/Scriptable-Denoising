"""Static, exportable comparison of the exact sibling representations."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    root=Path(__file__).resolve().parents[2]
    out=root/'build/true-superresolution'
    z=np.load(out/'probe.npz');fs=int(z['sample_rate'])
    fields=[('1  Ordinary STFT magnitude · N=512',np.abs(z['ordinary']),fs/512),
            ('2  Exact double inverse · abs(cosine) · N=2048',z['original'],fs/4094),
            ('3  Positive inverse ridges · hypot(cosine, quadrature) · N=2048',z['envelope'],fs/4094),
            ('4  Registered sub-hop magnitude · N=2048',z['phase_fused'],fs/4094),
            ('5  Later enhanced field · registered apertures + reassigned texture',z['trace_field'],fs/4094)]
    fig,axes=plt.subplots(len(fields),2,figsize=(15,13),layout='constrained')
    for row,(label,field,df) in enumerate(fields):
        keep=int(np.ceil(3000/df))+1
        data=field[:keep]
        scale=max(float(np.quantile(data,.995)),1e-30)
        display=np.log1p(data/(.015*scale))
        for col,ax in enumerate(axes[row]):
            ax.imshow(display,origin='lower',aspect='auto',interpolation='nearest',cmap='magma',
                      extent=(-256/fs,(data.shape[1]-.5)*512/fs,-df/2,(data.shape[0]-.5)*df),vmin=0,vmax=np.log1p(1/.015))
            ax.set_ylim(0,3000);ax.set_xlim(0,1)
            ax.set_title(label if col==0 else 'Zoom · same data, no smoothing',loc='left',fontsize=10)
            ax.set_ylabel('Hz')
            if col:ax.set_xlim(.35,.80);ax.set_ylim(500,1700)
    axes[-1,0].set_xlabel('Seconds');axes[-1,1].set_xlabel('Seconds')
    fig.suptitle('Which enhanced magnitude? · Dave and Simon, first second\nIndependent display normalization per representation; row density is not a measured resolving-power claim.',fontsize=13)
    fig.savefig(out/'representation-comparison.png',dpi=150)
    plt.close(fig)

if __name__=='__main__':main()
