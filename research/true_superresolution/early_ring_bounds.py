"""Measure bounds before completion of the FIRST Fourier axis.

This is an optimistic feasibility audit: oracle peak/competitor incumbents are
used to show how much a given bound can ever prune. It is not a speed claim or
an executable search schedule. Each bound is checked against every covered row.
"""
import argparse
import json
from pathlib import Path
import numpy as np

N=48
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--units',type=Path,default=Path('build/ring-pruning/units.bin'))
    p.add_argument('--report',type=Path,default=Path('build/ring-pruning/early-bounds.json'))
    args=p.parse_args()
    units=np.fromfile(args.units,dtype='<c16').reshape(-1,N,N)
    f=np.fft.fftfreq(N);r=np.hypot(f[:,None],f[None,:])
    masks=np.exp(-.5*((r[None]-np.linspace(.035,.46,7)[:,None,None])/.055)**2)
    data=[]
    for i,u in enumerate(units):
        s=u.T[None]*masks
        first=np.fft.ifft(s,axis=1)
        field=np.fft.ifft(first,axis=2).real
        peak=field.reshape(7,-1).argmax(axis=1)
        heights=field.reshape(7,-1).max(axis=1)
        competitors=[]
        for ring,best in enumerate(peak):
            y,x=divmod(int(best),N)
            dy=np.minimum((np.arange(N)-y)%N,(y-np.arange(N))%N)
            dx=np.minimum((np.arange(N)-x)%N,(x-np.arange(N))%N)
            allowed=(dy[:,None]>3)|(dx[None,:]>3)
            competitors.append(field[ring][allowed].max())
        competitors=np.asarray(competitors)
        consensus=bool(np.all(peak==0))
        for q in (2,3,4,6,8,12,16,24):
            length=N//q
            # k = r + length*m, y = t + q*l. Only the q-point m transform
            # has run. Each group t still needs a length-point transform.
            stage=np.fft.ifft(s.reshape(7,q,length,N),axis=1)*q
            mask_stage=np.fft.ifft(masks.reshape(7,q,length,N),axis=1)*q
            mass=masks.reshape(7,q,length,N).sum(axis=1)[:,None,:,:]
            envelope=abs(mask_stage)+.125*mass
            bounds={
                'triangle':np.sum(abs(stage),axis=(2,3))/(N*N)+1e-12,
                'weighted_energy':np.sqrt(mass.sum(axis=(2,3))*np.sum(abs(stage)**2/mass,axis=(2,3)))/(N*N)+1e-12,
                'envelope_energy':np.sqrt(envelope.sum(axis=(2,3))*np.sum(abs(stage)**2/envelope,axis=(2,3)))/(N*N)+1e-12,
            }
            for method,upper in bounds.items():
                for t in range(q):
                    assert np.all(field[:,t::q,:].max(axis=(1,2))<=upper[:,t])
                peak_groups=np.sum(upper>=heights[:,None],axis=1)
                needed_groups=peak_groups if consensus else np.sum(upper>=competitors[:,None],axis=1)
                data.append(dict(chart=i,q=q,method=method,peak_fraction=float(peak_groups.mean()/q),
                                 final_fraction=float(needed_groups.mean()/q)))
    summaries={str(q):{method:dict(
        optimistic_peak_branches_skipped=1-float(np.mean([d['peak_fraction'] for d in data if d['q']==q and d['chart']>=9 and d['method']==method])),
        optimistic_final_branches_skipped=1-float(np.mean([d['final_fraction'] for d in data if d['q']==q and d['chart']>=9 and d['method']==method])),
    ) for method in ('triangle','weighted_energy','envelope_energy')} for q in (2,3,4,6,8,12,16,24)}
    result=dict(charts=len(units),meaning=__doc__,speech=summaries)
    args.report.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
