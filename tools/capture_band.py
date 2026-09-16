"""Capture live same-grid input/output spectra and squelch evidence without changing controls."""
import argparse
import json
import math
from pathlib import Path
import time
from control import request


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--port',type=int,default=52381)
    p.add_argument('--seconds',type=float,default=5)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    snapshots=[];end=time.monotonic()+a.seconds;last=None
    while time.monotonic()<end:
        d=request(a.port,'diagnostic')
        key=(d.get('generation'),d.get('sequence'))
        if d.get('enabled') and key!=last and 'input' in d:
            snapshots.append(d);last=key
        time.sleep(.2)
    status=request(a.port,'status')
    curves=[]
    for d in snapshots:
        n=d['bins'];count=d['emit'];half=.5*d['half_bin'];df=d['sample_rate']/d['fft']
        bins=[]
        for b in range(n):
            hz=(b+half)*df
            if not 1500<=hz<=4500:continue
            incoming=sum(d['input'][t*n+b]**2 for t in range(count))/count
            outgoing=sum(d['filtered'][t*n+b]**2 for t in range(count))/count
            bins.append(dict(hz=hz,input_db=10*math.log10(max(incoming,1e-30)),
                             output_db=10*math.log10(max(outgoing,1e-30)),
                             gain_db=10*math.log10(max(outgoing,1e-30)/max(incoming,1e-30))))
        curves.append(dict(generation=d['generation'],sequence=d['sequence'],bypass=d['bypass'],
                           action=d['action'],squelch=d['squelch'],evidence=d.get('evidence'),bins=bins))
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(dict(status=status,curves=curves,snapshots=snapshots)))
    print(json.dumps(dict(path=str(a.output),snapshots=len(snapshots),status=status),indent=2))
    if not snapshots:raise SystemExit('No same-grid input diagnostic: install matching engine and enable viewer')


if __name__=='__main__':main()
