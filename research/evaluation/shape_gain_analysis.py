"""Separate uniform attenuation from waveform distortion in saved pair metrics."""
import argparse,json,math
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('reports',nargs='+',type=Path);p.add_argument('--output',required=True,type=Path);a=p.parse_args();rows=[]
for path in a.reports:
 for variant in json.loads(path.read_text())['variants']:
  for m in variant['metrics']:
   power=10**(m['clean_gain_db']/10);gain=(1+power-m['clean_distortion_ratio'])/2
   rows.append(dict(name=variant['name'],cutoff_hz=m['cutoff_hz'],kind=m['kind'],least_squares_gain=gain,clean_coherence=gain/math.sqrt(power),gain_compensated_distortion=max(0.,power-gain*gain)/max(gain*gain,1e-30),snr_change_db=m['clean_gain_db']-m['noise_gain_db']))
a.output.write_text(json.dumps(dict(method='Derived from recorded clean output/input energy and squared error: g=(1+Pout/Pin-D)/2; residual=Pout/Pin-g². Identical known-mixture adaptive mask on clean and noise; no refit.',rows=rows),indent=2))
for r in rows:
 if r['cutoff_hz']==3000:print(json.dumps(r))
