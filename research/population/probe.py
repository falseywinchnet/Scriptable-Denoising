"""Support selection on actual native registered fields; no oracle at inference."""
import argparse
import ctypes as C
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np
from scipy import signal
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'runtime'),str(ROOT/'tests')]
from compile_filter import bind_native_dsp
from test_registered_guide import library,ptr
spec=importlib.util.spec_from_file_location('population_probe',ROOT/'Filter.py')
m=importlib.util.module_from_spec(spec);bind_native_dsp(m);spec.loader.exec_module(m)
parser=argparse.ArgumentParser();parser.add_argument('--holdout',action='store_true');args=parser.parse_args()
out=ROOT/('build/population-holdout' if args.holdout else 'build/population');out.mkdir(exist_ok=True)
lib=library();plan=lib.cleanup_guide_create(2048,2048,48,512,0)
assert plan
rng=np.random.default_rng(97423 if args.holdout else 62489);fs=48000.;n=24576;t=np.arange(n)/fs
report=[];fields={}
cases=[('noise1200',1200.,'noise'),('noise3000',3000.,'noise'),('noise6000',6000.,'noise'),('fullband',24000.,'noise'),('voiced3000',3000.,'voiced'),('fm3000',3000.,'fm'),('carrier3000',3000.,'carrier'),('notch3000',3000.,'notch')]
if args.holdout:cases=[('order4_2000',2000.,'order4'),('order2_4000',4000.,'order2'),('order12_5000',5000.,'order12'),('pinkfull',24000.,'pink'),('blue3000',3000.,'blue'),('tone_only',3000.,'tone'),('impulses3000',3000.,'impulses')]
try:
 for name,cutoff,kind in cases:
  x=.02*rng.normal(size=n)
  if kind=='pink':x=signal.lfilter([1.],[1.,-.98],x)
  if kind=='blue':x=np.diff(x,prepend=x[0])
  if kind=='tone':x=.1*np.cos(2*np.pi*901*t)
  if kind=='impulses':x[::1091]+=1.
  if kind=='voiced':
   for h in range(1,50):x+=.05/h*np.cos(2*np.pi*h*(137*t+2*t*t))
  if kind=='fm':x+=.08*np.cos(2*np.pi*600*t+3*np.sin(2*np.pi*117*t))
  if kind=='carrier':x+=2.*np.cos(2*np.pi*711*t)
  if kind=='notch':x=signal.sosfilt(signal.butter(5,[800,1800],btype='bandstop',fs=fs,output='sos'),x)
  if cutoff<24000.:x=signal.sosfilt(signal.butter(int(kind[5:]) if kind.startswith('order') else 8,cutoff,fs=fs,output='sos'),x)
  guide=np.zeros((48,2048));assert lib.cleanup_guide_run(plan,ptr(np.ascontiguousarray(x)),len(x),ptr(guide))==0
  envelope=np.zeros(2048);prefix=np.zeros((97,5));scratch=np.zeros(2048+192);diag=np.zeros(8)
  nb=m.population_support(guide,fs,4094.,envelope,prefix,scratch,diag, np.zeros(192))
  row=dict(name=name,physical_cutoff_hz=cutoff,population_bins=nb,population_hz=(nb-1)*fs/4094.,diagnostics=diag.tolist())
  report.append(row);fields[name]=guide;fields[name+'_envelope']=envelope.copy();print(json.dumps(row),flush=True)
 # Test amplitude scaling and steady fields directly, without new registration.
 direct=[]
 for gain in (.01,1.,100.):
  diag=np.zeros(8);nb=m.population_support(fields[cases[0][0]]*gain,fs,4094.,envelope,prefix,scratch,diag, np.zeros(192))
  direct.append(dict(gain=gain,bins=nb))
 assert len(set(v['bins'] for v in direct))==1,direct
 (out/'probe.json').write_text(json.dumps(dict(cases=report,scale_invariance=direct),indent=2))
 np.savez_compressed(out/'probe.npz',**fields)
finally:lib.cleanup_guide_destroy(plan)
