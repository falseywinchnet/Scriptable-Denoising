"""Frozen observed-mixture masks applied separately to known clean/noise channels."""
import argparse,hashlib,json,sys,time
from pathlib import Path
import numpy as np
from scipy import signal
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'tools'))
from native_client import Engine
FS=48000;DURATION=2.;COUNT=int(FS*DURATION)
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--shape-only',action='store_true');parser.add_argument('--soft-only',action='store_true');args=parser.parse_args()
 out=ROOT/('build/shape-soft-pairs' if args.soft_only else 'build/shape-pairs' if args.shape_only else 'build/occupancy-pairs');out.mkdir(parents=True,exist_ok=True)
 rng=np.random.default_rng(10203);clean=[];noise=[];labels=[]
 for cutoff in [1200.,3000.,6000.]:
  for kind in ['voiced','continuous_fm','weak_chirp']:
   t=np.arange(COUNT)/FS
   if kind=='voiced':
    phase=2*np.pi*(137*t+1.5*t*t);s=np.zeros(COUNT)
    for h in range(1,45):
     hz=137*h;env=.2+np.exp(-.5*((hz-600)/130)**2)+.7*np.exp(-.5*((hz-1500)/250)**2)
     s+=(env/h)*np.cos(h*phase+.07*h*h)
    s*=.02/np.sqrt(np.mean(s*s))
   elif kind=='continuous_fm':s=.03*np.cos(2*np.pi*600*t+2.5*np.sin(2*np.pi*110*t))
   else:s=.009*np.cos(2*np.pi*(400*t+160*t*t))*np.exp(-.5*((t-1.1)/.025)**2)
   n=.04*rng.normal(size=COUNT)
   sos=signal.butter(8,cutoff,fs=FS,output='sos');s=signal.sosfilt(sos,s);n=signal.sosfilt(sos,n)
   clean.append(s);noise.append(n);labels.append((cutoff,kind))
 s=np.concatenate(clean).astype('float32');n=np.concatenate(noise).astype('float32');y=s+n
 content=np.stack((s,n),axis=1);observed=np.stack((y,y),axis=1)
 source=(ROOT/'Filter.py').read_text().replace('CLEANUP_POPULATION = "rolloff"','CLEANUP_POPULATION = "receiver"').replace('GUIDE_REGISTRATION = False','GUIDE_REGISTRATION = True').replace('GUIDE_BINS = GUIDE_FFT','GUIDE_BINS = 512',1);report=dict(seed=10203,sample_rate=FS,source_sha256=hashlib.sha256(source.encode()).hexdigest(),description='Identical observed-mixture analysis in both channels; one common adaptive mask filters known clean and noise separately. Synthetic fixtures, not speech-quality scores.',variants=[])
 variants=[('shape-soft' if args.soft_only else 'shape','registered','occupancy_surface')] if args.shape_only or args.soft_only else [('baseline','baseline','legacy'),('surface','registered','variance_surface'),('occupancy','registered','occupancy_surface')]
 for name,mode,floor in variants:
  text=source.replace('ANALYSIS_MODE = "registered"',f'ANALYSIS_MODE = "{mode}"').replace('REGISTERED_FLOOR = "legacy"',f'REGISTERED_FLOOR = "{floor}"').replace('VIEWER = True','VIEWER = False')
  if args.shape_only or args.soft_only:text=text.replace('OCCUPANCY_SHAPE = False','OCCUPANCY_SHAPE = True')
  if args.shape_only:text=text.replace('SHAPE_SOFT_ADMISSION = True','SHAPE_SOFT_ADMISSION = False')
  path=out/(name+'.py');path.write_text(text)
  with Engine(ROOT/'build/libCleanupNative.dylib',path,channels=2,sample_rate=FS) as e:
   st=e.wait();assert st['success'],st;e.lib.cleanup_set_harmonics(e.handle,0)
   delay=st['latency_samples'];pad=np.zeros((delay+8192,2),'float32');xs=np.concatenate((content,pad));ys=np.concatenate((observed,pad));outputs=[]
   start=time.monotonic()
   for at in range(0,len(xs),8192):
    # Give baseline the oracle receive setting; new estimators do not use it.
    scene=min(len(labels)-1,at//COUNT);e.lib.cleanup_set_bandwidth(e.handle,labels[scene][0])
    z,code=e.process(xs[at:at+8192],analysis=ys[at:at+8192]);assert code==0,e.status();outputs.append(z)
   z=np.concatenate(outputs)[delay:delay+len(s)];assert np.isfinite(z).all()
   metrics=[]
   for i,(cutoff,kind) in enumerate(labels):
    a=i*COUNT+int(.6*FS);b=i*COUNT+int(1.65*FS)
    ps=np.mean(s[a:b]**2);pn=np.mean(n[a:b]**2);os=np.mean(z[a:b,0]**2);on=np.mean(z[a:b,1]**2)
    metrics.append(dict(cutoff_hz=cutoff,kind=kind,clean_gain_db=float(10*np.log10(max(os,1e-30)/ps)),noise_gain_db=float(10*np.log10(max(on,1e-30)/pn)),clean_distortion_ratio=float(np.mean((z[a:b,0]-s[a:b])**2)/ps),mixture_error_ratio=float(np.mean((z[a:b].sum(axis=1)-s[a:b])**2)/ps)))
   row=dict(name=name,seconds=time.monotonic()-start,status=e.status(),metrics=metrics);report['variants'].append(row);print(json.dumps(row),flush=True)
 (out/'results.json').write_text(json.dumps(report,indent=2))
if __name__=='__main__':main()
