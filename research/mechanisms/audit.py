"""One-change ablations through the real native engine; no new denoiser here."""
import argparse,hashlib,json,sys,time
from pathlib import Path
import numpy as np
from scipy import signal
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'tools')]
from native_client import Engine
from experiment import read_wav,write_wav
from listening_demo import diagnostic
FS=48000
VARIANTS=[('registered-reference','legacy',0),('frozen-second-statistics','legacy',1),('binary-second-pass','legacy',2),('no-row-admission','legacy',4),('no-context-shape-scaling','legacy',8),('no-residual-floor','legacy',16),('row-statistics-only','legacy',32),('expanded-guide-same-cleanup','legacy',64),('occupancy-reference','occupancy_surface',0),('occupancy-residual-second','occupancy_surface',128)]
def fixtures():
 x,rate=read_wav(ROOT/'build/listening-demo-v15/original.wav');assert rate==FS
 radio=x[:8*FS,0];content=[np.stack((radio,np.zeros_like(radio)),axis=1)];observed=[np.stack((radio,radio),axis=1)];scenes=[];cursor=len(radio);rng=np.random.default_rng(71313)
 for cutoff,kind in [(3000.,'voiced'),(3000.,'continuous_fm'),(3000.,'weak_chirp'),(1200.,'weak_chirp')]:
  t=np.arange(2*FS)/FS
  if kind=='voiced':
   phase=2*np.pi*(137*t+1.5*t*t);s=np.zeros(len(t))
   for h in range(1,45):
    hz=137*h;env=.2+np.exp(-.5*((hz-600)/130)**2)+.7*np.exp(-.5*((hz-1500)/250)**2);s+=(env/h)*np.cos(h*phase+.07*h*h)
   s*=.02/np.sqrt(np.mean(s*s))
  elif kind=='continuous_fm':s=.03*np.cos(2*np.pi*600*t+2.5*np.sin(2*np.pi*110*t))
  else:s=.009*np.cos(2*np.pi*(400*t+160*t*t))*np.exp(-.5*((t-1.1)/.025)**2)
  sos=signal.butter(8,cutoff,fs=FS,output='sos');s=signal.sosfilt(sos,s);n=signal.sosfilt(sos,.04*rng.normal(size=len(t)));y=s+n
  content.append(np.stack((s,n),axis=1));observed.append(np.stack((y,y),axis=1));scenes.append(dict(start=cursor,stop=cursor+len(t),kind=kind,cutoff_hz=cutoff));cursor+=len(t)
 return np.concatenate(content).astype('float32'),np.concatenate(observed).astype('float32'),scenes

def compare_existing_guides(out):
 a=np.load(ROOT/'build/listening-demo-v15/registered-spectra.npz')['guide'];b=np.load(ROOT/'build/listening-demo-v15/occupancy-spectra.npz')['guide'];rows=min(len(a),len(b));bins=int(3400/(FS/4094))+1
 a=a[:rows,:bins].astype(float);b=b[:rows,:bins].astype(float);scale=np.sum(a*b)/np.sum(a*a)
 result=dict(compared_bins=bins,scale=float(scale),relative_error=float(np.linalg.norm(b-a)/np.linalg.norm(a)),residual_after_scalar=float(np.linalg.norm(b-scale*a)/np.linalg.norm(b)),correlation=float(np.corrcoef(a.ravel(),b.ravel())[0,1]))
 (out/'existing-guide-comparison.json').write_text(json.dumps(result,indent=2));print('existing guide comparison',json.dumps(result),flush=True)

def main():
 p=argparse.ArgumentParser();p.add_argument('--only',choices=[v[0] for v in VARIANTS]);p.add_argument('--population',action='store_true');a=p.parse_args()
 out=ROOT/('build/population-evaluation' if a.population else 'build/mechanism-audit');out.mkdir(exist_ok=True);compare_existing_guides(out)
 content,observed,scenes=fixtures();np.savez_compressed(out/'fixtures.npz',content=content,observed=observed);write_wav(out/'original.wav',content[:8*FS,:1],FS)
 source=(ROOT/'Filter.py').read_text().replace('CLEANUP_POPULATION = "rolloff"','CLEANUP_POPULATION = "receiver"').replace('GUIDE_REGISTRATION = False','GUIDE_REGISTRATION = True').replace('GUIDE_BINS = GUIDE_FFT','GUIDE_BINS = 512',1);report=dict(seed=71313,sample_rate=FS,description='Same mixture analysis on both channels; separately filter known clean and noise. Eight seconds of unchanged radio precede fixed synthetic fixtures. One change per registered-reference ablation.',stages=['first_seeds','second_combination','after_recursion','final_gain'],stage_fields=['mean','zero_fraction','soft_fraction','one_fraction','time_tv','frequency_tv','mean_square','maximum'],variants=[])
 variants=[('receiver-full-guide','legacy',64),('oracle-population','legacy',0),('rolloff-population','legacy',0)] if a.population else VARIANTS
 for name,floor,bits in variants:
  if a.only and name!=a.only:continue
  text=source.replace('REGISTERED_FLOOR = "legacy"',f'REGISTERED_FLOOR = "{floor}"').replace('CLEANUP_AUDIT_BITS = 0',f'CLEANUP_AUDIT_BITS = {bits}').replace('CLEANUP_AUDIT_TRACE = False','CLEANUP_AUDIT_TRACE = True').replace('VIEWER_AUTOSTART = True','VIEWER_AUTOSTART = False')
  if a.population:
   population='receiver' if name=='receiver-full-guide' else 'oracle' if name=='oracle-population' else 'rolloff'
   text=text.replace('CLEANUP_POPULATION = "receiver"',f'CLEANUP_POPULATION = "{population}"')
  path=out/(name+'.py');path.write_text(text);print('starting',name,flush=True)
  with Engine(ROOT/'build/libCleanupNative.dylib',path,channels=2,sample_rate=FS) as e:
   st=e.wait();assert st['success'],st;e.lib.cleanup_set_harmonics(e.handle,0);delay=st['latency_samples'];pad=np.zeros((delay+8192,2),'float32');xs=np.concatenate((content,pad));ys=np.concatenate((observed,pad));outputs=[];trace=[];guide=[];final=[];stats=[];population_trace=[]
   start=time.monotonic()
   for at in range(0,len(xs),8192):
    bandwidth=3400. if at<8*FS else next((s['cutoff_hz'] for s in scenes if s['start']<=at<s['stop']),1200.)
    e.lib.cleanup_set_bandwidth(e.handle,bandwidth)
    z,code=e.process(xs[at:at+8192],analysis=ys[at:at+8192]);assert code==0,e.status();outputs.append(z)
    if a.population:
     population_trace.append(dict(input_sample=at,values=diagnostic(st['port']).get('population_stats',[])))
    if at<8*FS and len(xs[at:at+8192])==8192:
     d=diagnostic(st['port']);assert d['valid'];trace.append(d['mask_stages']);stats.append(d['mask_stats']);guide.append(np.asarray(d['guide'],'float32').reshape(d['guide_emit'],d['guide_bins']));final.append(np.asarray(d['filtered'],'float32').reshape(d['emit'],d['bins']))
   z=np.concatenate(outputs)[delay:delay+len(content)];assert np.isfinite(z).all();write_wav(out/(name+'.wav'),z[:8*FS,:1],FS)
   metrics=[]
   for scene in scenes:
    lo=scene['start']+int(.6*FS);hi=scene['start']+int(1.65*FS);c=content[lo:hi,0].astype(float);n=content[lo:hi,1].astype(float);oc=z[lo:hi,0].astype(float);on=z[lo:hi,1].astype(float)
    pc=np.mean(c*c);pn=np.mean(n*n);poc=np.mean(oc*oc);pon=np.mean(on*on);g=np.mean(c*oc)/pc
    chunks=on[:len(on)//960*960].reshape(-1,960);energy=np.mean(chunks*chunks,axis=1)
    metrics.append(dict(kind=scene['kind'],cutoff_hz=scene['cutoff_hz'],clean_gain_db=float(10*np.log10(max(poc,1e-30)/pc)),noise_gain_db=float(10*np.log10(max(pon,1e-30)/pn)),clean_coherence=float(g/np.sqrt(max(poc/pc,1e-30))),gain_compensated_distortion=float(max(0,poc/pc-g*g)/max(g*g,1e-30)),noise_energy_modulation=float(np.std(energy)/max(np.mean(energy),1e-30))))
   stage=np.array(trace).reshape(-1,4,8);np.savez_compressed(out/(name+'-stages.npz'),stages=stage,guide=np.concatenate(guide),final=np.concatenate(final),statistics=np.array(stats))
   row=dict(name=name,floor=floor,bits=bits,script_sha256=hashlib.sha256(text.encode()).hexdigest(),seconds=time.monotonic()-start,status=e.status(),metrics=metrics,radio_stage_means=stage[4:-2].mean(axis=0).tolist(),population_trace=population_trace,radio_rms=float(np.sqrt(np.mean(z[int(.6*FS):int(7.5*FS),0]**2))))
   report['variants'].append(row);(out/'results.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in row.items() if k!='population_trace'}),flush=True)
if __name__=='__main__':main()
