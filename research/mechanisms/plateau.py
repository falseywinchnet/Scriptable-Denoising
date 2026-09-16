"""Probe the decision rule on controlled magnitude fields, not acoustic signals."""
import importlib.util,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'runtime')]
from compile_filter import bind_native_dsp
spec=importlib.util.spec_from_file_location('mechanism_probe',ROOT/'Filter.py');m=importlib.util.module_from_spec(spec);bind_native_dsp(m);spec.loader.exec_module(m)
out=ROOT/'build/mechanism-audit';out.mkdir(exist_ok=True);records=[];saved={}
for ripple in (0.,.005,.05):
 mag=np.full((192,512),.02);mag[:,290:]=0.
 # A wide foreground plateau inside the analysis population, bounded by
 # ordinary lower-amplitude observations. Ripple is fixed across time.
 rng=np.random.default_rng(403);mag[:,80:160]=2.+ripple*rng.normal(size=80)
 smooth=np.zeros_like(mag);mask=np.zeros_like(mag);scratch=np.zeros(12*512);values=np.zeros(mag.size);dev=values.copy()
 raw=np.zeros(192);sm=raw.copy();det=raw.copy();log1=np.zeros(512);log3=np.zeros(1536)
 _,maximum,_=m.guide_entropy(mag,290,raw,sm,det,log1,log3,scratch,True, np.zeros(192))
 m.sawtooth(mag,smooth,290,scratch);man,atd=m.statistics(smooth,290,values,dev)
 threshold,frac=m.peaks(smooth,mask,290,raw,det,maximum,False,man,atd,values,dev)
 means=np.zeros_like(mag);var=means.copy();floor=means.copy();clipped=means.copy();low=means.copy();occ=means.copy();ref=means.copy()
 m.variance_surface(smooth,means,var,floor,clipped,np.zeros((3,512)),np.zeros(2),scratch,48000.,0., np.zeros(192))
 m.occupancy_surface(clipped,means,var,floor,low,occ,ref,np.zeros(1024),np.zeros(2),scratch,48000.,0., np.zeros(192))
 alternative=np.zeros_like(mag);m.surface_peaks(smooth,alternative,floor,512)
 roi=np.s_[63:127,100:140]
 row=dict(ripple=ripple,reference_plateau_acceptance=float(mask[roi].mean()),occupancy_plateau_acceptance=float(alternative[roi].mean()),reference_threshold=float(threshold),occupancy_threshold=float(floor[roi].mean()),plateau_mean=float(means[roi].mean()),minimum_threshold_minus_mean=float(np.min(floor[roi]-means[roi])))
 records.append(row);saved['input_'+str(ripple)]=smooth[95];saved['reference_'+str(ripple)]=mask[95];saved['occupancy_'+str(ripple)]=alternative[95];saved['floor_'+str(ripple)]=floor[95]
 print(json.dumps(row),flush=True)
assert records[0]['reference_plateau_acceptance']==1.
assert records[0]['occupancy_plateau_acceptance']==0.
(out/'plateau.json').write_text(json.dumps(dict(description='Controlled, time-constant magnitude fields; this probes threshold selectivity, not audio perceptual quality.',records=records),indent=2));np.savez_compressed(out/'plateau.npz',**saved)
