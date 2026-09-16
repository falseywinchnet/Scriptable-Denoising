"""Analytical floor audit; no live deployment or synthetic-speech quality claim.

All candidate DSP is in explicitly compiled Filter.py helpers. This driver builds
known additive pairs, calls the actual native registered transform, measures the
candidate, and plots evidence. Cutoff labels are used ONLY by the scorer.
"""
import argparse
import ast
import ctypes as C
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
import numpy as np
from scipy import signal, special
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'runtime'))
from compile_filter import bind_native_dsp, _audit_tree
FS=48000; N=2048; HOP=512; ROWS=48; BINS=2048; SIZE=24576
FREQ=np.arange(BINS)*FS/(2*(N-1)); EMIT=slice(16,32); VALID=slice(3,45)
Q=.2

def load_filter():
    spec=importlib.util.spec_from_file_location('floor_research_filter',ROOT/'Filter.py')
    m=importlib.util.module_from_spec(spec);bind_native_dsp(m)
    tree=_audit_tree(ast.parse((ROOT/'Filter.py').read_text()))
    exec(compile(tree,str(ROOT/'Filter.py'),'exec'),m.__dict__)
    for name in ('research_floor_ratio','research_guarded_floor','research_joint_ratio','research_pool_score','research_tail_seeds'):
        assert len(getattr(m,name).nopython_signatures)==1, name
    return m

class Guide:
    def __init__(self,path):
        self.lib=C.CDLL(str(path.resolve()));p=C.POINTER(C.c_double)
        self.lib.cleanup_guide_create.argtypes=[C.c_size_t]*4+[C.c_long];self.lib.cleanup_guide_create.restype=C.c_void_p
        self.lib.cleanup_guide_run.argtypes=[C.c_void_p,p,C.c_size_t,p]
        self.lib.cleanup_guide_destroy.argtypes=[C.c_void_p]
        self.plan=self.lib.cleanup_guide_create(N,BINS,ROWS,HOP,0)
        if not self.plan:raise RuntimeError('guide allocation')
        self.p=p
    def __call__(self,x):
        x=np.ascontiguousarray(x,dtype='float64');y=np.empty((ROWS,BINS))
        if self.lib.cleanup_guide_run(self.plan,x.ctypes.data_as(self.p),len(x),y.ctypes.data_as(self.p)):raise RuntimeError('guide error')
        return y
    def close(self):self.lib.cleanup_guide_destroy(self.plan)

def candidate(m,data):
    floor=np.empty(data.shape[1]);ratio=np.empty_like(data);scratch=np.empty(data.shape[0])
    assert m.research_floor_ratio(np.ascontiguousarray(data),floor,ratio,scratch,Q,3,data.shape[0]-3, np.zeros(192))==0
    return floor,ratio

def joint_candidate(m,data,temporal):
    joint=np.empty_like(temporal);scratch=np.empty(16)
    ratio=np.empty_like(data)
    assert m.research_joint_ratio(data,temporal,joint,ratio,scratch,4,16, np.zeros(192))==0
    return joint,ratio

def pooled(m,ratio):
    # Preserve the original Cleanup time footprint: interpolate only after
    # quantile fitting, then call the exact padded paired-branch recurrence.
    dense=np.empty((192,ratio.shape[1]));output=np.empty_like(ratio)
    nr,nc=dense.shape
    v=np.zeros((nr+26,nc+6));h=v.copy();scratch=np.zeros(max(v.shape))
    assert m.research_pool_score(ratio,output,dense,v,h,scratch)==0
    return output

def filtered(x,cutoff,kind='butter'):
    # Filter acts on signal AND upstream noise. Labels never reach candidate().
    if cutoff is None:return x.copy()
    if kind=='brick':
        f=np.fft.rfftfreq(len(x),1/FS);return np.fft.irfft(np.fft.rfft(x)*(f<cutoff),len(x))
    return signal.sosfilt(signal.butter(8,cutoff,fs=FS,output='sos'),x)

def make_case(seed,cutoff,kind='noise',filter_kind='butter'):
    # Warm filter before the measured context, avoiding startup as an oracle.
    rng=np.random.default_rng(seed);length=SIZE+8192;t=np.arange(length)/FS
    noise=.02*rng.normal(size=length);clean=np.zeros(length)
    center=(8192+SIZE/2)/FS
    if kind in ('sparse_tone','weak_tone'):
        envelope=np.exp(-.5*((t-center)/.035)**2)
        clean=(.0015 if kind=='weak_tone' else .025)*envelope*np.cos(2*np.pi*900*t+.4)
    elif kind=='persistent_tone':clean=.025*np.cos(2*np.pi*900*t+.4)
    elif kind=='brief_chirp':
        local=t-center;envelope=np.exp(-.5*(local/.004)**2)
        clean=.025*envelope*np.cos(2*np.pi*(900*local+15000*local**2))
    elif kind=='impulses':
        for tc in [center-.11,center,center+.09]:noise[int(tc*FS):int(tc*FS)+3]+=.5
    elif kind=='colored':noise=signal.lfilter([1.],[1.,-.85],noise)*np.sqrt(1-.85**2)
    elif kind=='noise_step':noise[t>center]*=3
    clean=filtered(clean,cutoff,filter_kind)[8192:]
    noise=filtered(noise,cutoff,filter_kind)[8192:]
    return clean,noise

def null_summary(score,threshold,region):
    x=score[EMIT][:,region]
    flags=x>threshold
    return dict(seed_fraction=float(flags.mean()),frame_fraction_with_any_seed=float(np.any(flags,axis=1).mean()))

def invariants(m):
    rng=np.random.default_rng(39);x=np.exp(rng.normal(size=(48,85)))
    floor,r=candidate(m,x);h=np.exp(rng.uniform(-16,4,85));fh,rh=candidate(m,x*h)
    _,rp=candidate(m,np.pad(x,((0,0),(0,2000))))
    _,rz=candidate(m,np.zeros_like(x))
    assert np.max(abs(r-rh))<1e-10
    np.testing.assert_array_equal(r,rp[:,:85]);np.testing.assert_array_equal(rz,0)
    assert np.max(abs(fh-floor*h))<1e-10
    return dict(diagonal_amplitude_scale_max_ratio_error=float(np.max(abs(r-rh))),appended_zero_bins_max_ratio_error=float(np.max(abs(r-rp[:,:85]))),zero_input='abstention')

def main():
    p=argparse.ArgumentParser();p.add_argument('--library',type=Path,default=ROOT/'build/libCleanupNative.dylib');p.add_argument('--out',type=Path,default=ROOT/'build/floor-analysis');p.add_argument('--demo',type=Path,default=ROOT/'build/listening-demo-v5/crossing-spectra.npz');p.add_argument('--trials',type=int,default=12);p.add_argument('--calibration',type=int,default=8);args=p.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    m=load_filter();guide=Guide(args.library);started=time.monotonic()
    report=dict(source_sha256=hashlib.sha256((ROOT/'Filter.py').read_bytes()).hexdigest(),quantile=Q,context_seconds=SIZE/FS,emitted_frames=16,quantile_frames=42,calibration=f'{args.calibration} independent white-noise contexts; candidate local temporal q20; no cutoff input',heldout_trials_per_case=args.trials,geometry=dict(sample_rate=FS,guide_fft=N,guide_hop=HOP,guide_rows=ROWS,guide_bins=BINS,guard_bins=4,flank_bins=16),driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),native_sha256=hashlib.sha256(args.library.read_bytes()).hexdigest(),invariants=invariants(m))
    training=[];training_pool=[];training_joint=[];training_joint_pool=[];raw=[];example_white=None
    for seed in range(110,110+args.calibration):
        _,noise=make_case(seed,None);a=guide(noise);floor,r=candidate(m,a);s=pooled(m,r)
        interior=(FREQ>500)&(FREQ<20000)
        training.append(r[EMIT][:,interior].ravel());training_pool.append(s[EMIT][:,interior].ravel())
        jf,jr=joint_candidate(m,a,floor);js=pooled(m,jr)
        training_joint.append(jr[EMIT][:,interior].ravel());training_joint_pool.append(js[EMIT][:,interior].ravel())
        raw.append(a[EMIT][:,interior].ravel());example_white=(a,floor,r)
    calibration=np.concatenate(training);cal_pool=np.concatenate(training_pool)
    threshold=float(np.quantile(calibration,.99));threshold_pool=float(np.quantile(cal_pool,.99))
    threshold_joint=float(np.quantile(np.concatenate(training_joint),.99));threshold_joint_pool=float(np.quantile(np.concatenate(training_joint_pool),.99))
    report['null_thresholds']=dict(joint=threshold_joint,joint_pooled=threshold_joint_pool,cell=threshold,pooled=threshold_pool,nominal_cell_false_alarm=.01)
    # Compare an incorrect half-normal threshold to the ACTUAL registered law.
    half_ratio=special.ndtri(.995)/special.ndtri(.6)
    report['wrong_halfnormal_ratio_threshold']=float(half_ratio)
    report['wrong_halfnormal_empirical_false_alarm']=float(np.mean(calibration>half_ratio))
    records=[];examples={};truth_examples={}
    case_specs=[('noise',c,'butter') for c in [500.,1200.,3000.,6000.,12000.,None]]
    case_specs += [('noise',3000.,'brick'),('colored',3000.,'butter'),('impulses',3000.,'butter'),('noise_step',3000.,'butter'),('sparse_tone',3000.,'butter'),('weak_tone',3000.,'butter'),('persistent_tone',3000.,'butter'),('brief_chirp',3000.,'butter')]
    for index,(kind,cutoff,fkind) in enumerate(case_specs):
        trials=[]
        for repeat in range(args.trials):
            clean,noise=make_case(900+repeat,cutoff,kind,fkind);a=guide(clean+noise);floor,r=candidate(m,a);s=pooled(m,r)
            jf,jr=joint_candidate(m,a,floor);js=pooled(m,jr)
            upper=20000 if cutoff is None else cutoff*.70
            region=(FREQ>250)&(FREQ<upper)
            row=dict(**null_summary(r,threshold,region),pooled=null_summary(s,threshold_pool,region),joint=null_summary(jr,threshold_joint,region),joint_pooled=null_summary(js,threshold_joint_pool,region))
            if cutoff is not None:
                edge=(FREQ>.8*cutoff)&(FREQ<1.2*cutoff)
                row['transition']=null_summary(r,threshold,edge)
                row['transition_pooled']=null_summary(s,threshold_pool,edge)
                row['transition_joint']=null_summary(jr,threshold_joint,edge)
            if kind in ('sparse_tone','weak_tone','persistent_tone','brief_chirp'):
                only_noise=guide(noise);noise_floor,_=candidate(m,only_noise)
                at=np.argmin(abs(FREQ-900));support=(FREQ>830)&(FREQ<970)
                active=np.abs(np.arange(48)*HOP/FS-SIZE/(2*FS))<(.025 if kind in ('sparse_tone','weak_tone') else .010 if kind=='brief_chirp' else .5)
                region2=np.outer(active,support)
                row['joint_signal_region_seed_fraction']=float((jr[region2]>threshold_joint).mean())
                row['joint_pooled_signal_region_seed_fraction']=float((js[region2]>threshold_joint_pool).mean())
                row['joint_floor_bias_at_tone_db']=float(20*np.log10(jf[at]/max(noise_floor[at],1e-280)))
                row['signal_region_seed_fraction']=float((r[region2]>threshold).mean())
                row['signal_region_pooled_seed_fraction']=float((s[region2]>threshold_pool).mean())
                row['floor_bias_at_tone_db']=float(20*np.log10(floor[at]/max(noise_floor[at],1e-280)))
                truth_examples[kind]=(noise_floor,floor,jf)
            trials.append(row)
            if repeat==0:examples[f'{kind}_{cutoff}_{fkind}']=(a,floor,r,s,jf,jr,js)
        record=dict(kind=kind,cutoff_hz=cutoff,filter=fkind,trials=trials)
        records.append(record);print(json.dumps(record),flush=True)
    report['cases']=records
    # Exact global gain equivariance of registration vs local-LPF approximation.
    _,x=make_case(777,None);a=guide(x);a2=guide(3*x)
    report['registered_global_gain_equivariance_relative_error']=float(np.max(abs(a2-3*a))/np.max(3*a))
    # Correlation from native frame lattice, not the interpolated mask lattice.
    a,_,r=example_white;z=r[VALID,interior];z=z-z.mean(axis=0)
    report['lag1_time_correlation']=float(np.sum(z[:-1]*z[1:])/np.sqrt(np.sum(z[:-1]**2)*np.sum(z[1:]**2)))
    report['seconds']=time.monotonic()-started
    (args.out/'results.json').write_text(json.dumps(report,indent=2))
    np.savez_compressed(args.out/'examples.npz',**{key.replace('.','p')+'_'+label:array for key,values in examples.items() for label,array in zip(['magnitude','q20','ratio','pooled','joint_floor','joint_ratio','joint_pooled'],values)})
    plt.style.use('dark_background')
    fig,ax=plt.subplots(2,2,figsize=(14,9),layout='constrained')
    for c in [1200.,3000.,6000.,12000.,None]:
        a,f,r,s,jf,jr,js=examples[f'noise_{c}_butter'];ax[0,0].plot(FREQ,20*np.log10(np.maximum(f,1e-16)),label=f'{c or "full"} Hz')
    ax[0,0].set(title='Blind local q20 follows the filtered noise profile',xlabel='Hz',ylabel='Raw guide amplitude (dB)',xlim=(0,16000),ylim=(-160,-40));ax[0,0].legend(fontsize=8)
    for c in [1200.,3000.,6000.,12000.,None]:
        _,_,r,_,_,_,_=examples[f'noise_{c}_butter'];region=(FREQ>250)&(FREQ<(20000 if c is None else .7*c));v=np.sort(r[EMIT][:,region].ravel());ax[0,1].plot(v,np.arange(1,len(v)+1)/len(v),label=f'{c or "full"} Hz')
    ax[0,1].axvline(threshold,color='white',ls='--',label='White-noise 99% calibration');ax[0,1].set(title='Held-out normalized noise CDFs: interiors only',xlabel='Magnitude / local temporal q20',ylabel='CDF',xlim=(0,8),ylim=(0,1));ax[0,1].legend(fontsize=8)
    for name,(nf,f,jf) in truth_examples.items():
        if name=='persistent_tone':
            ax[1,0].plot(FREQ,20*np.log10(np.maximum(f,1e-16)/np.maximum(nf,1e-16)),label='Temporal floor: persistent tone')
            ax[1,0].plot(FREQ,20*np.log10(np.maximum(jf,1e-16)/np.maximum(nf,1e-16)),label='Guarded joint floor: persistent tone')
    ax[1,0].set(title='Signal contamination of estimated floor',xlabel='Hz',ylabel='q20 bias vs same noise alone (dB)',xlim=(500,1300),ylim=(-5,35));ax[1,0].legend(fontsize=8)
    noise_records=[v for v in records if v['kind']=='noise'];labels=[str(v['cutoff_hz'] or 'full')+(' brick' if v['filter']=='brick' else '') for v in noise_records]
    x=np.arange(len(labels));means=[np.mean([t['seed_fraction'] for t in v['trials']]) for v in noise_records];pm=[np.mean([t['pooled']['seed_fraction'] for t in v['trials']]) for v in noise_records]
    jm=[np.mean([t['joint']['seed_fraction'] for t in v['trials']]) for v in noise_records]
    ax[1,1].bar(x-.22,np.array(means)*100,.22,label='Temporal');ax[1,1].bar(x,np.array(pm)*100,.22,label='Temporal + pooling');ax[1,1].bar(x+.22,np.array(jm)*100,.22,label='Guarded joint');ax[1,1].axhline(1,color='white',ls='--');ax[1,1].set(xticks=x,xticklabels=labels,title='Held-out false seed rates; target 1%',ylabel='Percent',xlabel='Hidden LPF cutoff (Hz)');ax[1,1].legend(fontsize=8)
    fig.savefig(args.out/'analytical-results.png',dpi=150);plt.close(fig)
    if args.demo.exists():
        with np.load(args.demo) as d:field=d['guide'].astype(float);origin=float(d['first_center']+d['guide_center_offset']);rate=float(d['sample_rate'])
        center=int(round((2.7*rate-origin)/512));start=max(0,center-24);data=np.ascontiguousarray(field[start:start+48]);floor,r=candidate(m,data);s=pooled(m,r);jf,jr=joint_candidate(m,data,floor)
        cols=FREQ<=4000;times=(origin+(start+np.arange(48))*512)/rate
        fig,axs=plt.subplots(3,1,figsize=(13,10),layout='constrained');db=20*np.log10(np.maximum(data[:,cols],1e-12)/.500244)
        axs[0].imshow(db.T,origin='lower',aspect='auto',extent=[times[0],times[-1],0,FREQ[cols][-1]],cmap='magma',vmin=-90,vmax=0);axs[0].set(title='Actual registered analysis: Dave and Simon (not a clean reference)',ylabel='Hz')
        axs[1].plot(FREQ[cols],20*np.log10(np.maximum(np.median(data,axis=0)[cols],1e-12)),label='Temporal median');axs[1].plot(FREQ[cols],20*np.log10(np.maximum(floor[cols],1e-12)),label='Local q20 scale');axs[1].plot(FREQ[cols],20*np.log10(np.maximum(jf[cols]*threshold_joint,1e-12)),label='Joint noise-calibrated threshold');axs[1].legend();axs[1].set(ylabel='Raw guide amplitude dB',xlabel='Hz',ylim=(-110,-20))
        axs[2].imshow((jr[:,cols]>threshold_joint).T,origin='lower',aspect='auto',extent=[times[0],times[-1],0,FREQ[cols][-1]],cmap='gray',vmin=0,vmax=1);axs[2].set(title='Candidate event seeds ONLY: not an output spectrum or a denoising result',ylabel='Hz',xlabel='Seconds')
        fig.savefig(args.out/'recording-diagnostic.png',dpi=150);plt.close(fig)
        band_stats=[]
        for low,high in [(250,2800),(3100,4000),(4000,24001)]:
            band=(FREQ>=low)&(FREQ<high)
            band_stats.append(dict(low_hz=low,high_hz=high,selected_fraction=float((jr[:,band]>threshold_joint).mean()),mean_squared_guide_magnitude=float(np.mean(data[:,band]**2))))
        report['recording_diagnostic_bands']=band_stats
        (args.out/'results.json').write_text(json.dumps(report,indent=2))
    guide.close();print(json.dumps({'report':str(args.out/'results.json'),'seconds':report['seconds']}),flush=True)
if __name__=='__main__':main()
