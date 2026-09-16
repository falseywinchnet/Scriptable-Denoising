"""Live guide parity, full recursive mask, fixed admission, and native stream checks."""
import ctypes as C
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest
import numpy as np
from scipy.io import wavfile
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'runtime'),str(ROOT/'tools')]
from compile_filter import bind_native_dsp,compile_candidate,release
from native_client import Engine
from research.true_superresolution.oracle.uncertainty_fusion import phase_lattice_observations_at_centers,registered_perceptual_maximum
from research.true_superresolution.bridge import project_decisions
LIB=os.environ.get('CLEANUP_TEST_LIBRARY',str(ROOT/'build/libCleanupNative.dylib'))
OFFSETS=(-379,-241,-64,0,83,214,427)
PTR=C.POINTER(C.c_double)
REPORT={}

def ptr(x):return x.ctypes.data_as(PTR)

def library():
    lib=C.CDLL(LIB)
    lib.cleanup_guide_create.argtypes=[C.c_size_t]*4+[C.c_long]
    lib.cleanup_guide_create.restype=C.c_void_p
    lib.cleanup_guide_destroy.argtypes=[C.c_void_p]
    lib.cleanup_guide_run.argtypes=[C.c_void_p,PTR,C.c_size_t,PTR]
    lib.cleanup_guide_run_stream.argtypes=[C.c_void_p,PTR,C.c_size_t,C.c_size_t,PTR]
    lib.cleanup_guide_reused_frames.argtypes=[C.c_void_p]
    lib.cleanup_guide_reused_frames.restype=C.c_size_t
    lib.cleanup_guide_register.argtypes=[C.c_void_p,PTR,PTR]
    return lib

def module(transform='rfft'):
    source=(ROOT/'Filter.py').read_text().replace('CLEANUP_POPULATION = "rolloff"','CLEANUP_POPULATION = "receiver"').replace('GUIDE_REGISTRATION = False','GUIDE_REGISTRATION = True').replace('GUIDE_BINS = GUIDE_FFT','GUIDE_BINS = 512',1).replace('ANALYSIS_MODE = "baseline"','ANALYSIS_MODE = "registered"').replace('CLEANUP_POPULATION = "rolloff"','CLEANUP_POPULATION = "receiver"')
    if transform=='odft':source=source.replace('BINS = FFT_SIZE // 2 + 1','TRANSFORM = "odft"\nBINS = FFT_SIZE // 2')
    spec=importlib.util.spec_from_file_location('guide_'+transform,ROOT/'Filter.py')
    m=importlib.util.module_from_spec(spec);bind_native_dsp(m);exec(compile(source,str(ROOT/'Filter.py'),'exec'),m.__dict__)
    return m

class GuideTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.lib=library()

    def test_shared_reference_and_half_spectrum_odd_even_charts(self):
        # Odd chart sizes have no Nyquist endpoint; their last retained column
        # needs multiplicity two. Also exercise non-divisible chart job counts.
        rng=np.random.default_rng(9816)
        for rows,cols in ((13,13),(64,24),(64,48)):
            observations=np.ascontiguousarray(rng.uniform(.001,.1,(7,rows,cols)))
            expected=registered_perceptual_maximum(observations,offsets=OFFSETS).fused.T
            plan=self.lib.cleanup_guide_create(64,rows,cols,16,0)
            self.assertTrue(plan)
            try:
                actual=np.empty((cols,rows))
                self.assertEqual(self.lib.cleanup_guide_register(plan,ptr(observations),ptr(actual)),0)
                np.testing.assert_allclose(actual,expected,atol=2e-10,rtol=2e-7)
            finally:self.lib.cleanup_guide_destroy(plan)

    def test_unregistered_central_observation_and_cache(self):
        lib=self.lib
        lib.cleanup_guide_create_mode.argtypes=[C.c_size_t]*4+[C.c_long,C.c_int]
        lib.cleanup_guide_create_mode.restype=C.c_void_p
        lib.cleanup_guide_lane_count.argtypes=[C.c_void_p]
        lib.cleanup_guide_lane_count.restype=C.c_size_t
        lib.cleanup_guide_timings.argtypes=[C.c_void_p,C.POINTER(C.c_uint64)]
        cached=lib.cleanup_guide_create_mode(2048,512,48,512,0,0)
        fresh=lib.cleanup_guide_create_mode(2048,512,48,512,0,0)
        self.assertTrue(cached);self.assertTrue(fresh)
        self.assertFalse(lib.cleanup_guide_create_mode(2048,512,48,512,0,2))
        rng=np.random.default_rng(446);audio=rng.normal(0,.02,49152)
        actual=np.empty((48,512));expected=np.empty_like(actual)
        try:
            self.assertEqual(lib.cleanup_guide_lane_count(cached),1)
            for i in range(4):
                x=np.ascontiguousarray(audio[i*8192:i*8192+24576]) if i<3 else np.zeros(24576)
                oracle=phase_lattice_observations_at_centers(x,np.arange(48)*512,crop_rows=512,offsets=(0,))[0].T
                self.assertEqual(lib.cleanup_guide_run_stream(cached,ptr(x),len(x),8192,ptr(actual)),0)
                self.assertEqual(lib.cleanup_guide_run(fresh,ptr(x),len(x),ptr(expected)),0)
                np.testing.assert_array_equal(actual,expected)
                np.testing.assert_allclose(actual,oracle,atol=2e-12,rtol=2e-10)
                self.assertEqual(lib.cleanup_guide_reused_frames(cached)>0,i in (1,2))
                timing=(C.c_uint64*3)();lib.cleanup_guide_timings(cached,timing)
                self.assertEqual(list(timing)[1:],[0,0])
        finally:
            lib.cleanup_guide_destroy(cached);lib.cleanup_guide_destroy(fresh)

    def test_reference_units_and_temporal_kernel_support(self):
        m=module()
        # Independent literal double inverse on a constant frame, compared to
        # the actual reference Hann FFT. Exercises irfft endpoint weighting.
        window=.5-.5*np.cos(2*np.pi*np.arange(m.GUIDE_FFT)/m.GUIDE_FFT)
        recovered=np.fft.irfft(np.fft.rfft(window))
        guide=np.fft.irfft(recovered)
        reference=abs(np.fft.rfft(np.hanning(512))[0])
        self.assertAlmostEqual(guide[0]*m.GUIDE_MAGNITUDE_SCALE,reference,places=10)
        # Three recursive13 rounds must have the baseline time support even
        # though native observations are every512 samples. No extra delay.
        mask=np.zeros((m.MASK_FRAMES,m.GUIDE_BINS));center=mask.shape[0]//2
        mask[center,30]=1
        v=np.zeros((m.GUIDE_PAD_T,m.GUIDE_PAD_B));h=v.copy()
        scratch=np.zeros(max(v.shape))
        self.assertEqual(m.recursive_smooth(mask,v,h,100,scratch),0)
        support=np.flatnonzero(mask[:,30]>1e-14)
        self.assertEqual((support[0]-center,support[-1]-center),(-18,18))
        self.assertEqual(18*m.MASK_HOP,18*128)
        np.testing.assert_allclose(mask[center-18:center,30],mask[center+1:center+19,30][::-1],atol=1e-14)

    def test_00_cached_observations_match_fresh_context(self):
        rng=np.random.default_rng(712)
        signal=rng.normal(0,.02,65536)
        cached=self.lib.cleanup_guide_create(2048,512,48,512,0)
        fresh=self.lib.cleanup_guide_create(2048,512,48,512,0)
        self.assertTrue(cached);self.assertTrue(fresh)
        actual=np.zeros((48,512));expected=actual.copy()
        try:
            for step in range(4):
                audio=np.ascontiguousarray(signal[step*8192:step*8192+24576])
                self.assertEqual(self.lib.cleanup_guide_run_stream(cached,ptr(audio),len(audio),8192,ptr(actual)),0)
                self.assertEqual(self.lib.cleanup_guide_run(fresh,ptr(audio),len(audio),ptr(expected)),0)
                np.testing.assert_array_equal(actual,expected)
                self.assertEqual(self.lib.cleanup_guide_reused_frames(cached)>0,step>0)
            # A discontinuity defeats byte equality and forces all observations.
            audio=rng.normal(0,.02,24576)
            self.assertEqual(self.lib.cleanup_guide_run_stream(cached,ptr(audio),len(audio),8192,ptr(actual)),0)
            self.assertEqual(self.lib.cleanup_guide_reused_frames(cached),0)
            self.assertEqual(self.lib.cleanup_guide_run(fresh,ptr(audio),len(audio),ptr(expected)),0)
            np.testing.assert_array_equal(actual,expected)
        finally:
            self.lib.cleanup_guide_destroy(cached);self.lib.cleanup_guide_destroy(fresh)

    def test_01_exact_registered_oracle(self):
        rng=np.random.default_rng(781)
        fs,pcm=wavfile.read(ROOT/'research/true_superresolution/daveandsimon-first-second.wav')
        audio=pcm.astype(np.float64)/32768.
        inputs={'silence':np.zeros(24576),'noise':rng.normal(0,.1,24576),'tones':.12*np.sin(2*np.pi*711*np.arange(24576)/48000)+.03*np.cos(2*np.pi*1234*np.arange(24576)/48000),'radio':audio[:24576]}
        rows,cols=512,48
        plan=self.lib.cleanup_guide_create(2048,rows,cols,512,0);self.assertTrue(plan)
        records=[];saved=dict(aperture=np.array(2048),rows=np.array(rows),columns=np.array(cols),hop=np.array(512))
        try:
            for name,x in inputs.items():
                observations=phase_lattice_observations_at_centers(x,np.arange(cols)*512,crop_rows=rows,offsets=OFFSETS)
                expected=registered_perceptual_maximum(observations,OFFSETS).fused.T
                saved['audio_'+name]=x;saved['expected_'+name]=expected
                actual=np.empty((cols,rows))
                start=time.perf_counter();self.assertEqual(self.lib.cleanup_guide_run(plan,ptr(x),len(x),ptr(actual)),0);elapsed=time.perf_counter()-start
                error=float(np.max(abs(expected-actual)))
                records.append(dict(signal=name,max_absolute_error=error,seconds=elapsed))
                np.testing.assert_allclose(actual,expected,atol=2e-10,rtol=2e-7)
                direct=np.empty_like(actual)
                self.assertEqual(self.lib.cleanup_guide_register(plan,ptr(np.ascontiguousarray(observations)),ptr(direct)),0)
                np.testing.assert_allclose(direct,expected,atol=2e-10,rtol=2e-7)
        finally:self.lib.cleanup_guide_destroy(plan)
        np.savez_compressed(ROOT/'build/registered-oracle.npz',**saved)
        REPORT['oracle_parity']=records;print(records)

    def test_02_complete_cleanup_and_projection(self):
        m=module();rng=np.random.default_rng(42)
        state=np.zeros((m.MAX_ENTITIES,m.ENTITY_STORAGE))
        guide=rng.uniform(1e-6,1e-5,(m.GUIDE_FRAMES,m.GUIDE_BINS))
        guide[:,35]+=.25;guide[14:34,80]+=.1
        state[2,:guide.size]=guide.ravel()
        ref=state[3,64:64+m.REFERENCE_PLANE].reshape(m.REFERENCE_FRAMES,m.REFERENCE_BINS)
        ref[:]=rng.uniform(1e-5,1e-4,ref.shape);ref[:,5]+=.8
        analysis=rng.normal(0,.1,(m.FRAMES,m.BINS,2));content=analysis.copy()
        cfg=np.array([48000.,3400.,0.,0.,0.])
        self.assertEqual(m.Filter(analysis,content,state,cfg),m.PROCESS)
        nb=min(m.GUIDE_BINS,max(4,int(np.floor(3400/(48000/4094)+1.5))))
        # Independent orchestration of ALL original Cleanup stages on the guide.
        scaled=np.stack([np.interp(np.arange(m.MASK_FRAMES)*m.MASK_HOP,np.arange(m.GUIDE_FRAMES)*m.GUIDE_HOP,guide[:,b]) for b in range(m.GUIDE_BINS)],axis=1)*m.GUIDE_MAGNITUDE_SCALE
        mag=np.zeros_like(scaled);mag[:,:nb]=scaled[:,:nb]
        raw=np.zeros(m.MASK_FRAMES);smooth=raw.copy();det=raw.copy();log1=np.zeros(m.GUIDE_BINS);log3=np.zeros(3*m.GUIDE_BINS);scratch=np.zeros(max(3*m.GUIDE_BINS,m.GUIDE_PAD_T,m.GUIDE_PAD_B));values=np.zeros(mag.size);deviations=values.copy()
        _,maximum,maximum3=m.guide_entropy(mag,nb,raw,smooth,det,log1,log3,scratch,bool(state[0,1]), np.zeros(192))
        initial=mag.max();smoothed=mag.copy();mask=np.zeros_like(mag)
        m.sawtooth(mag,smoothed,nb,scratch);man,atd=m.statistics(smoothed,nb,values,deviations)
        m.peaks(smoothed,mask,nb,raw,det,maximum,False,man,atd,values,deviations)
        mag[mask==0]=0.;multiplier=min(1.,mag.max()/initial)
        man,atd=m.statistics(mag,nb,values,deviations);m.sawtooth(mag,smoothed,nb,scratch)
        m.peaks(smoothed,smoothed,nb,raw,det,maximum,False,man,atd,values,deviations)
        mask[:,:nb]=np.minimum(mask[:,:nb],smoothed[:,:nb]*multiplier)
        m.sawtooth(mask,mask,nb,scratch)
        vert=np.zeros((m.GUIDE_PAD_T,m.GUIDE_PAD_B));horiz=vert.copy();m.recursive_smooth(mask,vert,horiz,nb,scratch)
        mag[:,:nb]=scaled[:,:nb];man,atd=m.statistics(mag,nb,values,deviations);man/=mag.max();atd/=mag.max()
        for t in range(m.MASK_FRAMES):
            if det[t]==0:mask[t,:nb]=np.maximum(mask[t,:nb],man*smooth[t]/maximum3)
        mask[:,nb:]=0.;np.clip(mask,0,1,out=mask)
        actual=state[0,64+2*m.MASK_PLANE:64+3*m.MASK_PLANE].reshape(mask.shape)
        np.testing.assert_allclose(actual,mask,atol=1e-14,rtol=1e-13)
        REPORT['mask_max_error']=float(np.max(abs(actual-mask)))
        expected=project_decisions(mask.T,np.arange(m.GUIDE_BINS)*48000/4094,np.arange(m.MASK_FRAMES)*m.MASK_HOP/48000,np.arange(m.BINS)*48000/512,np.arange(m.FIRST,m.FIRST+m.EMIT)*128/48000)
        synth_nb=int(np.floor(3400/(48000/512)+1.5));expected[:,synth_nb:]=0.
        np.testing.assert_allclose(content[m.FIRST:m.FIRST+m.EMIT],analysis[m.FIRST:m.FIRST+m.EMIT]*expected[:,:,None],atol=2e-15)
        self.assertGreater(np.max(expected),0);self.assertLessEqual(np.max(expected),1)
        original=content.copy()
        # Ordinary analysis is NOT the guide used by the mask. Hold content,
        # guide and reference fixed while changing analysis arbitrarily.
        content[:]=analysis;m.Filter(analysis*200,content,state,cfg)
        np.testing.assert_array_equal(content,original)
        state[2,:guide.size]=np.roll(guide,33,axis=1).ravel();content[:]=analysis;m.Filter(analysis,content,state,cfg)
        self.assertGreater(np.max(abs(content-original)),1e-8)
        np.testing.assert_array_equal(state[2,:guide.size],np.roll(guide,33,axis=1).ravel())
        self.assertEqual((state[0,9],state[0,11]),(129,192))
        self.assertEqual((state[0,13],state[0,15]),(129,192))
        cfg[1]=7000.;self.assertEqual(m.Filter(analysis,content,state,cfg),m.ERROR)

    def test_03_gain_centers_rfft_odft_and_identity(self):
        for mode in ('rfft','odft'):
            m=module(mode);half=.5 if mode=='odft' else 0.
            content=np.ones((m.FRAMES,m.BINS,2));gain=np.ones((m.MASK_FRAMES,m.GUIDE_BINS));m.apply_guide_gain(gain,content,m.BINS,half)
            np.testing.assert_array_equal(content,1.)
            gain[:]=.2+np.arange(m.MASK_FRAMES)[:,None]*.001+np.arange(m.GUIDE_BINS)[None,:]*.0001
            content[:]=1.;m.apply_guide_gain(gain,content,37,half)
            for t in (m.FIRST,m.FIRST+31,m.FIRST+m.EMIT-1):
                expected=.2+(t*128/m.MASK_HOP)*.001+(np.arange(37)+half)*4094/512*.0001
                np.testing.assert_allclose(content[t,:37,0],expected,atol=2e-16)
                np.testing.assert_array_equal(content[t,37:],0.)

    def test_04_native_stream_viewer_reload(self):
        from viewer import snapshot
        source=(ROOT/'Filter.py').read_text().replace('CLEANUP_POPULATION = "rolloff"','CLEANUP_POPULATION = "receiver"').replace('GUIDE_REGISTRATION = False','GUIDE_REGISTRATION = True').replace('GUIDE_BINS = GUIDE_FFT','GUIDE_BINS = 512',1).replace('ANALYSIS_MODE = "baseline"','ANALYSIS_MODE = "registered"').replace('VIEWER = False','VIEWER = True').replace('VIEWER_AUTOSTART = True','VIEWER_AUTOSTART = False')
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'Filter.py';path.write_text(source)
            with Engine(LIB,path,channels=1) as e:
                status=e.wait();self.assertTrue(status['success'],status);self.assertEqual(status['analysis_mode'],'registered')
                timings=[];x=np.random.default_rng(31).normal(0,.01,(8192,1)).astype('float32');x[:,0]+=.1*np.sin(2*np.pi*700*np.arange(8192)/48000)
                for _ in range(5):
                    start=time.perf_counter();y,code=e.process(x);timings.append(time.perf_counter()-start);self.assertEqual(code,0);self.assertTrue(np.isfinite(y).all())
                d=snapshot(status['port']);self.assertEqual(d['guide_bins'],512);self.assertGreater(max(d['guide']),0)
                before=e.status();path.write_text(source.replace('GUIDE_SLOT = 2','GUIDE_SLOT = 3'));e.reload();bad=e.wait();self.assertFalse(bad['success']);self.assertEqual(bad['generation'],before['generation'])
                path.write_text(source.replace('GUIDE_BINS = 512','GUIDE_BINS = 8'));e.reload();bad=e.wait();self.assertFalse(bad['success']);self.assertEqual(bad['generation'],before['generation'])
                path.write_text(source.replace('GUIDE_BINS = 512','GUIDE_BINS = 384'));e.reload();good=e.wait();self.assertTrue(good['success'],good);self.assertEqual(good['generation'],before['generation']+1)
                e.process(np.zeros_like(x));d=snapshot(good['port']);self.assertEqual(d['guide_bins'],384);np.testing.assert_array_equal(d['guide'],0.)
                self.assertEqual(e.status()['faults'],0)
                REPORT['stream']=dict(block_seconds=8192/48000,timings=timings,peak_seconds=max(timings),latency_samples=status['latency_samples'],faults=e.status()['faults'])
                e.lib.cleanup_set_bandwidth(e.handle,16000.)
                _,code=e.process(x);self.assertEqual(code,-2)
                e.lib.cleanup_set_bandwidth(e.handle,3400.)
                fault=e.status();self.assertTrue(fault['runtime_fault'])
                self.assertIn('16000 Hz',fault['error']);self.assertIn('guide covers',fault['error'])
                e.reload();restored=e.wait();self.assertTrue(restored['success'],restored)
                self.assertFalse(restored['runtime_fault']);self.assertEqual(restored['error'],'')

    def test_05_stereo_deadlines(self):
        source=(ROOT/'Filter.py').read_text().replace('CLEANUP_POPULATION = "rolloff"','CLEANUP_POPULATION = "receiver"').replace('GUIDE_REGISTRATION = False','GUIDE_REGISTRATION = True').replace('GUIDE_BINS = GUIDE_FFT','GUIDE_BINS = 512',1).replace('ANALYSIS_MODE = "baseline"','ANALYSIS_MODE = "registered"').replace('VIEWER = True','VIEWER = False')
        records=[]
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'Filter.py';path.write_text(source)
            for identical in (True,False):
                with Engine(LIB,path,channels=2) as e:
                    status=e.wait();self.assertTrue(status['success'],status)
                    rng=np.random.default_rng(578);x=rng.normal(0,.03,(8192,2)).astype('float32')
                    if identical:x[:,1]=x[:,0]
                    timings=[]
                    for _ in range(6):
                        start=time.perf_counter();y,code=e.process(x);timings.append(time.perf_counter()-start)
                        self.assertEqual(code,0);self.assertTrue(np.isfinite(y).all())
                    records.append(dict(identical=identical,seconds=timings,peak_seconds=max(timings),native_peak_ns=e.status()['peak_block_ns'],faults=e.status()['faults']))
        REPORT['stereo']=records

    @classmethod
    def tearDownClass(cls):
        (ROOT/'build/registered-guide-results.json').write_text(json.dumps(REPORT,indent=2)+'\n')

if __name__=='__main__':unittest.main(verbosity=2)
