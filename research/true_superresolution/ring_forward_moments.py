"""Forward-spectrum ring moments: isolated research, never a live replacement.

The moment identity is exact. Replacing an argmax by a circular centroid is an
estimator change, explicitly measured here rather than hidden behind parity.
"""
import argparse
import json
from pathlib import Path
import sys
import time
import wave
import numpy as np
from numba import njit

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from research.true_superresolution.oracle.circles import prepare_phase_circle_spectrum, _phase_peak
from research.true_superresolution.oracle.uncertainty_fusion import phase_lattice_observations_at_centers, _perceptual_registration_view

def plan(n):
    f=np.fft.fftfreq(n);radius=np.hypot(f[:,None],f[None,:])
    centers=np.linspace(.035,.46,7)
    rings=np.exp(-.5*((radius[...,None]-centers)/.055)**2)
    # The same U(k)*conj(U(k-q)) supplies every ring. These products of masks
    # are fixed at construction; the dynamic pass has no exponential calls.
    weights=np.stack([rings*np.roll(rings,1,axis=1),rings*np.roll(rings,1,axis=0)],axis=-1)
    return rings,np.ascontiguousarray(weights)

@njit
def forward_moments(unit,joint,rings,weights):
    n=unit.shape[0]
    moments=np.zeros((7,2),np.complex128)
    bound=np.zeros((7,2));energy=np.zeros(7)
    for y in range(n):
        for x in range(n):
            u=unit[y,x]
            px=u*np.conj(unit[y,(x-1)%n]);py=u*np.conj(unit[(y-1)%n,x])
            ax=abs(px);ay=abs(py)
            for ring in range(7):
                wx=weights[y,x,ring,0];wy=weights[y,x,ring,1]
                moments[ring,0]+=px*wx;moments[ring,1]+=py*wy
                bound[ring,0]+=ax*wx;bound[ring,1]+=ay*wy
                energy[ring]+=joint[y,x]*rings[y,x,ring]
    return moments,bound,energy

def estimate(unit,joint,rings,weights):
    moments,bound,energy=forward_moments(unit,joint,rings,weights)
    vector=-np.angle(moments)*unit.shape[0]/(2*np.pi)
    coherence=np.divide(abs(moments),bound,out=np.zeros_like(bound),where=bound>1e-30)
    # Research reliability, NOT the legacy competing-peak reliability.
    confidence=np.minimum(coherence[:,0],coherence[:,1])**2
    weight=energy*np.maximum(confidence,1e-4)
    weight/=max(weight.sum(),1e-30)
    mean=np.sum(weight[:,None]*vector,axis=0)
    dispersion=np.sqrt(np.sum(weight*np.sum((vector-mean)**2,axis=1)))
    return mean,dict(rings=vector,coherence=coherence,dispersion=dispersion,moments=moments)

def inverse_reference(unit,joint,rings):
    fields=np.fft.ifft2(np.moveaxis(rings,2,0)*unit[None],axes=(-2,-1)).real
    vectors=[];weights=[];ambiguities=[]
    for ring,field in enumerate(fields):
        vector,peak,competitor=_phase_peak(field)
        ambiguity=np.clip(competitor/max(peak,1e-12),0.,1.)
        vectors.append(vector);ambiguities.append(ambiguity)
        weights.append(np.sum(rings[...,ring]*joint)*max((1-ambiguity)**2,1e-4))
    weights=np.asarray(weights);weights/=max(weights.sum(),1e-30)
    return np.sum(weights[:,None]*vectors,axis=0),fields,np.array(ambiguities)

def spectra(a,b):
    sa=prepare_phase_circle_spectrum(a).spectrum;sb=prepare_phase_circle_spectrum(b).spectrum
    cross=sb*np.conj(sa)
    return cross/np.maximum(abs(cross),1e-12),np.sqrt(abs(sa)*abs(sb))

def summary(values):
    return dict(median=float(np.median(values)),p95=float(np.quantile(values,.95)),maximum=float(np.max(values)))

@njit
def selected_ring_values(unit,rings,phase,points):
    """Exact original ring correlations at selected integer displacement pixels."""
    n=unit.shape[0];result=np.zeros((len(points),7))
    for p in range(len(points)):
        dy,dx=points[p]
        for y in range(n):
            for x in range(n):
                # Each Fourier atom is shared across all seven radial weights.
                value=(unit[y,x]*phase[dy,y]*phase[dx,x]).real
                for ring in range(7):result[p,ring]+=value*rings[y,x,ring]
    return result/(n*n)

def candidate_probe(unit,rings,info,fields,phase):
    n=unit.shape[0];negative=(-np.arange(n))%n
    # One forward transform gives the common unweighted correlation, with
    # reflected output coordinates and inverse normalization.
    common=np.fft.fft2(unit).real[np.ix_(negative,negative)]/(n*n)
    candidates=set(int(p) for p in np.argpartition(common.ravel(),-8)[-8:])
    for dx,dy in info['rings']:
        for y in (int(np.floor(dy)),int(np.ceil(dy))):
            for x in (int(np.floor(dx)),int(np.ceil(dx))):candidates.add((y%n)*n+x%n)
    points=np.array([divmod(p,n) for p in sorted(candidates)],dtype=np.int64)
    values=selected_ring_values(unit,rings,phase,points)
    expected=fields[:,points[:,0],points[:,1]].T
    error=float(np.max(abs(values-expected)))
    assert error<1e-12,error
    peaks=competitors=0
    for field in fields:
        best=int(np.argmax(field));py,px=divmod(best,n)
        peaks+=best in candidates
        excluded=field.copy()
        for dy in range(-3,4):
            for dx in range(-3,4):excluded[(py+dy)%n,(px+dx)%n]=-np.inf
        competitors+=int(np.argmax(excluded)) in candidates
    return dict(points=len(points),peak_hits=int(peaks),competitor_hits=int(competitors),max_exact_query_error=error)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--report',type=Path,default=ROOT/'build/ring-forward-moments.json')
    args=p.parse_args();rng=np.random.default_rng(20260916)
    proof=[]
    for n in (13,24,48):
        rings,weights=plan(n);worst=0.
        for _ in range(8):
            unit,joint=spectra(rng.normal(size=(n,n)),rng.normal(size=(n,n)))
            moments,_,_=forward_moments(unit,joint,rings,weights)
            fields=np.fft.ifft2(np.moveaxis(rings,2,0)*unit[None],axes=(-2,-1)).real
            # F(|c|^2)[q] = sum S[k] conj(S[k-q]) / n^2.
            characteristic=np.fft.fft2(fields**2,axes=(-2,-1))
            expected=characteristic[:,[0,1],[1,0]]
            worst=max(worst,float(np.max(abs(moments/n**2-expected))))
        assert worst<1e-12,(n,worst)
        proof.append(dict(extent=n,max_moment_identity_error=worst))
    n=48;rings,weights=plan(n);f=np.fft.fftfreq(n)
    ideal=[]
    for dx,dy in ((0,0),(1,-2),(-7,9),(23,-23)):
        unit=np.exp(-2j*np.pi*(f[None,:]*dx+f[:,None]*dy))
        mean,info=estimate(unit,np.ones((n,n)),rings,weights)
        error=float(np.max(abs(info['rings']-np.array([dx,dy]))))
        assert error<1e-10,error
        ideal.append(dict(shift_xy=[dx,dy],max_ring_error=error))
    singular=np.linalg.svd(rings.reshape(-1,7),compute_uv=False)
    stress=[]
    yy,xx=np.mgrid[:n,:n]
    texture=rng.normal(size=(n,n));periodic=np.cos(2*np.pi*xx/6)+.4*np.cos(2*np.pi*yy/8)
    for label,a,b in (
        ('windowed_translation',texture,np.roll(texture,(2,-3),(0,1))),
        ('periodic_ambiguous',periodic,np.roll(periodic,(2,-3),(0,1))),
        ('incoherent',texture,rng.normal(size=(n,n))),
        ('two_displacements',texture,np.roll(texture,(1,2),(0,1))+.9*np.roll(texture,(-7,-8),(0,1))),
        ('silence',np.zeros((n,n)),np.zeros((n,n))),
    ):
        unit,joint=spectra(a,b);old,_,amb=inverse_reference(unit,joint,rings);new,info=estimate(unit,joint,rings,weights)
        stress.append(dict(case=label,old_xy=old.tolist(),moment_xy=new.tolist(),difference=float(np.linalg.norm(old-new)),
            minimum_axis_coherence=info['coherence'].min(axis=1).tolist(),old_ambiguity=amb.tolist()))
    with wave.open(str(ROOT/'research/true_superresolution/daveandsimon-first-second.wav')) as wav:
        audio=np.frombuffer(wav.readframes(wav.getnframes()),'<i2').astype(float)/32768.
    offsets=(-379,-241,-64,0,83,214,427)
    observations=phase_lattice_observations_at_centers(audio[:24576],np.arange(48)*512,crop_rows=512,offsets=offsets)
    ref_q=max(np.quantile(observations[3],.995),1e-30)
    views=[_perceptual_registration_view(o*ref_q/max(np.quantile(o,.995),1e-30)) for o in observations]
    padded=[np.pad(v,((24,24),(24,24)),mode='reflect') for v in views]
    records=[];bench=[];candidate_records=[]
    phase=np.exp(2j*np.pi*np.arange(n)[:,None]*np.arange(n)[None,:]/n)
    for offset in (0,1,2,4,5,6):
        for y in (8,40,88,152,248,344,440,488):
            for x in (8,24,40):
                unit,joint=spectra(padded[3][y:y+48,x:x+48],padded[offset][y:y+48,x:x+48])
                old,fields,amb=inverse_reference(unit,joint,rings);new,info=estimate(unit,joint,rings,weights)
                candidate_records.append(candidate_probe(unit,rings,info,fields,phase))
                records.append(dict(offset=offsets[offset],chart_yx=[y,x],old_xy=old.tolist(),moment_xy=new.tolist(),
                    difference=float(np.linalg.norm(old-new)),coherence_mean=float(np.mean(info['coherence'])),
                    old_ambiguity_mean=float(np.mean(amb))))
                bench.append((unit,joint))
    times={}
    for label,method in (('seven_numpy_inverse_with_peak_search',lambda u,j:inverse_reference(u,j,rings)),
                         ('numba_shared_moments_only',lambda u,j:forward_moments(u,j,rings,weights))):
        samples=[]
        for repeat in range(5):
            start=time.perf_counter()
            for unit,joint in bench:method(unit,joint)
            samples.append((time.perf_counter()-start)/len(bench)*1e6)
        times[label]=float(np.median(samples))
    report=dict(identity=proof,ideal_translations=ideal,ring_mask_singular_values=singular.tolist(),
        ring_mask_relative_singular_values=(singular/singular[0]).tolist(),stress_cases=stress,
        speech_chart_difference_pixels=summary([r['difference'] for r in records]),speech_charts=records,
        one_forward_candidate_probe=dict(charts=len(candidate_records),
            average_queries=float(np.mean([r['points'] for r in candidate_records])),
            ring_peak_capture_fraction=sum(r['peak_hits'] for r in candidate_records)/(7*len(candidate_records)),
            ring_competitor_capture_fraction=sum(r['competitor_hits'] for r in candidate_records)/(7*len(candidate_records)),
            max_direct_query_error=max(r['max_exact_query_error'] for r in candidate_records)),
        prototype_microseconds_per_patch=times,
        timing_caveat='Numba moment accumulator vs NumPy transforms plus Python peak searches; not a native bfft speedup claim.',
        promotion='Research only: exact correlation-power moments do not preserve legacy maxima/competitors.')
    args.report.parent.mkdir(parents=True,exist_ok=True);args.report.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='speech_charts'},indent=2))

if __name__=='__main__':main()
