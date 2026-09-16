"""Reproduce the sibling magnitude field and exercise the decision bridge."""
from pathlib import Path
import json
import time
import numpy as np
from scipy.io import wavfile
from .oracle.fourier_unrolled_densified import fourier_unrolled_densified, reassigned_texture_baseline, DEFAULT_OFFSETS
from .oracle.uncertainty_fusion import centered_stft_lattice, double_irfft_low_rows, phase_lattice_observations, registered_perceptual_maximum
from .oracle.cleanup_shark_vad import analyze_cleanup_shark
from .bridge import guide_gain, project_decisions, odft_frequencies, unrolled_frequencies
from .native_adapter import native_observations

def main():
    root=Path(__file__).resolve().parents[2]
    out=root/'build/true-superresolution';out.mkdir(parents=True,exist_ok=True)
    fs,pcm=wavfile.read(Path(__file__).with_name('daveandsimon-first-second.wav'))
    samples=pcm.astype(np.float64)/32768.
    start=time.perf_counter()
    fusion=fourier_unrolled_densified(samples)
    baseline=reassigned_texture_baseline(fusion.fused,fusion.support_union)
    field_time=time.perf_counter()-start
    start=time.perf_counter()
    observations=phase_lattice_observations(samples)
    registered=registered_perceptual_maximum(observations,DEFAULT_OFFSETS)
    registered_time=time.perf_counter()-start
    np.testing.assert_allclose(registered.fused,fusion.aperture_fields[1],atol=1e-12,rtol=1e-10)
    start=time.perf_counter()
    native=native_observations(samples,fusion.centers,DEFAULT_OFFSETS)
    native_core_time=time.perf_counter()-start
    np.testing.assert_allclose(native,observations,atol=2e-12,rtol=2e-10)
    native_registered=registered_perceptual_maximum(native,DEFAULT_OFFSETS)
    native_registered_time=time.perf_counter()-start
    field_error=float(np.max(np.abs(native_registered.fused-registered.fused)))
    np.testing.assert_allclose(native_registered.fused,registered.fused,atol=2e-10,rtol=2e-8)
    start=time.perf_counter();vad=analyze_cleanup_shark(samples,fs);vad_time=time.perf_counter()-start
    ordinary=centered_stft_lattice(samples,n_fft=512,hop_length=512)
    spectrum=centered_stft_lattice(samples,n_fft=2048,hop_length=512)
    recovered=np.fft.irfft(spectrum,n=2048,axis=0)
    cosine=np.fft.irfft(recovered,axis=0)
    quadrature=np.fft.irfft(-1j*recovered,axis=0)
    envelope=np.hypot(cosine,quadrature)[:256]
    original=double_irfft_low_rows(spectrum,n_fft=2048,crop_rows=256)
    # User selected the final field as the reference, with registered sub-hop
    # magnitude as the sufficient/cheaper candidate for the first integration.
    gain=guide_gain(registered.fused)
    full_gain=guide_gain(baseline.merged)
    target_seconds=np.arange(1+len(samples)//128)*128/fs
    target_hz=odft_frequencies(512,fs)
    mapped=project_decisions(gain,unrolled_frequencies(2048,256,fs),fusion.centers/fs,
                             target_hz,target_seconds)
    mapped_full=project_decisions(full_gain,unrolled_frequencies(2048,256,fs),fusion.centers/fs,
                                  target_hz,target_seconds)
    np.savez_compressed(out/'probe.npz',samples=samples,sample_rate=fs,ordinary=ordinary,
                        original=original,envelope=envelope,phase_fused=fusion.aperture_fields[1],
                        multiscale=fusion.fused,trace_field=baseline.merged,
                        guide_gain=gain,mapped_odft_gain=mapped,
                        full_guide_gain=full_gain,mapped_full_odft_gain=mapped_full,
                        vad_state=vad.state,speech_mask=vad.speech_mask,
                        voiced_score=vad.fused_score,unvoiced_score=vad.unvoiced_score,
                        target_seconds=target_seconds,target_hz=target_hz)
    report=dict(source='daveandsimon.wav first second',sample_rate=int(fs),
                apertures=list(fusion.apertures),offsets=list(fusion.phase_fusions[0].offsets),
                field_shape=list(baseline.merged.shape),field_seconds=field_time,vad_seconds=vad_time,
                selected_candidate='registered_subhop',comparison_reference='full_enhanced_field',
                registered_subhop_seconds=registered_time,native_subhop_core_seconds=native_core_time,
                native_core_plus_python_registration_seconds=native_registered_time,
                native_registered_max_absolute_error=field_error,
                mapped_shape=list(mapped.shape),guidance_highest_hz=float(255*fs/4094),
                speech_intervals=[dict(start=v.seconds0,end=v.seconds1) for v in vad.intervals],
                stage='offline extraction and provisional decision projection, not live filtering')
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
