"""Executable derivation of an exact shared sliding-window observation bank.

Research only: this uses dense exponential sums to verify the recurrence, not
to propose that NumPy matrix products belong on the live audio path. It retains
the 2*(N-1) grid, periodic N-point Hann and half-weighted inverse endpoints.
"""
import json
from pathlib import Path
import time
import numpy as np


def trial(n, rng):
    rows=n
    starts=np.array([0,21,50,64,83,133,214,271,427,448,512])
    x=rng.normal(0,.05,n+int(starts[-1]))
    theta=np.pi*np.arange(rows)/(n-1)
    omega=2*np.pi/n
    q=theta[None,:]+np.array([0.,-omega,omega])[:,None]
    # T_s(q)=sum_j x[s+j] exp(-i*q*j), three modulations of one rectangle.
    moments=np.exp(-1j*q[:,:,None]*np.arange(n))@x[:n]
    phase=np.exp(1j*q)
    far=np.exp(-1j*q*n)
    hann=.5-.5*np.cos(omega*np.arange(n))
    position=0
    errors=[]
    begun=time.perf_counter()
    for start in starts:
        while position<start:
            moments=phase*(moments-x[position]+x[position+n]*far)
            position+=1
        cosine=.5*moments[0]-.25*moments[1]-.25*moments[2]
        # j=0 has zero Hann weight. The other endpoint is half weighted by
        # irfft's DC/Nyquist convention, not by the periodic Hann convention.
        cosine-=.5*hann[-1]*x[start+n-1]*np.where(np.arange(rows)%2,-1.,1.)
        got=np.abs(cosine.real/(n-1))
        expected=np.abs(np.fft.irfft(x[start:start+n]*hann)[:rows])
        errors.append(float(np.max(np.abs(got-expected))))
        np.testing.assert_allclose(got,expected,rtol=2e-9,atol=2e-13)
    return dict(aperture=n,rows=rows,starts=starts.tolist(),max_absolute_error=max(errors),
                recurrence_seconds=time.perf_counter()-begun,
                note='Numerical proof, not a realtime implementation or speed comparison.')


if __name__=='__main__':
    report=dict(method='Three sliding Fourier sums, exact Hann modulation and inverse endpoint correction',
                cases=[trial(n,np.random.default_rng(912+n)) for n in (64,512,2048)])
    path=Path('build/shared-window-proof.json');path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(report,indent=2)+'\n');print(path.read_text())
