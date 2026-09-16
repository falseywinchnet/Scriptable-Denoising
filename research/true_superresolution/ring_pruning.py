"""Measure certified unfinished-row pruning; no live DSP changes.

The full transforms below are an oracle, not the proposed execution schedule.
Only row upper bounds and previously evaluated rows may guide pruning. Exported
cross spectra let the native benchmark include the actual transform/bound cost.
"""
import argparse
import json
from pathlib import Path
import sys
import wave
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from research.true_superresolution.oracle.circles import prepare_phase_circle_spectrum
from research.true_superresolution.oracle.uncertainty_fusion import (
    phase_lattice_observations_at_centers, _perceptual_registration_view,
)

N = 48

def cross(a, b):
    a = prepare_phase_circle_spectrum(a).spectrum
    b = prepare_phase_circle_spectrum(b).spectrum
    z = b * np.conj(a)
    unit = z / np.maximum(abs(z), 1e-12)
    # Enforce the native primitive's Hermitian input contract. Independent
    # NumPy evaluations at +/-k can otherwise acquire amplified asymmetry in
    # near-zero cross-spectrum bins during phase normalization.
    for y in range(N):
        for x in range(N):
            other = ((-y) % N, (-x) % N)
            if y*N+x < other[0]*N+other[1]:
                unit[other] = unit[y,x].conjugate()
            elif y*N+x == other[0]*N+other[1]:
                unit[y,x] = unit[y,x].real
    return unit

def search_rows(upper, rows):
    """Complete a row only when its certified upper bound can change a result."""
    order = np.argsort(-upper, kind='stable')
    visited = np.zeros(N, bool)
    peak = -np.inf
    best = N*N
    for y in order:
        if upper[y] < peak:
            break
        visited[y] = True
        x = int(np.argmax(rows[y]))
        v = rows[y, x]
        if v > peak or (v == peak and y*N+x < best):
            peak, best = v, y*N+x
    peak_rows = int(visited.sum())
    py, px = divmod(best, N)
    dy = np.minimum((np.arange(N)-py) % N, (py-np.arange(N)) % N)
    dx = np.minimum((np.arange(N)-px) % N, (px-np.arange(N)) % N)
    allowed = (dy[:, None] > 3) | (dx[None, :] > 3)
    competitor = np.max(np.where(visited[:, None] & allowed, rows, -np.inf))
    for y in order:
        if upper[y] < competitor:
            break
        if not visited[y]:
            visited[y] = True
            competitor = max(competitor, np.max(rows[y, allowed[y]]))
    expected = int(np.argmax(rows))
    expected_competitor = np.max(np.where(allowed, rows, -np.inf))
    assert best == expected, (best, expected)
    assert competitor == expected_competitor
    return dict(peak_rows=peak_rows, total_rows=int(visited.sum()))

def measure(unit, rings):
    records = []
    for axis in (0, 1):
        u = unit if axis == 0 else unit.T
        for ring in rings:
            partial = np.fft.ifft(u * ring, axis=0)
            # The unfinished row is Hermitian. DC is a signed constant;
            # interior terms are conjugate pairs; Nyquist is a signed parity.
            upper = (partial[:, 0].real + abs(partial[:, N//2].real)
                     + 2*np.sum(abs(partial[:, 1:N//2]), axis=1))/N
            # Conservative numerical padding: uncertain ties are evaluated.
            upper += 1e-12*np.maximum(1., abs(upper))
            rows = np.fft.ifft(partial, axis=1).real
            assert np.all(rows.max(axis=1) <= upper)
            result = search_rows(upper, rows)
            records.append(dict(axis=axis, **result))
    return records

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT/'build/ring-pruning')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    f = np.fft.fftfreq(N)
    radius = np.hypot(f[:, None], f[None, :])
    rings = np.exp(-.5*((radius[None]-np.linspace(.035,.46,7)[:,None,None])/.055)**2)
    rng = np.random.default_rng(20260916)
    yy, xx = np.mgrid[:N,:N]
    texture = rng.normal(size=(N,N))
    periodic = np.cos(2*np.pi*xx/6)+.4*np.cos(2*np.pi*yy/8)
    units, labels = [], []
    def add(label, unit):
        labels.append(label)
        units.append(unit)
    add('silence', np.zeros((N,N), complex))
    for dx,dy in ((0,0), (1,-2), (-7,9), (23,-23)):
        add('pure_shift', np.exp(-2j*np.pi*(f[None,:]*dx+f[:,None]*dy)))
    for label,a,b in (
        ('windowed_translation', texture, np.roll(texture,(2,-3),(0,1))),
        ('periodic', periodic, np.roll(periodic,(2,-3),(0,1))),
        ('incoherent', texture, rng.normal(size=(N,N))),
        ('two_displacements', texture, np.roll(texture,(1,2),(0,1))+.9*np.roll(texture,(-7,-8),(0,1))),
    ):
        add(label, cross(a,b))
    with wave.open(str(ROOT/'research/true_superresolution/daveandsimon-first-second.wav')) as wav:
        audio = np.frombuffer(wav.readframes(wav.getnframes()),'<i2').astype(float)/32768.
    observations = phase_lattice_observations_at_centers(
        audio[:24576], np.arange(48)*512, crop_rows=512,
        offsets=(-379,-241,-64,0,83,214,427))
    level = max(np.quantile(observations[3], .995), 1e-30)
    views = [_perceptual_registration_view(o*level/max(np.quantile(o,.995),1e-30)) for o in observations]
    padded = [np.pad(v,((24,24),(24,24)),mode='reflect') for v in views]
    for offset in (0,1,2,4,5,6):
        for y in (8,40,88,152,248,344,440,488):
            for x in (8,24,40):
                add('speech', cross(padded[3][y:y+48,x:x+48], padded[offset][y:y+48,x:x+48]))
    records = [dict(case=label, rows=measure(unit,rings)) for label,unit in zip(labels,units)]
    summaries = {}
    for label in dict.fromkeys(labels):
        summaries[label] = {}
        for axis in (0,1):
            rows = [r for record in records if record['case']==label for r in record['rows'] if r['axis']==axis]
            summaries[label][str(axis)] = dict(
                mean_peak_rows=float(np.mean([r['peak_rows'] for r in rows])),
                mean_total_rows=float(np.mean([r['total_rows'] for r in rows])),
                outputs_skipped_fraction=1-np.mean([r['total_rows'] for r in rows])/N)
    np.asarray(units,dtype='<c16').tofile(args.out/'units.bin')
    report = dict(charts=len(units),ring_maps=7*len(units),extent=N,
        meaning='Exact mathematical row bound; full oracle rows used only when marked evaluated. No speed claim.',
        summaries=summaries, records=records)
    (args.out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='records'},indent=2))

if __name__ == '__main__':
    main()
