"""Verify frozen comparison identity and comfort-only spectral locality."""
from pathlib import Path
import json,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parents[1];base=ROOT/'build/listening-demo-v9';out=ROOT/'build/listening-demo-v15'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
old={p.name:sha(p) for p in base.glob('*.wav')}
assert all(sha(out/name)==digest for name,digest in old.items())
oldz=np.load(base/'occupancy-excitation-spectra.npz');newz=np.load(out/'occupancy-comfort-spectra.npz')
assert np.array_equal(oldz['guide'],newz['guide'])
for field in ('surface_mean','surface_variance','surface_floor','surface_mask','surface_low','surface_occupancy','surface_reference'):
 assert np.array_equal(oldz[field],newz[field]),field
oldf,newf=oldz['final'],newz['final'];meta=json.loads((out/'occupancy-comfort-spectra.json').read_text());emit=8192//128
changed=0;total=0
for i,stat in enumerate(meta['statistics']):
 last=int(stat['edge_hz']/(48000/512));a=i*emit;b=a+emit
 assert np.array_equal(oldf[a:b,:last+1],newf[a:b,:last+1]),i
 changed+=np.count_nonzero(oldf[a:b,last+1:]!=newf[a:b,last+1:]);total+=oldf[a:b,last+1:].size
manifest=json.loads((out/'manifest.json').read_text());renders=[]
for v in manifest['variants']:
 if v['name'] in ('shape','occupancy-comfort','shape-comfort'):
  assert sha(out/(v['name']+'.py'))==v['status']['sha256']
  assert v['status']['faults']==0 and v['clipped_samples']==0
  renders.append(v)
report=dict(previous_wavs_identical=len(old),comfort_analysis_and_mask_identical=True,comfort_protected_lower_magnitudes_identical=True,changed_upper_cells=int(changed),upper_cells=int(total),renders=renders)
(ROOT/'build/shape-demo-audit.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
