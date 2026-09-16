"""Ship only the new trial and updated UI; retain existing audio byte-for-byte."""
import json
import tarfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
out=ROOT/'build/listening-demo-v16'
meta=json.loads((out/'population-spectra.json').read_text())
files={'index.html','manifest.json','population.wav','population.py','population-spectra.json'}
files.update(p['file'] for p in meta['panels'])
with tarfile.open(ROOT/'build/population-demo-assets.tar.gz','w:gz') as tar:
    for name in sorted(files):tar.add(out/name,arcname=name)
print(json.dumps(dict(files=sorted(files),bytes=(ROOT/'build/population-demo-assets.tar.gz').stat().st_size)))
