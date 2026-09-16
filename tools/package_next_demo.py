"""Package new comparisons only, sharing byte-identical existing display rasters."""
import hashlib,json,tarfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
base=ROOT/'build/listening-demo-v6';out=ROOT/'build/listening-demo-v9'
names=['occupancy','surface-excitation','occupancy-excitation']
known={hashlib.sha256(p.read_bytes()).hexdigest():p.name for p in sorted(base.glob('*.u8'))}
files={'index.html','manifest.json'};shared=[]
for name in names:
 p=out/(name+'-spectra.json');meta=json.loads(p.read_text())
 for panel in meta['panels']+meta.get('surface_panels',[]):
  original=panel['file'];digest=hashlib.sha256((out/original).read_bytes()).hexdigest()
  if digest in known:
   panel['file']=known[digest];shared.append(dict(original=original,reuse=panel['file']))
  else:known[digest]=original;files.add(original)
 p.write_text(json.dumps(meta))
 files.update([p.name,name+'.wav',name+'.py'])
archive=ROOT/'build/next-demo-assets.tar.gz'
with tarfile.open(archive,'w:gz') as tar:
 for name in sorted(files):tar.add(out/name,arcname=name)
report=dict(archive_bytes=archive.stat().st_size,files=sorted(files),shared=shared)
(ROOT/'build/next-demo-package.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
