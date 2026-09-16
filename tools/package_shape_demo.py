"""Package new comparisons only, sharing byte-identical existing display rasters."""
import hashlib,json,tarfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
base=ROOT/'build/listening-demo-v9';out=ROOT/'build/listening-demo-v15'
names=['shape','occupancy-comfort','shape-comfort']
references=set()
for metadata in base.glob('*-spectra.json'):
 m=json.loads(metadata.read_text());references.update(p['file'] for p in m['panels']+m.get('surface_panels',[]))
known={hashlib.sha256((base/name).read_bytes()).hexdigest():name for name in sorted(references)}
files={'index.html','manifest.json'};shared=[]
for name in names:
 p=out/(name+'-spectra.json');meta=json.loads(p.read_text())
 for panel in meta['panels']+meta.get('surface_panels',[]):
  if panel['kind']=='surface_shape':panel['normalizer']=.2;panel['display_scale']='linear';panel['display_range']=[0.,.2]
  original=panel['file'];digest=hashlib.sha256((out/original).read_bytes()).hexdigest()
  if digest in known:
   panel['file']=known[digest];shared.append(dict(original=original,reuse=panel['file']))
  else:known[digest]=original;files.add(original)
 p.write_text(json.dumps(meta))
 files.update([p.name,name+'.wav',name+'.py'])
archive=ROOT/'build/shape-demo-assets.tar.gz'
with tarfile.open(archive,'w:gz') as tar:
 for name in sorted(files):tar.add(out/name,arcname=name)
report=dict(archive_bytes=archive.stat().st_size,files=sorted(files),shared=shared)
(ROOT/'build/shape-demo-package.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
