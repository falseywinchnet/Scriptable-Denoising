"""Stage a self-contained Windows x64 candidate, including its private runtime.

This does not publish. Test the staged bundle with tests/bundle_host.cpp and SDR#
before shipping. No C# source, SDR# assemblies, Git data or build trees are added.
"""
import argparse
import ast
import hashlib
import json
import re
from pathlib import Path
import shutil
import zipfile

ROOT=Path(__file__).resolve().parents[1]

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--python-runtime',type=Path,required=True)
    parser.add_argument('--output',type=Path,default=ROOT/'build/distribution')
    parser.add_argument('--zip',action='store_true')
    args=parser.parse_args()
    target=args.output/'Cleanup'
    if target.exists():
        raise SystemExit(f'Refusing to merge a candidate with stale files: {target}')
    target.mkdir(parents=True)
    def copy(source,dest):
        dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source,dest)
    for name in ('Cleanup.dll','CleanupNative.dll','libwinpthread-1.dll','cleanupctl.exe'):
        copy(ROOT/'build/windows'/name,target/name)
    wrapper=ROOT/'plugin/bin/Release/net10.0-windows/SDRSharp.Cleanup.dll'
    if not wrapper.is_file():wrapper=ROOT/'plugin/precompiled/windows-x64/SDRSharp.Cleanup.dll'
    copy(wrapper,target/'SDRSharp.Cleanup.dll')
    for name in ('CleanupViewer.exe','VIEWER-LICENSES.txt'):
        copy(ROOT/'build/viewer-windows'/name,target/name)
    ignore=shutil.ignore_patterns('__pycache__','*.pyc','*.log','.DS_Store')
    shutil.copytree(ROOT/'runtime',target/'runtime',ignore=ignore)
    py=args.python_runtime
    private=target/'_python';private.mkdir()
    for source in py.iterdir():
        if source.is_file() and (source.suffix.lower() in ('.exe','.dll') or source.name=='LICENSE.txt'):
            copy(source,private/source.name)
    shutil.copytree(py/'DLLs',private/'DLLs',ignore=ignore)
    def runtime_ignore(folder,names):
        excluded=set(ignore(folder,names))
        if Path(folder).name=='site-packages':
            excluded.update(n for n in names if n.startswith(('pip','dearpygui')))
        return excluded
    shutil.copytree(py/'Lib',private/'Lib',ignore=runtime_ignore)
    source=(ROOT/'Filter.py').read_text()
    # Preserve the tested script's population policy; packaging must not silently
    # select a more expensive experimental mask or statistical population.
    # Deployment default is independent of a developer's current opt-in trial.
    source=re.sub(r'(?m)^GUIDE_REGISTRATION = .*$', 'GUIDE_REGISTRATION = False', source)
    (target/'Filter.py').write_text(source)
    # Underscore directories keep the SDR# managed DLL scanner out of SDK and
    # source trees as well as the private interpreter/NumPy/LLVM libraries.
    for name in ('cleanup.h','cleanup_loader.h'):
        copy(ROOT/'native/include'/name,target/'_sdk/include'/name)
    copy(ROOT/'build/windows/libCleanup.dll.a',target/'_sdk/lib/libCleanup.dll.a')
    copy(ROOT/'tests/bundle_host.cpp',target/'_sdk/example/bundle_host.cpp')
    for name in ('main.cpp','CMakeLists.txt'):
        copy(ROOT/'viewer'/name,target/'_source/viewer'/name)
    for name in ('native','vendor/bfft'):
        shutil.copytree(ROOT/name,target/'_source'/name,ignore=shutil.ignore_patterns('.git','build','__pycache__','*.pyc','.DS_Store'))
    for name in ('CMakeLists.txt','requirements.txt','research/true_superresolution/unrolled.cpp','docs/algorithm.md','docs/population-cleanup.md','docs/registered-cleanup.md','docs/optional-registration.md','docs/mask-source.md','docs/mono-analysis-tap.md','docs/scratch-workspace.md','docs/release-2026-09-16.md'):
        copy(ROOT/name,target/'_source'/name)
    copy(ROOT/'docs/windows-bundle.md',target/'README.md')
    for name in ('vendor/bfft/LICENSE','reference/legacy-plugin/CleanupNative/CleanupNative.h','reference/streamcleaner-master/LICENSE_AND_MANUAL.txt'):
        p=ROOT/name
        if p.exists():copy(p,target/'_notices'/p.name)
    # Include MinGW runtime distribution terms with the redistributed pthread DLL.
    for base in (Path('/opt/homebrew/opt/mingw-w64'),):
        for name in ('COPYING','COPYING.RUNTIME','COPYING.LIB'):
            if (base/name).is_file():copy(base/name,target/'_notices'/('mingw-'+name))
    forbidden=list(target.rglob('*.cs'))+list(target.rglob('SDRSharp.exe'))
    if forbidden:raise RuntimeError(f'Forbidden release input: {forbidden}')
    files={str(p.relative_to(target)):dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted(target.rglob('*')) if p.is_file()}
    settings={}
    for node in ast.parse(source).body:
        if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name):
            if node.targets[0].id in ('ANALYSIS_MODE','MASK_SOURCE','CLEANUP_POPULATION','REGISTERED_FLOOR','GUIDE_REGISTRATION','CHAIN'):
                settings[node.targets[0].id]=ast.literal_eval(node.value)
    manifest=dict(format=1,platform='windows-x86_64',candidate=True,
        python='3.12.10',numpy='2.3.5',numba='0.63.1',llvmlite='0.46.0',
        script_settings=settings,files=files)
    (target/'MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
    if args.zip:
        archive=args.output/'Cleanup-windows-x64-candidate.zip'
        with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as out:
            for p in sorted(target.rglob('*')):
                if p.is_file():out.write(p,Path('Cleanup')/p.relative_to(target))
        print(json.dumps(dict(archive=str(archive),bytes=archive.stat().st_size,sha256=hashlib.sha256(archive.read_bytes()).hexdigest())))
    print(target)

if __name__=='__main__':main()
