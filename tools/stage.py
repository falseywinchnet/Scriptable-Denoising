"""Install a built plugin and its private runtime into an SDR# directory.
The --development option links Filter.py to the authoritative checkout on macOS;
Windows release staging always copies it. Existing files are backed up once per run.
"""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sdrsharp',type=Path,required=True)
    p.add_argument('--python-runtime',type=Path,required=True,help='Python 3.12 Windows directory, with NumPy/Numba already installed')
    p.add_argument('--native-build',type=Path,default=ROOT/'build/windows')
    p.add_argument('--managed-build',type=Path)
    p.add_argument('--viewer-build',type=Path,default=ROOT/'build/viewer-windows')
    p.add_argument('--development',action='store_true')
    args=p.parse_args()
    if args.managed_build is None:
        built=ROOT/'plugin/bin/Release/net10.0-windows'
        args.managed_build=built if (built/'SDRSharp.Cleanup.dll').is_file() else ROOT/'plugin/precompiled/windows-x64'
    if not (args.sdrsharp/'SDRSharp.exe').is_file():p.error('No SDRSharp.exe at that destination')
    for path in (args.native_build/'CleanupNative.dll',args.managed_build/'SDRSharp.Cleanup.dll',args.python_runtime/'python312.dll',args.viewer_build/'CleanupViewer.exe',args.viewer_build/'VIEWER-LICENSES.txt'):
        if not path.is_file():p.error(f'Missing build input: {path}')
    target=args.sdrsharp/'Plugins/Cleanup'
    if target.exists():
        stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        backup=args.sdrsharp/'_cleanup_backups'/stamp
        shutil.copytree(target,backup,symlinks=True)
        print(f'Backup: {backup}')
    target.mkdir(parents=True,exist_ok=True)
    for name in ('Cleanup.dll','CleanupNative.dll','libwinpthread-1.dll','cleanupctl.exe'):
        if (args.native_build/name).exists():
            shutil.copy2(args.native_build/name,target/name)
    shutil.copy2(args.managed_build/'SDRSharp.Cleanup.dll',target)
    shutil.copy2(args.viewer_build/'CleanupViewer.exe',target)
    shutil.copy2(args.viewer_build/'VIEWER-LICENSES.txt',target)
    shutil.copytree(ROOT/'runtime',target/'runtime',dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__','*.pyc','*.log'))
    # SDR# recursively scans plugin DLLs. A leading underscore excludes the runtime.
    shutil.copytree(args.python_runtime,target/'_python',dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__','*.pyc','*.log'))
    script=target/'Filter.py'
    if script.is_symlink() or script.exists():script.unlink()
    if args.development:script.symlink_to(ROOT/'Filter.py')
    else:shutil.copy2(ROOT/'Filter.py',script)
    shutil.copy2(ROOT/'docs/algorithm.md',target/'EXPERIMENT.md')
    print(f'Installed: {target.resolve()}')
    print('Restart SDR# to load a changed DLL or compiler. Edit Filter.py and Reload for DSP/geometry changes.')


if __name__=='__main__':main()
