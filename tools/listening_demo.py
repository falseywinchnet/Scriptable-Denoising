"""Render time-aligned listening comparisons through the actual native engine.

No individual normalization: every version retains the original recording gain.
The manifest records the source crop, exact script and native status for each run.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import socket
import time

import numpy as np

from experiment import read_wav, write_wav
from native_client import Engine


def diagnostic(port):
    with socket.create_connection(('127.0.0.1', port), timeout=10) as sock:
        sock.sendall(b'diagnostic\n')
        with sock.makefile('rb') as stream:
            return json.loads(stream.readline(64*1024*1024))


def save_spectra(directory, name, snapshots, state, rate):
    """Keep exact float32 publications; browser receives fixed-range dB bytes."""
    first = snapshots[0]
    guide = np.concatenate([np.asarray(d['guide'], np.float32).reshape(d['guide_emit'], d['guide_bins']) for d in snapshots])
    final = np.concatenate([np.asarray(d['filtered'], np.float32).reshape(d['emit'], d['bins']) for d in snapshots])
    gate = np.concatenate([np.full(d['emit'], d['gate_closed'], np.uint8) for d in snapshots])
    # First diagnostic context starts two blocks before source sample zero.
    # Its emitted synthesis centers start one hop before the middle block.
    start = -(state['block'] + state['hop'])
    fields = {}
    if 'surface_floor' in first:
        for key in ('surface_mean','surface_variance','surface_floor','surface_mask','surface_low','surface_occupancy','surface_reference','surface_shape'):
            if key not in first:continue
            fields[key]=np.concatenate([np.asarray(d[key],np.float32).reshape(d['guide_emit'],d['guide_bins']) for d in snapshots])
    np.savez_compressed(directory/(name+'-spectra.npz'), **fields, guide=guide, final=final, gate=gate,
                        sample_rate=rate, first_center=start, guide_center_offset=first['guide_center_offset'])
    result = dict(db_min=-90., db_max=0., gate=gate.tolist(), hop=first['hop'], sample_rate=rate,
                  first_center=start, frames=len(final), panels=[], block=state['block'],
                  mask_domain_hz=rate*.5 if name in ('crossing','surface','occupancy','surface-excitation','occupancy-excitation','shape','occupancy-comfort','shape-comfort') else state['bandwidth'],
                  population_mode=name.startswith('population'),
                  guide_registration=state.get('guide_registration',True),
                  statistics=[dict(population=d.get('population_stats',[]),mask=d.get('mask_stats',[]),crossing=d.get('crossing_stats',[]),edge_hz=d.get('harmonic_edge_hz',0),harmonic=d.get('harmonic_stats',[])) for d in snapshots])
    for key, data, hop, denominator, half, origin in (
        ('guide', guide, first['guide_hop'], first['guide_denominator'], first['guide_half_bin'], start+first['guide_center_offset']),
        ('final', final, first['hop'], first['fft'], first['half_bin'], start)):
        # Common coherent-window-amplitude convention, not calibrated PSD.
        if key == 'guide' and state['analysis_mode'] == 'registered':
            n=state['guide_fft']
            norm=(n*.5-.25*(1.-np.cos(2.*np.pi/n)))/(n-1.)
        else:
            norm=(state['fft']-1.)*.5
        bins=data.shape[1]  # Keep the full observed frequency axis for inspection.
        db=20*np.log10(np.maximum(data[:,:bins]/norm,1e-12))
        display=np.rint(np.clip((db+90.)/90.,0,1)*255).astype(np.uint8)
        filename=name+'-'+key+'.u8';display.tofile(directory/filename)
        result['panels'].append(dict(kind=key,file=filename,bins=bins,frames=len(data),hop=hop,
                                      frequency_step=rate/denominator,half_bin=.5*half,
                                      first_center=origin,normalizer=norm))
    if fields:
        result['distribution_weighted']=name in ('shape','shape-comfort')
        result['surface_panels']=[]
        for key in fields:
            data=fields[key]
            if key=='surface_variance':data=np.sqrt(data)  # amplitude units for display
            if key=='surface_shape':display=np.rint(np.clip(data/.2,0,1)*255).astype(np.uint8)
            elif key in ('surface_mask','surface_occupancy'):display=np.rint(np.clip(data,0,1)*255).astype(np.uint8)
            else:
                db=20*np.log10(np.maximum(data/255.5,1e-12))
                display=np.rint(np.clip((db+90.)/90.,0,1)*255).astype(np.uint8)
            filename=name+'-'+key+'.u8';display.tofile(directory/filename)
            result['surface_panels'].append(dict(kind=key,file=filename,bins=data.shape[1],frames=len(data),
                hop=first['guide_hop'],frequency_step=rate/first['guide_denominator'],half_bin=0,
                first_center=start+first['guide_center_offset'],normalizer=.2 if key=='surface_shape' else 255.5 if key not in ('surface_mask','surface_occupancy') else 1.))
    (directory/(name+'-spectra.json')).write_text(json.dumps(result))


def retain_previous(previous, output):
    """Retain an identified previous harmonic result for a controlled A/B."""
    if (previous/'original.wav').read_bytes() != (output/'original.wav').read_bytes():
        raise ValueError('Previous harmonic comparison requires the identical source crop')
    old, new = 'registered-harmonics', 'previous-harmonics'
    for suffix in ('.wav','.py','-guide.u8','-final.u8','-spectra.npz'):
        source=previous/(old+suffix)
        if source.exists():shutil.copy2(source,output/(new+suffix))
    meta=json.loads((previous/(old+'-spectra.json')).read_text())
    for panel in meta['panels']:panel['file']=panel['file'].replace(old,new)
    (output/(new+'-spectra.json')).write_text(json.dumps(meta))
    manifest=json.loads((previous/'manifest.json').read_text())
    record=next(item for item in manifest['variants'] if item['name']==old)
    return dict(source_directory=str(previous),render=record)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('source', type=Path)
    p.add_argument('--library', type=Path, required=True)
    p.add_argument('--output', type=Path, default=Path('build/listening-demo'))
    p.add_argument('--seconds', type=float, default=30.)
    p.add_argument('--previous', type=Path, help='Previous demo directory for harmonic A/B')
    p.add_argument('--only',choices=['baseline','registered','registered-harmonics','crossing','surface','occupancy','surface-excitation','occupancy-excitation','shape','occupancy-comfort','shape-comfort','population','population-unregistered'])
    p.add_argument('--hard-shape',action='store_true',help='Retain the hard-admission ablation instead of the softened trial')
    p.add_argument('--base',type=Path,help='Retain exact prior demo artifacts and render only the new variant')
    args = p.parse_args()
    if args.base:shutil.copytree(args.base,args.output,dirs_exist_ok=True)
    args.output.mkdir(parents=True, exist_ok=True)
    html=Path(__file__).with_suffix('.html').read_text()
    if args.previous is None and not (args.output/'previous-harmonics.wav').exists():html=html.replace('const HAS_PREVIOUS = true;', 'const HAS_PREVIOUS = false;')
    if args.only=='population' or (args.output/'population.wav').exists():html=html.replace('const HAS_POPULATION = false;', 'const HAS_POPULATION = true;')
    if args.only=='population-unregistered' or (args.output/'population-unregistered.wav').exists():html=html.replace('const HAS_UNREGISTERED = false;', 'const HAS_UNREGISTERED = true;')
    (args.output/'index.html').write_text(html)
    x, rate = read_wav(args.source)
    x = x[:int(args.seconds * rate)]
    if not len(x):
        raise ValueError('Listening crop is empty')
    write_wav(args.output / 'original.wav', x, rate)
    source = (Path(__file__).resolve().parents[1] / 'Filter.py').read_text()
    report = dict(source=str(args.source), source_sha256=hashlib.sha256(args.source.read_bytes()).hexdigest(),
                  samples=len(x), sample_rate=rate, seconds=len(x)/rate,
                  gain_policy='Original gain; no per-version normalization; engine latency removed', variants=[])
    if args.base:
        report=json.loads((args.base/'manifest.json').read_text())
        if (args.base/'original.wav').read_bytes() != (args.output/'original.wav').read_bytes():raise ValueError('Base demo source differs')
        report['variants']=[item for item in report['variants'] if item['name'] != args.only]
    if args.previous is not None:report['previous_harmonics']=retain_previous(args.previous,args.output)
    for name, mode, harmonics, floor in [('baseline', 'baseline', False, 'legacy'),
                                  ('registered', 'registered', False, 'legacy'),
                                  ('registered-harmonics', 'registered', True, 'legacy'),
                                  ('crossing', 'registered', False, 'zero_crossing'),
                                  ('surface','registered',False,'variance_surface'),
                                  ('occupancy','registered',False,'occupancy_surface'),
                                  ('surface-excitation','registered',True,'variance_surface'),
                                  ('occupancy-excitation','registered',True,'occupancy_surface'),
                                  ('shape','registered',False,'occupancy_surface'),
                                  ('occupancy-comfort','registered',True,'occupancy_surface'),
                                  ('shape-comfort','registered',True,'occupancy_surface'),
                                  ('population','registered',False,'legacy'),
                                  ('population-unregistered','registered',False,'legacy')]:
        if args.only and name != args.only:continue
        script = args.output / (name + '.py')
        import re
        experiment_source = ((Path(__file__).resolve().parents[1] / 'research/harmonics/Filter-retired.py').read_text()
                             if harmonics else source)
        text = re.sub(r'^ANALYSIS_MODE = .*$', 'ANALYSIS_MODE = "'+mode+'"', experiment_source, flags=re.M)
        text = re.sub(r'^GUIDE_REGISTRATION = .*$', 'GUIDE_REGISTRATION = '+str(name!='population-unregistered'), text, flags=re.M)
        text = re.sub(r'^REGISTERED_FLOOR = .*$', 'REGISTERED_FLOOR = "'+floor+'"', text, flags=re.M)
        text = re.sub(r'^HARMONIC_MODEL = .*$', 'HARMONIC_MODEL = \"'+('excitation' if 'excitation' in name or 'comfort' in name else 'packet')+'\"', text, flags=re.M)
        text = re.sub(r'^OCCUPANCY_SHAPE = .*$', 'OCCUPANCY_SHAPE = '+str(name in ('shape','shape-comfort')), text, flags=re.M)
        text = re.sub(r'^SHAPE_SOFT_ADMISSION = .*$', 'SHAPE_SOFT_ADMISSION = '+str(not args.hard_shape), text, flags=re.M)
        text = re.sub(r'^CENTROID_COMFORT_GAIN = .*$', 'CENTROID_COMFORT_GAIN = '+str(.035 if 'comfort' in name else 0.), text, flags=re.M)
        text = re.sub(r'^CLEANUP_POPULATION = .*$', 'CLEANUP_POPULATION = "'+('rolloff' if name.startswith('population') else 'receiver')+'"', text, flags=re.M)
        text = re.sub(r'^VIEWER = .*$', 'VIEWER = True', text, flags=re.M)
        text = re.sub(r'^VIEWER_AUTOSTART = .*$', 'VIEWER_AUTOSTART = False', text, flags=re.M)
        script.write_text(text)
        with Engine(args.library, script, channels=x.shape[1], sample_rate=rate) as engine:
            state = engine.wait()
            if not state['success']:
                raise RuntimeError(state)
            engine.lib.cleanup_set_harmonics(engine.handle, int(harmonics))
            delay = state['latency_samples']
            padded = np.concatenate((x, np.zeros((delay + state['block'], x.shape[1]), np.float32)))
            output = []
            snapshots = []
            start = time.perf_counter()
            for i in range(0, len(padded), state['block']):
                y, code = engine.process(padded[i:i+state['block']])
                if code:
                    raise RuntimeError(engine.status())
                output.append(y)
                if len(output)%50==0:print(json.dumps(dict(variant=name,rendered_input_seconds=min(len(x),(i+state['block']))/rate)),flush=True)
                if len(padded[i:i+state['block']]) == state['block']:
                    d=diagnostic(state['port'])
                    if not d['valid'] or d['sequence'] != len(snapshots)+1:
                        raise RuntimeError('Missing or invalid diagnostic block')
                    snapshots.append(d)
            y = np.concatenate(output)[delay:delay+len(x)]
            if not np.isfinite(y).all():
                raise RuntimeError('Nonfinite demo output')
            write_wav(args.output / (name + '.wav'), y, rate)
            save_spectra(args.output,name,snapshots,state,rate)
            item = dict(name=name, status=engine.status(), render_seconds=time.perf_counter()-start,
                        peak=float(np.max(np.abs(y))), rms=float(np.sqrt(np.mean(y*y))),
                        clipped_samples=int(np.count_nonzero(np.abs(y) >= 1.)))
            report['variants'].append(item)
            print(json.dumps(item), flush=True)
        (args.output / 'manifest.json').write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
