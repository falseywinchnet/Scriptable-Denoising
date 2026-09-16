"""Native viewer launcher and historical DearPyGui replay prototype.

Only launch() is used by the parent's reload worker. The GUI and socket polling
never run in the Numba callback or the host's audio thread.
"""
import argparse
import faulthandler
import json
import os
from pathlib import Path
import queue
import socket
import subprocess
import sys
import tempfile
import threading
import time
import numpy as np


def launch(port, generation):
    """Launch the native cross-platform executable and verify its first frame."""
    root=Path(__file__).resolve().parents[1]
    name='CleanupViewer.exe' if os.name=='nt' else 'CleanupViewer'
    override=os.environ.get('CLEANUP_VIEWER_EXECUTABLE')
    candidates=([Path(override)] if override else
                [root/name,root/'build'/('viewer-windows' if os.name=='nt' else 'viewer')/name,
                 root/'build'/'viewer-mac'/name])
    executable=next((p.resolve() for p in candidates if p.is_file()),None)
    if executable is None:
        raise RuntimeError(f'Native {name} is missing; build viewer/ or set CLEANUP_VIEWER_EXECUTABLE')
    with tempfile.TemporaryDirectory(prefix='cleanup-viewer-start-') as td:
        ready=Path(td)/'ready.json'
        log=Path(__file__).with_name('viewer.log')
        with log.open('a',encoding='utf-8') as output:
            process=subprocess.Popen([str(executable),'--port',str(port),'--generation',str(generation),
                                      '--ready-file',str(ready)],stdout=output,stderr=output,
                                     close_fds=True,creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
        deadline=time.monotonic()+15.
        while time.monotonic()<deadline:
            if ready.exists():
                result=json.loads(ready.read_text())
                if not result.get('success'):raise RuntimeError(result.get('error','Viewer startup failed'))
                return
            if process.poll() is not None:
                raise RuntimeError(f'Viewer exited during startup ({process.returncode}); see viewer.log')
            time.sleep(.05)
        process.terminate()
        raise RuntimeError('Viewer did not render its first frame within 15 seconds')


def snapshot(port):
    with socket.create_connection(('127.0.0.1',port),timeout=.8) as sock:
        sock.sendall(b'diagnostic\n')
        with sock.makefile('rb') as stream:
            raw=stream.readline(64*1024*1024)
        if not raw.endswith(b'\n'):raise ValueError('incomplete diagnostic snapshot')
        return json.loads(raw)


class History:
    """Bounded rolling display buffers; gaps stay visible, reload resets them."""
    def __init__(self, seconds=6.):
        self.seconds=seconds;self.generation=None;self.sequence=None

    def accept(self, item):
        if not item.get('enabled') or not item.get('sequence'):return False
        generation=item['generation'];seq=item['sequence']
        if generation==self.generation and self.sequence is not None and seq<=self.sequence:return False
        if generation!=self.generation:
            self.generation=generation;self.sequence=None
            self.width=max(1,int(np.ceil(self.seconds*item['sample_rate']/item['hop'])))
            self.guide_width=max(1,int(np.ceil(self.seconds*item['sample_rate']/item['guide_hop'])))
            self.guide=np.full((self.guide_width,item['guide_bins']),np.nan,dtype=np.float32)
            self.filtered=np.full((self.width,item['bins']),np.nan,dtype=np.float32)
            self.gate=np.full(self.width,-1.,dtype=np.float32)
        guide=np.asarray(item['guide'],dtype=np.float32).reshape(item['guide_emit'],item['guide_bins'])
        filtered=np.asarray(item['filtered'],dtype=np.float32).reshape(item['emit'],item['bins'])
        gaps=0 if self.sequence is None else max(0,seq-self.sequence-1)
        self._append(self.guide,guide,gaps*item['guide_emit'])
        self._append(self.filtered,filtered,gaps*item['emit'])
        gate=np.full(item['emit'],1. if item['gate_closed'] else 0.,dtype=np.float32)
        self._append(self.gate,gate,gaps*item['emit'])
        self.sequence=seq;self.item=item
        return True

    @staticmethod
    def _append(buffer,values,gap):
        fill=-1. if buffer.ndim==1 else np.nan
        shift=min(len(buffer),gap+len(values))
        if shift<len(buffer):buffer[:-shift]=buffer[shift:]
        buffer[-shift:]=fill
        count=min(len(buffer),len(values));buffer[-count:]=values[-count:]


def poll(port, expected_generation, mailbox, stopped):
    failures=0
    while not stopped.is_set():
        try:
            item=snapshot(port);failures=0
            if expected_generation is not None and (item.get('generation')!=expected_generation or not item.get('enabled')):
                item={'exit':True}
            if mailbox.full():mailbox.get_nowait()
            mailbox.put_nowait(item)
            if item.get('exit'):return
        except (OSError,ValueError) as error:
            failures+=1
            if failures>=12:
                if mailbox.full():mailbox.get_nowait()
                mailbox.put_nowait({'exit':True,'error':str(error)});return
        stopped.wait(.08)


def demo(path, mailbox, stopped):
    """Replay saved research fields; clearly labeled, independent of live SDR#."""
    z=np.load(path);fs=int(z['sample_rate']);guide=z['phase_fused'].T
    spectrum=z['ordinary'].T
    source_hz=np.arange(guide.shape[1])*fs/4094
    target_hz=np.arange(spectrum.shape[1])*fs/512
    gain=np.stack([np.interp(target_hz,source_hz,column,left=1.,right=1.)
                   for column in z['guide_gain'].T])
    filtered=np.abs(spectrum*gain) # actual final coefficients for this offline mask
    step=16;seq=0
    while not stopped.is_set():
        for start in range(0,len(guide)-step+1,step):
            seq+=1
            gate_closed=not bool(np.any(z['speech_mask'][start:start+step]))
            final=np.zeros_like(filtered[start:start+step]) if gate_closed else filtered[start:start+step]
            item=dict(enabled=True,generation=1,sequence=seq,channel=0,sample_rate=fs,
                      block=step*512,bins=filtered.shape[1],emit=step,hop=512,fft=512,half_bin=0,
                      guide_bins=guide.shape[1],guide_emit=step,guide_hop=512,guide_denominator=4094,
                      guide_half_bin=0,guide_center_offset=0,action=1,bypass=False,squelch=True,
                      gate_closed=gate_closed,valid=True,
                      guide=guide[start:start+step].ravel().tolist(),filtered=final.ravel().tolist())
            if mailbox.full():mailbox.get_nowait()
            mailbox.put_nowait(item)
            if stopped.wait(step*512/fs):return


INFERNO = np.array([[0,0,4],[31,12,72],[85,15,109],[136,34,106],
                    [186,54,85],[227,89,51],[249,140,10],[249,201,50],
                    [252,255,164]], dtype=np.float32) / 255.
_LUT = np.ones((256,4), dtype=np.float32)
for _channel in range(3):
    _LUT[:,_channel] = np.interp(np.linspace(0,8,256),np.arange(9),INFERNO[:,_channel])


def texture_rgba(values):
    """Time-major magnitudes to contiguous RGBA; high frequencies at the top."""
    db=20*np.log10(np.maximum(np.nan_to_num(values,nan=0.,posinf=1e10,neginf=0.),1e-6))
    indices=np.clip((db+120.)*(255./140.),0,255).astype(np.uint8)
    rgba=np.ascontiguousarray(_LUT[indices.T[::-1]])
    # Missing snapshots are distinguishable from measured silence.
    rgba[np.isnan(values.T[::-1])] = (0.08,0.08,0.09,1.)
    return rgba.ravel()


def main():
    faulthandler.enable()
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port',type=int,default=52381)
    parser.add_argument('--generation',type=int)
    parser.add_argument('--demo',type=Path)
    parser.add_argument('--seconds',type=float,default=6.)
    parser.add_argument('--exit-after',type=float,default=0.)
    parser.add_argument('--ready-file',type=Path,help=argparse.SUPPRESS)
    parser.add_argument('--capture',type=Path,help='Export one application framebuffer for visual verification')
    args=parser.parse_args()
    if not 1.<=args.seconds<=30.:parser.error('--seconds must be in [1,30]')
    import dearpygui.dearpygui as dpg
    stopped=threading.Event();mailbox=queue.Queue(maxsize=1);history=History(args.seconds)
    worker=threading.Thread(target=demo if args.demo else poll,
                            args=(args.demo,mailbox,stopped) if args.demo else (args.port,args.generation,mailbox,stopped),daemon=True)
    dpg.create_context();rendered=0
    try:
        # Run callbacks on the render thread, including Pause and texture changes.
        dpg.configure_app(manual_callback_management=True, auto_device=True)
        dpg.add_texture_registry(tag='textures')
        paused=False;dirty=False
        def toggle_pause():
            nonlocal paused,dirty
            paused=not paused;dirty=True
            dpg.set_item_label('pause','Resume' if paused else 'Pause')
            dpg.set_value('playback','Paused · filtering and polling continue' if paused else 'Live')
        dpg.create_viewport(title='Cleanup · live filter diagnostics',width=1180,height=820)
        with dpg.window(tag='main',label='Cleanup diagnostics'):
            dpg.add_text('OFFLINE REPLAY · analysis → experimental mask → final coefficients' if args.demo else 'Live · channel 1 · waiting for filter data',tag='status')
            with dpg.group(horizontal=True):
                dpg.add_button(label='Pause',tag='pause',callback=toggle_pause)
                dpg.add_text('Live',tag='playback')
            dpg.add_text('Top strip: 1 = squelch closed, 0 = open; gaps = dropped display updates. Reload recreates this viewer.')
            with dpg.plot(height=90,width=-1,no_menus=True):
                dpg.add_plot_axis(dpg.mvXAxis,label='Seconds before latest block',tag='gate_x')
                dpg.add_plot_axis(dpg.mvYAxis,label='Squelch',tag='gate_y')
                dpg.add_line_series([],[],parent='gate_y',tag='gate')
                dpg.set_axis_limits('gate_y',-.1,1.1)
            for key,label in [('guide','Analysis · before filtering'),('filtered','Final result · immediately before inversion')]:
                with dpg.plot(label=label,height=300,width=-1,tag=key+'_plot'):
                    dpg.add_plot_axis(dpg.mvXAxis,label='Seconds before latest block',tag=key+'_x')
                    dpg.add_plot_axis(dpg.mvYAxis,label='Hz',tag=key+'_y')
        dpg.setup_dearpygui();dpg.show_viewport();dpg.set_primary_window('main',True)
        worker.start();start=time.monotonic();generation=None;captured=False;ready=False;rendered=0
        while dpg.is_dearpygui_running():
            if args.exit_after and time.monotonic()-start>=args.exit_after:break
            dpg.run_callbacks(dpg.get_callback_queue())
            try:item=mailbox.get_nowait()
            except queue.Empty:item=None
            if item and item.get('exit'):break
            if item and history.accept(item):dirty=True
            if dirty and not paused and history.sequence is not None:
                item=history.item;dirty=False
                new_generation=generation!=history.generation;generation=history.generation
                fs=item['sample_rate']
                for key,values in [('guide',history.guide),('filtered',history.filtered)]:
                    if new_generation and dpg.does_item_exist(key+'_image'):
                        dpg.delete_item(key+'_image')
                        dpg.delete_item(key+'_texture')
                    hop=item['guide_hop'] if key=='guide' else item['hop']
                    denominator=item['guide_denominator'] if key=='guide' else item['fft']
                    half=item['guide_half_bin'] if key=='guide' else item['half_bin']
                    offset=item['guide_center_offset']/fs if key=='guide' else 0.
                    df=fs/denominator
                    # Fixed dB range and no per-frame peak normalization: changes
                    # in suppression remain visible. The enhanced guide has its
                    # own amplitude convention, identified in the panel label.
                    rgba=texture_rgba(values)
                    bounds_min=(-len(values)*hop/fs+offset,(.5*half-.5)*df)
                    bounds_max=(offset,(values.shape[1]+.5*half-.5)*df)
                    if not dpg.does_item_exist(key+'_image'):
                        dpg.add_dynamic_texture(len(values),values.shape[1],rgba,
                                                parent='textures',tag=key+'_texture')
                        dpg.add_image_series(key+'_texture',bounds_min=bounds_min,bounds_max=bounds_max,
                                             parent=key+'_y',tag=key+'_image')
                    else:
                        dpg.set_value(key+'_texture',rgba)
                    if new_generation:
                        dpg.set_axis_limits(key+'_x',-args.seconds,0.)
                        dpg.set_axis_limits(key+'_y',0.,min(4000.,bounds_max[1]))
                gate_x=(np.arange(history.width)-history.width)*item['hop']/fs
                dpg.set_value('gate',[np.repeat(gate_x,2)[1:].tolist(),np.repeat(history.gate,2)[:-1].tolist()])
                dpg.set_axis_limits('gate_x',-args.seconds,0.)
                prefix='OFFLINE REPLAY · ' if args.demo else ''
                dpg.set_value('status',f"{prefix}generation {generation} · block {item['sequence']} · action {item['action']} · "
                              f"{'BYPASS' if item['bypass'] else 'SQUELCHED' if item['gate_closed'] else 'OPEN'} · "
                              f"{'valid' if item['valid'] else 'invalid diagnostic data'}")
            dpg.render_dearpygui_frame()
            rendered+=1
            if args.ready_file and not ready:
                pending=args.ready_file.with_suffix('.tmp')
                pending.write_text(json.dumps(dict(success=True,pid=os.getpid())))
                pending.replace(args.ready_file);ready=True
            if args.capture and not captured and history.sequence is not None and time.monotonic()-start>2.:
                dpg.output_frame_buffer(str(args.capture.resolve()))
                captured=True
            time.sleep(.005)
    finally:
        stopped.set()
        if worker.is_alive():worker.join(timeout=1.)
        dpg.destroy_context()
        print(json.dumps(dict(viewer_exit=True,platform=sys.platform,rendered_frames=rendered,
                              generation=history.generation,last_sequence=history.sequence)),flush=True)


if __name__=='__main__':main()
