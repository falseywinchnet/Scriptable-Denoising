"""Exercise the real viewer callback/render loop with an offline fixture."""
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
import dearpygui.dearpygui as dpg

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'runtime'))
import viewer

report={};latest=[None];stage=[0];started=[None]
original_render=dpg.render_dearpygui_frame
original_accept=viewer.History.accept

def accept(self,item):
    result=original_accept(self,item)
    if result:latest[0]=self.sequence
    return result

def fingerprint():
    return hashlib.sha256(np.asarray(dpg.get_value('filtered_texture'),dtype=np.float32).tobytes()).hexdigest()

def render():
    original_render()
    if not dpg.does_item_exist('filtered_texture'):return
    if started[0] is None:started[0]=time.monotonic()
    elapsed=time.monotonic()-started[0]
    if stage[0]==0 and elapsed>=1.:
        report['frozen_status']=dpg.get_value('status')
        report['frozen_texture']=fingerprint()
        report['pause_sequence']=latest[0]
        dpg.get_item_callback('pause')()
        assert dpg.get_item_label('pause')=='Resume'
        stage[0]=1
    elif stage[0]==1 and elapsed>=3.:
        assert dpg.get_value('status')==report['frozen_status']
        assert fingerprint()==report['frozen_texture']
        assert latest[0]>report['pause_sequence']
        report['polling_while_paused']=True
        dpg.get_item_callback('pause')()
        assert dpg.get_item_label('pause')=='Pause'
        stage[0]=2
    elif stage[0]==2 and elapsed>=4.:
        assert dpg.get_value('status')!=report['frozen_status']
        report['resumed_status']=dpg.get_value('status')
        report['resumed_sequence']=latest[0]
        report['success']=True
        stage[0]=3
        dpg.stop_dearpygui()

viewer.History.accept=accept
dpg.render_dearpygui_frame=render
sys.argv=[sys.argv[0],'--demo',str(ROOT/'build/true-superresolution/probe.npz'),'--exit-after','10']
viewer.main()
assert stage[0]==3,report
print(json.dumps(report,indent=2))
