"""Feed the actual cross-platform native viewer an explicitly offline replay."""
import argparse
import json
import os
from pathlib import Path
import queue
import socket
import subprocess
import sys
import threading

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'runtime'))
from viewer import demo

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--viewer',type=Path,required=True)
p.add_argument('--wine',action='store_true')
p.add_argument('--report',default='build/native-viewer-smoke.json')
p.add_argument('--capture',default='build/native-viewer.ppm')
a=p.parse_args()
mailbox=queue.Queue(maxsize=1);stop=threading.Event()
thread=threading.Thread(target=demo,args=(ROOT/'build/true-superresolution/probe.npz',mailbox,stop),daemon=True)
listener=socket.socket();listener.bind(('127.0.0.1',0));listener.listen();listener.settimeout(.2)
port=listener.getsockname()[1]

def serve():
    latest=dict(enabled=True,generation=1,sequence=0)
    while not stop.is_set():
        try:client,_=listener.accept()
        except socket.timeout:continue
        with client:
            client.settimeout(.5)
            try:
                line=client.makefile('rb').readline(100)
                if line!=b'diagnostic\n':continue
                try:latest=mailbox.get_nowait()
                except queue.Empty:pass
                latest['source']='offline-replay'
                client.sendall((json.dumps(latest)+'\n').encode())
            except OSError:pass

server=threading.Thread(target=serve,daemon=True)
thread.start();server.start()
try:
    command=(['wine'] if a.wine else [])+[str(a.viewer),'--port',str(port),'--self-test','--exit-after','12','--report',a.report,'--capture',a.capture]
    env=dict(os.environ,WINEDEBUG='-all',MVK_CONFIG_LOG_LEVEL='0')
    subprocess.run(command,env=env,check=True,timeout=30)
    report=json.loads(Path(a.report).read_text())
    assert report['success'] and report['pause_resume_passed'] and report['received_sequence']>0,report
    print(json.dumps(report,indent=2))
finally:
    stop.set();server.join(timeout=1);thread.join(timeout=1);listener.close()
