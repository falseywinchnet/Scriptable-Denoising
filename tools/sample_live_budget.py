"""Read-only timing sample from a running Cleanup control server."""
import argparse
import json
from pathlib import Path
import socket
import statistics
import time

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--port',type=int,default=52381)
p.add_argument('--seconds',type=float,default=30.)
p.add_argument('--report',type=Path,required=True)
a=p.parse_args()
records=[];seen=set();errors=[];end=time.monotonic()+a.seconds
while time.monotonic()<end:
    try:
        with socket.create_connection(('127.0.0.1',a.port),timeout=1.) as sock:
            sock.sendall(b'status\n');data=b''
            while b'\n' not in data:
                part=sock.recv(65536)
                if not part:break
                data+=part
        status=json.loads(data)
        key=(status['generation'],status['blocks_processed'])
        if key not in seen:
            seen.add(key);records.append(status)
    except (OSError,ValueError,KeyError) as e:errors.append(str(e))
    time.sleep(.05)
active=[r for r in records if r['blocks_processed'] and r.get('last_calls',{}).get('filter',0)>0
        and not any(r[k] for k in ('loading','bypass','runtime_fault'))]
summary=dict(seconds=a.seconds,observed_blocks=len(records),active_blocks=len(active),
    fault_blocks=sum(r['runtime_fault'] for r in records),bypassed_blocks=sum(r['bypass'] for r in records),errors=errors)
if active:
    ms=sorted(r['last_block_ns']/1e6 for r in active)
    summary.update(median_ms=statistics.median(ms),peak_ms=max(ms),
        missed_deadlines=sum(r['last_block_ns']/1e9>r['block']/r['sample_rate'] for r in active),
        stage_median_ms={k:statistics.median(r['stage_ns'][k]/1e6 for r in active) for k in active[0]['stage_ns']},
        generations=sorted({r['generation'] for r in active}))
a.report.parent.mkdir(parents=True,exist_ok=True)
a.report.write_text(json.dumps(dict(summary=summary,records=records),indent=2)+'\n')
print(json.dumps(summary,indent=2))
