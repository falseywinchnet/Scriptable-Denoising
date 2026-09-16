"""Cross-platform controller for the command server inside CleanupNative.
Reload --wait reports compiler failures with a nonzero exit status.
"""
import argparse
import json
from pathlib import Path
import socket
import sys
import time


def request(port,command):
    with socket.create_connection(('127.0.0.1',port),timeout=5) as s:
        s.sendall((command+'\n').encode())
        return json.loads(s.makefile().readline(64*1024*1024))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--port',type=int,default=52381)
    p.add_argument('--discovery',type=Path)
    p.add_argument('--wait',action='store_true')
    p.add_argument('command',nargs='*',default=['status'])
    args=p.parse_args()
    port=json.loads(args.discovery.read_text())['port'] if args.discovery else args.port
    command=' '.join(args.command) or 'status'
    result=request(port,command)
    if args.wait and command=='reload':
        end=time.monotonic()+120
        while result.get('loading') and time.monotonic()<end:
            time.sleep(.1);result=request(port,'status')
    print(json.dumps(result,indent=2))
    if args.wait and command=='reload' and (result.get('loading') or not result.get('success')):
        sys.exit(1)


if __name__=='__main__':main()
