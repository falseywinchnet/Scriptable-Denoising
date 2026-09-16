"""Read-only D3D11 device probe; no window, registry edit, or renderer crash."""
import ctypes as c
import json
import sys

if sys.platform!='win32':raise SystemExit('Run with Windows Python, natively or under Wine.')
d3d=c.WinDLL('d3d11')
create=d3d.D3D11CreateDevice
create.argtypes=[c.c_void_p,c.c_uint,c.c_void_p,c.c_uint,c.POINTER(c.c_uint),c.c_uint,
                 c.c_uint,c.POINTER(c.c_void_p),c.POINTER(c.c_uint),c.POINTER(c.c_void_p)]
create.restype=c.c_int32

def release(pointer):
    if pointer:
        table=c.cast(pointer,c.POINTER(c.POINTER(c.c_void_p))).contents
        c.WINFUNCTYPE(c.c_ulong,c.c_void_p)(table[2])(pointer)

rows=[]
for driver,name in [(1,'hardware'),(5,'warp')]:
    for requested in [(0xb000,0xa000),(0xa000,),(0x9300,0x9200,0x9100)]:
        levels=(c.c_uint*len(requested))(*requested)
        device=c.c_void_p();context=c.c_void_p();actual=c.c_uint()
        hr=create(None,driver,None,0,levels,len(levels),7,c.byref(device),c.byref(actual),c.byref(context))
        rows.append(dict(driver=name,requested=[hex(x) for x in requested],
                         hresult=f'0x{hr & 0xffffffff:08x}',actual=hex(actual.value),
                         device_created=bool(device.value)))
        release(context);release(device)
print(json.dumps(rows,indent=2))
