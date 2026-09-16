"""Read PortAudio host/device names without starting an audio stream."""
import ctypes as C
import json
import sys
class Host(C.Structure):
    _fields_=[('version',C.c_int),('type',C.c_int),('name',C.c_char_p),('deviceCount',C.c_int),('input',C.c_int),('output',C.c_int)]
class Device(C.Structure):
    _fields_=[('version',C.c_int),('name',C.c_char_p),('host',C.c_int),('inputs',C.c_int),('outputs',C.c_int),('lowInputLatency',C.c_double),('lowOutputLatency',C.c_double),('highInputLatency',C.c_double),('highOutputLatency',C.c_double),('rate',C.c_double)]
p=C.CDLL(sys.argv[1]);p.Pa_GetHostApiInfo.restype=C.POINTER(Host);p.Pa_GetDeviceInfo.restype=C.POINTER(Device)
assert p.Pa_Initialize()==0
try:
    result=[]
    for i in range(p.Pa_GetDeviceCount()):
        d=p.Pa_GetDeviceInfo(i).contents;h=p.Pa_GetHostApiInfo(d.host).contents
        result.append(dict(index=i,name=d.name.decode(),host=h.name.decode(),host_type=h.type,inputs=d.inputs,outputs=d.outputs,rate=d.rate))
    print(json.dumps(result,indent=2))
finally:p.Pa_Terminate()
