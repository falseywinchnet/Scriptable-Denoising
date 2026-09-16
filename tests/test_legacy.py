import ctypes as C
import importlib.util
import sys
from pathlib import Path
import unittest
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'runtime'))
from compile_filter import bind_native_dsp
spec=importlib.util.spec_from_file_location('original_port',ROOT/'Filter.py')
m=importlib.util.module_from_spec(spec)
bind_native_dsp(m)
spec.loader.exec_module(m)

class LegacyParity(unittest.TestCase):
    def test_two_pass_mask_and_recursive_smoothing_match_legacy(self):
        lib=C.CDLL(str(ROOT/'build/legacy-oracle/legacy.dylib'))
        ptr=C.POINTER(C.c_double)
        lib.legacy_mask.argtypes=[ptr,ptr,ptr,C.c_int,C.c_int,ptr]
        records=[]
        for squelch in (False,True):
            for nb in (26,37,129):
                rng=np.random.default_rng(nb)
                a=rng.normal(0,.001,(192,257,2))
                a[:,5,0]+=.5
                a[70:115,12,0]+=.2
                state=np.zeros(m.ENTITY_STORAGE)
                cfg=np.array([48000.,(nb-1)*93.75,float(squelch),0.])
                # Legacy After=True: evidence from early audio, mask amplitudes
                # from a differently gained/filtered later stream.
                content=a.copy()*37.
                content[:,8:10,:]*=.07
                mask_input=content.copy()
                m.cleanup(a,content,state,cfg)
                mag=np.zeros((192,257));mag[:,:nb]=np.hypot(a[:,:nb,0],a[:,:nb,1])
                raw=np.zeros(192);smooth=raw.copy();detected=raw.copy()
                log1=np.zeros(257);log3=np.zeros(771);scratch=np.zeros(771)
                m.entropy(mag,nb,raw,smooth,detected,log1,log3,scratch, np.zeros(192))
                mag[:,:nb]=np.hypot(mask_input[:,:nb,0],mask_input[:,:nb,1])
                expected=np.zeros((192,257))
                lib.legacy_mask(*(x.ctypes.data_as(ptr) for x in (mag,raw,detected)),nb,int(squelch),expected.ctypes.data_as(ptr))
                actual=state[64+2*m.PLANE:64+3*m.PLANE].reshape(192,257)
                error=float(np.max(np.abs(actual[63:127]-expected[63:127])))
                records.append(dict(squelch=squelch,nb=nb,max_center_mask_error=error))
                self.assertLess(error,1e-10)
        import json
        (ROOT/'build/legacy-parity.json').write_text(json.dumps(records,indent=2))
        print(records)

if __name__=='__main__':unittest.main(verbosity=2)
