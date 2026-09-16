"""Low-tail/RMS identity and excitation physics, separate from listening claims."""
import importlib.util,json,sys,unittest
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'runtime'),str(ROOT/'tools')]
from compile_filter import bind_native_dsp,_geometry
REPORT={}
class Experiments(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  spec=importlib.util.spec_from_file_location('next_experiments',ROOT/'research/harmonics/Filter-retired.py');cls.m=importlib.util.module_from_spec(spec);bind_native_dsp(cls.m)
  source=(ROOT/'research/harmonics/Filter-retired.py').read_text().replace('CLEANUP_POPULATION = "rolloff"','CLEANUP_POPULATION = "receiver"').replace('GUIDE_REGISTRATION = False','GUIDE_REGISTRATION = True').replace('GUIDE_BINS = GUIDE_FFT','GUIDE_BINS = 512',1).replace('REGISTERED_FLOOR = "legacy"','REGISTERED_FLOOR = "occupancy_surface"').replace('HARMONIC_MODEL = "packet"','HARMONIC_MODEL = "excitation"').replace('VIEWER_AUTOSTART = True','VIEWER_AUTOSTART = False')
  exec(compile(source,str(ROOT/'research/harmonics/Filter-retired.py'),'exec'),cls.m.__dict__);_geometry(cls.m)
 def test_low_anchor_and_baseline_moment_identity(self):
  m=self.m;x=np.tile(np.linspace(.1,1.5,128),(192,1));mu=np.zeros_like(x);v=mu.copy();floor=mu.copy();clipped=mu.copy()
  m.variance_surface(x,mu,v,floor,clipped,np.zeros((3,128)),np.zeros(2),np.zeros(512),48000.,0., np.zeros(192))
  low=np.zeros_like(x);rho=low.copy();reference=low.copy()
  m.occupancy_surface(clipped,mu,v,floor,low,rho,reference,np.zeros(256),np.zeros(2),np.zeros(512),48000.,0., np.zeros(192))
  radius=int(m.SURFACE_FREQUENCY_HZ*2*(m.GUIDE_FFT-1)/48000)
  for b in range(128):
   a=np.sort(clipped[0,max(0,b-radius):min(128,b+radius+1)]);mass=.1*len(a);n=int(mass)
   expected=(a[:n].sum()+(mass-n)*a[n])/mass
   self.assertAlmostEqual(low[0,b],expected,places=12)
  np.testing.assert_allclose(floor,low+np.sqrt(v+(mu-low)**2),rtol=1e-10,atol=1e-12)
  np.testing.assert_allclose(rho,(mu-low)**2/(v+(mu-low)**2),rtol=1e-10)
  self.assertTrue(np.all(floor>=reference))
  REPORT['occupancy_identity']='T=L+RMS(A-L); lower-tail mean matches independent sorting'
 def test_two_level_occupancy_and_scale(self):
  m=self.m;shape=(192,80);mu=np.full(shape,1.);v=np.full(shape,3.);clipped=np.zeros(shape);floor=mu.copy();low=mu.copy();rho=mu.copy();reference=mu.copy()
  # Mass 1/4 at amplitude 4 and 3/4 at zero has mu=1,var=3.
  m.occupancy_surface(clipped,mu,v,floor,low,rho,reference,np.zeros(160),np.zeros(2),np.zeros(512),48000.,0., np.zeros(192))
  np.testing.assert_allclose(rho,.25);np.testing.assert_allclose(floor,2.)
  m.occupancy_surface(clipped,mu*7,v*49,floor,low,rho,reference,np.zeros(160),np.zeros(2),np.zeros(512),48000.,0., np.zeros(192))
  np.testing.assert_allclose(rho,.25);np.testing.assert_allclose(floor,14.)
 def test_exact_fractional_hann_response(self):
  m=self.m;n=m.FFT_SIZE;r=np.arange(n)-n/2;window=np.hanning(n)
  for delta in np.linspace(-3.8,3.8,77):
   expected=np.sum(window*np.exp(2j*np.pi*delta*r/n))/window.sum()
   re,im=m.harmonic_window_response(float(delta));np.testing.assert_allclose(complex(re,im),expected,atol=2e-13)
 def test_pitch_uses_multiple_components_and_rejects_single_line(self):
  m=self.m;freq=np.arange(m.GUIDE_BINS)*48000/(2*(m.GUIDE_FFT-1));cases=[]
  for pitch in [100.,154.,220.,302.]:
   field=np.full((m.GUIDE_FRAMES,m.GUIDE_BINS),.005)
   for h in range(1,int(2900/pitch)):
    field+=np.exp(-.5*((freq-h*pitch)/11)**2)[None,:]/np.sqrt(h)
   f,c=m.excitation_pitch(field,20,48000.,2900.,0.)
   self.assertLessEqual(abs(f-pitch),4.,(pitch,f,c));cases.append(dict(expected=pitch,pitch=f,confidence=c))
  field=np.tile(.005+np.exp(-.5*((freq-900)/10)**2),(m.GUIDE_FRAMES,1))
  f,c=m.excitation_pitch(field,20,48000.,2900.,0.);self.assertEqual((f,c),(0.,0.))
  REPORT['pitch']=cases
 def test_excitation_protected_band_budget_disabled_and_phase(self):
  m=self.m;rate=48000.;pitch=154.;df=rate/m.FFT_SIZE
  analysis=np.zeros((m.FRAMES,m.BINS,2));guide=np.full((m.GUIDE_FRAMES,m.GUIDE_BINS),.0001)
  freq=np.arange(m.GUIDE_BINS)*rate/(2*(m.GUIDE_FFT-1))
  for h in range(1,19):guide+=np.exp(-.5*((freq-h*pitch)/10)**2)[None,:]/h
  for t in range(m.FRAMES):
   for h in range(2,19):
    amplitude=1./h;theta=2*np.pi*h*pitch*(t*m.HOP)/rate+.2*h
    coord=h*pitch/df
    for b in range(max(1,int(coord)-3),min(m.BINS,int(coord)+5)):
     re,im=m.harmonic_window_response(coord-b)
     analysis[t,b,0]+=amplitude*(np.cos(theta)*re-np.sin(theta)*im)
     analysis[t,b,1]+=amplitude*(np.sin(theta)*re+np.cos(theta)*im)
  state=np.zeros((m.MAX_ENTITIES,m.ENTITY_STORAGE));state[m.GUIDE_SLOT,:m.GUIDE_PLANE]=guide.ravel()
  config=np.array([rate,3000.,0.,0.,0.]);content=analysis.copy()
  self.assertEqual(m.harmonics_excitation(analysis,content,state,config),m.PASSTHROUGH)
  np.testing.assert_array_equal(content,analysis)
  config[3]=1.;self.assertEqual(m.harmonics_excitation(analysis,content,state,config),m.PROCESS)
  edge=int(state[1,4]/df);np.testing.assert_array_equal(content[:,:edge+1],analysis[:,:edge+1])
  self.assertGreater(state[1,2],0.);self.assertLessEqual(state[1,2],m.HARMONIC_MAX_POWER_RATIO*state[1,3]*(1+1e-10))
  self.assertLess(abs(state[1,5]-pitch),4.)
  self.assertTrue(np.isfinite(content).all())
  REPORT['excitation']=dict(pitch=state[1,5],confidence=state[1,6],added_power=state[1,2],low_power=state[1,3],edge_hz=state[1,4])
  # A squelch reset clears the mix. A stale old phase must not survive that reset.
  state[1,0]=0.;fresh=state.copy();fresh[1,9]=2.;fresh[1,12]=0.
  stale=state.copy();stale[1,9]=-1.;stale[1,12]=1.
  a=analysis.copy();b=analysis.copy()
  m.harmonics_excitation(analysis,a,fresh,config);m.harmonics_excitation(analysis,b,stale,config)
  np.testing.assert_array_equal(a,b)

if __name__=='__main__':
 result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Experiments))
 (ROOT/'build/occupancy-excitation-tests.json').write_text(json.dumps(REPORT,indent=2));sys.exit(not result.wasSuccessful())
