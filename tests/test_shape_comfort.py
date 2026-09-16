"""Current-distribution weighting and bounded centroid comfort-noise checks."""
import importlib.util,json,sys,unittest
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'runtime'),str(ROOT/'tools')]
from compile_filter import bind_native_dsp,_geometry
REPORT={}
class ShapeComfort(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  spec=importlib.util.spec_from_file_location('shape_comfort',ROOT/'research/harmonics/Filter-retired.py');cls.m=importlib.util.module_from_spec(spec);bind_native_dsp(cls.m)
  source=(ROOT/'research/harmonics/Filter-retired.py').read_text().replace('CLEANUP_POPULATION = "rolloff"','CLEANUP_POPULATION = "receiver"').replace('GUIDE_REGISTRATION = False','GUIDE_REGISTRATION = True').replace('GUIDE_BINS = GUIDE_FFT','GUIDE_BINS = 512',1).replace('REGISTERED_FLOOR = "legacy"','REGISTERED_FLOOR = "occupancy_surface"').replace('HARMONIC_MODEL = "packet"','HARMONIC_MODEL = "excitation"').replace('OCCUPANCY_SHAPE = False','OCCUPANCY_SHAPE = True').replace('CENTROID_COMFORT_GAIN = 0.0','CENTROID_COMFORT_GAIN = 0.035').replace('VIEWER_AUTOSTART = True','VIEWER_AUTOSTART = False')
  exec(compile(source,str(ROOT/'research/harmonics/Filter-retired.py'),'exec'),cls.m.__dict__);_geometry(cls.m)
 def test_distribution_shape_scale_and_continual_line(self):
  m=self.m;shape=(192,512);mean=np.ones(shape);data=mean.copy();score=mean.copy();scratch=np.zeros(12*m.GUIDE_BINS)
  maximum=m.local_distribution_shape(data,mean,score,scratch,48000., np.zeros(192));np.testing.assert_allclose(score,0.,atol=1e-12)
  # An uninterrupted spectral line retains shape evidence without quiet frames.
  data[:,256]=80.;m.local_distribution_shape(data,mean,score,scratch,48000., np.zeros(192))
  self.assertGreater(score[80,256],m.VAD_THRESHOLD)
  scaled=np.zeros(shape);m.local_distribution_shape(data*7,mean*7,scaled,scratch,48000., np.zeros(192));np.testing.assert_allclose(scaled,score,atol=1e-12)
  # Remote out-of-band zeros cannot change this local patch's score.
  cut=data.copy();cut[:,400:]=0.;other=np.zeros(shape);m.local_distribution_shape(cut,mean,other,scratch,48000., np.zeros(192))
  np.testing.assert_array_equal(other[:,256],score[:,256])
  REPORT['shape']=dict(flat_score=0.,continual_line_score=float(score[80,256]),maximum=maximum)
 def test_exact_baseline_weight_structure_and_aliasing(self):
  m=self.m;rows,bins=9,96;data=np.tile(np.linspace(.2,3.,bins),(rows,1));low=np.full_like(data,.4);ref=np.full_like(data,2.);score=np.full_like(data,.2);contour=np.zeros_like(data);mask=np.zeros_like(data);scratch=np.zeros(12*m.GUIDE_BINS);maximum=.8
  m.shape_surface_peaks(data,data,mask,low,ref,contour,score,scratch,48000.,maximum,True, np.zeros(192))
  b=40;r=int(m.SURFACE_FREQUENCY_HZ*2*(m.GUIDE_FFT-1)/48000);values=data[4,b-r:b+r+1];mass=.1*len(values);whole=int(mass)
  rm=(values[:whole].sum()+(mass-whole)*values[whole])/mass;ra=np.sqrt(np.mean((values-rm)**2));f=1-.2/.8;gm=.4*f;ga=1.6*f
  wm=np.exp(-.5*abs(gm-rm));wa=np.exp(-.5*abs(ga-ra));expected=rm*wm+gm*(1-wm)+ra*wa+ga*(1-wa)
  self.assertAlmostEqual(contour[4,b],expected,places=12)
  a=data.copy();separate=data.copy();m.shape_surface_peaks(data,data,separate,low,ref,contour,score,scratch,48000.,maximum,False, np.zeros(192))
  m.shape_surface_peaks(a,a,a,low,ref,contour,score,scratch,48000.,maximum,False, np.zeros(192));np.testing.assert_array_equal(a,separate)
  mask[:]=0.;score[:]=0.;m.shape_surface_peaks(data,data,mask,low,ref,contour,score,scratch,48000.,maximum,True, np.zeros(192));np.testing.assert_array_equal(mask,0.)
 def test_soft_admission_uses_same_knee_without_erasing_subthreshold_support(self):
  m=self.m;data=np.ones((9,96));data[:,40]=3.;low=np.zeros_like(data);ref=np.ones_like(data);score=np.full_like(data,m.VAD_THRESHOLD*.5);contour=np.zeros_like(data);mask=np.zeros_like(data)
  m.shape_surface_peaks(data,data,mask,low,ref,contour,score,np.zeros(12*m.GUIDE_BINS),48000.,.8,True, np.zeros(192))
  np.testing.assert_allclose(mask[:,40],.5)
  self.assertTrue(np.all((mask>=0.)&(mask<=.5)))
 def test_comfort_locality_budget_determinism_and_ramp(self):
  m=self.m;last=32;original=np.zeros((m.FRAMES,m.BINS,2));original[:,:last+1,0]=1.;env=np.ones((m.FRAMES,m.BINS));holes=np.zeros_like(env);scratch=np.zeros((m.BINS,2));out=original.copy()
  p,low=m.centroid_comfort(out,original,env,holes,scratch,48000.,0.,2,last,1.,True);self.assertEqual(p,0.);np.testing.assert_array_equal(out,original)
  holes[:,last+1:]=.5;holes[m.FIRST:m.FIRST+10]=0.
  p,low=m.centroid_comfort(out,original,env,holes,scratch,48000.,0.,2,last,1.,True)
  self.assertGreater(p,0.);self.assertLessEqual(p,.01*m.HARMONIC_MAX_POWER_RATIO*low*(1+1e-12))
  np.testing.assert_array_equal(out[:,:last+1],original[:,:last+1]);np.testing.assert_array_equal(out[m.FIRST:m.FIRST+10],original[m.FIRST:m.FIRST+10])
  again=original.copy();m.centroid_comfort(again,original,env,holes,scratch,48000.,0.,2,last,1.,True);np.testing.assert_array_equal(out,again)
  zero=np.zeros_like(out);m.centroid_comfort(zero,zero.copy(),env,holes,scratch,48000.,0.,2,last,1.,True);np.testing.assert_array_equal(zero,0.)
  off=original.copy();m.centroid_comfort(off,original,env,holes,scratch,48000.,0.,2,last,1.,False);np.testing.assert_array_equal(off[m.FIRST+m.EMIT-1],original[m.FIRST+m.EMIT-1])
  REPORT['comfort']=dict(power=p,lower_power=low,max_ratio=.01*m.HARMONIC_MAX_POWER_RATIO)
if __name__=='__main__':
 result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(ShapeComfort))
 (ROOT/'build/shape-comfort-tests.json').write_text(json.dumps(REPORT,indent=2));sys.exit(not result.wasSuccessful())
