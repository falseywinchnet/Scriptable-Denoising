// Exact bounded-context port of oracle/uncertainty_fusion.py and circles.py.
// All allocations and Fourier planning occur at generation construction.
#include "registered_guide.h"
#include "guide_fft.h"
#include "guide_peak.h"
#include "ring_forward.h"
#include "ring_pruned.h"
#include "ring_early.h"
#include <bfft/bfft.h>
#include <algorithm>
#include <cmath>
#include <complex>
#include <limits>
#include <memory>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <cstring>
#include <stdexcept>
#include <vector>
#include <chrono>
#include <atomic>
extern "C" void* cleanup_unrolled_create(size_t,size_t);
extern "C" void cleanup_unrolled_destroy(void*);
extern "C" int cleanup_unrolled_frame(void*,const double*,double*);
namespace {
using Z=std::complex<double>;
constexpr double pi=3.141592653589793238462643383279502884;
constexpr int offsets[7]={-379,-241,-64,0,83,214,427};
void check(bfft_status s){if(s!=BFFT_OK)throw std::runtime_error(bfft_status_string(s));}
using FFT=CleanupGuideFFT;
#if CLEANUP_EARLY_RING_FORWARD
using PrunedGuide=cleanup_ring_research::Early48;
#else
using PrunedGuide=cleanup_ring_research::Pruned48;
#endif
#ifndef CLEANUP_PROFILE_GUIDE
#define CLEANUP_PROFILE_GUIDE 0
#endif
uint64_t stamp(){
 if constexpr(CLEANUP_PROFILE_GUIDE)return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
 return 0;
}
int reflect(int i,int n,bool whole=false){
 if(n==1)return 0; int period=whole?2*n-2:2*n;i%=period;if(i<0)i+=period;return i<n?i:(whole?period-i:period-1-i);
}
double sample(const double* a,int h,int w,double y,double x){
 int iy=int(std::floor(y)),ix=int(std::floor(x));double fy=y-iy,fx=x-ix;
 int y0=reflect(iy,h),y1=reflect(iy+1,h),x0=reflect(ix,w),x1=reflect(ix+1,w);
 return (a[y0*w+x0]*(1-fx)+a[y0*w+x1]*fx)*(1-fy)+(a[y1*w+x0]*(1-fx)+a[y1*w+x1]*fx)*fy;
}
struct Guide {
 size_t n,h,w,hop,count;long origin;int extent;size_t lanes;
 bool registered=true;
 std::unique_ptr<void,decltype(&cleanup_unrolled_destroy)> unrolled{nullptr,cleanup_unrolled_destroy};FFT fft;
 std::vector<double> stack,views,sorted,temp,frame,column,pa,pb,window,rings,mass,nx,ny,gauss_y,gauss_x,chart_y,chart_x,reference_patches,joint,real_line;
 std::vector<Z> sa,sb,corr,line,reference_spectra,unit;
 std::unique_ptr<cleanup_ring::RingForward48> ring_forward;
 std::unique_ptr<PrunedGuide> ring_pruned;
 std::vector<double> ring_fields;
 std::vector<int> cy,cx;
 std::vector<double> fused;
 std::vector<double> chart_mass,chart_dx,chart_dy;
 std::vector<double> atlas_mass,atlas_dx,atlas_dy;
 std::atomic<int> next_job{0};
 const double* observation_audio=nullptr;
 size_t observation_samples=0,observation_advance=0,observation_columns=0;
 bool observation_shifted=false;
 size_t chart_jobs=0,warp_jobs=0;
 std::vector<std::unique_ptr<Guide>> workers;
 Guide* reference_owner=nullptr;
 std::thread worker_thread;
 std::mutex job_mutex;std::condition_variable job_ready;
 bool pending=false,done=true,stopping=false;int job_first=0,job_last=0,job_error=0,job_kind=0;
 double reference_level=0.;
 const double* job_stack=nullptr;const double* job_views=nullptr;
 std::vector<double> previous_audio,previous_raw;
 bool previous_valid=false;size_t reused_frames=0;
 uint64_t timing[3]{};
 uint64_t work_detail[8]{};
 uint64_t lane_timing[3]{}; // view preparation, reference, registration/warp
 Guide(size_t aperture,size_t rows,size_t cols,size_t step,long start,Guide* owner=nullptr,bool registration=true):n(aperture),h(rows),w(cols),hop(step),count(rows*cols),origin(start),extent(std::max(12,int(std::min({size_t(48),rows,cols})))),lanes(owner?owner->lanes:std::max(1u,std::min(unsigned(CLEANUP_GUIDE_THREADS),std::thread::hardware_concurrency()))),registered(registration),fft(extent),reference_owner(owner){
  const bool child=owner!=nullptr;
  if(n<4||n>8192||h<12||h>2*(n-1)||w<12||w>2048||!hop||hop>8192||count>2000000)throw std::runtime_error("invalid registered guide geometry");
  if(!registered){
   lanes=1;frame.resize(n);column.resize(h);
   previous_audio.resize(w*hop);previous_raw.resize(count);
   unrolled.reset(cleanup_unrolled_create(n,h));if(!unrolled)throw std::runtime_error("unrolled plan failed");
   return;
  }
  sorted.resize(count);temp.resize(count);
  if(!child){stack.resize(7*count);views.resize(7*count);
   previous_audio.resize(w*hop);previous_raw.resize(7*count);}
  fused.resize(count);pa.resize(extent*extent);pb.resize(extent*extent);window.resize(extent*extent);
  rings.resize(7*extent*extent);sa.resize(extent*extent);sb.resize(extent*extent);corr.resize(extent*extent);line.resize(extent);real_line.resize(extent);
  mass.resize(count);nx.resize(count);ny.resize(count);
  for(int y=0;y<extent;++y)for(int x=0;x<extent;++x){int q=y*extent+x;
   window[q]=(.5-.5*std::cos(2*pi*y/(extent-1)))*(.5-.5*std::cos(2*pi*x/(extent-1)));
   double fy=double(y<(extent+1)/2?y:y-extent)/extent,fx=double(x<(extent+1)/2?x:x-extent)/extent;
   double r=std::hypot(fx,fy);for(int ring=0;ring<7;++ring)rings[ring*extent*extent+q]=std::exp(-.5*std::pow((r-(.035+(.46-.035)*ring/6))/.055,2));
  }
  for(int j=8;j<int(h);j+=16)cy.push_back(j);if(cy.empty())cy.push_back(int(h)/2);if(cy.back()<int(h)-9)cy.push_back(int(h)-9);
#if CLEANUP_EARLY_RING_FORWARD || CLEANUP_PRUNED_RING_FORWARD
  if(extent==48)ring_pruned=std::make_unique<PrunedGuide>(rings.data());
#elif CLEANUP_FUSED_RING_FORWARD
  if(extent==48){ring_forward=std::make_unique<cleanup_ring::RingForward48>(rings.data());ring_fields.resize(7*extent*extent);}
#endif
  for(int j=8;j<int(w);j+=16)cx.push_back(j);if(cx.empty())cx.push_back(int(w)/2);if(cx.back()<int(w)-9)cx.push_back(int(w)-9);
  if(!child){reference_patches.resize(cy.size()*cx.size()*extent*extent);reference_spectra.resize(reference_patches.size());}
  chart_y.resize(cy.size()*h);chart_x.resize(cx.size()*w);joint.resize(extent*extent);unit.resize(extent*extent);
  chart_mass.resize(cy.size()*w);chart_dx.resize(cy.size()*w);chart_dy.resize(cy.size()*w);
#if CLEANUP_DYNAMIC_GUIDE
  if(!child){atlas_mass.resize(6*cy.size()*w);atlas_dx.resize(atlas_mass.size());atlas_dy.resize(atlas_mass.size());}
#endif
  const double sigma=.42*extent;
  for(size_t c=0;c<cy.size();++c)for(size_t y=0;y<h;++y)chart_y[c*h+y]=std::exp(-.5*std::pow((double(y)-cy[c])/sigma,2));
  for(size_t c=0;c<cx.size();++c)for(size_t x=0;x<w;++x)chart_x[c*w+x]=std::exp(-.5*std::pow((double(x)-cx[c])/sigma,2));
  gauss_y.resize(9);gauss_x.resize(25);
  for(int axis=0;axis<2;++axis){auto& g=axis?gauss_x:gauss_y;double s=axis?3.:1.,sum=0.;int radius=int(g.size()/2);for(int j=-radius;j<=radius;++j)sum+=(g[j+radius]=std::exp(-.5*j*j/(s*s)));for(double& v:g)v/=sum;}
  if(!child||CLEANUP_PARALLEL_OBSERVATIONS){
   frame.resize(n);column.resize(h);
   unrolled.reset(cleanup_unrolled_create(n,h));if(!unrolled)throw std::runtime_error("unrolled plan failed");
  }
  if(child)worker_thread=std::thread([this]{worker_loop();});
  else{
   for(size_t j=1;j<lanes;++j)workers.push_back(std::make_unique<Guide>(n,h,w,hop,origin,this));
  }
 }
 ~Guide(){
  {std::lock_guard<std::mutex> lock(job_mutex);stopping=true;}
  job_ready.notify_all();if(worker_thread.joinable())worker_thread.join();
 }
 void worker_loop(){
  for(;;){
   std::unique_lock<std::mutex> lock(job_mutex);job_ready.wait(lock,[this]{return pending||stopping;});if(stopping)return;
   const double* data=job_stack;const double* view=job_views;int first=job_first,last=job_last,kind=job_kind;pending=false;lock.unlock();
   auto began=stamp();
   int error=0;try{
    if(kind==1)prepare_reference(view+3*count,first,last);
    else if(kind==2)prepare_views(first,last);
    else if(kind==3)dynamic_charts();
    else if(kind==4)dynamic_warp();
    else if(kind==5)dynamic_views();
    else if(kind==6)dynamic_observations();
    else merge_fields(data,view,first,last,fused.data());
   }catch(...){error=1;}
   if(kind==4)lane_timing[2]+=stamp()-began;
   else lane_timing[(kind==2||kind==5)?0:kind==1?1:2]=stamp()-began;
   lock.lock();job_error=error;done=true;lock.unlock();job_ready.notify_one();
  }
 }
 void dispatch(const double* data,const double* view,int first,int last,int kind=0){
  {std::lock_guard<std::mutex> lock(job_mutex);job_stack=data;job_views=view;job_first=first;job_last=last;job_kind=kind;done=false;pending=true;job_error=0;}job_ready.notify_one();
 }
 void finish(){std::unique_lock<std::mutex> lock(job_mutex);job_ready.wait(lock,[this]{return done;});if(job_error)throw std::runtime_error("Registered guide worker failed");}

 double quantile(const double* values,double p,bool positive=false){
  size_t total=0;for(size_t j=0;j<count;++j)if(!positive||values[j]>0)sorted[total++]=values[j];if(!total)return 1.;
  double at=p*(total-1);size_t lo=size_t(at);
  // Select exactly the same two order statistics as a full sort. Registration
  // needs their interpolated quantile, not an ordering of every other value.
  std::nth_element(sorted.begin(),sorted.begin()+lo,sorted.begin()+total);
  double low=sorted[lo];if(lo+1==total)return low;
  double high=*std::min_element(sorted.begin()+lo+1,sorted.begin()+total);
  return low+(at-lo)*(high-low);
 }
 void perceptual(const double* field,double* out){
  double scale=std::max(quantile(field,.75,true),std::numeric_limits<double>::min());
  for(size_t j=0;j<count;++j)out[j]=std::log1p(std::max(field[j],0.)/scale);
  for(size_t y=0;y<h;++y)for(size_t x=0;x<w;++x){double v=0;for(int k=-4;k<=4;++k)v+=out[reflect(int(y)+k,int(h))*w+x]*gauss_y[k+4];temp[y*w+x]=v;}
  for(size_t y=0;y<h;++y)for(size_t x=0;x<w;++x){double v=0;for(int k=-12;k<=12;++k)v+=temp[y*w+reflect(int(x)+k,int(w))]*gauss_x[k+12];out[y*w+x]=v;}
 }
 // Real images and Hermitian cross spectra need only half of the columns.
 // Complete the conjugate half exactly, rather than evaluating redundant DFTs.
 void fft2(std::vector<Z>& a,bool inverse=false){
  if(!inverse){
   for(int y=0;y<extent;++y){for(int x=0;x<extent;++x)real_line[x]=a[y*extent+x].real();fft.forward_real(real_line.data(),a.data()+y*extent);}
   for(int x=0;x<=extent/2;++x){for(int y=0;y<extent;++y)line[y]=a[y*extent+x];fft.run(line.data(),false);for(int y=0;y<extent;++y)a[y*extent+x]=line[y];}
   for(int y=0;y<extent;++y)for(int x=extent/2+1;x<extent;++x)a[y*extent+x]=std::conj(a[((extent-y)%extent)*extent+extent-x]);
  }else{
   for(int x=0;x<=extent/2;++x){
    for(int y=0;y<extent;++y)line[y]=a[y*extent+x];
    // DC and the even-size Nyquist column are themselves Hermitian.
    if(x==0 || (extent%2==0 && x==extent/2)){
     fft.inverse_real(line.data(),real_line.data());for(int y=0;y<extent;++y)a[y*extent+x]=real_line[y];
    }else{fft.run(line.data(),true);for(int y=0;y<extent;++y)a[y*extent+x]=line[y];}
   }
   for(int y=0;y<extent;++y){for(int x=extent/2+1;x<extent;++x)a[y*extent+x]=std::conj(a[y*extent+extent-x]);fft.inverse_real(a.data()+y*extent,real_line.data());for(int x=0;x<extent;++x)a[y*extent+x]=real_line[x];}
  }
 }
 void spectrum(const std::vector<double>& p,std::vector<Z>& out){double mean=0;for(double v:p)mean+=v;mean/=p.size();for(size_t j=0;j<p.size();++j)out[j]=(p[j]-mean)*window[j];fft2(out);}
 void translation(double& dx,double& dy,double& dispersion){
  auto start=stamp();spectrum(pb,sb);work_detail[1]+=stamp()-start;
  double weights[7],vx[7],vy[7],total=0;int e2=extent*extent;
  start=stamp();
  // Finite log-compressed registration patches have bounded squared spectra.
  // Direct norms avoid millions of general overflow-scaled hypot calls.
  // Both patch images are real. Opposite Fourier cells are conjugates,
  // including across the frequency axis: their ring/joint energies are equal.
  // Evaluate the nonnegative time-frequency half once; fft2 already completes
  // its conjugates during inversion. Endpoint columns have multiplicity one.
  for(int y=0;y<extent;++y)for(int x=0;x<=extent/2;++x){int j=y*extent+x;
   Z c=sb[j]*std::conj(sa[j]);unit[j]=c/std::max(std::sqrt(std::norm(c)),1e-12);
   double multiplicity=(x==0||(extent%2==0&&x==extent/2))?1.:2.;
   joint[j]=multiplicity*std::sqrt(std::sqrt(std::norm(sa[j]))*std::sqrt(std::norm(sb[j])));
  }
  work_detail[2]+=stamp()-start;
  cleanup_ring_research::Peak pruned_peaks[7];
  if(ring_forward||ring_pruned){
   start=stamp();
   // Complete the COMMON cross spectrum once. The full bank emits all pixels;
   // the experimental query bank returns only the required peak decisions.
   for(int y=0;y<extent;++y)for(int x=extent/2+1;x<extent;++x)unit[y*extent+x]=std::conj(unit[((extent-y)%extent)*extent+extent-x]);
   if(ring_pruned){
    bool consensus=ring_pruned->run(unit.data(),pruned_peaks,true);work_detail[4]+=stamp()-start;
    // All weighted positions and their dispersion are identically zero,
    // including the total-weight floor. Confidence cannot affect this chart.
    if(consensus){dx=dy=dispersion=0.;return;}
   }else{ring_forward->run(unit.data(),ring_fields.data());work_detail[4]+=stamp()-start;}
  }
  for(int ring=0;ring<7;++ring){double energy=0;
   start=stamp();
   for(int y=0;y<extent;++y)for(int x=0;x<=extent/2;++x){int j=y*extent+x;double mask=rings[ring*e2+j];if(!ring_forward&&!ring_pruned)corr[j]=unit[j]*mask;energy+=mask*joint[j];}
   work_detail[3]+=stamp()-start;
   if(!ring_forward&&!ring_pruned){start=stamp();fft2(corr,true);work_detail[4]+=stamp()-start;}
   start=stamp();int best;double competitor,peak;
   if(ring_pruned){const auto& p=pruned_peaks[ring];best=p.best;competitor=p.competitor;peak=p.peak;}
   else if(ring_forward){auto values=ring_fields.data()+ring*e2;cleanup_phase_peak(values,extent,best,competitor);peak=values[best];}
   else{cleanup_phase_peak(corr.data(),extent,best,competitor);peak=corr[best].real();}
   int py=best/extent,px=best%extent;
   double ambiguity=std::clamp(competitor/std::max(peak,1e-12),0.,1.);
   weights[ring]=energy*std::max((1-ambiguity)*(1-ambiguity),1e-4);total+=weights[ring];vx[ring]=px<=extent/2?px:px-extent;vy[ring]=py<=extent/2?py:py-extent;
   work_detail[5]+=stamp()-start;
  }
  dx=dy=dispersion=0;total=std::max(total,1e-12);for(int j=0;j<7;++j){weights[j]/=total;dx+=weights[j]*vx[j];dy+=weights[j]*vy[j];}
  for(int j=0;j<7;++j)dispersion+=weights[j]*(std::pow(vx[j]-dx,2)+std::pow(vy[j]-dy,2));dispersion=std::sqrt(dispersion);
 }
 void prepare_reference(const double* a,int first,int last){
  const auto start=stamp();
  auto& owner=reference_owner?*reference_owner:*this;
  // All six offset registrations share the SAME reference spectra. Each lane
  // computes a disjoint range once, then a barrier makes the arrays read-only.
  for(int chart=first;chart<last;++chart){int y=cy[chart/cx.size()],x=cx[chart%cx.size()];
   for(int j=0;j<extent;++j)for(int k=0;k<extent;++k)pa[j*extent+k]=a[reflect(y+j-extent/2,int(h),true)*w+reflect(x+k-extent/2,int(w),true)];
   spectrum(pa,sa);std::copy(pa.begin(),pa.end(),owner.reference_patches.begin()+chart*pa.size());std::copy(sa.begin(),sa.end(),owner.reference_spectra.begin()+chart*sa.size());
  }
  work_detail[0]=stamp()-start;
 }
 void estimate_chart_rows(const double* b,size_t first,size_t last){
  const auto& reference=reference_owner?*reference_owner:*this;
  std::fill(chart_mass.begin()+first*w,chart_mass.begin()+last*w,0.);
  std::fill(chart_dx.begin()+first*w,chart_dx.begin()+last*w,0.);
  std::fill(chart_dy.begin()+first*w,chart_dy.begin()+last*w,0.);
  int margin=std::max(2,extent/10);
  for(size_t yi=first;yi<last;++yi)for(size_t xi=0;xi<cx.size();++xi){int y=cy[yi],x=cx[xi];size_t chart_index=yi*cx.size()+xi;
   std::copy_n(reference.reference_patches.data()+chart_index*pa.size(),pa.size(),pa.data());std::copy_n(reference.reference_spectra.data()+chart_index*sa.size(),sa.size(),sa.data());
   for(int j=0;j<extent;++j)for(int k=0;k<extent;++k){int q=j*extent+k;size_t i=reflect(y+j-extent/2,int(h),true)*w+reflect(x+k-extent/2,int(w),true);pb[q]=b[i];}
   double dx,dy,disp;translation(dx,dy,disp);double zero=0,aligned=0;
   // One translation applies to the whole patch. Reuse its reflected indices
   // and fractional weights along each axis instead of deriving them per cell.
   // Evaluate j+dy / k+dx exactly as sample() did, preserving rounding/order.
   int x0[48],x1[48];double fx[48];
   for(int k=margin;k<extent-margin;++k){double x=k+dx;int ix=int(std::floor(x));fx[k]=x-ix;x0[k]=reflect(ix,extent);x1[k]=reflect(ix+1,extent);}
   for(int j=margin;j<extent-margin;++j){double y=j+dy;int iy=int(std::floor(y));double fy=y-iy;int y0=reflect(iy,extent),y1=reflect(iy+1,extent);
    for(int k=margin;k<extent-margin;++k){
     double value=(pb[y0*extent+x0[k]]*(1-fx[k])+pb[y0*extent+x1[k]]*fx[k])*(1-fy)+(pb[y1*extent+x0[k]]*(1-fx[k])+pb[y1*extent+x1[k]]*fx[k])*fy;
     zero+=std::pow(pb[j*extent+k]-pa[j*extent+k],2);aligned+=std::pow(value-pa[j*extent+k],2);
    }
   }
   double cells=(extent-2*margin)*(extent-2*margin);zero/=cells;aligned/=cells;
   double improvement=std::clamp((zero-aligned)/std::max(zero,1e-8),0.,1.);
   double chart=std::exp(-std::pow(disp/(.5+std::hypot(dx,dy)),2))*(.1+.9*std::sqrt(improvement));
   // The Gaussian chart window is separable. Sum its time factors once per
   // chart row, then expand along frequency once per row (not once per chart).
   // Keep all nonzero Gaussian tails; no changed aperture, charts or cutoff.
   for(size_t k=0;k<w;++k){size_t q=yi*w+k;double weight=chart*chart_x[xi*w+k];chart_mass[q]+=weight;chart_dx[q]+=weight*dx;chart_dy[q]+=weight*dy;}
  }
 }
 void registration(const double* b){
  std::fill(mass.begin(),mass.end(),0.);std::fill(nx.begin(),nx.end(),0.);std::fill(ny.begin(),ny.end(),0.);
  estimate_chart_rows(b,0,cy.size());
  for(size_t yi=0;yi<cy.size();++yi)for(size_t j=0;j<h;++j){
   double weight=chart_y[yi*h+j];if(weight==0.)continue;
   for(size_t k=0;k<w;++k){size_t q=j*w+k,r=yi*w+k;mass[q]+=weight*chart_mass[r];nx[q]+=weight*chart_dx[r];ny[q]+=weight*chart_dy[r];}
  }
 }
 void merge_fields(const double* data,const double* view,int first,int last,double* output){
  constexpr int lattice[6]={0,1,2,4,5,6};
  std::fill_n(work_detail+1,7,0);
  for(size_t y=0;y<h;++y)for(size_t x=0;x<w;++x)output[x*h+y]=data[3*count+y*w+x];
  for(int index=first;index<last;++index){int l=lattice[index];auto start=stamp();
   uint64_t previous=0;for(int k=1;k<=5;++k)previous+=work_detail[k];
   registration(view+l*count);
   uint64_t current=0;for(int k=1;k<=5;++k)current+=work_detail[k];
   work_detail[6]+=stamp()-start-(current-previous);start=stamp();
   for(size_t y=0;y<h;++y)for(size_t x=0;x<w;++x){size_t q=y*w+x;double m=std::max(mass[q],std::numeric_limits<double>::min());double value=sample(data+l*count,int(h),int(w),y+ny[q]/m,x+nx[q]/m);output[x*h+y]=std::max(output[x*h+y],value);}
   work_detail[7]+=stamp()-start;
  }
 }
 // A task owns a complete chart row, preserving the xi accumulation order.
 // Workers publish disjoint atlas rows; the barrier precedes all atlas reads.
 void dynamic_charts(){
  auto& owner=reference_owner?*reference_owner:*this;
  constexpr int lattice[6]={0,1,2,4,5,6};
  std::fill_n(work_detail+1,7,0);chart_jobs=0;
  for(int task;(task=owner.next_job.fetch_add(1,std::memory_order_relaxed))<int(6*cy.size());){
   const int index=task%6;const size_t yi=size_t(task/6);auto start=stamp();
   uint64_t before=0;for(int k=1;k<=5;++k)before+=work_detail[k];
   estimate_chart_rows(owner.views.data()+lattice[index]*count,yi,yi+1);
   const size_t at=(index*cy.size()+yi)*w;
   std::copy_n(chart_mass.data()+yi*w,w,owner.atlas_mass.data()+at);
   std::copy_n(chart_dx.data()+yi*w,w,owner.atlas_dx.data()+at);
   std::copy_n(chart_dy.data()+yi*w,w,owner.atlas_dy.data()+at);
   uint64_t after=0;for(int k=1;k<=5;++k)after+=work_detail[k];
   work_detail[6]+=stamp()-start-(after-before);++chart_jobs;
  }
 }
 // A task owns output rows across all views. It reduces chart rows and views
 // in exactly their original order; scheduling never changes a sample's sum.
 void dynamic_warp(){
  auto& owner=reference_owner?*reference_owner:*this;
  constexpr int lattice[6]={0,1,2,4,5,6};constexpr size_t stripe=16;
  warp_jobs=0;
  for(int task;(task=owner.next_job.fetch_add(1,std::memory_order_relaxed))<int((h+stripe-1)/stripe);){
   size_t first=size_t(task)*stripe,last=std::min(h,first+stripe);
   for(size_t y=first;y<last;++y)for(size_t x=0;x<w;++x)owner.fused[x*h+y]=owner.stack[3*count+y*w+x];
   for(int index=0;index<6;++index){auto start=stamp();
    std::fill(mass.begin()+first*w,mass.begin()+last*w,0.);
    std::fill(nx.begin()+first*w,nx.begin()+last*w,0.);std::fill(ny.begin()+first*w,ny.begin()+last*w,0.);
    const size_t base=index*cy.size()*w;
    for(size_t yi=0;yi<cy.size();++yi)for(size_t y=first;y<last;++y){double weight=chart_y[yi*h+y];if(weight==0.)continue;
     for(size_t x=0;x<w;++x){size_t q=y*w+x,r=base+yi*w+x;
      mass[q]+=weight*owner.atlas_mass[r];nx[q]+=weight*owner.atlas_dx[r];ny[q]+=weight*owner.atlas_dy[r];}}
    work_detail[6]+=stamp()-start;start=stamp();
    for(size_t y=first;y<last;++y)for(size_t x=0;x<w;++x){size_t q=y*w+x;double m=std::max(mass[q],std::numeric_limits<double>::min());
     double value=sample(owner.stack.data()+lattice[index]*count,int(h),int(w),y+ny[q]/m,x+nx[q]/m);
     owner.fused[x*h+y]=std::max(owner.fused[x*h+y],value);}
    work_detail[7]+=stamp()-start;
   }
   ++warp_jobs;
  }
 }
 void dynamic_views(){
  auto& owner=reference_owner?*reference_owner:*this;
  for(int task;(task=owner.next_job.fetch_add(1,std::memory_order_relaxed))<7;)prepare_views(task,task+1);
 }
 void prepare_views(int first,int last){
  auto& owner=reference_owner?*reference_owner:*this;
  for(int l=first;l<last;++l){double* field=owner.stack.data()+l*count;
   double scale=owner.reference_level/std::max(quantile(field,.995),1e-30);
   for(size_t j=0;j<count;++j)field[j]*=scale;
   perceptual(field,owner.views.data()+l*count);
  }
 }
 void fuse(const double* observations,double* output){
  if(!registered){
   for(size_t t=0;t<w;++t)for(size_t b=0;b<h;++b)output[t*h+b]=observations[3*count+b*w+t];
   return;
  }
  auto started=std::chrono::steady_clock::now();timing[0]=0;
  if(observations!=stack.data())std::copy_n(observations,7*count,stack.data());reference_level=std::max(quantile(stack.data()+3*count,.995),1e-30);
#if CLEANUP_DYNAMIC_GUIDE
  next_job.store(0,std::memory_order_relaxed);
  for(auto& worker:workers)worker->dispatch(stack.data(),views.data(),0,0,5);
#else
  for(size_t i=0;i<workers.size();++i)workers[i]->dispatch(stack.data(),views.data(),int(7*(i+1)/lanes),int(7*(i+2)/lanes),2);
#endif
  auto lane_started=stamp();
  int view_error=0;try{
#if CLEANUP_DYNAMIC_GUIDE
   dynamic_views();
#else
   prepare_views(0,7/lanes);
#endif
  }catch(...){view_error=1;}
  lane_timing[0]=stamp()-lane_started;
  for(auto& worker:workers)try{worker->finish();}catch(...){view_error=1;}
  if(view_error)throw std::runtime_error("Registered view preparation failed");
  auto prepared=std::chrono::steady_clock::now();timing[1]=std::chrono::duration_cast<std::chrono::nanoseconds>(prepared-started).count();
  const int charts=int(cy.size()*cx.size());
  for(size_t i=0;i<workers.size();++i)workers[i]->dispatch(stack.data(),views.data(),int(charts*(i+1)/lanes),int(charts*(i+2)/lanes),true);
  int reference_error=0;
  lane_started=stamp();
  try{prepare_reference(views.data()+3*count,0,charts/lanes);}catch(...){reference_error=1;}
  lane_timing[1]=stamp()-lane_started;
  for(auto& worker:workers)try{worker->finish();}catch(...){reference_error=1;}
  if(reference_error)throw std::runtime_error("Registered reference preparation failed");
#if CLEANUP_DYNAMIC_GUIDE
  for(int kind:{3,4}){
   next_job.store(0,std::memory_order_relaxed);
   for(auto& worker:workers)worker->dispatch(stack.data(),views.data(),0,0,kind);
   lane_started=stamp();int error=0;
   try{if(kind==3)dynamic_charts();else dynamic_warp();}catch(...){error=1;}
   if(kind==3)lane_timing[2]=stamp()-lane_started;else lane_timing[2]+=stamp()-lane_started;
   for(auto& worker:workers)try{worker->finish();}catch(...){error=1;}
   if(error)throw std::runtime_error("Dynamic registered guide processing failed");
  }
  std::copy(fused.begin(),fused.end(),output);
#else
  for(size_t i=0;i<workers.size();++i)workers[i]->dispatch(stack.data(),views.data(),int(6*(i+1)/lanes),int(6*(i+2)/lanes));
  // Even on a rare FFT failure, join the current jobs before caller data can move.
  lane_started=stamp();
  int error=0;try{merge_fields(stack.data(),views.data(),0,6/lanes,output);}catch(...){error=1;}
  lane_timing[2]=stamp()-lane_started;
  for(auto& worker:workers){try{worker->finish();}catch(...){error=1;}for(size_t j=0;j<count;++j)output[j]=std::max(output[j],worker->fused[j]);}
  if(error)throw std::runtime_error("Registered guide processing failed");
#endif
  timing[2]=std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now()-prepared).count();
 }
 void observe_lattice(int l){
  auto& owner=reference_owner?*reference_owner:*this;
  for(size_t t=0;t<w;++t){
   long start=origin+long(t*hop)+offsets[l]-long(n/2);
   // Reuse only apertures wholly inside the common sample interval. Edge
   // observations have different zero padding in the two contexts and MUST be
   // recomputed. The byte comparison proves the common samples are identical.
   if(owner.observation_shifted&&t+owner.observation_columns<w&&start>=0&&size_t(start)+n<=owner.observation_samples-owner.observation_advance){
    for(size_t b=0;b<h;++b)owner.stack[(l*h+b)*w+t]=owner.previous_raw[(l*h+b)*w+t+owner.observation_columns];
    ++reused_frames;
   }else{
    for(size_t j=0;j<n;++j){long at=start+long(j);frame[j]=(at>=0&&at<long(owner.observation_samples))?owner.observation_audio[at]:0.;}
    if(cleanup_unrolled_frame(unrolled.get(),frame.data(),column.data()))throw std::runtime_error("unrolled failed");
    for(size_t b=0;b<h;++b)owner.stack[(l*h+b)*w+t]=column[b];
   }
  }
 }
 void dynamic_observations(){
  auto& owner=reference_owner?*reference_owner:*this;reused_frames=0;
  // Each task owns a whole row-major lattice, avoiding cache-line contention
  // between workers writing neighboring time columns in every frequency row.
  for(int l;(l=owner.next_job.fetch_add(1,std::memory_order_relaxed))<7;)observe_lattice(l);
 }
 void run_reference(const double* audio,size_t samples,double* output,size_t advance){
  const auto started=std::chrono::steady_clock::now();reused_frames=0;
  const bool can_cache=samples==previous_audio.size();
  const bool shifted=can_cache&&previous_valid&&advance>0&&advance<samples&&advance%hop==0&&
   std::memcmp(audio,previous_audio.data()+advance,(samples-advance)*sizeof(double))==0;
  const size_t columns=advance/hop;
  for(size_t t=0;t<w;++t){
   const long start=origin+long(t*hop)-long(n/2);
   if(shifted&&t+columns<w&&start>=0&&size_t(start)+n<=samples-advance){
    std::copy_n(previous_raw.data()+(t+columns)*h,h,output+t*h);++reused_frames;
   }else{
    for(size_t j=0;j<n;++j){long at=start+long(j);frame[j]=(at>=0&&at<long(samples))?audio[at]:0.;}
    if(cleanup_unrolled_frame(unrolled.get(),frame.data(),output+t*h))throw std::runtime_error("unrolled failed");
   }
  }
  previous_valid=can_cache;
  if(can_cache){std::copy_n(audio,samples,previous_audio.data());std::copy_n(output,count,previous_raw.data());}
  timing[0]=std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now()-started).count();
  timing[1]=timing[2]=0;
 }
 void run(const double* audio,size_t samples,double* output,size_t advance=0){
  if(!registered){run_reference(audio,samples,output,advance);return;}
  const auto started=std::chrono::steady_clock::now();
  reused_frames=0;
  const bool can_cache=samples==previous_audio.size();
  const bool shifted=can_cache&&previous_valid&&advance>0&&advance<samples&&advance%hop==0&&
   std::memcmp(audio,previous_audio.data()+advance,(samples-advance)*sizeof(double))==0;
  const size_t columns=advance/hop;
  observation_audio=audio;observation_samples=samples;observation_advance=advance;
  observation_columns=columns;observation_shifted=shifted;
#if CLEANUP_PARALLEL_OBSERVATIONS
  next_job.store(0,std::memory_order_relaxed);
  for(auto& worker:workers)worker->dispatch(nullptr,nullptr,0,0,6);
  int error=0;try{dynamic_observations();}catch(...){error=1;}
  for(auto& worker:workers){try{worker->finish();}catch(...){error=1;}reused_frames+=worker->reused_frames;}
  if(error)throw std::runtime_error("Parallel observation construction failed");
#else
  for(int l=0;l<7;++l)observe_lattice(l);
#endif
  previous_valid=can_cache;
  if(can_cache){std::copy_n(audio,samples,previous_audio.data());std::copy(stack.begin(),stack.end(),previous_raw.begin());}
  // Quantiles, registration and fusion still use the entire current context.
  const auto observed=std::chrono::steady_clock::now();
  fuse(stack.data(),output);
  timing[0]=std::chrono::duration_cast<std::chrono::nanoseconds>(observed-started).count();
 }
};
}
#ifdef _WIN32
#define EXPORT extern "C" __declspec(dllexport)
#else
#define EXPORT extern "C" __attribute__((visibility("default")))
#endif
EXPORT void* cleanup_guide_create(size_t n,size_t rows,size_t cols,size_t hop,long origin){try{return new Guide(n,rows,cols,hop,origin);}catch(...){return nullptr;}}
EXPORT void* cleanup_guide_create_mode(size_t n,size_t rows,size_t cols,size_t hop,long origin,int registration){if(registration!=0&&registration!=1)return nullptr;try{return new Guide(n,rows,cols,hop,origin,nullptr,registration!=0);}catch(...){return nullptr;}}
EXPORT void cleanup_guide_destroy(void* p){delete static_cast<Guide*>(p);}
EXPORT int cleanup_guide_run(void* p,const double* audio,size_t samples,double* output){if(!p||!audio||!output)return 1;try{static_cast<Guide*>(p)->run(audio,samples,output);return 0;}catch(...){return 2;}}
EXPORT int cleanup_guide_run_stream(void* p,const double* audio,size_t samples,size_t advance,double* output){if(!p||!audio||!output)return 1;try{static_cast<Guide*>(p)->run(audio,samples,output,advance);return 0;}catch(...){return 2;}}
EXPORT size_t cleanup_guide_reused_frames(void* p){return p?static_cast<Guide*>(p)->reused_frames:0;}
EXPORT void cleanup_guide_timings(void* p,uint64_t* out){if(p&&out)std::copy_n(static_cast<Guide*>(p)->timing,3,out);}
EXPORT void cleanup_guide_work_timings(void* p,uint64_t* out){if(p&&out){auto& g=*static_cast<Guide*>(p);std::copy_n(g.work_detail,8,out);for(auto& w:g.workers)for(int k=0;k<8;++k)out[k]+=w->work_detail[k];}}
EXPORT size_t cleanup_guide_lane_count(void* p){return p?1+static_cast<Guide*>(p)->workers.size():0;}
EXPORT void cleanup_guide_lane_timings(void* p,uint64_t* out){if(p&&out){auto& g=*static_cast<Guide*>(p);std::copy_n(g.lane_timing,3,out);for(size_t i=0;i<g.workers.size();++i)std::copy_n(g.workers[i]->lane_timing,3,out+3*(i+1));}}
EXPORT int cleanup_guide_register(void* p,const double* stack,double* output){if(!p||!stack||!output)return 1;try{static_cast<Guide*>(p)->fuse(stack,output);return 0;}catch(...){return 2;}}
