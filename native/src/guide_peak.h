#pragma once
#include <algorithm>
#include <cmath>
#include <complex>
#include <limits>

// Exact first-argmax and strongest competitor outside a wrapped 7x7 square.
// Registration charts are finite and at most 48 by 48. Cache each row maximum
// during argmax: only the seven intersecting rows need a second cell scan.
inline double cleanup_real_value(double value){return value;}
inline double cleanup_real_value(std::complex<double> value){return value.real();}
template <typename Value>
inline void cleanup_phase_peak(const Value* values,int extent,
                               int& best,double& competitor){
 double row_peak[48];
 best=0;double peak=cleanup_real_value(values[0]);
 for(int y=0;y<extent;++y){
  double maximum=cleanup_real_value(values[y*extent]);int column=0;
  for(int x=1;x<extent;++x){double v=cleanup_real_value(values[y*extent+x]);if(v>maximum){maximum=v;column=x;}}
  row_peak[y]=maximum;
  if(maximum>peak){peak=maximum;best=y*extent+column;}
 }
 int py=best/extent,px=best%extent;
 int allowed[48],count=0;
 for(int x=0;x<extent;++x){int d=std::abs(x-px);if(std::min(d,extent-d)>3)allowed[count++]=x;}
 competitor=-std::numeric_limits<double>::infinity();
 for(int y=0;y<extent;++y){int d=std::abs(y-py);
  if(std::min(d,extent-d)>3)competitor=std::max(competitor,row_peak[y]);
  else for(int k=0;k<count;++k)competitor=std::max(competitor,cleanup_real_value(values[y*extent+allowed[k]]));
 }
}
