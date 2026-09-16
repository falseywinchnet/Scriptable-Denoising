// Exact finite-Zak/DIP packet projection, modulation, and completion.
// The stage equations follow the user's bfft experiments/dip_numba.py.
// bfft owns the audio RFFT/ODFT and synthesis; this helper transports packet
// members between those boundaries. All storage/tables are supplied by Filter.py.
#include "cleanup.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <utility>
namespace {
struct Z {double r,i;};
// Caller storage is a double array. Access pairs without treating that storage
// as live Z objects (which would violate C++ aliasing/object-lifetime rules).
struct PairRef {
    double* p;
    operator Z() const {return {p[0],p[1]};}
    PairRef& operator=(Z z){p[0]=z.r;p[1]=z.i;return *this;}
};
struct Pairs {
    double* p;
    PairRef operator[](int64_t i) const {return {p+2*i};}
};
Z add(Z a,Z b){return {a.r+b.r,a.i+b.i};}
Z sub(Z a,Z b){return {a.r-b.r,a.i-b.i};}
Z mul(Z a,Z b){return {a.r*b.r-a.i*b.i,a.r*b.i+a.i*b.r};}
Z scale(Z a,double s){return {a.r*s,a.i*s};}
}
extern "C" CLEANUP_API int32_t cleanup_dip_transport(
    const double* source,const double* donors,const double* weights,double* output,
    double* scratch,const double* cosine,const double* sine,int64_t n,int64_t e,int64_t half_bin) {
    if(!source||!donors||!weights||!output||!scratch||!cosine||!sine || n<16||n>8192 ||
       (n&(n-1)) || e<1||e>n/2 || (e&(e-1)) || half_bin<0||half_bin>1)return -1;
    const int64_t q=n/e,bins=n/2+1-half_bin;
    Pairs a{scratch},b{scratch+2*n};
    for(int64_t k=0;k<bins;++k)a[k]={source[2*k],source[2*k+1]};
    for(int64_t k=bins;k<n;++k){int64_t other=n-half_bin-k;a[k]={source[2*other],-source[2*other+1]};}
    // Descend from the final DFT to the chosen packet level. For ODFT this
    // is the DFT of the half-bin-modulated real frame; its mirror is N-1-k.
    for(int64_t span=n/2;span>=e;span/=2){
        int64_t width=n/span,half=width/2;
        for(int64_t d=0;d<span;++d){
            int64_t angle=d*n/(2*span);Z rotation{cosine[angle*q+1],sine[angle*q+1]};
            for(int64_t j=0;j<half;++j){Z lo=a[d*half+j],hi=a[(d+span)*half+j];
                b[d*width+j]=scale(add(lo,hi),.5);
                b[d*width+half+j]=scale(mul(sub(lo,hi),rotation),.5);
            }
        }
        std::swap(a,b);
    }
    for(int64_t j=0;j<n;++j)b[j]=Z{0.,0.};
    for(int64_t k=0;k<bins;++k){
        output[2*k]=output[2*k+1]=0.;
        if(donors[k]<0.)continue;
        if(!std::isfinite(donors[k])||donors[k]>=k)return -2;
        int64_t donor=static_cast<int64_t>(donors[k]);
        if(!std::isfinite(donors[k])||donors[k]!=double(donor)||donor<0||donor>=k ||
           donor%e!=k%e || (!half_bin && k==n/2) ||
           !std::isfinite(weights[2*k])||!std::isfinite(weights[2*k+1]))return -2;
        // Project the existing packet's periodic component. Orthogonality
        // recovers the complex donor coefficient without a magnitude-only fit.
        Z coefficient{0.,0.};int64_t row=donor%e;
        for(int64_t j=0;j<q;++j)
            coefficient=add(coefficient,mul(a[row*q+j],Z{cosine[donor*q+j],-sine[donor*q+j]}));
        coefficient=scale(mul(coefficient,Z{weights[2*k],weights[2*k+1]}),1./q);
        int64_t mirror=n-half_bin-k;
        for(int64_t j=0;j<q;++j){
            b[(k%e)*q+j]=add(b[(k%e)*q+j],mul(coefficient,Z{cosine[k*q+j],sine[k*q+j]}));
            b[(mirror%e)*q+j]=add(b[(mirror%e)*q+j],mul(Z{coefficient.r,-coefficient.i},Z{cosine[mirror*q+j],sine[mirror*q+j]}));
        }
    }
    std::swap(a,b);
    // Finish the DIP walk, returning the added positive-frequency coefficients.
    for(int64_t span=e;span<n;span*=2){
        int64_t width=n/span,half=width/2;
        for(int64_t d=0;d<span;++d){
            int64_t angle=d*n/(2*span);Z rotation{cosine[angle*q+1],-sine[angle*q+1]};
            for(int64_t j=0;j<half;++j){Z lo=a[d*width+j],hi=mul(a[d*width+half+j],rotation);
                b[d*half+j]=add(lo,hi);b[(d+span)*half+j]=sub(lo,hi);
            }
        }
        std::swap(a,b);
    }
    for(int64_t k=0;k<bins;++k){Z value=a[k];output[2*k]=value.r;output[2*k+1]=value.i;}
    return 0;
}
