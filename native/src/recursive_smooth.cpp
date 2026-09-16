// Cleanup's paired recursive 2-D smoothing, with caller-owned fixed workspaces.
// The branches are averaged each round; this is NOT separable sequential blur.
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

#ifdef _WIN32
#define EXPORT extern "C" __declspec(dllexport)
#else
#define EXPORT extern "C" __attribute__((visibility("default")))
#endif

namespace {
bool overlaps(const double* a, size_t an, const double* b, size_t bn) {
    const auto aa=reinterpret_cast<uintptr_t>(a), bb=reinterpret_cast<uintptr_t>(b);
    // Validate arithmetic before forming the half-open byte ranges.
    if(an>std::numeric_limits<uintptr_t>::max()/sizeof(double) ||
       bn>std::numeric_limits<uintptr_t>::max()/sizeof(double)) return true;
    const auto as=an*sizeof(double),bs=bn*sizeof(double);
    if(aa>std::numeric_limits<uintptr_t>::max()-as || bb>std::numeric_limits<uintptr_t>::max()-bs)return true;
    return aa<bb+bs && bb<aa+as;
}
}

// Row-major mask[frames,bins], planes[frames+2*time_pad,bins+2*freq_pad].
// Scratch contains at least max(padded_rows,padded_cols) doubles. All four
// arrays must be distinct/nonoverlapping; the caller owns their capacities.
// Returns 0 on success, -1 invalid geometry/aliasing, -2 nonfinite/negative mask.
// No allocation, Python dispatch, thread creation, or hidden persistent state.
EXPORT int32_t cleanup_recursive_smooth(double* mask,double* vertical,double* horizontal,double* scratch,
                                       int64_t frames,int64_t bins,int64_t nb,
                                       int64_t time_pad,int64_t freq_pad,int64_t iterations) {
    if(!mask||!vertical||!horizontal||!scratch||frames<1||bins<1||nb<0||nb>bins||
       time_pad<0||freq_pad<0||iterations<0)return -1;
    constexpr auto limit=std::numeric_limits<int64_t>::max();
    if(time_pad>(limit-frames)/2 || freq_pad>(limit-bins)/2)return -1;
    const auto rows64=frames+2*time_pad,cols64=bins+2*freq_pad;
    if(rows64>limit/cols64 || frames>limit/bins)return -1;
    if(static_cast<uint64_t>(rows64)>std::numeric_limits<size_t>::max() ||
       static_cast<uint64_t>(cols64)>std::numeric_limits<size_t>::max() ||
       static_cast<uint64_t>(rows64*cols64)>std::numeric_limits<size_t>::max())return -1;
    const size_t rows=static_cast<size_t>(rows64),cols=static_cast<size_t>(cols64);
    const size_t plane=rows*cols,points=static_cast<size_t>(frames*bins),work=std::max(rows,cols);
    if(plane>std::numeric_limits<size_t>::max()/sizeof(double))return -1;
    if(overlaps(mask,points,vertical,plane)||overlaps(mask,points,horizontal,plane)||
       overlaps(mask,points,scratch,work)||overlaps(vertical,plane,horizontal,plane)||
       overlaps(vertical,plane,scratch,work)||overlaps(horizontal,plane,scratch,work))return -1;
    for(size_t j=0;j<points;++j)if(!std::isfinite(mask[j])||mask[j]<0.)return -2;
    std::fill_n(vertical,plane,0.);std::fill_n(horizontal,plane,0.);
    for(int64_t t=0;t<frames;++t){
        const size_t target=static_cast<size_t>(t+time_pad)*cols+static_cast<size_t>(freq_pad);
        std::copy_n(mask+static_cast<size_t>(t*bins),static_cast<size_t>(bins),vertical+target);
        std::copy_n(mask+static_cast<size_t>(t*bins),static_cast<size_t>(bins),horizontal+target);
    }
    const size_t filtered_cols=static_cast<size_t>(nb+2*freq_pad);
    for(int64_t iteration=0;iteration<iterations;++iteration){
        // Frequency branch, using the same center,left,right addition order as
        // Filter.py. A temporary row keeps reads independent of output writes.
        for(size_t t=0;t<rows;++t){
            double* row=vertical+t*cols;
            for(size_t b=0;b<cols;++b){
                double value=row[b];
                if(b>0)value+=row[b-1];
                if(b+1<cols)value+=row[b+1];
                scratch[b]=value/3.;
            }
            std::copy_n(scratch,cols,row);
        }
        // Time branch. Keep horizontal ORIGINAL throughout this sweep so a
        // rolling sum can read its outgoing row after earlier outputs finish.
        // Merge into vertical immediately; copying the merged plane at the end
        // implements the original simultaneous two-branch recurrence exactly.
        std::fill_n(scratch,filtered_cols,0.);
        for(size_t t=0;t<std::min<size_t>(rows,7);++t){
            const double* row=horizontal+t*cols;
            for(size_t b=0;b<filtered_cols;++b)scratch[b]+=row[b];
        }
        for(size_t t=0;t<rows;++t){
            double* out=vertical+t*cols;
            const double* original=horizontal+t*cols;
            for(size_t b=0;b<filtered_cols;++b){
                // Exact arithmetic is nonnegative; a sliding subtraction can
                // leave a few negative ulps at the trailing edge of an impulse.
                const double time_value=std::max(0.,scratch[b])/13.;
                out[b]=(out[b]+time_value)*.5;
            }
            // The legacy temporal branch leaves columns outside this limit
            // untouched; they must still participate in the arithmetic merge.
            for(size_t b=filtered_cols;b<cols;++b)out[b]=(out[b]+original[b])*.5;
            if(t+1<rows){
                if(t>=6){const double* leaving=horizontal+(t-6)*cols;
                    for(size_t b=0;b<filtered_cols;++b)scratch[b]-=leaving[b];}
                if(t+7<rows){const double* entering=horizontal+(t+7)*cols;
                    for(size_t b=0;b<filtered_cols;++b)scratch[b]+=entering[b];}
            }
        }
        std::copy_n(vertical,plane,horizontal);
    }
    for(int64_t t=0;t<frames;++t)
        std::copy_n(vertical+static_cast<size_t>(t+time_pad)*cols+static_cast<size_t>(freq_pad),
                    static_cast<size_t>(bins),mask+static_cast<size_t>(t*bins));
    return 0;
}
