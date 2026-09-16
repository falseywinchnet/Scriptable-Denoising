// Exact low rows of abs(irfft(irfft(rfft(window * frame)))).
// The first forward/inverse cancel. The second inverse is a scaled DCT-I,
// length 2*(N-1), evaluated by a Bluestein convolution using bfft only.
// This is the literal cosine observation, not a hypot replacement.
#include <bfft/bfft.h>
#include <algorithm>
#include <cmath>
#include <complex>
#include <memory>
#include <stdexcept>
#include <vector>

namespace {
using Z = std::complex<double>;
constexpr double pi = 3.141592653589793238462643383279502884;
void check(bfft_status s) { if (s != BFFT_OK) throw std::runtime_error(bfft_status_string(s)); }
struct Plan {
    size_t n, rows, length = 1;
    bfft_plan* fft = nullptr;
    bfft_workspace* work = nullptr;
    std::vector<double> real, imag, window;
    std::vector<bfft_complex> re_fft, im_fft;
    std::vector<Z> a, spectrum, kernel, chirp;
    Plan(size_t count, size_t crop): n(count), rows(crop) {
        if (n < 4 || n > 8192 || rows < 1 || rows > 2*(n-1)) throw std::runtime_error("invalid geometry");
        while (length < n + rows - 1) length *= 2;
        real.resize(length); imag.resize(length);
        re_fft.resize(length/2+1); im_fft.resize(length/2+1);
        a.resize(length); spectrum.resize(length); kernel.resize(length);
        chirp.resize(std::max(n,rows)); window.resize(n);
        check(bfft_plan_create(length,&fft));
        try { check(bfft_workspace_create(fft,&work)); }
        catch (...) { bfft_plan_destroy(fft); fft=nullptr; throw; }
        const double denominator = 2.*(n-1);
        for (size_t j=0;j<chirp.size();++j)
            chirp[j]=std::polar(1.,-pi*double(j)*double(j)/denominator);
        for (size_t j=0;j<n;++j) window[j]=.5-.5*std::cos(2*pi*j/n); // periodic Hann oracle
        for (size_t j=0;j<rows;++j) a[j]=std::conj(chirp[j]);
        for (size_t j=1;j<n;++j) a[length-j]=std::conj(chirp[j]);
        forward(a,kernel);
    }
    ~Plan() { bfft_workspace_destroy(work); bfft_plan_destroy(fft); }
    Plan(const Plan&)=delete;
    Plan& operator=(const Plan&)=delete;
    void forward(const std::vector<Z>& input,std::vector<Z>& output) {
        for(size_t i=0;i<length;++i){real[i]=input[i].real();imag[i]=input[i].imag();}
        check(bfft_forward_workspace(fft,work,real.data(),re_fft.data()));
        check(bfft_forward_workspace(fft,work,imag.data(),im_fft.data()));
        for(size_t k=0;k<length;++k){
            const size_t q=k<=length/2?k:length-k;
            const double sign=k<=length/2?1.:-1.;
            const Z x(re_fft[q].re,sign*re_fft[q].im), y(im_fft[q].re,sign*im_fft[q].im);
            output[k]=x+Z(0.,1.)*y;
        }
    }
    void run(const double* input,double* output,bool envelope=false) {
        std::fill(a.begin(),a.end(),Z{});
        for(size_t j=0;j<n;++j){
            const double endpoint=(j==0 || j==n-1)?.5:1.;
            a[j]=endpoint*window[j]*input[j]*chirp[j];
        }
        forward(a,spectrum);
        for(size_t k=0;k<length;++k) a[k]=std::conj(spectrum[k]*kernel[k]);
        forward(a,spectrum); // inverse by conjugation, reusing the same workspace
        for(size_t r=0;r<rows;++r) {
            const Z observation=std::conj(spectrum[r])*chirp[r]/(length*(n-1.));
            output[r]=envelope?std::abs(observation):std::abs(observation.real());
        }
    }
};
}
#ifdef _WIN32
#define EXPORT extern "C" __declspec(dllexport)
#else
#define EXPORT extern "C" __attribute__((visibility("default")))
#endif
EXPORT void* cleanup_unrolled_create(size_t n,size_t rows) {
    try {return new Plan(n,rows);} catch(...) {return nullptr;}
}
EXPORT void cleanup_unrolled_destroy(void* p){delete static_cast<Plan*>(p);}
EXPORT int cleanup_unrolled_frame(void* p,const double* input,double* output){
    if(!p || !input || !output) return 1;
    try{static_cast<Plan*>(p)->run(input,output);return 0;}catch(...){return 2;}
}
// The earlier sibling's exact positive ridge construction: hypot of the
// cosine inverse and irfft(-i*frame), retaining the same 2*(N-1) grid.
EXPORT int cleanup_unrolled_envelope(void* p,const double* input,double* output){
    if(!p || !input || !output) return 1;
    try{static_cast<Plan*>(p)->run(input,output,true);return 0;}catch(...){return 2;}
}
