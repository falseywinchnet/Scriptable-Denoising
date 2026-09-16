#pragma once
// Exact seven-output ring transform, using one lane-valued Bruun forward walk.
// If S=R*U is Hermitian, h=Re(S)-Im(S) is real and
// inverseDFT2(S) = (Re(DFT2(h))-Im(DFT2(h))) / N^2.
// Thus the real common input Re(U)-Im(U) is lifted into seven filter lanes at
// intake. No ring-specific complex inverse workspace or inverse calls remain.
#include "detail/bruun_dit_kernel.hpp"
#include "detail/complex_dit16_kernel.hpp"
#include <array>
#include <complex>
#include <vector>
#include <stdexcept>

namespace cleanup_ring {
#ifndef CLEANUP_RING_LANES
#define CLEANUP_RING_LANES 4
#endif
static_assert(CLEANUP_RING_LANES==2||CLEANUP_RING_LANES==4||CLEANUP_RING_LANES==8,"Ring lanes must be 2, 4 or 8");
struct Pack {
    static constexpr int width=CLEANUP_RING_LANES;
#if BRUUN_LEVEL >= 1
    bruun_v2 v[width/2];
    Pack()=default;
    Pack(double x){for(auto& a:v)a=V2_SET1(x);}
    static Pack load(const double* x){Pack p;for(int j=0;j<width/2;++j)p.v[j]=V2_LD(x+2*j);return p;}
    void store(double* x)const{for(int j=0;j<width/2;++j)V2_ST(x+2*j,v[j]);}
    friend Pack operator+(const Pack&a,const Pack&b){Pack p;for(int j=0;j<width/2;++j)p.v[j]=V2_ADD(a.v[j],b.v[j]);return p;}
    friend Pack operator-(const Pack&a,const Pack&b){Pack p;for(int j=0;j<width/2;++j)p.v[j]=V2_SUB(a.v[j],b.v[j]);return p;}
    friend Pack operator*(const Pack&a,const Pack&b){Pack p;for(int j=0;j<width/2;++j)p.v[j]=V2_MUL(a.v[j],b.v[j]);return p;}
#else
    double v[width];
    Pack()=default;
    Pack(double x){for(auto& a:v)a=x;}
    static Pack load(const double* x){Pack p;for(int j=0;j<width;++j)p.v[j]=x[j];return p;}
    void store(double* x)const{for(int j=0;j<width;++j)x[j]=v[j];}
    friend Pack operator+(const Pack&a,const Pack&b){Pack p;for(int j=0;j<width;++j)p.v[j]=a.v[j]+b.v[j];return p;}
    friend Pack operator-(const Pack&a,const Pack&b){Pack p;for(int j=0;j<width;++j)p.v[j]=a.v[j]-b.v[j];return p;}
    friend Pack operator*(const Pack&a,const Pack&b){Pack p;for(int j=0;j<width;++j)p.v[j]=a.v[j]*b.v[j];return p;}
#endif
    friend Pack operator-(const Pack&a){return Pack(0.)-a;}
};
struct ComplexPack {
    Pack re,im;
    friend ComplexPack operator+(const ComplexPack&a,const ComplexPack&b){return {a.re+b.re,a.im+b.im};}
    friend ComplexPack operator-(const ComplexPack&a,const ComplexPack&b){return {a.re-b.re,a.im-b.im};}
    friend ComplexPack operator*(const ComplexPack&a,const std::complex<double>&b){return {a.re*Pack(b.real())-a.im*Pack(b.imag()),a.re*Pack(b.imag())+a.im*Pack(b.real())};}
};
inline ComplexPack conjugate(const ComplexPack&a){return {a.re,-a.im};}

class Forward48 {
    bruun::DIT_RFFT_kernel power;
    std::array<Pack,16> input,work;
    std::array<ComplexPack,9> frequency;
    std::array<ComplexPack,48> residues;
    std::array<std::complex<double>,16> phase,phase2;
    void combine(ComplexPack* output,bool half){
        const std::complex<double> omega(-.5,-.866025403784438646763723170752936183);
        for(int k=0;k<16;++k){
            const auto a=residues[k],b=residues[k+16]*phase[k],c=residues[k+32]*phase2[k];
            output[k]=a+b+c;
            if(!half||k<=8)output[k+16]=a+b*omega+c*std::conj(omega);
            if(!half)output[k+32]=a+b*std::conj(omega)+c*omega;
        }
    }
public:
    Forward48(){
        if(!power.init(16))throw std::runtime_error("Bruun lane plan failed");
        for(int k=0;k<16;++k){phase[k]=std::polar(1.,-2*3.14159265358979323846*k/48);phase2[k]=phase[k]*phase[k];}
    }
    void real(const Pack* source,ComplexPack* output){
        for(int r=0;r<3;++r){
            for(int t=0;t<16;++t)input[t]=source[r+3*t];
            power.forward_lanes(input.data(),frequency.data(),work.data());
            for(int k=0;k<16;++k)residues[r*16+k]=k<=8?frequency[k]:conjugate(frequency[16-k]);
        }
        combine(output,true);
    }
    void complex(const ComplexPack* source,ComplexPack* output){
#ifndef CLEANUP_RING_COMPLEX16
#define CLEANUP_RING_COMPLEX16 1
#endif
#if CLEANUP_RING_COMPLEX16
        for(int r=0;r<3;++r)bruun::complex_dit16_forward(source+r,residues.data()+r*16,3);
#else
        // The same Bruun tree operates on all ring lanes at once. Preserve
        // the real-part transforms until the imaginary transforms complete.
        for(int r=0;r<3;++r){
            for(int t=0;t<16;++t)input[t]=source[r+3*t].re;
            power.forward_lanes(input.data(),frequency.data(),work.data());
            for(int k=0;k<16;++k)residues[r*16+k]=k<=8?frequency[k]:conjugate(frequency[16-k]);
            for(int t=0;t<16;++t)input[t]=source[r+3*t].im;
            power.forward_lanes(input.data(),frequency.data(),work.data());
            for(int k=0;k<16;++k){auto b=k<=8?frequency[k]:conjugate(frequency[16-k]);auto& a=residues[r*16+k];a={a.re-b.im,a.im+b.re};}
        }
#endif
        combine(output,false);
    }
};

class RingForward48 {
    static constexpr int groups=(7+Pack::width-1)/Pack::width;
    Forward48 fft;
    std::vector<Pack> weights;
    std::vector<ComplexPack> rows;
    std::array<Pack,48> input;
    std::array<ComplexPack,48> line,output;
public:
    // Masks use the existing ring-major [7][48][48] storage.
    explicit RingForward48(const double* masks):weights(groups*48*48),rows(48*25){
        for(int group=0;group<groups;++group)for(int j=0;j<48*48;++j){double w[Pack::width]{};
            for(int lane=0;lane<Pack::width;++lane){int ring=group*Pack::width+lane;if(ring<7)w[lane]=masks[ring*48*48+j];}
            weights[group*48*48+j]=Pack::load(w);}
    }
    // U is full Hermitian input. Results are ring-major, with ALL 2304 real
    // correlation values present per ring, including global competitors.
    void run(const std::complex<double>* unit,double* result){
      for(int group=0;group<groups;++group){
        for(int y=0;y<48;++y){
            for(int x=0;x<48;++x){int j=y*48+x;input[x]=weights[group*48*48+j]*Pack(unit[j].real()-unit[j].imag());}
            fft.real(input.data(),rows.data()+y*25);
        }
        const Pack scale(1./(48*48));
        for(int x=0;x<=24;++x){
            for(int y=0;y<48;++y)line[y]=rows[y*25+x];
            fft.complex(line.data(),output.data());
            for(int y=0;y<48;++y){
                double a[Pack::width],b[Pack::width];((output[y].re-output[y].im)*scale).store(a);
                const int opposite=((48-y)%48)*48+(48-x)%48;
                if(x>0&&x<24)((output[y].re+output[y].im)*scale).store(b);
                for(int lane=0;lane<Pack::width;++lane){int ring=group*Pack::width+lane;if(ring<7){result[ring*48*48+y*48+x]=a[lane];if(x>0&&x<24)result[ring*48*48+opposite]=b[lane];}}
            }
        }
      }
    }
};
} // namespace cleanup_ring
