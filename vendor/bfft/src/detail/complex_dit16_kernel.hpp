#pragma once
// Cleanup local bfft extension: exact radix-4 x radix-4 complex forward DFT.
// Coefficients may be SIMD ring lanes. No plan, allocation, real/imaginary
// transforms or full-spectrum reconstruction in this 16-point codelet.
#include <complex>
#include <cstddef>

namespace bruun {
template <class Complex>
inline void complex_dit4(const Complex& x0,const Complex& x1,const Complex& x2,const Complex& x3,Complex* out){
    const auto a=x0+x2,b=x0-x2,c=x1+x3,d=x1-x3;
    const Complex minus_i_d{d.im,-d.re};
    out[0]=a+c;out[1]=b+minus_i_d;out[2]=a-c;out[3]=b-minus_i_d;
}
template <class Complex>
inline void complex_dit16_forward(const Complex* input,Complex* output,size_t stride=1){
    constexpr double c=.923879532511286756128183189396788286;
    constexpr double s=.382683432365089771728459984030398866;
    constexpr double q=.707106781186547524400844362104849039;
    const std::complex<double> phase[10]={{1,0},{c,-s},{q,-q},{s,-c},{0,-1},{-s,-c},{-q,-q},{-c,-s},{-1,0},{-c,s}};
    Complex stage[4][4],column[4];
    for(int r=0;r<4;++r)complex_dit4(input[r*stride],input[(r+4)*stride],input[(r+8)*stride],input[(r+12)*stride],stage[r]);
    for(int k=0;k<4;++k){
        complex_dit4(stage[0][k],stage[1][k]*phase[k],stage[2][k]*phase[2*k],stage[3][k]*phase[3*k],column);
        for(int l=0;l<4;++l)output[k+4*l]=column[l];
    }
}
} // namespace bruun
