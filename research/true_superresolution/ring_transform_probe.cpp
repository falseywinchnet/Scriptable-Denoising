// Exact native target-product test and same-compiler/same-bfft benchmark.
#include "guide_fft.h"
#include "guide_peak.h"
#include "ring_forward.h"
#include <chrono>
#include <iostream>
#include <random>
#include <iomanip>

using Z=std::complex<double>;
constexpr int N=48,M=N*N;
struct Reference {
    CleanupGuideFFT fft{N};
    std::vector<Z> a=std::vector<Z>(M),line=std::vector<Z>(N);
    std::vector<double> real=std::vector<double>(N);
    void run(const Z* unit,const double* rings,double* result){
        for(int ring=0;ring<7;++ring){
            for(int y=0;y<N;++y)for(int x=0;x<=N/2;++x){int j=y*N+x;a[j]=unit[j]*rings[ring*M+j];}
            for(int x=0;x<=N/2;++x){
                for(int y=0;y<N;++y)line[y]=a[y*N+x];
                if(x==0||x==N/2){fft.inverse_real(line.data(),real.data());for(int y=0;y<N;++y)a[y*N+x]=real[y];}
                else{fft.run(line.data(),true);for(int y=0;y<N;++y)a[y*N+x]=line[y];}
            }
            for(int y=0;y<N;++y){
                for(int x=N/2+1;x<N;++x)a[y*N+x]=std::conj(a[y*N+N-x]);
                fft.inverse_real(a.data()+y*N,real.data());
                std::copy(real.begin(),real.end(),result+ring*M+y*N);
            }
        }
    }
};

int main(){
    constexpr double pi=3.141592653589793238462643383279502884;
    std::vector<double> rings(7*M),expected(7*M),actual(7*M);
    for(int y=0;y<N;++y)for(int x=0;x<N;++x){
        double fy=double(y<(N+1)/2?y:y-N)/N,fx=double(x<(N+1)/2?x:x-N)/N;
        double radius=std::hypot(fx,fy);
        for(int r=0;r<7;++r)rings[r*M+y*N+x]=std::exp(-.5*std::pow((radius-(.035+(.46-.035)*r/6))/.055,2));
    }
    cleanup_ring::RingForward48 plan(rings.data());Reference reference;
    std::mt19937 random(9137);std::normal_distribution<double> normal;
    std::vector<std::vector<Z>> inputs(40,std::vector<Z>(M));
    for(int trial=0;trial<40;++trial){
        auto& u=inputs[trial];
        for(int y=0;y<N;++y)for(int x=0;x<N;++x){
            int j=y*N+x,p=((N-y)%N)*N+(N-x)%N;
            if(j>p)continue;
            if(trial==0)u[j]=0.;
            else if(trial<8)u[j]=std::polar(1.,-2*pi*(x*(trial-3)+y*(trial+1))/N);
            else {u[j]=Z(normal(random),normal(random));u[j]/=std::max(std::abs(u[j]),1e-12);}
            if(j==p)u[j]=u[j].real();u[p]=std::conj(u[j]);
        }
    }
    double worst=0.,direct_error=0.,competitor_error=0.;int peak_mismatches=0;
    std::vector<Z> oldmap(M),newmap(M);
    for(auto& u:inputs){
        reference.run(u.data(),rings.data(),expected.data());plan.run(u.data(),actual.data());
        for(int j=0;j<7*M;++j)worst=std::max(worst,std::abs(expected[j]-actual[j]));
        for(int ring=0;ring<7;++ring){
            for(int j=0;j<M;++j){oldmap[j]=expected[ring*M+j];newmap[j]=actual[ring*M+j];}
            int oldpeak,newpeak;double oldcomp,newcomp;
            cleanup_phase_peak(oldmap.data(),N,oldpeak,oldcomp);cleanup_phase_peak(newmap.data(),N,newpeak,newcomp);
            peak_mismatches+=oldpeak!=newpeak;competitor_error=std::max(competitor_error,std::abs(oldcomp-newcomp));
        }
    }
    // Independent literal DFT checks, separate from the existing FFT factorization.
    const auto& u=inputs.back();plan.run(u.data(),actual.data());
    for(int r=0;r<7;++r)for(auto d:std::array<std::pair<int,int>,5>{{{0,0},{1,7},{24,24},{47,0},{11,33}}}){
        Z value{};for(int y=0;y<N;++y)for(int x=0;x<N;++x)value+=u[y*N+x]*rings[r*M+y*N+x]*std::polar(1.,2*pi*(y*d.first+x*d.second)/N);
        direct_error=std::max(direct_error,std::abs(actual[r*M+d.first*N+d.second]-value.real()/M));
    }
    volatile double checksum=0.;std::vector<double> oldtime,newtime;
    for(int repeat=0;repeat<12;++repeat){
        for(int method=0;method<2;++method){
            auto begin=std::chrono::steady_clock::now();
            for(int t=8;t<40;++t){if(method==0)reference.run(inputs[t].data(),rings.data(),actual.data());else plan.run(inputs[t].data(),actual.data());checksum+=actual[t];}
            double us=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-begin).count()/32;
            (method==0?oldtime:newtime).push_back(us);
        }
    }
    std::sort(oldtime.begin(),oldtime.end());std::sort(newtime.begin(),newtime.end());
    std::cout<<std::setprecision(17)<<"{\n  \"max_field_error\": "<<worst<<",\n  \"max_direct_dft_error\": "<<direct_error
        <<",\n  \"peak_mismatches\": "<<peak_mismatches<<",\n  \"max_competitor_error\": "<<competitor_error
        <<",\n  \"seven_inverse_us\": "<<oldtime[6]<<",\n  \"fused_forward_us\": "<<newtime[6]
        <<",\n  \"speedup\": "<<oldtime[6]/newtime[6]<<"\n}\n";
    return worst<1e-11 && direct_error<1e-11 && peak_mismatches==0?0:1;
}
