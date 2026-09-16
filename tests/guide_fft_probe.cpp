// Standalone numerical test for the exact registration transform optimization.
#include "guide_fft.h"
#include "guide_peak.h"
#include <iostream>
#include <random>
int main() {
    using Z = std::complex<double>;
    std::mt19937 generator(2718);
    std::normal_distribution<double> normal;
    for(int n : {12,13,16,24,31,48})for(int trial=0;trial<40;++trial){
        std::vector<Z> a(n*n);
        for(auto& v:a)v=trial==0?0.:normal(generator);
        if(trial>1){a[(trial*n+trial)%(n*n)]=99.;a[(trial*n+trial+1)%(n*n)]=99.;}
        int expected=0;for(int j=1;j<n*n;++j)if(a[j].real()>a[expected].real())expected=j;
        double competitor=-std::numeric_limits<double>::infinity();
        for(int y=0;y<n;++y)for(int x=0;x<n;++x){
            int dy=std::abs(y-expected/n),dx=std::abs(x-expected%n);
            if(std::min(dy,n-dy)>3||std::min(dx,n-dx)>3)competitor=std::max(competitor,a[y*n+x].real());
        }
        int actual;double value;cleanup_phase_peak(a.data(),n,actual,value);
        if(actual!=expected||value!=competitor){std::cerr<<"phase peak mismatch\n";return 1;}
    }
    double worst = 0.;
    for (size_t n : {12, 13, 16, 24, 31, 32, 33, 47, 48, 64}) {
        CleanupGuideFFT plan(n);
        std::vector<Z> input(n), actual(n), expected(n);
        for (auto& v : input) v = {normal(generator), normal(generator)};
        for (bool inverse : {false, true}) {
            for (size_t k = 0; k < n; ++k) {
                expected[k] = {};
                for (size_t j = 0; j < n; ++j)
                    expected[k] += input[j] * std::polar(1., (inverse ? 2. : -2.) * cleanup_guide_detail::pi * k * j / n);
                if (inverse) expected[k] /= double(n);
            }
            actual = input;
            plan.run(actual.data(), inverse);
            for (size_t j = 0; j < n; ++j) worst = std::max(worst, std::abs(actual[j] - expected[j]));
        }
        actual = input;
        plan.run(actual.data());
        plan.run(actual.data(), true);
        for (size_t j = 0; j < n; ++j) worst = std::max(worst, std::abs(actual[j] - input[j]));
        std::vector<double> real(n), restored(n);
        for (size_t j = 0; j < n; ++j) {real[j] = input[j].real();expected[j] = real[j];}
        plan.run(expected.data());
        plan.forward_real(real.data(), actual.data());
        for (size_t j = 0; j < n; ++j) worst = std::max(worst, std::abs(actual[j] - expected[j]));
        plan.inverse_real(actual.data(), restored.data());
        for (size_t j = 0; j < n; ++j) worst = std::max(worst, std::abs(restored[j] - real[j]));
    }
    std::cout << "guide_fft max error vs direct DFT: " << worst << '\n';
    return worst < 1e-10 ? 0 : 1;
}
