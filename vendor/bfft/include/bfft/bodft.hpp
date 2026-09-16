#ifndef BFFT_BODFT_HPP
#define BFFT_BODFT_HPP

#include <bfft/bodft.h>
#include <bfft/bfft.hpp>

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace bfft {

/* SIMD backend selected by the build target. */
inline std::string bodft_backend() {
    return bodft_backend_name();
}

/* Reusable BODFT: the native half-bin-shifted real transform (odd-frequency
   DFT). Forward maps N real samples to N/2 packed complex bins; inverse recovers
   the N real samples exactly. A single instance serves double and float. */
class bodft {
public:
    explicit bodft(std::size_t n) {
        bodft_plan* raw = nullptr;
        status result = bodft_plan_create(n, &raw);
        if (result != BFFT_OK) throw error(result);
        impl_.reset(raw);
    }

    std::size_t size() const noexcept { return bodft_plan_size(impl_.get()); }
    std::size_t bins() const noexcept { return bodft_plan_bins(impl_.get()); }

    void forward(const double* input, complex* output) const {
        check(bodft_forward(impl_.get(), input, output));
    }
    void inverse(const complex* input, double* output) const {
        check(bodft_inverse(impl_.get(), input, output));
    }
    void forward_f32(const float* input, complex_f32* output) const {
        check(bodft_forward_f32(impl_.get(), input, output));
    }
    void inverse_f32(const complex_f32* input, float* output) const {
        check(bodft_inverse_f32(impl_.get(), input, output));
    }

    /* Convenience overloads that allocate the result vector. */
    std::vector<complex> forward(const std::vector<double>& input) const {
        if (input.size() != size()) throw error(BFFT_ERROR_INVALID_ARGUMENT);
        std::vector<complex> output(bins());
        forward(input.data(), output.data());
        return output;
    }
    std::vector<double> inverse(const std::vector<complex>& input) const {
        if (input.size() != bins()) throw error(BFFT_ERROR_INVALID_ARGUMENT);
        std::vector<double> output(size());
        inverse(input.data(), output.data());
        return output;
    }

private:
    struct deleter {
        void operator()(bodft_plan* p) const noexcept { bodft_plan_destroy(p); }
    };
    static void check(status result) {
        if (result != BFFT_OK) throw error(result);
    }
    std::unique_ptr<bodft_plan, deleter> impl_;
};

} // namespace bfft

#endif
