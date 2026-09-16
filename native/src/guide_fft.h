#pragma once
// Exact small complex DFTs built from bfft real transforms. The registration
// oracle uses 48-point charts: radix 3 x 16 avoids a 128-point Bluestein
// convolution for every row/column while retaining its exact sampling grid.
#include <bfft/bfft.h>
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <vector>

namespace cleanup_guide_detail {
using Complex = std::complex<double>;
constexpr double pi = 3.141592653589793238462643383279502884;
inline void check(bfft_status status) {
    if (status != BFFT_OK) throw std::runtime_error(bfft_status_string(status));
}
inline bool power_of_two(size_t n) { return n >= 4 && !(n & (n - 1)); }
inline size_t checked_size(size_t n) {
    if (n < 4 || n > 8192) throw std::invalid_argument("Guide DFT size must be 4..8192");
    return n;
}
inline size_t convolution_size(size_t n) {
    size_t result = 1;
    while (result < 2 * n - 1) result *= 2;
    return result;
}

class RealPlan {
    size_t n_;
    std::vector<double> real_, imag_;
    std::vector<bfft_complex> real_frequency_, imag_frequency_;
    std::unique_ptr<bfft_plan, decltype(&bfft_plan_destroy)> plan_{nullptr, bfft_plan_destroy};
    std::unique_ptr<bfft_workspace, decltype(&bfft_workspace_destroy)> workspace_{nullptr, bfft_workspace_destroy};
public:
    explicit RealPlan(size_t n) : n_(n), real_(n), imag_(n),
        real_frequency_(n / 2 + 1), imag_frequency_(n / 2 + 1) {
        bfft_plan* plan = nullptr;
        auto status = bfft_plan_create(n, &plan);
        plan_.reset(plan);
        check(status);
        bfft_workspace* workspace = nullptr;
        status = bfft_workspace_create(plan_.get(), &workspace);
        workspace_.reset(workspace);
        check(status);
    }
    // Two real bfft calls form an ordinary complex forward transform. All input
    // is read before output is written, so in-place and strided inputs are safe.
    void forward(const Complex* input, Complex* output, size_t stride = 1) {
        for (size_t j = 0; j < n_; ++j) {
            real_[j] = input[j * stride].real();
            imag_[j] = input[j * stride].imag();
        }
        check(bfft_forward_workspace(plan_.get(), workspace_.get(), real_.data(), real_frequency_.data()));
        check(bfft_forward_workspace(plan_.get(), workspace_.get(), imag_.data(), imag_frequency_.data()));
        for (size_t k = 0; k < n_; ++k) {
            const size_t q = k <= n_ / 2 ? k : n_ - k;
            const double sign = k <= n_ / 2 ? 1. : -1.;
            const auto r = real_frequency_[q], i = imag_frequency_[q];
            output[k] = Complex(r.re - sign * i.im, sign * r.im + i.re);
        }
    }
    void forward_real(const double* input, Complex* output, size_t stride = 1) {
        for (size_t j = 0; j < n_; ++j) real_[j] = input[j * stride];
        check(bfft_forward_workspace(plan_.get(), workspace_.get(), real_.data(), real_frequency_.data()));
        for (size_t k = 0; k < n_; ++k) {
            const size_t q = k <= n_ / 2 ? k : n_ - k;
            output[k] = Complex(real_frequency_[q].re, (k <= n_ / 2 ? 1. : -1.) * real_frequency_[q].im);
        }
    }
    void inverse_real(const Complex* input, double* output, size_t stride = 1) {
        for (size_t k = 0; k <= n_ / 2; ++k)
            real_frequency_[k] = {input[k].real(), input[k].imag()};
        check(bfft_inverse_workspace(plan_.get(), workspace_.get(), real_frequency_.data(), real_.data()));
        for (size_t j = 0; j < n_; ++j) output[j * stride] = real_[j];
    }
};
} // namespace cleanup_guide_detail

class CleanupGuideFFT {
    using Z = cleanup_guide_detail::Complex;
    size_t n_, m_;
    bool direct_, radix_three_;
    cleanup_guide_detail::RealPlan power_;
    std::vector<Z> work_, frequency_, kernel_, chirp_, real_fallback_;

    void combine_radix(Z* output) {
        constexpr double root = 0.866025403784438646763723170752936183;
        const Z omega(-.5, -root), omega2(-.5, root);
        for (size_t k = 0; k < m_; ++k) {
            const Z u = work_[k];
            const Z v = work_[m_ + k] * chirp_[k];
            const Z w = work_[2 * m_ + k] * chirp_[k] * chirp_[k];
            output[k] = u + v + w;
            output[k + m_] = u + omega * v + omega2 * w;
            output[k + 2 * m_] = u + omega2 * v + omega * w;
        }
    }

    void forward(Z* input) {
        if (direct_) {
            power_.forward(input, input);
        } else if (radix_three_) {
            // j=r+3t; first perform each t-transform. Twiddle each result by
            // exp(-2 pi i r k/N), then perform the three-member outer DFT.
            for (size_t r = 0; r < 3; ++r)
                power_.forward(input + r, work_.data() + r * m_, 3);
            combine_radix(input);
        } else {
            std::fill(work_.begin(), work_.end(), Z{});
            for (size_t j = 0; j < n_; ++j) work_[j] = input[j] * chirp_[j];
            power_.forward(work_.data(), frequency_.data());
            for (size_t j = 0; j < m_; ++j) work_[j] = std::conj(frequency_[j] * kernel_[j]);
            power_.forward(work_.data(), frequency_.data());
            for (size_t j = 0; j < n_; ++j)
                input[j] = std::conj(frequency_[j]) * chirp_[j] / double(m_);
        }
    }
public:
    explicit CleanupGuideFFT(size_t size)
        : n_(cleanup_guide_detail::checked_size(size)),
          m_(cleanup_guide_detail::power_of_two(n_) ? n_ :
              n_ % 3 == 0 && cleanup_guide_detail::power_of_two(n_ / 3) ? n_ / 3 :
              cleanup_guide_detail::convolution_size(n_)),
          direct_(m_ == n_), radix_three_(m_ * 3 == n_), power_(m_),
          work_(radix_three_ ? n_ : m_), frequency_(direct_ || radix_three_ ? 0 : m_),
          kernel_(direct_ || radix_three_ ? 0 : m_), chirp_(direct_ ? 0 : n_),
          real_fallback_(direct_ || radix_three_ ? 0 : n_) {
        if (radix_three_) {
            for (size_t k = 0; k < m_; ++k)
                chirp_[k] = std::polar(1., -2 * cleanup_guide_detail::pi * k / n_);
        } else if (!direct_) {
            for (size_t j = 0; j < n_; ++j) {
                chirp_[j] = std::polar(1., -cleanup_guide_detail::pi * double(j) * j / n_);
                work_[j] = std::conj(chirp_[j]);
                if (j) work_[m_ - j] = work_[j];
            }
            power_.forward(work_.data(), kernel_.data());
        }
    }
    void run(Z* input, bool inverse = false) {
        if (inverse) for (size_t j = 0; j < n_; ++j) input[j] = std::conj(input[j]);
        forward(input);
        if (inverse) for (size_t j = 0; j < n_; ++j) input[j] = std::conj(input[j]) / double(n_);
    }
    // The real/Hermitian boundary of a 2D transform needs half as many bfft
    // calls. Input/output each contain N values; inverse requires ordinary
    // Hermitian symmetry and returns its real inverse with 1/N normalization.
    void forward_real(const double* input, Z* output) {
        if (direct_) power_.forward_real(input, output);
        else if (radix_three_) {
            for (size_t r = 0; r < 3; ++r)
                power_.forward_real(input + r, work_.data() + r*m_, 3);
            combine_radix(output);
        } else {
            for (size_t j = 0; j < n_; ++j) real_fallback_[j] = input[j];
            run(real_fallback_.data());
            std::copy(real_fallback_.begin(), real_fallback_.end(), output);
        }
    }
    void inverse_real(const Z* input, double* output) {
        if (direct_) power_.inverse_real(input, output);
        else if (radix_three_) {
            constexpr double root = 0.866025403784438646763723170752936183;
            const Z omega(-.5, -root), omega2(-.5, root);
            // Each real inverse below consumes only its nonnegative half.
            // The omitted residues are conjugates and were never read.
            for (size_t k = 0; k <= m_ / 2; ++k) {
                const Z u = input[k], v = input[k+m_], w = input[k+2*m_];
                const Z phase = std::conj(chirp_[k]);
                work_[k] = (u + v + w) / 3.;
                work_[m_+k] = (u + omega2*v + omega*w) * phase / 3.;
                work_[2*m_+k] = (u + omega*v + omega2*w) * phase * phase / 3.;
            }
            for (size_t r = 0; r < 3; ++r)
                power_.inverse_real(work_.data() + r*m_, output+r, 3);
        } else {
            std::copy_n(input, n_, real_fallback_.begin());
            run(real_fallback_.data(), true);
            for (size_t j = 0; j < n_; ++j) output[j] = real_fallback_[j].real();
        }
    }
};
