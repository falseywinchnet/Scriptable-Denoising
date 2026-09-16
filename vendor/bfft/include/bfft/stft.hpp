#ifndef BFFT_STFT_HPP
#define BFFT_STFT_HPP

#include <bfft/stft.h>
#include <bfft/bfft.hpp>
#include <cstddef>
#include <memory>
#include <vector>

namespace bfft {

inline void stft_check(status result) {
    if (result != BFFT_OK) throw error(result);
}

using stft_transform = bfft_stft_transform;
constexpr stft_transform stft_rfft = BFFT_STFT_RFFT;
constexpr stft_transform stft_odft = BFFT_STFT_ODFT;
/* Forward-only Fast Correlated Transform frames (see <bfft/fct.h>). */
constexpr stft_transform stft_fct = BFFT_STFT_FCT;

inline std::vector<double> hann_window(std::size_t n_fft) {
    std::vector<double> out(n_fft);
    stft_check(bfft_stft_hann_window(n_fft, out.data()));
    return out;
}

class stft_plan {
public:
    stft_plan(std::size_t n,
              std::size_t n_fft,
              std::size_t hop_length,
              const double* window = nullptr,
              stft_transform transform = stft_rfft) {
        bfft_stft_plan* raw = nullptr;
        status result = bfft_stft_plan_create(n, n_fft, hop_length, window, transform, &raw);
        if (result != BFFT_OK) throw error(result);
        impl_.reset(raw);
    }

    stft_plan(std::size_t n,
              std::size_t n_fft,
              std::size_t hop_length,
              const std::vector<double>& window,
              stft_transform transform = stft_rfft)
        : stft_plan(n, n_fft, hop_length, window.data(), transform) {
        if (window.size() != n_fft) throw error(BFFT_ERROR_INVALID_ARGUMENT);
    }

    std::size_t n() const noexcept { return bfft_stft_plan_n(impl_.get()); }
    std::size_t n_fft() const noexcept { return bfft_stft_plan_n_fft(impl_.get()); }
    std::size_t hop_length() const noexcept { return bfft_stft_plan_hop_length(impl_.get()); }
    std::size_t bins() const noexcept { return bfft_stft_plan_bins(impl_.get()); }
    std::size_t segments() const noexcept { return bfft_stft_plan_segments(impl_.get()); }
    std::size_t buffer_length() const noexcept { return bfft_stft_plan_buffer_length(impl_.get()); }
    stft_transform transform() const noexcept { return bfft_stft_plan_transform(impl_.get()); }

    void reset_buffer() { stft_check(bfft_stft_reset_buffer(impl_.get())); }

    void forward(const double* input, complex* output) { stft_check(bfft_stft_forward(impl_.get(), input, output)); }
    void inverse(const complex* input, double* output) { stft_check(bfft_stft_inverse(impl_.get(), input, output)); }

    std::vector<complex> forward(const std::vector<double>& input) {
        if (input.size() != n()) throw error(BFFT_ERROR_INVALID_ARGUMENT);
        std::vector<complex> output(bins() * segments());
        forward(input.data(), output.data());
        return output;
    }

    std::vector<double> inverse(const std::vector<complex>& input) {
        if (input.size() != bins() * segments()) throw error(BFFT_ERROR_INVALID_ARGUMENT);
        std::vector<double> output(segments() * hop_length());
        inverse(input.data(), output.data());
        return output;
    }

private:
    struct deleter { void operator()(bfft_stft_plan* p) const noexcept { bfft_stft_plan_destroy(p); } };
    std::unique_ptr<bfft_stft_plan, deleter> impl_;
};

} // namespace bfft

#endif
