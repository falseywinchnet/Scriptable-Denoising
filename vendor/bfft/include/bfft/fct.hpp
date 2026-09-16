#ifndef BFFT_FCT_HPP
#define BFFT_FCT_HPP

#include <bfft/fct.h>
#include <bfft/bfft.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

namespace bfft {

/* C++ wrapper over the FCT (Fast Correlated Transform) C ABI.
   Forward-only: each standard bin k is emitted at its maximally correlated
   leading-edge slice tau_k (see <bfft/fct.h>).  There is no inverse. */
class fct_plan {
public:
    explicit fct_plan(std::size_t n) {
        ::fct_plan* raw = nullptr;
        status result = ::fct_plan_create(n, &raw);
        if (result != BFFT_OK) throw error(result);
        impl_.reset(raw);
    }

    fct_plan(std::size_t n, int t_min, double rel, double act) {
        ::fct_plan* raw = nullptr;
        status result = ::fct_plan_create_ex(n, t_min, rel, act, &raw);
        if (result != BFFT_OK) throw error(result);
        impl_.reset(raw);
    }

    std::size_t size() const noexcept { return ::fct_plan_size(impl_.get()); }
    std::size_t bins() const noexcept { return ::fct_plan_bins(impl_.get()); }

    void forward(const double* input, complex* output,
                 std::int64_t* tau = nullptr) {
        status result = ::fct_forward(impl_.get(), input, output, tau);
        if (result != BFFT_OK) throw error(result);
    }

    void forward_complex(const complex* input, complex* output,
                         std::int64_t* tau = nullptr) {
        status result = ::fct_forward_complex(impl_.get(), input, output, tau);
        if (result != BFFT_OK) throw error(result);
    }

    /* Returns the correlated spectrum and the selected slice per bin. */
    std::pair<std::vector<complex>, std::vector<std::int64_t>>
    forward(const std::vector<double>& input) {
        if (input.size() != size()) throw error(BFFT_ERROR_INVALID_ARGUMENT);
        std::vector<complex> output(bins());
        std::vector<std::int64_t> tau(bins());
        forward(input.data(), output.data(), tau.data());
        return {std::move(output), std::move(tau)};
    }


    std::pair<std::vector<complex>, std::vector<std::int64_t>>
    forward_complex(const std::vector<complex>& input) {
        if (input.size() != size()) throw error(BFFT_ERROR_INVALID_ARGUMENT);
        std::vector<complex> output(size());
        std::vector<std::int64_t> tau(size());
        forward_complex(input.data(), output.data(), tau.data());
        return {std::move(output), std::move(tau)};
    }

private:
    struct deleter {
        void operator()(::fct_plan* p) const noexcept { ::fct_plan_destroy(p); }
    };
    std::unique_ptr<::fct_plan, deleter> impl_;
};

} // namespace bfft

#endif
