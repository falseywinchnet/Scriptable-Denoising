#include <bfft/fct.h>

#include "detail/fct_intrinsic_kernel.hpp"

#include <cstddef>
#include <new>
#include <vector>

static_assert(sizeof(bfft_complex) == sizeof(fct::fractional::value),
              "bfft_complex / intrinsic value size mismatch");
static_assert(offsetof(bfft_complex, re) == offsetof(fct::fractional::value, re),
              "bfft_complex / intrinsic value re offset mismatch");
static_assert(offsetof(bfft_complex, im) == offsetof(fct::fractional::value, im),
              "bfft_complex / intrinsic value im offset mismatch");

struct fct_plan {
    int n;
    int bins;
    int min_tau;
    double activity;
    fct::intrinsic::plan p;
    std::vector<fct::fractional::value> real_input;
    std::vector<fct::fractional::value> full_output;
    std::vector<int64_t> full_tau;

    fct_plan(int size, int minimum, double act)
        : n(size), bins(size / 2 + 1), min_tau(minimum), activity(act), p(size),
          real_input(static_cast<size_t>(size)),
          full_output(static_cast<size_t>(size)),
          full_tau(static_cast<size_t>(size)) {}
};

namespace {

fct::fractional::value* as_intrinsic(bfft_complex* v) {
    return reinterpret_cast<fct::fractional::value*>(v);
}
const fct::fractional::value* as_intrinsic(const bfft_complex* v) {
    return reinterpret_cast<const fct::fractional::value*>(v);
}

bool valid_size(size_t n) {
    if (n > static_cast<size_t>(2147483647)) return false;
    const int ni = static_cast<int>(n);
    return ni >= 16 && (ni & (ni - 1)) == 0;
}

} // namespace

bfft_status fct_plan_create_ex(size_t n, int t_min, double rel, double act,
                               fct_plan** plan) {
    if (plan == nullptr) return BFFT_ERROR_INVALID_ARGUMENT;
    *plan = nullptr;
    if (!valid_size(n)) return BFFT_ERROR_INVALID_ARGUMENT;
    if (t_min < 1 || t_min > static_cast<int>(n) ||
        !(rel > 0.0 && rel <= 1.0) || !(act >= 0.0))
        return BFFT_ERROR_INVALID_ARGUMENT;
    (void)rel;
    fct_plan* p = new (std::nothrow)
        fct_plan(static_cast<int>(n), t_min, act);
    if (!p) return BFFT_ERROR_ALLOCATION;
    *plan = p;
    return BFFT_OK;
}

bfft_status fct_plan_create(size_t n, fct_plan** plan) {
    return fct_plan_create_ex(n, 1, 0.5, 0.0, plan);
}

void fct_plan_destroy(fct_plan* plan) { delete plan; }

size_t fct_plan_size(const fct_plan* plan) {
    return plan ? static_cast<size_t>(plan->n) : 0;
}

size_t fct_plan_bins(const fct_plan* plan) {
    return plan ? static_cast<size_t>(plan->bins) : 0;
}

bfft_status fct_forward(fct_plan* plan, const double* input,
                        bfft_complex* output, int64_t* tau) {
    if (!plan || !input || !output) return BFFT_ERROR_INVALID_ARGUMENT;
    for (int i = 0; i < plan->n; ++i)
        plan->real_input[static_cast<size_t>(i)] =
            fct::fractional::value{input[i], 0.0};
    if (!plan->p.forward(plan->real_input.data(), plan->full_output.data(),
                         plan->full_tau.data(), plan->activity, plan->min_tau))
        return BFFT_ERROR_INTERNAL;
    for (int k = 0; k < plan->bins; ++k) {
        output[k].re = plan->full_output[static_cast<size_t>(k)].re;
        output[k].im = plan->full_output[static_cast<size_t>(k)].im;
        if (tau) tau[k] = plan->full_tau[static_cast<size_t>(k)];
    }
    return BFFT_OK;
}

bfft_status fct_forward_complex(fct_plan* plan, const bfft_complex* input,
                                bfft_complex* output, int64_t* tau) {
    if (!plan || !input || !output) return BFFT_ERROR_INVALID_ARGUMENT;
    if (!plan->p.forward(as_intrinsic(input), as_intrinsic(output), tau,
                         plan->activity, plan->min_tau))
        return BFFT_ERROR_INTERNAL;
    return BFFT_OK;
}

bfft_status fct_forward_complex_moment(fct_plan* plan,
                                       const bfft_complex* input,
                                       bfft_complex* output,
                                       int64_t* tau,
                                       bfft_complex* moment) {
    if (!plan || !input || !output || !moment)
        return BFFT_ERROR_INVALID_ARGUMENT;
    if (!plan->p.forward(as_intrinsic(input), as_intrinsic(output), tau,
                         plan->activity, plan->min_tau,
                         as_intrinsic(moment)))
        return BFFT_ERROR_INTERNAL;
    return BFFT_OK;
}

bfft_status fct_forward_numba(fct_plan* plan, const double* input,
                              bfft_complex* output, double* /*work*/,
                              bfft_complex* /*native_scratch*/) {
    return fct_forward(plan, input, output, nullptr);
}
