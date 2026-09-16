#include <bfft/bodft.h>

#include "detail/bodft_kernel.hpp"

#include <cstddef>
#include <new>

static_assert(sizeof(bfft_complex) == sizeof(bodft::complex_t),
              "bfft_complex / bodft::complex_t size mismatch");
static_assert(offsetof(bfft_complex, re) == offsetof(bodft::complex_t, re),
              "bfft_complex / bodft::complex_t re offset mismatch");
static_assert(offsetof(bfft_complex, im) == offsetof(bodft::complex_t, im),
              "bfft_complex / bodft::complex_t im offset mismatch");
static_assert(sizeof(bfft_complex_f32) == sizeof(bodft::complex_f32_t),
              "bfft_complex_f32 / bodft::complex_f32_t size mismatch");
static_assert(offsetof(bfft_complex_f32, re) == offsetof(bodft::complex_f32_t, re),
              "bfft_complex_f32 / bodft::complex_f32_t re offset mismatch");
static_assert(offsetof(bfft_complex_f32, im) == offsetof(bodft::complex_f32_t, im),
              "bfft_complex_f32 / bodft::complex_f32_t im offset mismatch");

// One plan carries a double and a single precision kernel so callers can mix
// precisions through one handle, mirroring the BFFT real plan.
struct bodft_plan {
    bodft::plan f64;
    bodft::plan_f32 f32;
    explicit bodft_plan(int n) : f64(n), f32(n) {}
};

namespace {

bodft::complex_t* as_c(bfft_complex* v) { return reinterpret_cast<bodft::complex_t*>(v); }
const bodft::complex_t* as_c(const bfft_complex* v) {
    return reinterpret_cast<const bodft::complex_t*>(v);
}
bodft::complex_f32_t* as_c(bfft_complex_f32* v) {
    return reinterpret_cast<bodft::complex_f32_t*>(v);
}
const bodft::complex_f32_t* as_c(const bfft_complex_f32* v) {
    return reinterpret_cast<const bodft::complex_f32_t*>(v);
}

bool valid_size(size_t n) {
    if (n > static_cast<size_t>(2147483647)) return false;
    const int ni = static_cast<int>(n);
    return ni >= 2 && (ni & (ni - 1)) == 0;
}

} // namespace

const char* bodft_backend_name(void) {
    return bruun::simd_backend_name();
}

bfft_status bodft_plan_create(size_t n, bodft_plan** plan) {
    if (plan == nullptr) return BFFT_ERROR_INVALID_ARGUMENT;
    *plan = nullptr;
    if (!valid_size(n)) return BFFT_ERROR_INVALID_ARGUMENT;
    bodft_plan* p = new (std::nothrow) bodft_plan(static_cast<int>(n));
    if (!p) return BFFT_ERROR_ALLOCATION;
    *plan = p;
    return BFFT_OK;
}

void bodft_plan_destroy(bodft_plan* plan) { delete plan; }

size_t bodft_plan_size(const bodft_plan* plan) {
    return plan ? static_cast<size_t>(plan->f64.size()) : 0;
}

size_t bodft_plan_bins(const bodft_plan* plan) {
    return plan ? static_cast<size_t>(plan->f64.bins()) : 0;
}

bfft_status bodft_forward(const bodft_plan* plan, const double* input, bfft_complex* output) {
    if (!plan || !input || !output) return BFFT_ERROR_INVALID_ARGUMENT;
    plan->f64.forward(input, as_c(output));
    return BFFT_OK;
}

bfft_status bodft_forward_numba(const bodft_plan* plan, const double* input, bfft_complex* output,
                                double* /*work*/, bfft_complex* /*native_scratch*/) {
    return bodft_forward(plan, input, output);
}

bfft_status bodft_inverse(const bodft_plan* plan, const bfft_complex* input, double* output) {
    if (!plan || !input || !output) return BFFT_ERROR_INVALID_ARGUMENT;
    plan->f64.inverse(as_c(input), output);
    return BFFT_OK;
}

bfft_status bodft_forward_f32(const bodft_plan* plan, const float* input, bfft_complex_f32* output) {
    if (!plan || !input || !output) return BFFT_ERROR_INVALID_ARGUMENT;
    plan->f32.forward(input, as_c(output));
    return BFFT_OK;
}

bfft_status bodft_forward_numba_f32(const bodft_plan* plan, const float* input, bfft_complex_f32* output,
                                    float* /*work*/, bfft_complex_f32* /*native_scratch*/) {
    return bodft_forward_f32(plan, input, output);
}

bfft_status bodft_inverse_f32(const bodft_plan* plan, const bfft_complex_f32* input, float* output) {
    if (!plan || !input || !output) return BFFT_ERROR_INVALID_ARGUMENT;
    plan->f32.inverse(as_c(input), output);
    return BFFT_OK;
}
