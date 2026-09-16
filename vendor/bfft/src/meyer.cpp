#include <bfft/meyer.h>

#include "detail/meyer_kernel.hpp"

#include <new>

struct bfft_meyer_plan {
    meyer::engine eng;
    bool configured = true;
};

namespace {

bool pow2_ge8(size_t n) { return n >= 8 && (n & (n - 1)) == 0; }

}  // namespace

bfft_status bfft_meyer_plan_create(size_t height, size_t width, double lam,
                                   double mu, int passes, int rung_sweeps,
                                   double rung_tol, int threads,
                                   bfft_meyer_plan** plan) {
    if (plan == nullptr) return BFFT_ERROR_INVALID_ARGUMENT;
    *plan = nullptr;
    if (height < 2 || width < 2 ||
        (!pow2_ge8(height) && !pow2_ge8(width)))
        return BFFT_ERROR_INVALID_ARGUMENT;
    if (!(lam > 0.0) || !(mu > 0.0) || passes < 1 || rung_sweeps < 1 ||
        !(rung_tol >= 0.0) || threads < 0 || threads > 64)
        return BFFT_ERROR_INVALID_ARGUMENT;
    bfft_meyer_plan* p = new (std::nothrow) bfft_meyer_plan();
    if (p == nullptr) return BFFT_ERROR_ALLOCATION;
    const bfft_status st =
        p->eng.init(height, width, lam, mu, passes, rung_sweeps, rung_tol,
                    threads);
    if (st != BFFT_OK) {
        delete p;
        return st;
    }
    p->configured = pow2_ge8(height) && pow2_ge8(width);
    *plan = p;
    return BFFT_OK;
}

void bfft_meyer_plan_destroy(bfft_meyer_plan* plan) { delete plan; }

size_t bfft_meyer_plan_height(const bfft_meyer_plan* plan) {
    return plan ? plan->eng.H : 0;
}

size_t bfft_meyer_plan_width(const bfft_meyer_plan* plan) {
    return plan ? plan->eng.W : 0;
}

bfft_status bfft_meyer_plan_set_passes(bfft_meyer_plan* plan, int passes) {
    if (plan == nullptr || passes < 1) return BFFT_ERROR_INVALID_ARGUMENT;
    plan->eng.passes = passes;
    return BFFT_OK;
}

bfft_status bfft_meyer_plan_set_solver(bfft_meyer_plan* plan, int mode) {
    if (plan == nullptr || !plan->eng.set_solver(mode))
        return BFFT_ERROR_INVALID_ARGUMENT;
    plan->configured = true;
    return BFFT_OK;
}

int bfft_meyer_plan_solver(const bfft_meyer_plan* plan) {
    return plan ? plan->eng.solver : 0;
}

bfft_status bfft_meyer_split(bfft_meyer_plan* plan, const double* image,
                             double* cartoon, double* texture) {
    if (plan == nullptr || !plan->configured || image == nullptr ||
        cartoon == nullptr ||
        texture == nullptr || cartoon == texture)
        return BFFT_ERROR_INVALID_ARGUMENT;
    if (!plan->eng.facr_active) {
        if (!plan->eng.split_flow_jump(
                image, cartoon, texture, 4, 10, 2, 5))
            return BFFT_ERROR_INVALID_ARGUMENT;
    } else {
        // The semismooth chart currently uses the complete 2-D spectrum.
        // Periodic and Neumann FACR therefore retain the configured fused
        // alternation rather than falling back to the defective scalar hard
        // jump.
        plan->eng.split_effective(image, cartoon, texture);
    }
    return BFFT_OK;
}

bfft_status bfft_meyer_split_legacy(
        bfft_meyer_plan* plan, const double* image,
        double* cartoon, double* texture) {
    if (plan == nullptr || !plan->configured || image == nullptr ||
        cartoon == nullptr || texture == nullptr)
        return BFFT_ERROR_INVALID_ARGUMENT;
    plan->eng.split(image, cartoon, texture);
    return BFFT_OK;
}

bfft_status bfft_meyer_split_conditioned_first(
        bfft_meyer_plan* plan, const double* image,
        double* cartoon, double* texture, double strength) {
    if (plan == nullptr || !plan->configured || image == nullptr ||
        cartoon == nullptr || texture == nullptr || !(strength >= 0.0))
        return BFFT_ERROR_INVALID_ARGUMENT;
    if (!plan->eng.split_conditioned_first(
            image, cartoon, texture, strength))
        return BFFT_ERROR_INVALID_ARGUMENT;
    return BFFT_OK;
}

bfft_status bfft_meyer_split_preconditioned(
        bfft_meyer_plan* plan, const double* image,
        double* cartoon, double* texture, double strength,
        int virtual_passes, int gate_power) {
    if (plan == nullptr || !plan->configured || image == nullptr ||
        cartoon == nullptr || texture == nullptr || !(strength >= 0.0) ||
        virtual_passes < 1 || virtual_passes > 64 || gate_power < 1 ||
        gate_power > 64)
        return BFFT_ERROR_INVALID_ARGUMENT;
    if (!plan->eng.split_preconditioned(
            image, cartoon, texture, strength, virtual_passes, gate_power))
        return BFFT_ERROR_INVALID_ARGUMENT;
    return BFFT_OK;
}

bfft_status bfft_meyer_split_jump_measure(
        bfft_meyer_plan* plan, const double* image,
        double* cartoon, double* texture, int virtual_passes) {
    if (plan == nullptr || !plan->configured || image == nullptr ||
        cartoon == nullptr || texture == nullptr || cartoon == texture ||
        virtual_passes < 1 || virtual_passes > 64)
        return BFFT_ERROR_INVALID_ARGUMENT;
    if (!plan->eng.split_jump_measure(
            image, cartoon, texture, virtual_passes))
        return BFFT_ERROR_INVALID_ARGUMENT;
    return BFFT_OK;
}

bfft_status bfft_meyer_split_flow_jump(
        bfft_meyer_plan* plan, const double* image,
        double* cartoon, double* texture,
        int prefix_passes, int horizon,
        int settle_passes, int jump_count) {
    if (plan == nullptr || !plan->configured || image == nullptr ||
        cartoon == nullptr || texture == nullptr || cartoon == texture ||
        prefix_passes < 1 || prefix_passes > 64 ||
        horizon < 1 || horizon > 64 ||
        settle_passes < 0 || settle_passes > 64 ||
        jump_count < 1 || jump_count > 16)
        return BFFT_ERROR_INVALID_ARGUMENT;
    if (!plan->eng.split_flow_jump(
            image, cartoon, texture, prefix_passes, horizon,
            settle_passes, jump_count))
        return BFFT_ERROR_INVALID_ARGUMENT;
    return BFFT_OK;
}

bfft_status bfft_meyer_split_trace(bfft_meyer_plan* plan,
                                   const double* image,
                                   double* cartoon_trace,
                                   double* texture_trace) {
    if (plan == nullptr || !plan->configured || image == nullptr ||
        cartoon_trace == nullptr ||
        texture_trace == nullptr)
        return BFFT_ERROR_INVALID_ARGUMENT;
    plan->eng.split_trace(image, cartoon_trace, texture_trace);
    return BFFT_OK;
}

bfft_status bfft_meyer_split_visit(bfft_meyer_plan* plan,
                                   const double* image,
                                   bfft_meyer_trace_visitor visitor,
                                   void* user) {
    if (plan == nullptr || !plan->configured || image == nullptr ||
        visitor == nullptr)
        return BFFT_ERROR_INVALID_ARGUMENT;
    plan->eng.split_visit(image, visitor, user);
    return BFFT_OK;
}

bfft_status bfft_meyer_decompose(bfft_meyer_plan* plan, const double* image,
                                 double* cartoon, double* texture,
                                 double* band_coarse, double* band_mid,
                                 double* band_fine) {
    if (plan == nullptr || !plan->configured || image == nullptr ||
        cartoon == nullptr ||
        texture == nullptr || band_coarse == nullptr || band_mid == nullptr ||
        band_fine == nullptr)
        return BFFT_ERROR_INVALID_ARGUMENT;
    plan->eng.decompose(image, cartoon, texture, band_coarse, band_mid,
                        band_fine);
    return BFFT_OK;
}

bfft_status bfft_meyer_rof(bfft_meyer_plan* plan, const double* image,
                           double* smooth, double c, double eta, int sweeps,
                           double tol) {
    if (plan == nullptr || !plan->configured || image == nullptr ||
        smooth == nullptr)
        return BFFT_ERROR_INVALID_ARGUMENT;
    if (!(c > 0.0) || sweeps < 1 || !(tol >= 0.0))
        return BFFT_ERROR_INVALID_ARGUMENT;
    if (!(eta > 0.0)) eta = 10.0 * c;
    plan->eng.rof(image, smooth, c, eta, sweeps, tol);
    return BFFT_OK;
}

bfft_status bfft_meyer_rof_accelerated(
        bfft_meyer_plan* plan, const double* image, double* smooth,
        double c, double eta, int sweeps, double tol, int hodge_after) {
    if (plan == nullptr || !plan->configured || image == nullptr ||
        smooth == nullptr)
        return BFFT_ERROR_INVALID_ARGUMENT;
    if (!(c > 0.0) || sweeps < 1 || !(tol >= 0.0) ||
        hodge_after < 1 || hodge_after > sweeps)
        return BFFT_ERROR_INVALID_ARGUMENT;
    if (!(eta > 0.0)) eta = 10.0 * c;
    if (!plan->eng.rof_accelerated(
            image, smooth, c, eta, sweeps, tol, hodge_after))
        return BFFT_ERROR_INVALID_ARGUMENT;
    return BFFT_OK;
}

int bfft_meyer_plan_last_rof_sweeps(const bfft_meyer_plan* plan) {
    return plan == nullptr ? 0 : plan->eng.last_rof_sweeps;
}

int bfft_meyer_plan_last_rof_hodge_applied(
        const bfft_meyer_plan* plan) {
    return plan != nullptr && plan->eng.last_rof_hodge_applied ? 1 : 0;
}
