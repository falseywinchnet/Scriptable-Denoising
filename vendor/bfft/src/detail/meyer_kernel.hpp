#pragma once

// Internal Meyer G-norm decomposer kernel (transport geometry fusion
// descent).  See include/bfft/meyer.h for the contract,
// notes/meyer_bregman_ladder.md for the algorithm measurements, and
// notes/meyer_accel_theory.md for the reduced-composite theory (the
// alternation is ISTA at step 1/lambda on a static convex composite).
//
// TRANSFORM ECONOMY.  One Split Bregman sweep of ROF(g, c) is
//
//     u = F^-1[ (c*g_hat - eta*div(d-b)_hat) / (c - eta*lap_hat) ]
//     t = grad(u) + b;  coef = max(|t| - 1/eta, 0)/|t|
//     d = coef*t;       b <- b + grad(u) - d
//
// Only div(d-b) ever needs a forward transform: g_hat is maintained
// spectrally.  In the outer alternation g_u = f - v = u + w and
// g_v = f - u, so keeping the spectra of f, u and w makes every sweep cost
// exactly one forward + one inverse 2-D transform.  The texture image v is
// never materialized until the end (v = f - u - w, one fused pass).
//
// SHRINK ALGEBRA.  The full ladder path uses the fused update
//
//     b_new  = (1 - coef) * t
//     d - b_new = (2*coef - 1) * t      (the only combination ever used)
//
// The split-only API closes the same recursion on t itself:
//
//     p = t - 2*proj(t);  u <- solve(c*g - eta*div(p))
//     t <- grad(u) + proj(t)
//
// It carries two planes per subproblem instead of b plus materialized db.
// Phase contract (x is u or w): incoming b = t - grad(x), p = proj(t),
// d = t - p.  An ordinary pass satisfies
//     p - b = grad(x) - d,
//     t_next - t = grad(x_next - x) + (p - b).
// Thus a small primal change alone does not certify a settled full state.
// Keep the complete vector t: a divergence-free component can affect a
// later pointwise projection and hence a later divergence.  These identities
// and the exact finite-increment oracle are recorded in
// experiments/meyer_transport_audit/PRE_REFRESH_INTEGRATION.md.
// The scalar divergence is streamed into the soon-to-be-overwritten iterate
// plane, so the reflected vector field never lands in memory.
//
// 2-D TRANSFORMS over the library's 1-D real plans, k-major layout.
// Forward: rows are transformed in panels of eight into a small complex
// stage, then panel-transposed into column-major Re/Im planes
// (reT/imT[k*H + i]); the column stage is ZERO-COPY -- each column rfft
// reads a contiguous H-run and writes its half-spectrum directly into the
// spectrum array, whose layout is complex (k, m) = a[2*(k*HB + m)].  The
// screened-Poisson symbol is real and even in both indices, so scaling the
// two planes entrywise IS the 2-D spectral solve, and linear combinations
// of spectra are exact; symbol tables are pre-expanded to interleaved
// (re, im) stride so every spectral loop is a pure unit-stride stream.
// Inverse reverses the two stages.
//
// THREADING.  Every stage is embarrassingly parallel: row panels, column
// transforms, shrink rows, and solve ranges partition disjointly.  A
// persistent pool of T-1 workers plus the caller executes each stage as
// one barrier region (~6 regions per sweep); each lane owns its own row
// and column plans, work buffers, stage panel, and line buffer, so no
// library state is ever shared between lanes.  The rung tolerance
// reduction stays SERIAL by design: outputs are bit-identical for every
// thread count.
//
// Everything is double; the library's inverse is numpy-normalized, so no
// scale factors appear anywhere.

#include <bfft/bfft.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cassert>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <numeric>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

namespace meyer {

using trace_visitor = void (*)(int pass, const double* cartoon,
                               const double* texture, std::size_t count,
                               void* user);

constexpr std::size_t PANEL = 8;   // rows per panel transpose; H,W >= 8
constexpr double PI = 3.141592653589793238462643383279502884;

// ---- persistent worker pool: run(f) executes f(tid) on all T lanes ------

class pool {
public:
    void start(int threads) {
        T = threads < 1 ? 1 : threads;
        for (int t = 1; t < T; ++t)
            workers.emplace_back([this, t] { worker(t); });
    }
    ~pool() {
        if (!workers.empty()) {
            {
                std::lock_guard<std::mutex> lk(m);
                stopping = true;
                ++epoch;
            }
            cv.notify_all();
            for (auto& w : workers) w.join();
        }
    }
    int lanes() const { return T; }

    template <typename Function>
    void run(Function&& f) {
        if (T <= 1) {
            f(0);
            return;
        }
        using function_type = std::remove_reference_t<Function>;
        {
            std::lock_guard<std::mutex> lk(m);
            job_context = std::addressof(f);
            job_invoke = &invoke<function_type>;
            done = 0;
            ++epoch;
        }
        cv.notify_all();
        f(0);
        std::unique_lock<std::mutex> lk(m);
        cv_done.wait(lk, [this] { return done == T - 1; });
        job_context = nullptr;
        job_invoke = nullptr;
    }

private:
    template <typename Function>
    static void invoke(void* context, int tid) {
        (*static_cast<Function*>(context))(tid);
    }

    void worker(int tid) {
        long seen = 0;
        for (;;) {
            void* context = nullptr;
            void (*call)(void*, int) = nullptr;
            {
                std::unique_lock<std::mutex> lk(m);
                cv.wait(lk, [&] { return epoch != seen; });
                seen = epoch;
                if (stopping) return;
                context = job_context;
                call = job_invoke;
            }
            call(context, tid);
            {
                std::lock_guard<std::mutex> lk(m);
                ++done;
            }
            cv_done.notify_one();
        }
    }

    std::vector<std::thread> workers;
    std::mutex m;
    std::condition_variable cv, cv_done;
    void* job_context = nullptr;
    void (*job_invoke)(void*, int) = nullptr;
    long epoch = 0;
    int done = 0, T = 1;
    bool stopping = false;
};

struct fft1 {
    bfft_plan* plan = nullptr;
    bfft_workspace* workspace = nullptr;
    std::size_t n = 0, bins = 0;

    bfft_status init(std::size_t size) {
        n = size;
        bfft_status st = bfft_plan_create(size, &plan);
        if (st != BFFT_OK) return st;
        bins = bfft_plan_bins(plan);
        return bfft_workspace_create(plan, &workspace);
    }
    ~fft1() {
        bfft_workspace_destroy(workspace);
        bfft_plan_destroy(plan);
    }

    void fwd(const double* in, bfft_complex* out) {
        bfft_forward_workspace(plan, workspace, in, out);
    }
    void inv(const bfft_complex* in, double* out) {
        bfft_inverse_workspace(plan, workspace, in, out);
    }
};

// Spectrum of a real H x W image, k-major: plane a = column spectra of the
// row-rfft Re plane, plane b likewise for Im; a[2*(k*HB + m)] is the real
// part of column-frequency m at row-frequency k.
struct spectrum {
    std::vector<double> a, b;   // each 2*WB*HB doubles
    void alloc(std::size_t hb, std::size_t wb) {
        a.assign(2 * wb * hb, 0.0);
        b.assign(2 * wb * hb, 0.0);
    }
    void zero() {
        std::memset(a.data(), 0, a.size() * sizeof(double));
        std::memset(b.data(), 0, b.size() * sizeof(double));
    }
};

// One vector in the six-field reduced Meyer state and the spectra of its
// two primal coordinates.  The finite-flow accelerator allocates three of
// these lazily: two Arnoldi basis vectors and one work vector.  Keeping the
// primal spectra with the basis makes each tangent action cost the same two
// forward/two inverse transforms as one ordinary fused pass.
struct flow_vector {
    std::array<std::vector<double>, 6> field;
    spectrum us, ws;

    void alloc(std::size_t count, std::size_t hb, std::size_t wb) {
        for (auto& value : field) value.assign(count, 0.0);
        us.alloc(hb, wb);
        ws.alloc(hb, wb);
    }
};

// One-axis real spectra for FACR.  The swept coordinate is unit-stride
// within each transformed-axis bin, independently for real and imaginary
// parts.  This makes both Thomas passes streaming.
struct facr_spectrum {
    std::vector<double> a, b;  // each bins * swept
    void alloc(std::size_t swept, std::size_t bins) {
        a.assign(swept * bins, 0.0);
        b.assign(swept * bins, 0.0);
    }
    void zero() {
        std::memset(a.data(), 0, a.size() * sizeof(double));
        std::memset(b.data(), 0, b.size() * sizeof(double));
    }
};

struct tri_factors {
    double c = 0.0, eta = 0.0;
    std::vector<double> pivot;  // [k * swept + s]
    std::vector<double> diagonal;  // unmodified periodic diagonal per bin
    // Optional cached cyclic Sherman-Morrison term. The fixed jump path
    // reuses its three factor families often enough to earn this storage.
    std::vector<double> correction;  // [k * swept + s]
    std::vector<double> correction_inverse;  // final denominator per bin
};

// per-lane transform state: own plans, work buffers, stage, line
struct lane {
    fft1 row, col;
    std::vector<bfft_complex> stage;   // PANEL * WB
    std::vector<double> line;          // max(H, W)
    std::vector<double> reflect_x, reflect_y;
    std::vector<double> correction;    // swept-axis Sherman-Morrison vector

    bfft_status init(std::size_t H, std::size_t W, std::size_t WB) {
        bfft_status st = BFFT_OK;
        if (W >= 8 && (W & (W - 1)) == 0) {
            st = row.init(W);
            if (st != BFFT_OK) return st;
        }
        if (H >= 8 && (H & (H - 1)) == 0) {
            st = col.init(H);
            if (st != BFFT_OK) return st;
        }
        const std::size_t max_bins = std::max(H / 2 + 1, WB);
        stage.assign(PANEL * max_bins, bfft_complex{0.0, 0.0});
        line.assign(std::max(H, W), 0.0);
        reflect_x.assign(std::max(H, W), 0.0);
        reflect_y.assign(std::max(H, W), 0.0);
        correction.assign(std::max(H, W), 0.0);
        return BFFT_OK;
    }
};

struct engine {
    std::size_t H = 0, W = 0, HB = 0, WB = 0;
    double lam = 0.05, mu = 40.0, rung_tol = 1e-5;
    int passes = 64, rung_sweeps = 600;
    int solver = 0;
    bool facr_active = false;
    bool sweep_height = true;
    std::size_t FS = 0, FT = 0, FB = 0;

    pool P;
    std::vector<std::unique_ptr<lane>> lanes;

    // symbol tables 1/(c - eta*lap_hat), expanded to interleaved (re, im)
    // stride: s[2*(k*HB+m)] == s[2*(k*HB+m)+1], one per (c, eta) pair
    std::vector<double> s_u, s_v, s_r0, s_r1, s_r2;
    // table for the general ROF entry point, rebuilt only when (c, eta)
    // changes
    std::vector<double> s_gen;
    double gen_c = 0.0, gen_eta = 0.0;

    tri_factors t_u, t_v, t_r0, t_r1, t_r2, t_gen, t_poisson, t_virtual;
    int facr_virtual_passes = 0;

    // spatial planes, H*W
    std::vector<double> u, w, xit;             // xit = generic ROF iterate
    // bu*/bv* are the reduced t fields for split, and are reused as the
    // conventional b fields if the lazy ladder path is requested.
    std::vector<double> bux, buy, dbux, dbuy;
    std::vector<double> bvx, bvy, dbvx, dbvy;
    std::vector<double> rbx, rby, rdbx, rdby;  // rung solver (reused)
    std::vector<double> vplane, prev, rhodge, rhodge_x, rhodge_y;
    // Lazy scratch for the opt-in first-pass structural conditioner.
    std::vector<double> condition_gate;
    double jump_confidence_boundary = 0.0;
    double jump_confidence_inverse_span = 0.0;
    int last_rof_sweeps = 0;
    bool last_rof_hodge_applied = false;

    // column-major stage planes for the 2-D transforms, WB*H each
    std::vector<double> reT, imT;

    spectrum f_spec, u_spec, w_spec, d_spec, q_spec, v_spec;
    flow_vector flow_q0, flow_q1, flow_work;
    facr_spectrum ff_spec, fu_spec, fw_spec, fd_spec, fq_spec, fv_spec;

    bfft_status init(std::size_t h, std::size_t wdt, double lam_, double mu_,
                     int passes_, int rung_sweeps_, double rung_tol_,
                     int threads) {
        H = h; W = wdt; HB = H / 2 + 1; WB = W / 2 + 1;
        lam = lam_; mu = mu_; passes = passes_;
        rung_sweeps = rung_sweeps_; rung_tol = rung_tol_;

        if (threads < 1) {
            const unsigned hw = std::thread::hardware_concurrency();
            // Transform panels are dynamically claimed, so heterogeneous
            // cores no longer have to finish equal fixed quotas. Use the
            // available small-core complement too, while avoiding excessive
            // pools on large servers unless the caller explicitly asks.
            threads = hw ? int(std::min(hw, 8U)) : 1;
        }
        // no more lanes than transform-line panels
        const std::size_t max_lanes =
            std::max<std::size_t>(1, std::max(H, W) / PANEL);
        if (std::size_t(threads) > max_lanes) threads = int(max_lanes);
        P.start(threads);
        lanes.clear();
        for (int t = 0; t < P.lanes(); ++t) {
            lanes.emplace_back(new lane());
            bfft_status st = lanes.back()->init(H, W, WB);
            if (st != BFFT_OK) return st;
        }

        const std::size_t n = H * W;
        // The split-only hot path owns just its two iterates and two
        // R^2-valued reflected-dual fields.  Ladder/ROF state is lazy.
        for (auto* p : {&u, &w, &bux, &buy, &bvx, &bvy})
            p->assign(n, 0.0);
        if (pow2_ge8(H) && pow2_ge8(W)) ensure_spectral_storage();
        return BFFT_OK;
    }

    static bool pow2_ge8(std::size_t n) {
        return n >= 8 && (n & (n - 1)) == 0;
    }

    void ensure_spectral_storage() {
        if (!s_u.empty()) return;
        symbol(s_u, lam, 2.0 * lam);
        symbol(s_v, 1.0 / mu, 10.0 / mu);
        reT.assign(WB * H, 0.0);
        imT.assign(WB * H, 0.0);
        // q_spec holds the second reflected-divergence spectrum in the
        // split-only triangular Meyer pass.
        for (auto* s : {&f_spec, &u_spec, &w_spec, &d_spec, &q_spec})
            s->alloc(HB, WB);
    }

    void ensure_decompose_storage() {
        const std::size_t n = H * W;
        if (vplane.empty()) vplane.assign(n, 0.0);
        ensure_rof_storage();
        if (facr_active) {
            if (fv_spec.a.empty()) fv_spec.alloc(FS, FB);
            if (t_r0.pivot.empty()) {
                const double m0 = mu, m1 = mu / 4.0, m2 = mu / 16.0;
                build_factors(t_r0, 1.0 / m0, 10.0 / m0);
                build_factors(t_r1, 1.0 / m1, 10.0 / m1);
                build_factors(t_r2, 1.0 / m2, 10.0 / m2);
            }
        } else {
            if (v_spec.a.empty()) v_spec.alloc(HB, WB);
            if (s_r0.empty()) {
                const double m0 = mu, m1 = mu / 4.0, m2 = mu / 16.0;
                symbol(s_r0, 1.0 / m0, 10.0 / m0);
                symbol(s_r1, 1.0 / m1, 10.0 / m1);
                symbol(s_r2, 1.0 / m2, 10.0 / m2);
            }
        }
    }

    void ensure_rof_storage() {
        const std::size_t n = H * W;
        for (auto* p : {&xit, &rbx, &rby, &rdbx, &rdby, &prev})
            if (p->empty()) p->assign(n, 0.0);
    }

    void ensure_rof_hodge_storage() {
        ensure_rof_storage();
        const std::size_t n = H * W;
        for (auto* p : {&rhodge, &rhodge_x, &rhodge_y})
            if (p->empty()) p->assign(n, 0.0);
    }

    void ensure_visit_storage() {
        if (vplane.empty()) vplane.assign(H * W, 0.0);
    }

    void ensure_conditioning_storage() {
        if (condition_gate.empty()) condition_gate.assign(H * W, 0.0);
    }

    void ensure_jump_measure_storage() {
        ensure_conditioning_storage();
    }

    void ensure_flow_storage() {
        if (!flow_q0.field[0].empty()) return;
        const std::size_t n = H * W;
        flow_q0.alloc(n, HB, WB);
        flow_q1.alloc(n, HB, WB);
        flow_work.alloc(n, HB, WB);
    }

    void clear_spectral_storage() {
        for (auto* v : {&s_u, &s_v, &s_r0, &s_r1, &s_r2, &s_gen,
                        &reT, &imT})
            std::vector<double>().swap(*v);
        for (auto* s : {&f_spec, &u_spec, &w_spec, &d_spec, &q_spec,
                        &v_spec}) {
            std::vector<double>().swap(s->a);
            std::vector<double>().swap(s->b);
        }
    }

    void clear_facr_storage() {
        for (auto* s : {&ff_spec, &fu_spec, &fw_spec, &fd_spec, &fq_spec,
                        &fv_spec}) {
            std::vector<double>().swap(s->a);
            std::vector<double>().swap(s->b);
        }
        for (auto* t : {&t_u, &t_v, &t_r0, &t_r1, &t_r2, &t_gen,
                        &t_poisson, &t_virtual}) {
            std::vector<double>().swap(t->pivot);
            std::vector<double>().swap(t->diagonal);
            std::vector<double>().swap(t->correction);
            std::vector<double>().swap(t->correction_inverse);
        }
        facr_virtual_passes = 0;
    }

    bool set_solver(int mode) {
        if (mode < 0 || mode > 2) return false;
        if (mode != 0 && !pow2_ge8(H) && !pow2_ge8(W)) return false;
        if (mode == 0 && (!pow2_ge8(H) || !pow2_ge8(W))) return false;
        solver = mode;
        facr_active = mode != 0 && (mode == 2 ||
            !pow2_ge8(H) || !pow2_ge8(W));
        if (!facr_active) {
            clear_facr_storage();
            ensure_spectral_storage();
            return true;
        }
        clear_spectral_storage();

        // Sweep the axis whose removal saves the most padding.  If only one
        // dimension is transformable, the choice is forced.
        if (!pow2_ge8(H)) {
            sweep_height = true;
        } else if (!pow2_ge8(W)) {
            sweep_height = false;
        } else {
            sweep_height = true;  // equal transformed area: deterministic tie
        }
        FS = sweep_height ? H : W;
        FT = sweep_height ? W : H;
        FB = FT / 2 + 1;

        for (auto* s : {&ff_spec, &fu_spec, &fw_spec, &fd_spec, &fq_spec})
            s->alloc(FS, FB);
        build_factors(t_u, lam, 2.0 * lam, solver == 1);
        build_factors(t_v, 1.0 / mu, 10.0 / mu);
        // The periodic Poisson right-hand sides below are divergences and
        // therefore have zero mean.  A tiny screened DC term makes the
        // cyclic Thomas factor nonsingular; facr_poisson removes the DC
        // component before and after the solve, so this term never enters
        // the represented field.
        if (solver == 1)
            build_factors(t_poisson, 1e-8, 1.0, true);
        if (solver == 1) {
            // Default jump depth K=8 represented as two equal backward
            // resolvents.  (1 + K*x/2)^-2 matches the exact (1+x)^-K
            // generator at x=0 while retaining quadratic high-frequency
            // decay.  This removes six cyclic solves from each of the two
            // virtual observations.
            build_factors(t_virtual, lam, 8.0 * lam, true);
            facr_virtual_passes = 8;
        }
        t_gen.pivot.clear();
        return true;
    }

    void build_factors(tri_factors& t, double c, double eta,
                       bool cache_correction = false) {
        t.c = c;
        t.eta = eta;
        t.pivot.resize(FB * FS);
        t.diagonal.resize(FB);
        if (solver == 1 && cache_correction) {
            t.correction.resize(FB * FS);
            t.correction_inverse.resize(FB);
        } else {
            t.correction.clear();
            t.correction_inverse.clear();
        }
        const double tau = 2.0 * PI / double(FT);
        for (std::size_t k = 0; k < FB; ++k) {
            const double lt = 2.0 * std::cos(tau * double(k)) - 2.0;
            const double base = c - eta * lt;
            const double d = base + 2.0 * eta;
            t.diagonal[k] = d;
            assert(d > 2.0 * eta);
            double* p = t.pivot.data() + k * FS;
            if (solver == 2) {
                p[0] = 1.0 / (base + eta);
                for (std::size_t s = 1; s < FS; ++s) {
                    const double diag =
                        (s + 1 == FS) ? base + eta : d;
                    p[s] = 1.0 / (diag - eta * eta * p[s - 1]);
                }
            } else {
                // Cyclic Thomas via one Sherman-Morrison correction.
                const double gamma = -d;
                p[0] = 1.0 / (d - gamma);
                for (std::size_t s = 1; s < FS; ++s) {
                    const double diag = (s + 1 == FS)
                        ? d - eta * eta / gamma : d;
                    p[s] = 1.0 / (diag - eta * eta * p[s - 1]);
                }
                if (cache_correction && FS > 2) {
                    double* z = t.correction.data() + k * FS;
                    const double gamma = -d;
                    z[0] = gamma * p[0];
                    for (std::size_t s = 1; s < FS; ++s) {
                        const double urhs =
                            (s + 1 == FS) ? -eta : 0.0;
                        z[s] = (urhs + eta * z[s - 1]) * p[s];
                    }
                    for (std::size_t s = FS - 1; s-- > 0;)
                        z[s] += eta * p[s] * z[s + 1];
                    const double q = eta / d;
                    t.correction_inverse[k] =
                        1.0 / (1.0 + z[0] + q * z[FS - 1]);
                }
            }
        }
    }

    void symbol(std::vector<double>& s, double c, double eta) {
        s.resize(2 * WB * HB);
        const double tau_h = 2.0 * PI / double(H);
        const double tau_w = 2.0 * PI / double(W);
        for (std::size_t k = 0; k < WB; ++k) {
            const double lx = 2.0 * std::cos(tau_w * double(k)) - 2.0;
            double* srow = s.data() + 2 * k * HB;
            for (std::size_t m = 0; m < HB; ++m) {
                const double ly = 2.0 * std::cos(tau_h * double(m)) - 2.0;
                const double val = 1.0 / (c - eta * (ly + lx));
                srow[2 * m] = val;
                srow[2 * m + 1] = val;
            }
        }
    }

    // ---- 2-D transforms, panel row stage + zero-copy column stage -------

    // stage panel (PANEL rows of row-spectra) -> column-major planes
    void panel_scatter(const lane& L, std::size_t i0) {
        const bfft_complex* __restrict st = L.stage.data();
        double* __restrict re = reT.data();
        double* __restrict im = imT.data();
        for (std::size_t k = 0; k < WB; ++k) {
            double* __restrict rk = re + k * H + i0;
            double* __restrict ik = im + k * H + i0;
            for (std::size_t r = 0; r < PANEL; ++r) {
                rk[r] = st[r * WB + k].re;
                ik[r] = st[r * WB + k].im;
            }
        }
    }

    // column-major planes -> stage panel of complex rows
    void panel_gather(lane& L, std::size_t i0) {
        bfft_complex* __restrict st = L.stage.data();
        const double* __restrict re = reT.data();
        const double* __restrict im = imT.data();
        for (std::size_t k = 0; k < WB; ++k) {
            const double* __restrict rk = re + k * H + i0;
            const double* __restrict ik = im + k * H + i0;
            for (std::size_t r = 0; r < PANEL; ++r) {
                st[r * WB + k].re = rk[r];
                st[r * WB + k].im = ik[r];
            }
        }
    }

    void cols_fwd(spectrum& spec) {
        std::atomic<std::size_t> next_column{0};
        P.run([&](int tid) {
            lane& L = *lanes[tid];
            for (;;) {
                const std::size_t k =
                    next_column.fetch_add(1, std::memory_order_relaxed);
                if (k >= WB) break;
                L.col.fwd(reT.data() + k * H,
                          reinterpret_cast<bfft_complex*>(
                              spec.a.data() + 2 * k * HB));
                L.col.fwd(imT.data() + k * H,
                          reinterpret_cast<bfft_complex*>(
                              spec.b.data() + 2 * k * HB));
            }
        });
    }

    void cols_inv(const spectrum& spec) {
        std::atomic<std::size_t> next_column{0};
        P.run([&](int tid) {
            lane& L = *lanes[tid];
            for (;;) {
                const std::size_t k =
                    next_column.fetch_add(1, std::memory_order_relaxed);
                if (k >= WB) break;
                L.col.inv(reinterpret_cast<const bfft_complex*>(
                              spec.a.data() + 2 * k * HB),
                          reT.data() + k * H);
                L.col.inv(reinterpret_cast<const bfft_complex*>(
                              spec.b.data() + 2 * k * HB),
                          imT.data() + k * H);
            }
        });
    }

    void fwd2d(const double* x, spectrum& spec) {
        std::atomic<std::size_t> next_panel{0};
        P.run([&](int tid) {
            lane& L = *lanes[tid];
            for (;;) {
                const std::size_t i0 =
                    next_panel.fetch_add(PANEL, std::memory_order_relaxed);
                if (i0 >= H) break;
                for (std::size_t r = 0; r < PANEL; ++r)
                    L.row.fwd(x + (i0 + r) * W, L.stage.data() + r * WB);
                panel_scatter(L, i0);
            }
        });
        cols_fwd(spec);
    }

    void inv2d(const spectrum& spec, double* x) {
        cols_inv(spec);
        std::atomic<std::size_t> next_panel{0};
        P.run([&](int tid) {
            lane& L = *lanes[tid];
            for (;;) {
                const std::size_t i0 =
                    next_panel.fetch_add(PANEL, std::memory_order_relaxed);
                if (i0 >= H) break;
                panel_gather(L, i0);
                for (std::size_t r = 0; r < PANEL; ++r)
                    L.row.inv(L.stage.data() + r * WB, x + (i0 + r) * W);
            }
        });
    }

    // Apply one directional-difference / oriented-Gaussian multiplier to
    // the split real-2D spectrum.  f_spec stores separate real transforms
    // of the row-spectrum real and imaginary planes.  The A +/- iB pair
    // reconstructs the positive/negative vertical frequencies; applying
    // both symbols and folding them back preserves that layout exactly.
    void directional_gaussian(const spectrum& src, spectrum& out,
                              int dy, int dx, double theta,
                              double sigma_long = 12.0,
                              double sigma_width = 0.75) {
        const double tau_h = 2.0 * PI / double(H);
        const double tau_w = 2.0 * PI / double(W);
        const double ct = std::cos(theta), st = std::sin(theta);
        P.run([&](int tid) {
            for (std::size_t k = std::size_t(tid); k < WB;
                 k += std::size_t(P.lanes())) {
                const double wx = tau_w * double(k);
                for (std::size_t m = 0; m < HB; ++m) {
                    const std::size_t r = 2 * (k * HB + m);
                    const double ar = src.a[r], ai = src.a[r + 1];
                    const double br = src.b[r], bi = src.b[r + 1];
                    const double wy = tau_h * double(m);

                    auto symbol = [&](double signed_wy,
                                      double& sr, double& si) {
                        const double phase =
                            wx * double(dx) + signed_wy * double(dy);
                        const double along = wx * ct + signed_wy * st;
                        const double across = -wx * st + signed_wy * ct;
                        const double gaussian = std::exp(-0.5 * (
                            sigma_long * along * along
                            + sigma_width * across * across));
                        sr = gaussian * (std::cos(phase) - 1.0);
                        si = gaussian * std::sin(phase);
                    };

                    double spr, spi, smr, smi;
                    symbol(wy, spr, spi);
                    symbol(-wy, smr, smi);

                    // P=A+iB and M=A-iB=conj(F(k,-m)).
                    const double p0r = ar - bi, p0i = ai + br;
                    const double m0r = ar + bi, m0i = ai - br;
                    const double pr = spr * p0r - spi * p0i;
                    const double pi = spr * p0i + spi * p0r;
                    // M is multiplied by conj(S(k,-m)).
                    const double mr = smr * m0r + smi * m0i;
                    const double mi = smr * m0i - smi * m0r;

                    out.a[r] = 0.5 * (pr + mr);
                    out.a[r + 1] = 0.5 * (pi + mi);
                    out.b[r] = 0.5 * (pi - mi);
                    out.b[r + 1] = -0.5 * (pr - mr);
                }
            }
        });
    }

    void fwd2d_conditioned_source_reflection(
            const double* image, const std::vector<double>& gate,
            double eta, double strength, spectrum& spec) {
        const double threshold = 1.0 / eta;
        std::atomic<std::size_t> next_panel{0};
        P.run([&](int tid) {
            lane& L = *lanes[tid];
            for (;;) {
                const std::size_t i0 =
                    next_panel.fetch_add(PANEL, std::memory_order_relaxed);
                if (i0 >= H) break;
                const std::size_t yp0 = (i0 == 0 ? H : i0) - 1;
                for (std::size_t x = 0; x < W; ++x) {
                    const std::size_t xp = x + 1 == W ? 0 : x + 1;
                    const std::size_t i = yp0 * W + x;
                    double unused, ry;
                    reflected(image[yp0 * W + xp] - image[i],
                              image[i0 * W + x] - image[i], threshold,
                              unused, ry);
                    L.correction[x] = strength * gate[i] * ry;
                }
                for (std::size_t row = 0; row < PANEL; ++row) {
                    const std::size_t y = i0 + row;
                    const std::size_t yn = y + 1 == H ? 0 : y + 1;
                    for (std::size_t x = 0; x < W; ++x) {
                        const std::size_t xn = x + 1 == W ? 0 : x + 1;
                        const std::size_t i = y * W + x;
                        double rx, ry;
                        reflected(image[y * W + xn] - image[i],
                                  image[yn * W + x] - image[i], threshold,
                                  rx, ry);
                        const double scale = strength * gate[i];
                        L.reflect_x[x] = scale * rx;
                        L.reflect_y[x] = scale * ry;
                    }
                    for (std::size_t x = 0; x < W; ++x) {
                        const std::size_t xp = (x == 0 ? W : x) - 1;
                        L.line[x] = L.reflect_x[x] - L.reflect_x[xp]
                            + L.reflect_y[x] - L.correction[x];
                    }
                    L.row.fwd(L.line.data(), L.stage.data() + row * WB);
                    std::memcpy(L.correction.data(), L.reflect_y.data(),
                                W * sizeof(double));
                }
                panel_scatter(L, i0);
            }
        });
        cols_fwd(spec);
    }

    // rhs = div(db), computed row-by-row into the lane's line buffer and
    // row-transformed: no full-plane rhs buffer exists.
    void fwd2d_div(const std::vector<double>& dbx,
                   const std::vector<double>& dby, spectrum& spec) {
        std::atomic<std::size_t> next_panel{0};
        P.run([&](int tid) {
            lane& L = *lanes[tid];
            double* __restrict ln = L.line.data();
            for (;;) {
                const std::size_t i0 =
                    next_panel.fetch_add(PANEL, std::memory_order_relaxed);
                if (i0 >= H) break;
                for (std::size_t r = 0; r < PANEL; ++r) {
                    const std::size_t i = i0 + r;
                    const double* __restrict px = dbx.data() + i * W;
                    const double* __restrict py = dby.data() + i * W;
                    const double* __restrict pyn =
                        dby.data() + ((i == 0 ? H : i) - 1) * W;
                    ln[0] = px[0] - px[W - 1] + py[0] - pyn[0];
                    for (std::size_t j = 1; j < W; ++j)
                        ln[j] = px[j] - px[j - 1] + py[j] - pyn[j];
                    L.row.fwd(ln, L.stage.data() + r * WB);
                }
                panel_scatter(L, i0);
            }
        });
        cols_fwd(spec);
    }

    static void reflected(double tx, double ty, double threshold,
                          double& px, double& py) {
        const double radius = std::sqrt(tx * tx + ty * ty);
        const double shrink =
            std::fmax(radius - threshold, 0.0) /
            std::fmax(radius, 1e-12);
        const double scale = 2.0 * shrink - 1.0;
        px = scale * tx;
        py = scale * ty;
    }

    // Transform div(t - 2*proj(t)) without materializing the reflected
    // vector field.  This closes Split Bregman on one R^2 state.
    void fwd2d_reflection(const std::vector<double>& tx,
                          const std::vector<double>& ty, double eta,
                          spectrum& spec) {
        const double threshold = 1.0 / eta;
        std::atomic<std::size_t> next_panel{0};
        P.run([&](int tid) {
            lane& L = *lanes[tid];
            for (;;) {
                const std::size_t i0 =
                    next_panel.fetch_add(PANEL, std::memory_order_relaxed);
                if (i0 >= H) break;
                const std::size_t yp0 = (i0 == 0 ? H : i0) - 1;
                for (std::size_t x = 0; x < W; ++x) {
                    double unused;
                    reflected(tx[yp0 * W + x], ty[yp0 * W + x],
                              threshold, unused, L.correction[x]);
                }
                for (std::size_t r = 0; r < PANEL; ++r) {
                    const std::size_t y = i0 + r;
                    for (std::size_t x = 0; x < W; ++x)
                        reflected(tx[y * W + x], ty[y * W + x], threshold,
                                  L.reflect_x[x], L.reflect_y[x]);
                    for (std::size_t x = 0; x < W; ++x) {
                        const std::size_t xp = (x == 0 ? W : x) - 1;
                        L.line[x] = L.reflect_x[x] - L.reflect_x[xp] +
                            L.reflect_y[x] - L.correction[x];
                    }
                    L.row.fwd(L.line.data(), L.stage.data() + r * WB);
                    std::memcpy(L.correction.data(), L.reflect_y.data(),
                                W * sizeof(double));
                }
                panel_scatter(L, i0);
            }
        });
        cols_fwd(spec);
    }

    static void projected_tangent(
            double tx, double ty, double hx, double hy, double radius,
            double& px, double& py) {
        const double magnitude = std::sqrt(tx * tx + ty * ty);
        if (!(magnitude > radius)) {
            px = hx;
            py = hy;
            return;
        }
        const double inverse = 1.0 / magnitude;
        const double nx = tx * inverse;
        const double ny = ty * inverse;
        const double tangent = -ny * hx + nx * hy;
        const double factor = radius * inverse;
        px = -factor * ny * tangent;
        py = factor * nx * tangent;
    }

    static void reflected_tangent(
            double tx, double ty, double hx, double hy, double radius,
            double& rx, double& ry) {
        double px, py;
        projected_tangent(tx, ty, hx, hy, radius, px, py);
        rx = hx - 2.0 * px;
        ry = hy - 2.0 * py;
    }

    // Transform the divergence of one semismooth reflected-map tangent
    // without materializing either vector component.  The branch is exactly
    // the Clarke derivative of the Euclidean disk projection already used
    // by the reduced Split-Bregman recurrence.
    void fwd2d_reflection_tangent(
            const std::vector<double>& tx,
            const std::vector<double>& ty,
            const std::vector<double>& hx,
            const std::vector<double>& hy,
            double eta, spectrum& spec) {
        const double radius = 1.0 / eta;
        std::atomic<std::size_t> next_panel{0};
        P.run([&](int tid) {
            lane& L = *lanes[tid];
            for (;;) {
                const std::size_t i0 =
                    next_panel.fetch_add(PANEL, std::memory_order_relaxed);
                if (i0 >= H) break;
                const std::size_t yp0 = (i0 == 0 ? H : i0) - 1;
                for (std::size_t x = 0; x < W; ++x) {
                    double unused;
                    reflected_tangent(
                        tx[yp0 * W + x], ty[yp0 * W + x],
                        hx[yp0 * W + x], hy[yp0 * W + x], radius,
                        unused, L.correction[x]);
                }
                for (std::size_t r = 0; r < PANEL; ++r) {
                    const std::size_t y = i0 + r;
                    for (std::size_t x = 0; x < W; ++x)
                        reflected_tangent(
                            tx[y * W + x], ty[y * W + x],
                            hx[y * W + x], hy[y * W + x], radius,
                            L.reflect_x[x], L.reflect_y[x]);
                    for (std::size_t x = 0; x < W; ++x) {
                        const std::size_t xp = (x == 0 ? W : x) - 1;
                        L.line[x] = L.reflect_x[x] - L.reflect_x[xp] +
                            L.reflect_y[x] - L.correction[x];
                    }
                    L.row.fwd(L.line.data(), L.stage.data() + r * WB);
                    std::memcpy(L.correction.data(), L.reflect_y.data(),
                                W * sizeof(double));
                }
                panel_scatter(L, i0);
            }
        });
        cols_fwd(spec);
    }

    // ---- one-axis transforms + FACR sweeps ------------------------------

    fft1& facr_fft(lane& L) { return sweep_height ? L.row : L.col; }

    template <bool Divergence>
    void facr_fwd_impl(const double* x, const std::vector<double>* dbx,
                       const std::vector<double>* dby, facr_spectrum& spec) {
        std::atomic<std::size_t> next_panel{0};
        P.run([&](int tid) {
            lane& L = *lanes[tid];
            fft1& F = facr_fft(L);
            for (;;) {
                const std::size_t s0 =
                    next_panel.fetch_add(PANEL, std::memory_order_relaxed);
                if (s0 >= FS) break;
                const std::size_t nr = std::min(PANEL, FS - s0);
                for (std::size_t r = 0; r < nr; ++r) {
                    const std::size_t s = s0 + r;
                    double* line = L.line.data();
                    const double* input = nullptr;
                    if constexpr (!Divergence) {
                        if (sweep_height) {
                            input = x + s * W;
                        } else {
                            for (std::size_t y = 0; y < H; ++y)
                                line[y] = x[y * W + s];
                            input = line;
                        }
                    } else {
                        const double* px = dbx->data();
                        const double* py = dby->data();
                        if (sweep_height) {
                            const std::size_t y = s;
                            const std::size_t yp = (y == 0 ? H : y) - 1;
                            const double* xr = px + y * W;
                            const double* yr = py + y * W;
                            const double* yprev = py + yp * W;
                            line[0] = xr[0] - xr[W - 1] + yr[0] -
                                (solver == 2 && y == 0 ? 0.0 : yprev[0]);
                            for (std::size_t j = 1; j < W; ++j)
                                line[j] = xr[j] - xr[j - 1] + yr[j] -
                                    (solver == 2 && y == 0
                                         ? 0.0 : yprev[j]);
                        } else {
                            const std::size_t col = s;
                            const std::size_t xp =
                                (col == 0 ? W : col) - 1;
                            for (std::size_t y = 0; y < H; ++y) {
                                const std::size_t yp =
                                    (y == 0 ? H : y) - 1;
                                const std::size_t i = y * W + col;
                                line[y] = px[i] - px[y * W + xp] + py[i] -
                                    py[yp * W + col];
                                if (solver == 2 && col == 0)
                                    line[y] += px[y * W + xp];
                            }
                        }
                        input = line;
                    }
                        F.fwd(input, L.stage.data() + r * FB);
                }
                for (std::size_t k = 0; k < FB; ++k) {
                    double* ar = spec.a.data() + k * FS + s0;
                    double* br = spec.b.data() + k * FS + s0;
                    for (std::size_t r = 0; r < nr; ++r) {
                        const bfft_complex z = L.stage[r * FB + k];
                        ar[r] = z.re;
                        br[r] = z.im;
                    }
                }
            }
        });
    }

    void facr_fwd(const double* x, facr_spectrum& spec) {
        facr_fwd_impl<false>(x, nullptr, nullptr, spec);
    }

    void facr_fwd_div(const std::vector<double>& dbx,
                      const std::vector<double>& dby, facr_spectrum& spec) {
        facr_fwd_impl<true>(nullptr, &dbx, &dby, spec);
    }

    void facr_fwd_reflection(const std::vector<double>& tx,
                             const std::vector<double>& ty, double eta,
                             facr_spectrum& spec) {
        const double threshold = 1.0 / eta;
        std::atomic<std::size_t> next_panel{0};
        P.run([&](int tid) {
            lane& L = *lanes[tid];
            fft1& F = facr_fft(L);
            for (;;) {
                const std::size_t s0 =
                    next_panel.fetch_add(PANEL, std::memory_order_relaxed);
                if (s0 >= FS) break;
                const std::size_t nr = std::min(PANEL, FS - s0);
                if (sweep_height) {
                    const std::size_t yp0 = (s0 == 0 ? H : s0) - 1;
                    for (std::size_t x = 0; x < W; ++x) {
                        if (solver == 2 && s0 == 0) {
                            L.correction[x] = 0.0;
                        } else {
                            double unused;
                            reflected(tx[yp0 * W + x], ty[yp0 * W + x],
                                      threshold, unused, L.correction[x]);
                        }
                    }
                    for (std::size_t r = 0; r < nr; ++r) {
                        const std::size_t y = s0 + r;
                        for (std::size_t x = 0; x < W; ++x)
                            reflected(tx[y * W + x], ty[y * W + x],
                                      threshold, L.reflect_x[x],
                                      L.reflect_y[x]);
                        for (std::size_t x = 0; x < W; ++x) {
                            const std::size_t xp = (x == 0 ? W : x) - 1;
                            L.line[x] =
                                L.reflect_x[x] - L.reflect_x[xp] +
                                L.reflect_y[x] - L.correction[x];
                        }
                        std::memcpy(L.correction.data(), L.reflect_y.data(),
                                    W * sizeof(double));
                        F.fwd(L.line.data(), L.stage.data() + r * FB);
                    }
                } else {
                    for (std::size_t r = 0; r < nr; ++r) {
                        const std::size_t x = s0 + r;
                        const std::size_t xp = (x == 0 ? W : x) - 1;
                        for (std::size_t y = 0; y < H; ++y)
                            reflected(tx[y * W + x], ty[y * W + x],
                                      threshold, L.reflect_x[y],
                                      L.reflect_y[y]);
                        for (std::size_t y = 0; y < H; ++y) {
                            const std::size_t yp = (y == 0 ? H : y) - 1;
                            double prev_x = 0.0, unused;
                            if (!(solver == 2 && x == 0))
                                reflected(tx[y * W + xp], ty[y * W + xp],
                                          threshold, prev_x, unused);
                            L.line[y] =
                                L.reflect_x[y] - prev_x +
                                L.reflect_y[y] - L.reflect_y[yp];
                        }
                        F.fwd(L.line.data(), L.stage.data() + r * FB);
                    }
                }
                for (std::size_t k = 0; k < FB; ++k) {
                    double* ar = spec.a.data() + k * FS + s0;
                    double* br = spec.b.data() + k * FS + s0;
                    for (std::size_t r = 0; r < nr; ++r) {
                        const bfft_complex z = L.stage[r * FB + k];
                        ar[r] = z.re;
                        br[r] = z.im;
                    }
                }
            }
        });
    }

    template <bool Subtract>
    void facr_inv_impl(const facr_spectrum& spec,
                       const facr_spectrum* subtract, double* x) {
        std::atomic<std::size_t> next_panel{0};
        P.run([&](int tid) {
            lane& L = *lanes[tid];
            fft1& F = facr_fft(L);
            for (;;) {
                const std::size_t s0 =
                    next_panel.fetch_add(PANEL, std::memory_order_relaxed);
                if (s0 >= FS) break;
                const std::size_t nr = std::min(PANEL, FS - s0);
                for (std::size_t k = 0; k < FB; ++k) {
                    const double* ar = spec.a.data() + k * FS + s0;
                    const double* br = spec.b.data() + k * FS + s0;
                    if constexpr (Subtract) {
                        const double* cr =
                            subtract->a.data() + k * FS + s0;
                        const double* dr =
                            subtract->b.data() + k * FS + s0;
                        for (std::size_t r = 0; r < nr; ++r) {
                            L.stage[r * FB + k].re = ar[r] - cr[r];
                            L.stage[r * FB + k].im = br[r] - dr[r];
                        }
                    } else {
                        for (std::size_t r = 0; r < nr; ++r) {
                            L.stage[r * FB + k].re = ar[r];
                            L.stage[r * FB + k].im = br[r];
                        }
                    }
                }
                for (std::size_t r = 0; r < nr; ++r) {
                    const std::size_t s = s0 + r;
                    if (sweep_height) {
                        F.inv(L.stage.data() + r * FB, x + s * W);
                    } else {
                        F.inv(L.stage.data() + r * FB, L.line.data());
                        for (std::size_t y = 0; y < H; ++y)
                            x[y * W + s] = L.line[y];
                    }
                }
            }
        });
    }

    void facr_inv(const facr_spectrum& spec, double* x) {
        facr_inv_impl<false>(spec, nullptr, x);
    }

    void facr_inv_difference(const facr_spectrum& lhs,
                             const facr_spectrum& rhs, double* x) {
        facr_inv_impl<true>(lhs, &rhs, x);
    }

    template <typename RhsA, typename RhsB>
    void facr_solve_pair(lane& L, std::size_t k, facr_spectrum& out,
                         const tri_factors& t, RhsA&& ra, RhsB&& rb) {
                double* va = out.a.data() + k * FS;
                double* vb = out.b.data() + k * FS;
                const double* p = t.pivot.data() + k * FS;
                if (solver == 1 && FS == 2) {
                    // The two periodic neighbors coincide at length two,
                    // so their couplings add instead of forming distinct
                    // tridiagonal and corner entries.
                    const double d = t.diagonal[k];
                    const double off = 2.0 * t.eta;
                    const double inv_det = 1.0 / (d * d - off * off);
                    const std::size_t i = k * FS;
                    const double ar0 = ra(i), ar1 = ra(i + 1);
                    const double br0 = rb(i), br1 = rb(i + 1);
                    va[0] = (d * ar0 + off * ar1) * inv_det;
                    va[1] = (off * ar0 + d * ar1) * inv_det;
                    vb[0] = (d * br0 + off * br1) * inv_det;
                    vb[1] = (off * br0 + d * br1) * inv_det;
                    return;
                }
                va[0] = ra(k * FS) * p[0];
                vb[0] = rb(k * FS) * p[0];
                for (std::size_t s = 1; s < FS; ++s) {
                    const double ps = p[s];
                    va[s] = (ra(k * FS + s) + t.eta * va[s - 1]) * ps;
                    vb[s] = (rb(k * FS + s) + t.eta * vb[s - 1]) * ps;
                }
                for (std::size_t s = FS - 1; s-- > 0;) {
                    const double ep = t.eta * p[s];
                    va[s] += ep * va[s + 1];
                    vb[s] += ep * vb[s + 1];
                }

                if (solver == 1) {
                    const double d = t.diagonal[k];
                    const double q = t.eta / d;
                    const bool cached = !t.correction.empty();
                    double* local_z = L.correction.data();
                    const double* z = cached
                        ? t.correction.data() + k * FS : local_z;
                    double inv_denom;
                    if (cached) {
                        inv_denom = t.correction_inverse[k];
                    } else {
                        const double gamma = -d;
                        local_z[0] = gamma * p[0];
                        for (std::size_t s = 1; s < FS; ++s) {
                            const double urhs =
                                (s + 1 == FS) ? -t.eta : 0.0;
                            local_z[s] =
                                (urhs + t.eta * local_z[s - 1]) * p[s];
                        }
                        for (std::size_t s = FS - 1; s-- > 0;)
                            local_z[s] +=
                                t.eta * p[s] * local_z[s + 1];
                        inv_denom = 1.0 /
                            (1.0 + local_z[0] + q * local_z[FS - 1]);
                    }
                    const double scale_a =
                        (va[0] + q * va[FS - 1]) * inv_denom;
                    const double scale_b =
                        (vb[0] + q * vb[FS - 1]) * inv_denom;
                    for (std::size_t s = 0; s < FS; ++s) {
                        va[s] -= scale_a * z[s];
                        vb[s] -= scale_b * z[s];
                    }
                }
    }

    template <typename RhsA, typename RhsB>
    void facr_solve(facr_spectrum& out, const tri_factors& t,
                    RhsA&& ra, RhsB&& rb) {
        P.run([&](int tid) {
            lane& L = *lanes[tid];
            for (std::size_t k = std::size_t(tid); k < FB;
                 k += std::size_t(P.lanes())) {
                facr_solve_pair(L, k, out, t, ra, rb);
            }
        });
    }

    void facr_scale(const facr_spectrum& src, double c,
                    const tri_factors& t, facr_spectrum& out) {
        facr_solve(out, t,
            [&](std::size_t i) { return c * src.a[i]; },
            [&](std::size_t i) { return c * src.b[i]; });
    }

    void facr_sum_inplace(facr_spectrum& io, const facr_spectrum& q,
                          const facr_spectrum& d, double c, double eta,
                          const tri_factors& t) {
        facr_solve(io, t,
            [&](std::size_t i) {
                return c * (io.a[i] + q.a[i]) - eta * d.a[i];
            },
            [&](std::size_t i) {
                return c * (io.b[i] + q.b[i]) - eta * d.b[i];
            });
    }

    void facr_diff(const facr_spectrum& p, const facr_spectrum& q,
                   const facr_spectrum* d, double c, double eta,
                   const tri_factors& t, facr_spectrum& out) {
        facr_solve(out, t,
            [&](std::size_t i) {
                return c * (p.a[i] - q.a[i]) -
                    (d == nullptr ? 0.0 : eta * d->a[i]);
            },
            [&](std::size_t i) {
                return c * (p.b[i] - q.b[i]) -
                    (d == nullptr ? 0.0 : eta * d->b[i]);
            });
    }

    void facr_g(const facr_spectrum& g, const facr_spectrum& d,
                double c, double eta, const tri_factors& t,
                facr_spectrum& out) {
        facr_solve(out, t,
            [&](std::size_t i) { return c * g.a[i] - eta * d.a[i]; },
            [&](std::size_t i) { return c * g.b[i] - eta * d.b[i]; });
    }

    // Mean-zero inverse of the periodic scalar Laplacian in the one-axis
    // FACR representation.  The transformed-axis DC line is projected to
    // zero before the screened solve and de-gauged afterward.  Every caller
    // supplies a divergence, but doing this explicitly prevents accumulated
    // transform roundoff from being magnified by the intentionally tiny DC
    // screen in t_poisson.
    void facr_poisson(const facr_spectrum& divergence,
                      facr_spectrum& potential) {
        double mean_a = 0.0, mean_b = 0.0;
        for (std::size_t s = 0; s < FS; ++s) {
            mean_a += divergence.a[s];
            mean_b += divergence.b[s];
        }
        mean_a /= double(FS);
        mean_b /= double(FS);
        facr_solve(potential, t_poisson,
            [&](std::size_t i) {
                return -divergence.a[i] + (i < FS ? mean_a : 0.0);
            },
            [&](std::size_t i) {
                return -divergence.b[i] + (i < FS ? mean_b : 0.0);
            });
        double gauge_a = 0.0, gauge_b = 0.0;
        for (std::size_t s = 0; s < FS; ++s) {
            gauge_a += potential.a[s];
            gauge_b += potential.b[s];
        }
        gauge_a /= double(FS);
        gauge_b /= double(FS);
        for (std::size_t s = 0; s < FS; ++s) {
            potential.a[s] -= gauge_a;
            potential.b[s] -= gauge_b;
        }
    }

    void facr_poisson_difference(const facr_spectrum& lhs,
                                 const facr_spectrum& rhs,
                                 facr_spectrum& potential) {
        double mean_a = 0.0, mean_b = 0.0;
        for (std::size_t s = 0; s < FS; ++s) {
            mean_a += lhs.a[s] - rhs.a[s];
            mean_b += lhs.b[s] - rhs.b[s];
        }
        mean_a /= double(FS);
        mean_b /= double(FS);
        facr_solve(potential, t_poisson,
            [&](std::size_t i) {
                return -(lhs.a[i] - rhs.a[i]) +
                    (i < FS ? mean_a : 0.0);
            },
            [&](std::size_t i) {
                return -(lhs.b[i] - rhs.b[i]) +
                    (i < FS ? mean_b : 0.0);
            });
        double gauge_a = 0.0, gauge_b = 0.0;
        for (std::size_t s = 0; s < FS; ++s) {
            gauge_a += potential.a[s];
            gauge_b += potential.b[s];
        }
        gauge_a /= double(FS);
        gauge_b /= double(FS);
        for (std::size_t s = 0; s < FS; ++s) {
            potential.a[s] -= gauge_a;
            potential.b[s] -= gauge_b;
        }
    }

    void facr_difference(const facr_spectrum& lhs,
                         const facr_spectrum& rhs,
                         facr_spectrum& out) {
        const std::size_t count = FS * FB;
        P.run([&](int tid) {
            std::size_t lo, hi;
            split(tid, count, lo, hi);
            for (std::size_t i = lo; i < hi; ++i) {
                out.a[i] = lhs.a[i] - rhs.a[i];
                out.b[i] = lhs.b[i] - rhs.b[i];
            }
        });
    }

    // Materialize source-potential once and return its lowpass. Consumers
    // take the final difference lazily, avoiding an otherwise write-only
    // oscillation spectrum.
    const facr_spectrum& build_virtual_lowpass_facr(
            const facr_spectrum& source,
            const facr_spectrum& potential, int virtual_passes) {
        facr_difference(source, potential, fd_spec);
        const int poles = std::min(virtual_passes, 2);
        if (facr_virtual_passes != virtual_passes) {
            const double eta = 2.0 * lam * double(virtual_passes) /
                double(poles);
            build_factors(t_virtual, lam, eta, true);
            facr_virtual_passes = virtual_passes;
        }
        facr_scale(fd_spec, lam, t_virtual, fu_spec);
        const facr_spectrum* lowpass = &fu_spec;
        if (poles == 2) {
            facr_scale(fu_spec, lam, t_virtual, fw_spec);
            lowpass = &fw_spec;
        }
        return *lowpass;
    }

    void update_reflected_dual_rows(const std::vector<double>& x,
                                    std::vector<double>& tx,
                                    std::vector<double>& ty, double eta,
                                    int tid) {
        const double threshold = 1.0 / eta;
        for (std::size_t y = std::size_t(tid); y < H;
             y += std::size_t(P.lanes())) {
            const double* xi = x.data() + y * W;
            const double* xn =
                x.data() + ((y + 1 == H) ? 0 : y + 1) * W;
            double* px = tx.data() + y * W;
            double* py = ty.data() + y * W;
            const bool neumann_y =
                solver == 2 && sweep_height && y + 1 == H;
                for (std::size_t j = 0; j < W; ++j) {
                    const double old_x = px[j], old_y = py[j];
                    const double radius =
                        std::sqrt(old_x * old_x + old_y * old_y);
                    const double shrink =
                        std::fmax(radius - threshold, 0.0) /
                        std::fmax(radius, 1e-12);
                    const double project = 1.0 - shrink;
                const bool neumann_x =
                    solver == 2 && !sweep_height && j + 1 == W;
                const double gx = neumann_x ? 0.0 :
                    xi[j + 1 == W ? 0 : j + 1] - xi[j];
                const double gy = neumann_y ? 0.0 : xn[j] - xi[j];
                px[j] = neumann_x ? 0.0 : gx + project * old_x;
                py[j] = neumann_y ? 0.0 : gy + project * old_y;
            }
        }
    }

    // t_next = grad(x) + proj(t, 1/eta), in place.  Together with
    // fwd*_reflection this is exactly the reduced Split-Bregman recursion.
    void update_reflected_dual(const std::vector<double>& x,
                               std::vector<double>& tx,
                               std::vector<double>& ty, double eta) {
        P.run([&](int tid) {
            update_reflected_dual_rows(x, tx, ty, eta, tid);
        });
    }

    void update_reflected_dual_pair(
            const std::vector<double>& x0, std::vector<double>& tx0,
            std::vector<double>& ty0, double eta0,
            const std::vector<double>& x1, std::vector<double>& tx1,
            std::vector<double>& ty1, double eta1) {
        P.run([&](int tid) {
            update_reflected_dual_rows(x0, tx0, ty0, eta0, tid);
            update_reflected_dual_rows(x1, tx1, ty1, eta1, tid);
        });
    }

    // Row-streamed divergence for the transpose-oriented FACR path.  The
    // current reflected row and one predecessor row are enough, so the
    // iterate plane can serve as scalar scratch without restoring either
    // eliminated vector-field plane.
    void reflection_divergence_rows(const std::vector<double>& tx,
                                    const std::vector<double>& ty,
                                    double eta, double* out, int tid) {
        const double threshold = 1.0 / eta;
        const std::size_t lanes_n = std::size_t(P.lanes());
        const std::size_t lo = H * std::size_t(tid) / lanes_n;
        const std::size_t hi = H * (std::size_t(tid) + 1) / lanes_n;
        if (lo == hi) return;
        lane& L = *lanes[tid];
        const std::size_t yp0 = (lo == 0 ? H : lo) - 1;
        for (std::size_t x = 0; x < W; ++x) {
            double unused;
            if (solver == 2 && sweep_height && lo == 0) {
                L.line[x] = 0.0;
            } else {
                reflected(tx[yp0 * W + x], ty[yp0 * W + x], threshold,
                          unused, L.line[x]);
            }
        }
        for (std::size_t y = lo; y < hi; ++y) {
            for (std::size_t x = 0; x < W; ++x) {
                reflected(tx[y * W + x], ty[y * W + x], threshold,
                          L.reflect_x[x], L.reflect_y[x]);
            }
            for (std::size_t x = 0; x < W; ++x) {
                const std::size_t xp = (x == 0 ? W : x) - 1;
                const double prev_x =
                    (solver == 2 && x == 0) ? 0.0 : L.reflect_x[xp];
                out[y * W + x] =
                    L.reflect_x[x] - prev_x +
                    L.reflect_y[x] - L.line[x];
            }
            std::memcpy(L.line.data(), L.reflect_y.data(),
                        W * sizeof(double));
        }
    }

    void reflection_divergence_plane(const std::vector<double>& tx,
                                     const std::vector<double>& ty,
                                     double eta, double* out) {
        P.run([&](int tid) {
            reflection_divergence_rows(tx, ty, eta, out, tid);
        });
    }

    void reflection_divergence_pair(
            const std::vector<double>& tx0, const std::vector<double>& ty0,
            double eta0, double* out0, const std::vector<double>& tx1,
            const std::vector<double>& ty1, double eta1, double* out1) {
        P.run([&](int tid) {
            reflection_divergence_rows(tx0, ty0, eta0, out0, tid);
            reflection_divergence_rows(tx1, ty1, eta1, out1, tid);
        });
    }

    // ---- fused spatial shrink (rows partition across lanes) --------------
    // t = grad(x) + b;  coef = max(|t|-1/eta,0)/max(|t|,eps);
    // b <- (1-coef)*t;  db <- (2*coef-1)*t.   Wrap column peeled.
    void shrink(const std::vector<double>& x, std::vector<double>& bx,
                std::vector<double>& by, std::vector<double>& dbx,
                std::vector<double>& dby, double eta) {
        const double thr = 1.0 / eta;
        P.run([&](int tid) {
            for (std::size_t i = tid; i < H; i += P.lanes()) {
                const double* __restrict xi = x.data() + i * W;
                const double* __restrict xn =
                    x.data() + ((i + 1 == H) ? 0 : i + 1) * W;
                double* __restrict pbx = bx.data() + i * W;
                double* __restrict pby = by.data() + i * W;
                double* __restrict pdx = dbx.data() + i * W;
                double* __restrict pdy = dby.data() + i * W;
                const bool neumann_y =
                    solver == 2 && sweep_height && i + 1 == H;
                for (std::size_t j = 0; j < W - 1; ++j) {
                    const double tx = xi[j + 1] - xi[j] + pbx[j];
                    const double ty = neumann_y
                        ? 0.0 : xn[j] - xi[j] + pby[j];
                    const double r = std::sqrt(tx * tx + ty * ty);
                    const double coef =
                        std::fmax(r - thr, 0.0) / std::fmax(r, 1e-12);
                    pbx[j] = (1.0 - coef) * tx;
                    pdx[j] = (2.0 * coef - 1.0) * tx;
                    if (neumann_y) {
                        pby[j] = 0.0;
                        pdy[j] = 0.0;
                    } else {
                        pby[j] = (1.0 - coef) * ty;
                        pdy[j] = (2.0 * coef - 1.0) * ty;
                    }
                }
                {
                    const std::size_t j = W - 1;
                    const bool neumann_x = solver == 2 && !sweep_height;
                    const double tx = neumann_x
                        ? 0.0 : xi[0] - xi[j] + pbx[j];
                    const double ty = neumann_y
                        ? 0.0 : xn[j] - xi[j] + pby[j];
                    const double r = std::sqrt(tx * tx + ty * ty);
                    const double coef =
                        std::fmax(r - thr, 0.0) / std::fmax(r, 1e-12);
                    if (neumann_x) {
                        pbx[j] = 0.0;
                        pdx[j] = 0.0;
                    } else {
                        pbx[j] = (1.0 - coef) * tx;
                        pdx[j] = (2.0 * coef - 1.0) * tx;
                    }
                    if (neumann_y) {
                        pby[j] = 0.0;
                        pdy[j] = 0.0;
                    } else {
                        pby[j] = (1.0 - coef) * ty;
                        pdy[j] = (2.0 * coef - 1.0) * ty;
                    }
                }
            }
        });
    }

    // ---- spectral solves: concrete unit-stride streams, range-split ------

    std::size_t n2() const { return 2 * WB * HB; }

    void split(int tid, std::size_t n, std::size_t& lo, std::size_t& hi) {
        const std::size_t T = std::size_t(P.lanes());
        const std::size_t chunk = ((n / 2) / T) * 2;   // even split
        lo = tid * chunk;
        hi = (tid == int(T) - 1) ? n : lo + chunk;
    }

    // out = c * src * s          (first sweep: Bregman state is zero)
    void solve_scale(const double* srcA, const double* srcB, double c,
                     const double* s, double* outA, double* outB) {
        P.run([&](int tid) {
            std::size_t lo, hi;
            split(tid, n2(), lo, hi);
            const double* __restrict pa = srcA;
            const double* __restrict pb = srcB;
            const double* __restrict ps = s;
            double* __restrict oa = outA;
            double* __restrict ob = outB;
            for (std::size_t r = lo; r < hi; ++r) oa[r] = c * pa[r] * ps[r];
            for (std::size_t r = lo; r < hi; ++r) ob[r] = c * pb[r] * ps[r];
        });
    }

    // io = (c * (io + q) - eta * d) * s     (u-step: g = u + w, in place)
    void solve_sum_inplace(double* ioA, double* ioB, const double* qA,
                           const double* qB, const double* dA,
                           const double* dB, double c, double eta,
                           const double* s) {
        P.run([&](int tid) {
            std::size_t lo, hi;
            split(tid, n2(), lo, hi);
            const double* __restrict qa = qA;
            const double* __restrict qb = qB;
            const double* __restrict da = dA;
            const double* __restrict db = dB;
            const double* __restrict ps = s;
            double* ia = ioA;
            double* ib = ioB;
            for (std::size_t r = lo; r < hi; ++r)
                ia[r] = (c * (ia[r] + qa[r]) - eta * da[r]) * ps[r];
            for (std::size_t r = lo; r < hi; ++r)
                ib[r] = (c * (ib[r] + qb[r]) - eta * db[r]) * ps[r];
        });
    }

    // out = c * (p - q) * s      (v-step first sweep: g = f - u)
    void solve_diff_scale(const double* pA, const double* pB,
                          const double* qA, const double* qB, double c,
                          const double* s, double* outA, double* outB) {
        P.run([&](int tid) {
            std::size_t lo, hi;
            split(tid, n2(), lo, hi);
            const double* __restrict pa = pA;
            const double* __restrict pb = pB;
            const double* __restrict qa = qA;
            const double* __restrict qb = qB;
            const double* __restrict ps = s;
            double* __restrict oa = outA;
            double* __restrict ob = outB;
            for (std::size_t r = lo; r < hi; ++r)
                oa[r] = c * (pa[r] - qa[r]) * ps[r];
            for (std::size_t r = lo; r < hi; ++r)
                ob[r] = c * (pb[r] - qb[r]) * ps[r];
        });
    }

    // out = (c * (p - q) - eta * d) * s      (v-step: g = f - u)
    void solve_diff(const double* pA, const double* pB, const double* qA,
                    const double* qB, const double* dA, const double* dB,
                    double c, double eta, const double* s, double* outA,
                    double* outB) {
        P.run([&](int tid) {
            std::size_t lo, hi;
            split(tid, n2(), lo, hi);
            const double* __restrict pa = pA;
            const double* __restrict pb = pB;
            const double* __restrict qa = qA;
            const double* __restrict qb = qB;
            const double* __restrict da = dA;
            const double* __restrict db = dB;
            const double* __restrict ps = s;
            double* __restrict oa = outA;
            double* __restrict ob = outB;
            for (std::size_t r = lo; r < hi; ++r)
                oa[r] = (c * (pa[r] - qa[r]) - eta * da[r]) * ps[r];
            for (std::size_t r = lo; r < hi; ++r)
                ob[r] = (c * (pb[r] - qb[r]) - eta * db[r]) * ps[r];
        });
    }

    // The two Meyer ROF sweeps are lower triangular in spectral space:
    //
    //   U' = S_u [c_u(U + W) - eta_u D_u]
    //   W' = S_v [c_v(F - U') - eta_v D_v].
    //
    // D_u and D_v depend only on the reflected-dual state at the beginning
    // of the pass.  Consequently both may be transformed first and the two
    // solves may share one range traversal before either inverse transform.
    void solve_meyer_triangle(
            const spectrum& du, const spectrum& dv, double c_u,
            double eta_u, double c_v, double eta_v) {
        P.run([&](int tid) {
            std::size_t lo, hi;
            split(tid, n2(), lo, hi);
            const double* __restrict fa = f_spec.a.data();
            const double* __restrict fb = f_spec.b.data();
            const double* __restrict dua = du.a.data();
            const double* __restrict dub = du.b.data();
            const double* __restrict dva = dv.a.data();
            const double* __restrict dvb = dv.b.data();
            const double* __restrict su = s_u.data();
            const double* __restrict sv = s_v.data();
            double* __restrict ua = u_spec.a.data();
            double* __restrict ub = u_spec.b.data();
            double* __restrict wa = w_spec.a.data();
            double* __restrict wb = w_spec.b.data();
            for (std::size_t r = lo; r < hi; ++r) {
                const double un =
                    (c_u * (ua[r] + wa[r]) - eta_u * dua[r]) * su[r];
                ua[r] = un;
                wa[r] =
                    (c_v * (fa[r] - un) - eta_v * dva[r]) * sv[r];
            }
            for (std::size_t r = lo; r < hi; ++r) {
                const double un =
                    (c_u * (ub[r] + wb[r]) - eta_u * dub[r]) * su[r];
                ub[r] = un;
                wb[r] =
                    (c_v * (fb[r] - un) - eta_v * dvb[r]) * sv[r];
            }
        });
    }

    // Apply one ordinary nonlinear triangle without overwriting the live
    // primal spectra.  This produces the exact fixed-flow residual used as
    // the Arnoldi seed.
    void solve_meyer_triangle_to(
            const spectrum& state_u, const spectrum& state_w,
            const spectrum& du, const spectrum& dv, double c_u,
            double eta_u, double c_v, double eta_v,
            spectrum& out_u, spectrum& out_w) {
        P.run([&](int tid) {
            std::size_t lo, hi;
            split(tid, n2(), lo, hi);
            for (int plane = 0; plane < 2; ++plane) {
                const double* f = plane == 0
                    ? f_spec.a.data() : f_spec.b.data();
                const double* iu = plane == 0
                    ? state_u.a.data() : state_u.b.data();
                const double* iw = plane == 0
                    ? state_w.a.data() : state_w.b.data();
                const double* ud = plane == 0
                    ? du.a.data() : du.b.data();
                const double* wd = plane == 0
                    ? dv.a.data() : dv.b.data();
                double* ou = plane == 0
                    ? out_u.a.data() : out_u.b.data();
                double* ow = plane == 0
                    ? out_w.a.data() : out_w.b.data();
                for (std::size_t r = lo; r < hi; ++r) {
                    const double un =
                        (c_u * (iu[r] + iw[r]) - eta_u * ud[r]) * s_u[r];
                    ou[r] = un;
                    ow[r] =
                        (c_v * (f[r] - un) - eta_v * wd[r]) * s_v[r];
                }
            }
        });
    }

    // Semismooth tangent of the same lower-triangular spectral map.
    void solve_meyer_tangent_to(
            const flow_vector& input, const spectrum& du,
            const spectrum& dv, double c_u, double eta_u, double c_v,
            double eta_v, spectrum& out_u, spectrum& out_w) {
        P.run([&](int tid) {
            std::size_t lo, hi;
            split(tid, n2(), lo, hi);
            for (int plane = 0; plane < 2; ++plane) {
                const double* iu = plane == 0
                    ? input.us.a.data() : input.us.b.data();
                const double* iw = plane == 0
                    ? input.ws.a.data() : input.ws.b.data();
                const double* ud = plane == 0
                    ? du.a.data() : du.b.data();
                const double* wd = plane == 0
                    ? dv.a.data() : dv.b.data();
                double* ou = plane == 0
                    ? out_u.a.data() : out_u.b.data();
                double* ow = plane == 0
                    ? out_w.a.data() : out_w.b.data();
                for (std::size_t r = lo; r < hi; ++r) {
                    const double un =
                        (c_u * (iu[r] + iw[r]) - eta_u * ud[r]) * s_u[r];
                    ou[r] = un;
                    ow[r] = (-c_v * un - eta_v * wd[r]) * s_v[r];
                }
            }
        });
    }

    void solve_meyer_triangle_first(double c_u, double c_v) {
        P.run([&](int tid) {
            std::size_t lo, hi;
            split(tid, n2(), lo, hi);
            const double* __restrict fa = f_spec.a.data();
            const double* __restrict fb = f_spec.b.data();
            const double* __restrict su = s_u.data();
            const double* __restrict sv = s_v.data();
            double* __restrict ua = u_spec.a.data();
            double* __restrict ub = u_spec.b.data();
            double* __restrict wa = w_spec.a.data();
            double* __restrict wb = w_spec.b.data();
            for (std::size_t r = lo; r < hi; ++r) {
                const double un = c_u * fa[r] * su[r];
                ua[r] = un;
                wa[r] = c_v * (fa[r] - un) * sv[r];
            }
            for (std::size_t r = lo; r < hi; ++r) {
                const double un = c_u * fb[r] * su[r];
                ub[r] = un;
                wb[r] = c_v * (fb[r] - un) * sv[r];
            }
        });
    }

    void solve_meyer_triangle_conditioned_first(
            const spectrum& du, double c_u, double eta_u, double c_v) {
        P.run([&](int tid) {
            std::size_t lo, hi;
            split(tid, n2(), lo, hi);
            const double* __restrict fa = f_spec.a.data();
            const double* __restrict fb = f_spec.b.data();
            const double* __restrict da = du.a.data();
            const double* __restrict db = du.b.data();
            const double* __restrict su = s_u.data();
            const double* __restrict sv = s_v.data();
            double* __restrict ua = u_spec.a.data();
            double* __restrict ub = u_spec.b.data();
            double* __restrict wa = w_spec.a.data();
            double* __restrict wb = w_spec.b.data();
            for (std::size_t r = lo; r < hi; ++r) {
                const double un = (c_u * fa[r] - eta_u * da[r]) * su[r];
                ua[r] = un;
                wa[r] = c_v * (fa[r] - un) * sv[r];
            }
            for (std::size_t r = lo; r < hi; ++r) {
                const double un = (c_u * fb[r] - eta_u * db[r]) * su[r];
                ub[r] = un;
                wb[r] = c_v * (fb[r] - un) * sv[r];
            }
        });
    }

    void solve_conditioned_source(const spectrum& source,
                                  const spectrum& divergence,
                                  double c, double eta,
                                  spectrum& out) {
        P.run([&](int tid) {
            std::size_t lo, hi;
            split(tid, n2(), lo, hi);
            const double* __restrict sa = source.a.data();
            const double* __restrict sb = source.b.data();
            const double* __restrict da = divergence.a.data();
            const double* __restrict db = divergence.b.data();
            const double* __restrict symbol_u = s_u.data();
            double* __restrict oa = out.a.data();
            double* __restrict ob = out.b.data();
            for (std::size_t r = lo; r < hi; ++r)
                oa[r] = (c * sa[r] - eta * da[r]) * symbol_u[r];
            for (std::size_t r = lo; r < hi; ++r)
                ob[r] = (c * sb[r] - eta * db[r]) * symbol_u[r];
        });
    }

    // out = (c * g - eta * d) * s            (rung sweeps: g fixed)
    void solve_g(const double* gA, const double* gB, const double* dA,
                 const double* dB, double c, double eta, const double* s,
                 double* outA, double* outB) {
        P.run([&](int tid) {
            std::size_t lo, hi;
            split(tid, n2(), lo, hi);
            const double* __restrict ga = gA;
            const double* __restrict gb = gB;
            const double* __restrict da = dA;
            const double* __restrict db = dB;
            const double* __restrict ps = s;
            double* __restrict oa = outA;
            double* __restrict ob = outB;
            for (std::size_t r = lo; r < hi; ++r)
                oa[r] = (c * ga[r] - eta * da[r]) * ps[r];
            for (std::size_t r = lo; r < hi; ++r)
                ob[r] = (c * gb[r] - eta * db[r]) * ps[r];
        });
    }

    void emit_split_state(const double* image, int pass,
                          double* cartoon_trace, double* texture_trace,
                          trace_visitor visitor, void* user) {
        if ((cartoon_trace == nullptr || texture_trace == nullptr) &&
            visitor == nullptr)
            return;
        const std::size_t n = H * W;
        double* cp = cartoon_trace == nullptr ? nullptr :
            cartoon_trace + std::size_t(pass) * n;
        double* tp = texture_trace == nullptr ? vplane.data() :
            texture_trace + std::size_t(pass) * n;
        for (std::size_t i = 0; i < n; ++i) {
            if (cp != nullptr) cp[i] = u[i];
            tp[i] = image[i] - u[i] - w[i];
        }
        if (visitor != nullptr)
            visitor(pass + 1, u.data(), tp, n, user);
    }

    void finish_split_texture(const double* image, double* texture) {
        if (texture == nullptr) return;
        const std::size_t n = H * W;
        for (std::size_t i = 0; i < n; ++i)
            texture[i] = image[i] - u[i] - w[i];
    }

    std::array<double, 5> flow_gram() {
        // [<r,r>, <r,s>, <s,s>, <r,t>, <s,t>] for
        // r=flow_q0, s=Jr=flow_q1, t=J^2r=flow_work.  Sixty-four fixed
        // pixel chunks make each local summation independent of the active
        // worker count; the final fixed-order reduction is bit-identical.
        constexpr std::size_t chunks = 64;
        std::array<std::array<double, 5>, chunks> partial{};
        const std::size_t n = H * W;
        P.run([&](int tid) {
            for (std::size_t chunk = std::size_t(tid); chunk < chunks;
                 chunk += std::size_t(P.lanes())) {
                const std::size_t lo = n * chunk / chunks;
                const std::size_t hi = n * (chunk + 1) / chunks;
                auto& value = partial[chunk];
                for (std::size_t i = lo; i < hi; ++i) {
                    for (std::size_t field = 0; field < 6; ++field) {
                        const double r = flow_q0.field[field][i];
                        const double s = flow_q1.field[field][i];
                        const double t = flow_work.field[field][i];
                        value[0] += r * r;
                        value[1] += r * s;
                        value[2] += s * s;
                        value[3] += r * t;
                        value[4] += s * t;
                    }
                }
            }
        });
        std::array<double, 5> result{};
        for (const auto& chunk : partial)
            for (std::size_t i = 0; i < result.size(); ++i)
                result[i] += chunk[i];
        return result;
    }

    void build_flow_residual(double c_u, double eta_u,
                             double c_v, double eta_v) {
        fwd2d_reflection(bux, buy, eta_u, d_spec);
        fwd2d_reflection(bvx, bvy, eta_v, q_spec);
        solve_meyer_triangle_to(
            u_spec, w_spec, d_spec, q_spec, c_u, eta_u, c_v, eta_v,
            flow_q0.us, flow_q0.ws);
        inv2d(flow_q0.us, flow_q0.field[0].data());
        inv2d(flow_q0.ws, flow_q0.field[1].data());

        const std::size_t n = H * W;
        P.run([&](int tid) {
            std::size_t lo, hi;
            split(tid, n, lo, hi);
            for (std::size_t i = lo; i < hi; ++i) {
                flow_q0.field[0][i] -= u[i];
                flow_q0.field[1][i] -= w[i];
            }
        });
        P.run([&](int tid) {
            std::size_t lo, hi;
            split(tid, n2(), lo, hi);
            for (std::size_t i = lo; i < hi; ++i) {
                flow_q0.us.a[i] -= u_spec.a[i];
                flow_q0.us.b[i] -= u_spec.b[i];
                flow_q0.ws.a[i] -= w_spec.a[i];
                flow_q0.ws.b[i] -= w_spec.b[i];
            }
        });

        const double radius_u = 1.0 / eta_u;
        const double radius_v = 1.0 / eta_v;
        P.run([&](int tid) {
            for (std::size_t y = std::size_t(tid); y < H;
                 y += std::size_t(P.lanes())) {
                const std::size_t yn = y + 1 == H ? 0 : y + 1;
                for (std::size_t x = 0; x < W; ++x) {
                    const std::size_t xn = x + 1 == W ? 0 : x + 1;
                    const std::size_t i = y * W + x;
                    const std::size_t ix = y * W + xn;
                    const std::size_t iy = yn * W + x;
                    const double next_ux =
                        (u[ix] + flow_q0.field[0][ix]) -
                        (u[i] + flow_q0.field[0][i]);
                    const double next_uy =
                        (u[iy] + flow_q0.field[0][iy]) -
                        (u[i] + flow_q0.field[0][i]);
                    const double next_wx =
                        (w[ix] + flow_q0.field[1][ix]) -
                        (w[i] + flow_q0.field[1][i]);
                    const double next_wy =
                        (w[iy] + flow_q0.field[1][iy]) -
                        (w[i] + flow_q0.field[1][i]);
                    const double magnitude_u =
                        std::sqrt(bux[i] * bux[i] + buy[i] * buy[i]);
                    const double scale_u = std::fmin(
                        1.0, radius_u / std::fmax(magnitude_u, 1e-30));
                    const double magnitude_v =
                        std::sqrt(bvx[i] * bvx[i] + bvy[i] * bvy[i]);
                    const double scale_v = std::fmin(
                        1.0, radius_v / std::fmax(magnitude_v, 1e-30));
                    flow_q0.field[2][i] =
                        next_ux + scale_u * bux[i] - bux[i];
                    flow_q0.field[3][i] =
                        next_uy + scale_u * buy[i] - buy[i];
                    flow_q0.field[4][i] =
                        next_wx + scale_v * bvx[i] - bvx[i];
                    flow_q0.field[5][i] =
                        next_wy + scale_v * bvy[i] - bvy[i];
                }
            }
        });
    }

    void apply_flow_tangent(const flow_vector& input, flow_vector& output,
                            double c_u, double eta_u,
                            double c_v, double eta_v) {
        fwd2d_reflection_tangent(
            bux, buy, input.field[2], input.field[3], eta_u, d_spec);
        fwd2d_reflection_tangent(
            bvx, bvy, input.field[4], input.field[5], eta_v, q_spec);
        solve_meyer_tangent_to(
            input, d_spec, q_spec, c_u, eta_u, c_v, eta_v,
            output.us, output.ws);
        inv2d(output.us, output.field[0].data());
        inv2d(output.ws, output.field[1].data());

        const double radius_u = 1.0 / eta_u;
        const double radius_v = 1.0 / eta_v;
        P.run([&](int tid) {
            for (std::size_t y = std::size_t(tid); y < H;
                 y += std::size_t(P.lanes())) {
                const std::size_t yn = y + 1 == H ? 0 : y + 1;
                for (std::size_t x = 0; x < W; ++x) {
                    const std::size_t xn = x + 1 == W ? 0 : x + 1;
                    const std::size_t i = y * W + x;
                    const std::size_t ix = y * W + xn;
                    const std::size_t iy = yn * W + x;
                    double pux, puy, pvx, pvy;
                    projected_tangent(
                        bux[i], buy[i], input.field[2][i],
                        input.field[3][i], radius_u, pux, puy);
                    projected_tangent(
                        bvx[i], bvy[i], input.field[4][i],
                        input.field[5][i], radius_v, pvx, pvy);
                    output.field[2][i] =
                        output.field[0][ix] - output.field[0][i] + pux;
                    output.field[3][i] =
                        output.field[0][iy] - output.field[0][i] + puy;
                    output.field[4][i] =
                        output.field[1][ix] - output.field[1][i] + pvx;
                    output.field[5][i] =
                        output.field[1][iy] - output.field[1][i] + pvy;
                }
            }
        });
    }

    void apply_flow_displacement(double coefficient0, double coefficient1) {
        const std::size_t n = H * W;
        std::array<std::vector<double>*, 6> state = {
            &u, &w, &bux, &buy, &bvx, &bvy};
        P.run([&](int tid) {
            std::size_t lo, hi;
            split(tid, n, lo, hi);
            for (std::size_t field = 0; field < 6; ++field)
                for (std::size_t i = lo; i < hi; ++i)
                    (*state[field])[i] +=
                        coefficient0 * flow_q0.field[field][i] +
                        coefficient1 * flow_q1.field[field][i];
        });
        P.run([&](int tid) {
            std::size_t lo, hi;
            split(tid, n2(), lo, hi);
            for (std::size_t i = lo; i < hi; ++i) {
                u_spec.a[i] += coefficient0 * flow_q0.us.a[i] +
                    coefficient1 * flow_q1.us.a[i];
                u_spec.b[i] += coefficient0 * flow_q0.us.b[i] +
                    coefficient1 * flow_q1.us.b[i];
                w_spec.a[i] += coefficient0 * flow_q0.ws.a[i] +
                    coefficient1 * flow_q1.ws.a[i];
                w_spec.b[i] += coefficient0 * flow_q0.ws.b[i] +
                    coefficient1 * flow_q1.ws.b[i];
            }
        });
    }

    void flow_ordinary_step(double c_u, double eta_u,
                            double c_v, double eta_v) {
        fwd2d_reflection(bux, buy, eta_u, d_spec);
        fwd2d_reflection(bvx, bvy, eta_v, q_spec);
        solve_meyer_triangle(d_spec, q_spec, c_u, eta_u, c_v, eta_v);
        inv2d(u_spec, u.data());
        inv2d(w_spec, w.data());
        update_reflected_dual_pair(
            u, bux, buy, eta_u, w, bvx, bvy, eta_v);
    }

    bool split_flow_jump(const double* image, double* cartoon,
                         double* texture, int prefix_passes,
                         int horizon, int settle_passes, int jump_count) {
        if (facr_active || prefix_passes < 1 || prefix_passes > 64 ||
            horizon < 1 || horizon > 64 || settle_passes < 0 ||
            settle_passes > 64 || jump_count < 1 || jump_count > 16)
            return false;
        ensure_flow_storage();
        const std::size_t n = H * W;
        for (auto* value : {&u, &w, &bux, &buy, &bvx, &bvy})
            std::memset(value->data(), 0, n * sizeof(double));
        u_spec.zero();
        w_spec.zero();
        fwd2d(image, f_spec);

        const double c_u = lam, eta_u = 2.0 * lam;
        const double c_v = 1.0 / mu, eta_v = 10.0 / mu;
        solve_meyer_triangle_first(c_u, c_v);
        inv2d(u_spec, u.data());
        inv2d(w_spec, w.data());
        update_reflected_dual_pair(
            u, bux, buy, eta_u, w, bvx, bvy, eta_v);
        for (int pass = 1; pass < prefix_passes; ++pass)
            flow_ordinary_step(c_u, eta_u, c_v, eta_v);

        for (int jump = 0; jump < jump_count; ++jump) {
            build_flow_residual(c_u, eta_u, c_v, eta_v);
            apply_flow_tangent(
                flow_q0, flow_q1, c_u, eta_u, c_v, eta_v);
            apply_flow_tangent(
                flow_q1, flow_work, c_u, eta_u, c_v, eta_v);
            const auto gram = flow_gram();
            const double g00 = gram[0], g01 = gram[1], g11 = gram[2];
            const double k01 = gram[3], k11 = gram[4];
            if (!(g00 > 1e-30)) break;

            // Galerkin projection in the nonorthogonal Krylov basis
            // B=[r,Jr].  Since J*r is the second basis vector, H's first
            // column is exactly (0,1); only the second column requires the
            // 2x2 Gram solve.  This is algebraically the depth-two Arnoldi
            // polynomial without normalization or full-field MGS sweeps.
            const double determinant = g00 * g11 - g01 * g01;
            double h00 = 0.0, h10 = 1.0, h01 = 0.0, h11 = 0.0;
            const bool rank_two = determinant >
                1e-13 * std::fmax(g00 * g11, 1e-300);
            if (rank_two) {
                h01 = (g11 * k01 - g01 * k11) / determinant;
                h11 = (-g01 * k01 + g00 * k11) / determinant;
            } else {
                // One-dimensional Krylov breakdown: J*r is collinear with
                // r, so the exact projected recurrence is scalar.
                h00 = g01 / g00;
                h10 = 0.0;
            }

            double power0 = 1.0, power1 = 0.0;
            double sum0 = 0.0, sum1 = 0.0;
            for (int step = 0; step < horizon; ++step) {
                sum0 += power0;
                sum1 += power1;
                const double next0 = h00 * power0 + h01 * power1;
                const double next1 = h10 * power0 + h11 * power1;
                power0 = next0;
                power1 = next1;
            }
            apply_flow_displacement(sum0, sum1);
            for (int pass = 0; pass < settle_passes; ++pass)
                flow_ordinary_step(c_u, eta_u, c_v, eta_v);
        }

        finish_split_texture(image, texture);
        for (std::size_t i = 0; i < n; ++i)
            cartoon[i] = image[i] - texture[i];
        return true;
    }

    void run_split_reduced_spectral(
            const double* image, double* texture_out = nullptr,
            double* cartoon_trace = nullptr, double* texture_trace = nullptr,
            trace_visitor visitor = nullptr, void* user = nullptr) {
        const std::size_t n = H * W;
        for (auto* p : {&u, &w, &bux, &buy, &bvx, &bvy})
            std::memset(p->data(), 0, n * sizeof(double));
        u_spec.zero();
        w_spec.zero();
        fwd2d(image, f_spec);
        const double c_u = lam, eta_u = 2.0 * lam;
        const double c_v = 1.0 / mu, eta_v = 10.0 / mu;
        for (int pass = 0; pass < passes; ++pass) {
            if (pass == 0) {
                solve_meyer_triangle_first(c_u, c_v);
            } else {
                fwd2d_reflection(bux, buy, eta_u, d_spec);
                fwd2d_reflection(bvx, bvy, eta_v, q_spec);
                solve_meyer_triangle(d_spec, q_spec, c_u, eta_u, c_v,
                                     eta_v);
            }
            inv2d(u_spec, u.data());
            inv2d(w_spec, w.data());
            update_reflected_dual_pair(
                u, bux, buy, eta_u, w, bvx, bvy, eta_v);
            emit_split_state(image, pass, cartoon_trace, texture_trace,
                             visitor, user);
        }
        finish_split_texture(image, texture_out);
    }

    void run_split_reduced_facr(
            const double* image, double* texture_out = nullptr,
            double* cartoon_trace = nullptr, double* texture_trace = nullptr,
            trace_visitor visitor = nullptr, void* user = nullptr) {
        const std::size_t n = H * W;
        for (auto* p : {&u, &w, &bux, &buy, &bvx, &bvy})
            std::memset(p->data(), 0, n * sizeof(double));
        fu_spec.zero();
        fw_spec.zero();
        facr_fwd(image, ff_spec);
        const double c_u = lam, eta_u = 2.0 * lam;
        const double c_v = 1.0 / mu, eta_v = 10.0 / mu;
        for (int pass = 0; pass < passes; ++pass) {
            if (pass == 0) {
                facr_scale(ff_spec, c_u, t_u, fu_spec);
                facr_diff(ff_spec, fu_spec, nullptr, c_v, eta_v, t_v,
                          fw_spec);
            } else {
                if (sweep_height) {
                    facr_fwd_reflection(bux, buy, eta_u, fd_spec);
                    facr_fwd_reflection(bvx, bvy, eta_v, fq_spec);
                } else {
                    // Column transforms make all four reflected-dual
                    // streams strided.  A row-major materialization is
                    // faster here despite the scalar scratch traffic.
                    reflection_divergence_pair(
                        bux, buy, eta_u, u.data(),
                        bvx, bvy, eta_v, w.data());
                    facr_fwd(u.data(), fd_spec);
                    facr_fwd(w.data(), fq_spec);
                }
                facr_sum_inplace(fu_spec, fw_spec, fd_spec, c_u, eta_u,
                                  t_u);
                facr_diff(ff_spec, fu_spec, &fq_spec, c_v, eta_v, t_v,
                          fw_spec);
            }
            facr_inv(fu_spec, u.data());
            facr_inv(fw_spec, w.data());
            update_reflected_dual_pair(
                u, bux, buy, eta_u, w, bvx, bvy, eta_v);
            emit_split_state(image, pass, cartoon_trace, texture_trace,
                             visitor, user);
        }
        finish_split_texture(image, texture_out);
    }

    void run_split_reduced(
            const double* image, double* texture_out = nullptr,
            double* cartoon_trace = nullptr, double* texture_trace = nullptr,
            trace_visitor visitor = nullptr, void* user = nullptr) {
        if (facr_active)
            run_split_reduced_facr(image, texture_out, cartoon_trace,
                                   texture_trace, visitor, user);
        else
            run_split_reduced_spectral(image, texture_out, cartoon_trace,
                                       texture_trace, visitor, user);
    }

    // ---- the alternation (shared by split and decompose) -----------------
    //
    // Runs the TGFD passes and leaves u (cartoon layer), w (texture-side
    // ROF survivor), vplane = f - u - w (texture layer), and the spectra
    // of f, u, w in place.

    void run_passes_spectral(const double* image,
                             double* cartoon_trace = nullptr,
                             double* texture_trace = nullptr,
                             trace_visitor visitor = nullptr,
                             void* user = nullptr) {
        const std::size_t n = H * W;
        for (auto* p : {&dbux, &dbuy, &dbvx, &dbvy})
            if (p->empty()) p->assign(n, 0.0);
        for (auto* p : {&u, &w, &bux, &buy, &dbux, &dbuy, &bvx, &bvy, &dbvx,
                        &dbvy})
            std::memset(p->data(), 0, n * sizeof(double));
        u_spec.zero();
        w_spec.zero();

        fwd2d(image, f_spec);

        const double c_u = lam, eta_u = 2.0 * lam;
        const double c_v = 1.0 / mu, eta_v = 10.0 / mu;

        for (int p = 0; p < passes; ++p) {
            // u-step: g = f - v = u + w (both spectra maintained)
            if (p == 0) {
                solve_scale(f_spec.a.data(), f_spec.b.data(), c_u,
                            s_u.data(), u_spec.a.data(), u_spec.b.data());
            } else {
                fwd2d_div(dbux, dbuy, d_spec);
                solve_sum_inplace(u_spec.a.data(), u_spec.b.data(),
                                  w_spec.a.data(), w_spec.b.data(),
                                  d_spec.a.data(), d_spec.b.data(), c_u,
                                  eta_u, s_u.data());
            }
            inv2d(u_spec, u.data());
            shrink(u, bux, buy, dbux, dbuy, eta_u);

            // v-step: g = f - u
            if (p == 0) {
                solve_diff_scale(f_spec.a.data(), f_spec.b.data(),
                                 u_spec.a.data(), u_spec.b.data(), c_v,
                                 s_v.data(), w_spec.a.data(),
                                 w_spec.b.data());
            } else {
                fwd2d_div(dbvx, dbvy, d_spec);
                solve_diff(f_spec.a.data(), f_spec.b.data(),
                           u_spec.a.data(), u_spec.b.data(),
                           d_spec.a.data(), d_spec.b.data(), c_v, eta_v,
                           s_v.data(), w_spec.a.data(), w_spec.b.data());
            }
            inv2d(w_spec, w.data());
            shrink(w, bvx, bvy, dbvx, dbvy, eta_v);

            if ((cartoon_trace != nullptr && texture_trace != nullptr) ||
                visitor != nullptr) {
                double* __restrict cp = cartoon_trace == nullptr
                    ? nullptr
                    : cartoon_trace + static_cast<std::size_t>(p) * n;
                double* __restrict tp = texture_trace == nullptr
                    ? vplane.data()
                    : texture_trace + static_cast<std::size_t>(p) * n;
                const double* __restrict fp = image;
                const double* __restrict up = u.data();
                const double* __restrict wp = w.data();
                for (std::size_t i = 0; i < n; ++i) {
                    if (cp != nullptr) cp[i] = up[i];
                    tp[i] = fp[i] - up[i] - wp[i];
                }
                if (visitor != nullptr)
                    visitor(p + 1, u.data(), tp, n, user);
            }
        }

        // v = f - u - w, single fused pass
        {
            double* __restrict vp = vplane.data();
            const double* __restrict fp = image;
            const double* __restrict up = u.data();
            const double* __restrict wp = w.data();
            for (std::size_t i = 0; i < n; ++i)
                vp[i] = fp[i] - up[i] - wp[i];
        }
    }

    void run_passes_facr(const double* image,
                         double* cartoon_trace = nullptr,
                         double* texture_trace = nullptr,
                         trace_visitor visitor = nullptr,
                         void* user = nullptr) {
        const std::size_t n = H * W;
        for (auto* p : {&dbux, &dbuy, &dbvx, &dbvy})
            if (p->empty()) p->assign(n, 0.0);
        for (auto* p : {&u, &w, &bux, &buy, &dbux, &dbuy, &bvx, &bvy,
                        &dbvx, &dbvy})
            std::memset(p->data(), 0, n * sizeof(double));
        fu_spec.zero();
        fw_spec.zero();
        facr_fwd(image, ff_spec);

        const double c_u = lam, eta_u = 2.0 * lam;
        const double c_v = 1.0 / mu, eta_v = 10.0 / mu;
        for (int pass = 0; pass < passes; ++pass) {
            if (pass == 0) {
                facr_scale(ff_spec, c_u, t_u, fu_spec);
            } else {
                facr_fwd_div(dbux, dbuy, fd_spec);
                facr_sum_inplace(fu_spec, fw_spec, fd_spec, c_u, eta_u,
                                  t_u);
            }
            facr_inv(fu_spec, u.data());
            shrink(u, bux, buy, dbux, dbuy, eta_u);

            if (pass == 0) {
                facr_diff(ff_spec, fu_spec, nullptr, c_v, eta_v, t_v,
                          fw_spec);
            } else {
                facr_fwd_div(dbvx, dbvy, fd_spec);
                facr_diff(ff_spec, fu_spec, &fd_spec, c_v, eta_v, t_v,
                          fw_spec);
            }
            facr_inv(fw_spec, w.data());
            shrink(w, bvx, bvy, dbvx, dbvy, eta_v);

            if ((cartoon_trace != nullptr && texture_trace != nullptr) ||
                visitor != nullptr) {
                double* cp = cartoon_trace == nullptr ? nullptr :
                    cartoon_trace + std::size_t(pass) * n;
                double* tp = texture_trace == nullptr ? vplane.data() :
                    texture_trace + std::size_t(pass) * n;
                for (std::size_t i = 0; i < n; ++i) {
                    if (cp != nullptr) cp[i] = u[i];
                    tp[i] = image[i] - u[i] - w[i];
                }
                if (visitor != nullptr)
                    visitor(pass + 1, u.data(), tp, n, user);
            }
        }
        for (std::size_t i = 0; i < n; ++i)
            vplane[i] = image[i] - u[i] - w[i];
    }

    void run_passes(const double* image, double* cartoon_trace = nullptr,
                    double* texture_trace = nullptr,
                    trace_visitor visitor = nullptr, void* user = nullptr) {
        if (!facr_active)
            run_passes_spectral(image, cartoon_trace, texture_trace,
                                visitor, user);
        else
            run_passes_facr(image, cartoon_trace, texture_trace,
                            visitor, user);
    }

    // ---- a plain ROF solve from a spectrum already in hand ---------------
    //
    // out <- argmin_x TV(x) + (c/2)|x - g|^2 by Split Bregman sweeps from a
    // FRESH state (Bregman states are c- and eta-scaled and must never be
    // carried between different fidelity constants).  gs is the spectrum of
    // g and gplane its samples (needed only to seed the change test); s is
    // the symbol table for this (c, eta) pair.  Sweeps stop early once the
    // relative iterate change falls below tol (tol <= 0 disables the test).
    // u_spec is used as spectral scratch: the u plane is left intact, so
    // decompose() may call this after run_passes().

    bool rof_hodge_drop(const double* gplane, double c, double eta,
                        const std::vector<double>& s) {
        const std::size_t n = H * W;

        // The current feasible dual flux is p = eta*b.  Ask Fourier for the
        // minimum-energy longitudinal correction whose divergence closes
        // c*(u-g)-div(p), preserving the current transverse route.
        P.run([&](int tid) {
            for (std::size_t y = std::size_t(tid); y < H;
                 y += std::size_t(P.lanes())) {
                const std::size_t yp = (y == 0 ? H : y) - 1;
                for (std::size_t x = 0; x < W; ++x) {
                    const std::size_t xp = (x == 0 ? W : x) - 1;
                    const std::size_t i = y * W + x;
                    const double div_b =
                        rbx[i] - rbx[y * W + xp] +
                        rby[i] - rby[yp * W + x];
                    rhodge[i] = c * (xit[i] - gplane[i]) - eta * div_b;
                }
            }
        });

        // Solve Delta(phi)=mismatch.  Since
        // s=1/(c+eta*L), -1/L = -eta*s/(1-c*s).  The zero mode is pinned.
        fwd2d(rhodge.data(), d_spec);
        P.run([&](int tid) {
            std::size_t lo, hi;
            split(tid, n2(), lo, hi);
            double* __restrict da = d_spec.a.data();
            double* __restrict db = d_spec.b.data();
            const double* __restrict ps = s.data();
            for (std::size_t i = lo; i < hi; ++i) {
                const double denominator = 1.0 - c * ps[i];
                const double factor = denominator > 1e-13
                    ? -eta * ps[i] / denominator : 0.0;
                da[i] *= factor;
                db[i] *= factor;
            }
        });
        inv2d(d_spec, rhodge.data());

        // p0 = p + grad(phi), followed by the one allowed unit-disk hit.
        P.run([&](int tid) {
            for (std::size_t y = std::size_t(tid); y < H;
                 y += std::size_t(P.lanes())) {
                const std::size_t yn = (y + 1 == H) ? 0 : y + 1;
                for (std::size_t x = 0; x < W; ++x) {
                    const std::size_t xn = (x + 1 == W) ? 0 : x + 1;
                    const std::size_t i = y * W + x;
                    double px = eta * rbx[i] +
                        rhodge[y * W + xn] - rhodge[i];
                    double py = eta * rby[i] +
                        rhodge[yn * W + x] - rhodge[i];
                    const double scale =
                        std::fmax(1.0, std::sqrt(px * px + py * py));
                    rhodge_x[i] = px / scale;
                    rhodge_y[i] = py / scale;
                }
            }
        });

        // Dual-induced primal.  rhodge is no longer needed as the potential.
        P.run([&](int tid) {
            for (std::size_t y = std::size_t(tid); y < H;
                 y += std::size_t(P.lanes())) {
                const std::size_t yp = (y == 0 ? H : y) - 1;
                for (std::size_t x = 0; x < W; ++x) {
                    const std::size_t xp = (x == 0 ? W : x) - 1;
                    const std::size_t i = y * W + x;
                    const double div_p =
                        rhodge_x[i] - rhodge_x[y * W + xp] +
                        rhodge_y[i] - rhodge_y[yp * W + x];
                    rhodge[i] = gplane[i] + div_p / c;
                }
            }
        });

        double minimum = xit[0], maximum = xit[0];
        for (std::size_t i = 1; i < n; ++i) {
            minimum = std::min(minimum, xit[i]);
            maximum = std::max(maximum, xit[i]);
        }
        const double gradient_tolerance =
            1e-10 * std::max(maximum - minimum, 1.0);
        double first = 0.0;
        double second = 0.0;
        for (std::size_t y = 0; y < H; ++y) {
            const std::size_t yn = (y + 1 == H) ? 0 : y + 1;
            for (std::size_t x = 0; x < W; ++x) {
                const std::size_t xn = (x + 1 == W) ? 0 : x + 1;
                const std::size_t i = y * W + x;
                const double direction = rhodge[i] - xit[i];
                const double gx = xit[y * W + xn] - xit[i];
                const double gy = xit[yn * W + x] - xit[i];
                const double dx =
                    (rhodge[y * W + xn] - xit[y * W + xn]) - direction;
                const double dy =
                    (rhodge[yn * W + x] - xit[yn * W + x]) - direction;
                const double magnitude = std::sqrt(gx * gx + gy * gy);
                first += c * (xit[i] - gplane[i]) * direction;
                if (magnitude > gradient_tolerance) {
                    first += (gx * dx + gy * dy) / magnitude;
                    const double cross = gx * dy - gy * dx;
                    second += cross * cross /
                        (magnitude * magnitude * magnitude);
                } else {
                    first += std::sqrt(dx * dx + dy * dy);
                }
                second += c * direction * direction;
            }
        }
        if (!(first < 0.0) || !(second > 0.0)) return false;
        const double alpha =
            std::min(1.0, std::max(0.0, -first / second));
        if (!(alpha > 0.0)) return false;

        double old_objective = 0.0;
        double new_objective = 0.0;
        for (std::size_t y = 0; y < H; ++y) {
            const std::size_t yn = (y + 1 == H) ? 0 : y + 1;
            for (std::size_t x = 0; x < W; ++x) {
                const std::size_t xn = (x + 1 == W) ? 0 : x + 1;
                const std::size_t i = y * W + x;
                const double direction = rhodge[i] - xit[i];
                const double gx = xit[y * W + xn] - xit[i];
                const double gy = xit[yn * W + x] - xit[i];
                const double dx =
                    (rhodge[y * W + xn] - xit[y * W + xn]) - direction;
                const double dy =
                    (rhodge[yn * W + x] - xit[yn * W + x]) - direction;
                const double old_residual = xit[i] - gplane[i];
                const double new_residual = old_residual + alpha * direction;
                old_objective += std::sqrt(gx * gx + gy * gy) +
                    0.5 * c * old_residual * old_residual;
                new_objective +=
                    std::sqrt((gx + alpha * dx) * (gx + alpha * dx) +
                              (gy + alpha * dy) * (gy + alpha * dy)) +
                    0.5 * c * new_residual * new_residual;
            }
        }
        if (!(new_objective < old_objective)) return false;

        // Re-seat ADMM at the same ROF target:
        // b <- feasible dual / eta, d <- grad(accelerated primal).
        P.run([&](int tid) {
            for (std::size_t y = std::size_t(tid); y < H;
                 y += std::size_t(P.lanes())) {
                for (std::size_t x = 0; x < W; ++x) {
                    const std::size_t i = y * W + x;
                    xit[i] += alpha * (rhodge[i] - xit[i]);
                    rbx[i] = rhodge_x[i] / eta;
                    rby[i] = rhodge_y[i] / eta;
                }
                // xit must be fully updated before gradients are read.
            }
        });
        P.run([&](int tid) {
            for (std::size_t y = std::size_t(tid); y < H;
                 y += std::size_t(P.lanes())) {
                const std::size_t yn = (y + 1 == H) ? 0 : y + 1;
                for (std::size_t x = 0; x < W; ++x) {
                    const std::size_t xn = (x + 1 == W) ? 0 : x + 1;
                    const std::size_t i = y * W + x;
                    rdbx[i] = xit[y * W + xn] - xit[i] - rbx[i];
                    rdby[i] = xit[yn * W + x] - xit[i] - rby[i];
                }
            }
        });
        return true;
    }

    void rof_from_spec(const spectrum& gs, const double* gplane, double c,
                       double eta, const std::vector<double>& s, int sweeps,
                       double tol, double* out, int hodge_after = 0) {
        const std::size_t n = H * W;
        std::memset(rbx.data(), 0, n * sizeof(double));
        std::memset(rby.data(), 0, n * sizeof(double));
        std::memcpy(prev.data(), gplane, n * sizeof(double));
        last_rof_sweeps = 0;
        last_rof_hodge_applied = false;
        bool done = false;
        for (int sweep = 0; sweep < sweeps && !done; ++sweep) {
            if (sweep == 0) {
                solve_scale(gs.a.data(), gs.b.data(), c, s.data(),
                            u_spec.a.data(), u_spec.b.data());
            } else {
                fwd2d_div(rdbx, rdby, d_spec);
                solve_g(gs.a.data(), gs.b.data(), d_spec.a.data(),
                        d_spec.b.data(), c, eta, s.data(), u_spec.a.data(),
                        u_spec.b.data());
            }
            inv2d(u_spec, xit.data());
            shrink(xit, rbx, rby, rdbx, rdby, eta);
            ++last_rof_sweeps;
            if (hodge_after > 0 && sweep + 1 == hodge_after) {
                last_rof_hodge_applied =
                    rof_hodge_drop(gplane, c, eta, s);
            }
            if (tol > 0.0) {
                // SERIAL by design: bit-identical for all thread counts
                const double* __restrict xi = xit.data();
                double* __restrict pv = prev.data();
                double d0 = 0, d1 = 0, x0 = 0, x1 = 0;
                for (std::size_t i = 0; i < n; i += 2) {
                    const double a0 = xi[i] - pv[i];
                    const double a1 = xi[i + 1] - pv[i + 1];
                    d0 += a0 * a0;
                    d1 += a1 * a1;
                    x0 += xi[i] * xi[i];
                    x1 += xi[i + 1] * xi[i + 1];
                    pv[i] = xi[i];
                    pv[i + 1] = xi[i + 1];
                }
                done = (d0 + d1) <= tol * tol * (x0 + x1);
            }
        }
        std::memcpy(out, xit.data(), n * sizeof(double));
    }

    void rof_from_facr(const facr_spectrum& gs, const double* gplane,
                       double c, double eta, const tri_factors& t,
                       int sweeps, double tol, double* out) {
        const std::size_t n = H * W;
        std::memset(rbx.data(), 0, n * sizeof(double));
        std::memset(rby.data(), 0, n * sizeof(double));
        std::memset(rdbx.data(), 0, n * sizeof(double));
        std::memset(rdby.data(), 0, n * sizeof(double));
        std::memcpy(prev.data(), gplane, n * sizeof(double));
        last_rof_sweeps = 0;
        last_rof_hodge_applied = false;
        bool done = false;
        for (int sweep = 0; sweep < sweeps && !done; ++sweep) {
            if (sweep == 0) {
                facr_scale(gs, c, t, fu_spec);
            } else {
                facr_fwd_div(rdbx, rdby, fd_spec);
                facr_g(gs, fd_spec, c, eta, t, fu_spec);
            }
            facr_inv(fu_spec, xit.data());
            shrink(xit, rbx, rby, rdbx, rdby, eta);
            ++last_rof_sweeps;
            if (tol > 0.0) {
                double d0 = 0.0, d1 = 0.0, x0 = 0.0, x1 = 0.0;
                std::size_t i = 0;
                for (; i + 1 < n; i += 2) {
                    const double a0 = xit[i] - prev[i];
                    const double a1 = xit[i + 1] - prev[i + 1];
                    d0 += a0 * a0; d1 += a1 * a1;
                    x0 += xit[i] * xit[i]; x1 += xit[i + 1] * xit[i + 1];
                    prev[i] = xit[i]; prev[i + 1] = xit[i + 1];
                }
                if (i < n) {
                    const double a = xit[i] - prev[i];
                    d0 += a * a; x0 += xit[i] * xit[i]; prev[i] = xit[i];
                }
                done = (d0 + d1) <= tol * tol * (x0 + x1);
            }
        }
        std::memcpy(out, xit.data(), n * sizeof(double));
    }

    // ---- public ROF: transform the image, then solve --------------------
    //
    // The symbol table depends on (c, eta) and costs WB*HB cosine pairs to
    // build, so the last one is cached: effect pipelines call this
    // repeatedly with fixed constants.

    void rof(const double* image, double* smooth, double c, double eta,
             int sweeps, double tol) {
        ensure_rof_storage();
        if (facr_active) {
            if (t_gen.pivot.empty() || c != t_gen.c || eta != t_gen.eta)
                build_factors(t_gen, c, eta);
            facr_fwd(image, ff_spec);
            rof_from_facr(ff_spec, image, c, eta, t_gen, sweeps, tol,
                           smooth);
            return;
        }
        if (s_gen.empty() || c != gen_c || eta != gen_eta) {
            symbol(s_gen, c, eta);
            gen_c = c;
            gen_eta = eta;
        }
        fwd2d(image, f_spec);
        rof_from_spec(f_spec, image, c, eta, s_gen, sweeps, tol, smooth);
    }

    bool rof_accelerated(const double* image, double* smooth, double c,
                         double eta, int sweeps, double tol,
                         int hodge_after) {
        if (facr_active || hodge_after < 1 || hodge_after > sweeps)
            return false;
        ensure_rof_hodge_storage();
        if (s_gen.empty() || c != gen_c || eta != gen_eta) {
            symbol(s_gen, c, eta);
            gen_c = c;
            gen_eta = eta;
        }
        fwd2d(image, f_spec);
        rof_from_spec(f_spec, image, c, eta, s_gen, sweeps, tol, smooth,
                      hodge_after);
        return true;
    }

    // ---- split: the model decomposition alone, no ladder ----------------
    //
    // cartoon = u, texture = v: exactly the pair Gilles' Algorithm 3
    // produces.  (The five-output decompose() instead reports
    // cartoon = u + s0, folding the ladder's coarsest survivor back into
    // the cartoon so that cartoon + bands = u + v.)

    void split(const double* image, double* cartoon, double* texture) {
        run_split_reduced(image, texture);
        const std::size_t n = H * W;
        std::memcpy(cartoon, u.data(), n * sizeof(double));
    }

    // Exact two-product readout used by the public default on FACR plans.
    // Texture is finalized before cartoon is written, so image may alias
    // cartoon without losing the source needed for cartoon = image-texture.
    void split_effective(const double* image, double* cartoon,
                         double* texture) {
        run_split_reduced(image, texture);
        const std::size_t n = H * W;
        for (std::size_t i = 0; i < n; ++i)
            cartoon[i] = image[i] - texture[i];
    }

    static std::size_t periodic_index(std::size_t coordinate, int offset,
                                      std::size_t length) {
        std::int64_t value = static_cast<std::int64_t>(coordinate) + offset;
        value %= static_cast<std::int64_t>(length);
        if (value < 0) value += static_cast<std::int64_t>(length);
        return static_cast<std::size_t>(value);
    }

    static inline std::size_t short_periodic_offset(
            std::size_t coordinate, int delta, std::size_t extent) {
        std::ptrdiff_t value =
            static_cast<std::ptrdiff_t>(coordinate) + delta;
        const std::ptrdiff_t period = static_cast<std::ptrdiff_t>(extent);
        if (value < 0 || value >= period) {
            value %= period;
            if (value < 0) value += period;
        }
        return static_cast<std::size_t>(value);
    }

    template <int Step>
    static inline std::size_t periodic_unit_step(
            std::size_t coordinate, std::size_t extent) {
        static_assert(Step >= -1 && Step <= 1,
                      "periodic unit step must be -1, 0, or 1");
        if constexpr (Step > 0)
            return coordinate + 1 == extent ? 0 : coordinate + 1;
        if constexpr (Step < 0)
            return coordinate == 0 ? extent - 1 : coordinate - 1;
        return coordinate;
    }

    template <int DY, int DX>
    inline void advance_direction(std::size_t& y, std::size_t& x) const {
        y = periodic_unit_step<DY>(y, H);
        x = periodic_unit_step<DX>(x, W);
    }

    template <int Radius>
    static inline void cyclic_box_line(const double* source,
                                       double* destination,
                                       std::size_t length) {
        const double inverse = 1.0 / double(2 * Radius + 1);
        double sum = 0.0;
        for (int step = -Radius; step <= Radius; ++step)
            sum += source[periodic_index(0, step, length)];
        std::size_t leave = periodic_index(0, -Radius, length);
        std::size_t enter = periodic_index(0, Radius + 1, length);
        for (std::size_t position = 0; position < length; ++position) {
            destination[position] = sum * inverse;
            sum += source[enter] - source[leave];
            if (++leave == length) leave = 0;
            if (++enter == length) enter = 0;
        }
    }

    // Fuse the three long-axis box stages while each independent row,
    // column family, or torus cycle is owned by one worker. Diagonal cycles
    // are packed once into u/w, processed contiguously, and unpacked once;
    // their seed and accumulation order match the spatial traversal exactly.
    template <int DY, int DX, int Radius>
    void directional_box_three_pass(const double* image) {
        static_assert(Radius > 0, "box radius must be positive");
        if constexpr (DY == 0) {
            P.run([&](int tid) {
                for (std::size_t y = std::size_t(tid); y < H;
                     y += std::size_t(P.lanes())) {
                    const double* source = image + y * W;
                    double* first = u.data() + y * W;
                    double* second = w.data() + y * W;
                    cyclic_box_line<Radius>(source, first, W);
                    cyclic_box_line<Radius>(first, second, W);
                    cyclic_box_line<Radius>(second, first, W);
                }
            });
        } else if constexpr (DX == 0) {
            // Carry a contiguous vector of column sums down the rows so each
            // of the three vertical passes remains unit-stride and SIMD-able.
            P.run([&](int tid) {
                std::size_t xlo, xhi;
                split(tid, W, xlo, xhi);
                double* sums = lanes[tid]->line.data();
                auto pass = [&](const double* source,
                                double* destination) {
                    for (std::size_t x = xlo; x < xhi; ++x)
                        sums[x] = 0.0;
                    for (int step = -Radius; step <= Radius; ++step) {
                        const std::size_t sy =
                            periodic_index(0, step * DY, H);
                        const double* row = source + sy * W;
                        for (std::size_t x = xlo; x < xhi; ++x)
                            sums[x] += row[x];
                    }
                    std::size_t leave_y = periodic_index(
                        0, -Radius * DY, H);
                    std::size_t enter_y = periodic_index(
                        0, (Radius + 1) * DY, H);
                    const double inverse =
                        1.0 / double(2 * Radius + 1);
                    for (std::size_t y = 0; y < H; ++y) {
                        double* out = destination + y * W;
                        const double* enter = source + enter_y * W;
                        const double* leave = source + leave_y * W;
                        for (std::size_t x = xlo; x < xhi; ++x)
                            out[x] = sums[x] * inverse;
                        for (std::size_t x = xlo; x < xhi; ++x)
                            sums[x] += enter[x] - leave[x];
                        leave_y = periodic_unit_step<DY>(leave_y, H);
                        enter_y = periodic_unit_step<DY>(enter_y, H);
                    }
                };
                pass(image, u.data());
                pass(u.data(), w.data());
                pass(w.data(), u.data());
            });
        } else {
            const std::size_t cycles = std::gcd(H, W);
            const std::size_t length = (H / cycles) * W;
            P.run([&](int tid) {
                for (std::size_t cycle = std::size_t(tid); cycle < cycles;
                     cycle += std::size_t(P.lanes())) {
                    const std::size_t base = cycle * length;
                    std::size_t y = 0, x = cycle;
                    for (std::size_t position = 0; position < length;
                         ++position) {
                        u[base + position] = image[y * W + x];
                        advance_direction<DY, DX>(y, x);
                    }
                }
            });
            P.run([&](int tid) {
                for (std::size_t cycle = std::size_t(tid); cycle < cycles;
                     cycle += std::size_t(P.lanes())) {
                    const std::size_t base = cycle * length;
                    cyclic_box_line<Radius>(
                        u.data() + base, w.data() + base, length);
                    cyclic_box_line<Radius>(
                        w.data() + base, u.data() + base, length);
                    cyclic_box_line<Radius>(
                        u.data() + base, w.data() + base, length);
                }
            });
            P.run([&](int tid) {
                for (std::size_t cycle = std::size_t(tid); cycle < cycles;
                     cycle += std::size_t(P.lanes())) {
                    const std::size_t base = cycle * length;
                    std::size_t y = 0, x = cycle;
                    for (std::size_t position = 0; position < length;
                         ++position) {
                        u[y * W + x] = w[base + position];
                        advance_direction<DY, DX>(y, x);
                    }
                }
            });
        }
    }

    template <int DY, int DX>
    void directional_three_pass(const double* source, double* destination,
                                double side) {
        const double center = 1.0 - 2.0 * side;
        // Unlike the box, this stencil has no recurrence. Always traverse
        // output rows in memory order; cycle traversal made vertical and
        // diagonal taps needlessly cache-hostile.
        P.run([&](int tid) {
            for (std::size_t y = std::size_t(tid); y < H;
                 y += std::size_t(P.lanes())) {
                const std::size_t ym = periodic_unit_step<-DY>(y, H);
                const std::size_t yp = periodic_unit_step<DY>(y, H);
                const double* minus = source + ym * W;
                const double* middle = source + y * W;
                const double* plus = source + yp * W;
                double* out = destination + y * W;
                if constexpr (DX == 0) {
                    for (std::size_t x = 0; x < W; ++x)
                        out[x] = side * minus[x] + center * middle[x] +
                                 side * plus[x];
                } else if constexpr (DX > 0) {
                    out[0] = side * minus[W - 1] + center * middle[0] +
                             side * plus[1];
                    for (std::size_t x = 1; x + 1 < W; ++x)
                        out[x] = side * minus[x - 1] + center * middle[x] +
                                 side * plus[x + 1];
                    out[W - 1] = side * minus[W - 2] +
                        center * middle[W - 1] + side * plus[0];
                } else {
                    out[0] = side * minus[1] + center * middle[0] +
                             side * plus[W - 1];
                    for (std::size_t x = 1; x + 1 < W; ++x)
                        out[x] = side * minus[x + 1] + center * middle[x] +
                                 side * plus[x - 1];
                    out[W - 1] = side * minus[0] +
                        center * middle[W - 1] + side * plus[W - 2];
                }
            }
        });
    }

    template <std::size_t NA, std::size_t NB>
    static std::array<double, NA + NB - 1> convolve_short_fir(
            const std::array<double, NA>& a,
            const std::array<double, NB>& b) {
        std::array<double, NA + NB - 1> out{};
        for (std::size_t i = 0; i < NA; ++i)
            for (std::size_t j = 0; j < NB; ++j)
                out[i + j] += a[i] * b[j];
        return out;
    }

    static std::array<double, 7> cross_cube(double side) {
        const std::array<double, 3> kernel{
            side, 1.0 - 2.0 * side, side};
        return convolve_short_fir(
            convolve_short_fir(kernel, kernel), kernel);
    }

    // Apply a short directional FIR row-major. Interior pixels bypass all
    // periodic-boundary branches, which lets the compiler vectorize axial
    // and diagonal taps. Coefficient k addresses offset k-Origin.
    template <int DY, int DX, int Origin, std::size_t N>
    void directional_short_fir(
            const double* source, double* destination,
            const std::array<double, N>& coefficients) {
        P.run([&](int tid) {
            for (std::size_t y = std::size_t(tid); y < H;
                 y += std::size_t(P.lanes())) {
                std::array<std::size_t, N> rows{};
                for (std::size_t tap = 0; tap < N; ++tap) {
                    const int offset = int(tap) - Origin;
                    rows[tap] = short_periodic_offset(
                        y, offset * DY, H);
                }
                constexpr int first_offset = -Origin * DX;
                constexpr int last_offset =
                    (int(N) - 1 - Origin) * DX;
                constexpr int minimum_offset =
                    first_offset < last_offset
                    ? first_offset : last_offset;
                constexpr int maximum_offset =
                    first_offset > last_offset
                    ? first_offset : last_offset;
                std::size_t interior_begin = minimum_offset < 0
                    ? std::size_t(-minimum_offset) : 0;
                const std::size_t right_margin = maximum_offset > 0
                    ? std::size_t(maximum_offset) : 0;
                std::size_t interior_end = right_margin <= W
                    ? W - right_margin : 0;
                if (interior_begin > interior_end) {
                    // Tiny legal dimensions can be shorter than the FIR.
                    // Route the entire row through the fully periodic path.
                    interior_begin = 0;
                    interior_end = 0;
                }
                double* out = destination + y * W;

                for (std::size_t x = 0; x < interior_begin; ++x) {
                    double sum = 0.0;
                    for (std::size_t tap = 0; tap < N; ++tap) {
                        const int offset = int(tap) - Origin;
                        const std::size_t xx = short_periodic_offset(
                            x, offset * DX, W);
                        sum += coefficients[tap]
                            * source[rows[tap] * W + xx];
                    }
                    out[x] = sum;
                }
                for (std::size_t x = interior_begin;
                     x < interior_end; ++x) {
                    double sum = 0.0;
                    for (std::size_t tap = 0; tap < N; ++tap) {
                        const int offset = int(tap) - Origin;
                        const std::size_t xx = std::size_t(
                            std::ptrdiff_t(x) + offset * DX);
                        sum += coefficients[tap]
                            * source[rows[tap] * W + xx];
                    }
                    out[x] = sum;
                }
                for (std::size_t x = interior_end; x < W; ++x) {
                    double sum = 0.0;
                    for (std::size_t tap = 0; tap < N; ++tap) {
                        const int offset = int(tap) - Origin;
                        const std::size_t xx = short_periodic_offset(
                            x, offset * DX, W);
                        sum += coefficients[tap]
                            * source[rows[tap] * W + xx];
                    }
                    out[x] = sum;
                }
            }
        });
    }

    void accumulate_absolute_response(const double* response) {
        const std::size_t n = H * W;
        P.run([&](int tid) {
            std::size_t lo, hi;
            split(tid, n, lo, hi);
            for (std::size_t i = lo; i < hi; ++i)
                condition_gate[i] += std::fabs(response[i]);
        });
    }

    template <int LongDY, int LongDX, int CrossDY, int CrossDX,
              int DifferenceDY, int DifferenceDX, int Radius>
    void accumulate_condition_direction(const double* image,
                                        double cross_side) {
        directional_box_three_pass<LongDY, LongDX, Radius>(image);
        directional_three_pass<CrossDY, CrossDX>(
            u.data(), w.data(), cross_side);
        directional_three_pass<CrossDY, CrossDX>(
            w.data(), u.data(), cross_side);
        directional_three_pass<CrossDY, CrossDX>(
            u.data(), w.data(), cross_side);
        P.run([&](int tid) {
            for (std::size_t y = std::size_t(tid); y < H;
                 y += std::size_t(P.lanes())) {
                const std::size_t yn =
                    periodic_unit_step<DifferenceDY>(y, H);
                for (std::size_t x = 0; x < W; ++x) {
                    const std::size_t xn =
                        periodic_unit_step<DifferenceDX>(x, W);
                    const std::size_t i = y * W + x;
                    condition_gate[i] +=
                        std::fabs(w[yn * W + xn] - w[i]);
                }
            }
        });
    }

    // Exact algebraic collapse of the transverse three-pass filter. K^3 is
    // one seven-tap FIR. For the axial responses, whose final difference is
    // transverse too, D*K^3 is one eight-tap FIR. This preserves the staged
    // transfer function to roundoff while removing two full image passes per
    // direction; the long moving-box cascade remains unchanged.
    template <int LongDY, int LongDX, int CrossDY, int CrossDX,
              int DifferenceDY, int DifferenceDX, int Radius>
    void accumulate_condition_direction_collapsed(
            const double* image, double cross_side) {
        directional_box_three_pass<LongDY, LongDX, Radius>(image);
        const std::array<double, 7> cross3 = cross_cube(cross_side);
        if constexpr (CrossDY == DifferenceDY &&
                      CrossDX == DifferenceDX) {
            const std::array<double, 2> difference{-1.0, 1.0};
            const std::array<double, 8> fused =
                convolve_short_fir(cross3, difference);
            directional_short_fir<CrossDY, CrossDX, 3>(
                u.data(), w.data(), fused);
            accumulate_absolute_response(w.data());
        } else {
            directional_short_fir<CrossDY, CrossDX, 3>(
                u.data(), w.data(), cross3);
            P.run([&](int tid) {
                for (std::size_t y = std::size_t(tid); y < H;
                     y += std::size_t(P.lanes())) {
                    const std::size_t yn =
                        periodic_unit_step<DifferenceDY>(y, H);
                    for (std::size_t x = 0; x < W; ++x) {
                        const std::size_t xn =
                            periodic_unit_step<DifferenceDX>(x, W);
                        const std::size_t i = y * W + x;
                        condition_gate[i] += std::fabs(
                            w[yn * W + xn] - w[i]);
                    }
                }
            });
        }
    }

    void normalize_condition_gate() {
        const std::size_t n = H * W;
        double sum = 0.0;
        for (double value : condition_gate) sum += value;
        const double scale = std::fmax(1.6 * sum / double(n), 1e-12);
        P.run([&](int tid) {
            std::size_t lo, hi;
            split(tid, n, lo, hi);
            for (std::size_t i = lo; i < hi; ++i) {
                const double ratio = condition_gate[i] / scale;
                const double square = ratio * ratio;
                const double base = square / (1.0 + square);
                const double base2 = base * base;
                condition_gate[i] = base2 * base2 * base2;
            }
        });
    }

    // FACR leaves one coordinate spatial, so the full two-axis spectral
    // Gaussian is not diagonal.  Three short periodic box passes reproduce
    // the long-axis variance (12, or 6 in diagonal lattice coordinates),
    // while three weighted 3-tap passes reproduce the transverse variance
    // (0.75, or 0.375 diagonally).  This keeps the four-direction symmetric
    // statistic O(N), cache-local, and independent of which axis FACR
    // transforms.
    void build_condition_gate_facr(const double* image) {
        const std::size_t n = H * W;
        std::memset(condition_gate.data(), 0, n * sizeof(double));
        accumulate_condition_direction_collapsed<0, 1, 1, 0, 1, 0, 3>(
            image, 0.125);
        accumulate_condition_direction_collapsed<1, 0, 0, 1, 0, 1, 3>(
            image, 0.125);
        accumulate_condition_direction_collapsed<1, 1, 1, -1, 1, 1, 2>(
            image, 0.0625);
        accumulate_condition_direction_collapsed<1, -1, 1, 1, 1, -1, 2>(
            image, 0.0625);
        normalize_condition_gate();
    }

    void build_condition_gate_from_source_spec() {
        const std::size_t n = H * W;
        std::memset(condition_gate.data(), 0, n * sizeof(double));
        struct direction { int dy, dx; double theta; };
        const direction directions[4] = {
            {1, 0, 0.0},
            {0, 1, 0.5 * PI},
            {1, 1, 0.25 * PI},
            {1, -1, 0.75 * PI},
        };
        for (const direction& d : directions) {
            directional_gaussian(
                f_spec, q_spec, d.dy, d.dx, d.theta);
            inv2d(q_spec, u.data());
            P.run([&](int tid) {
                for (std::size_t i = std::size_t(tid); i < n;
                     i += std::size_t(P.lanes()))
                    condition_gate[i] += std::fabs(u[i]);
            });
        }

        normalize_condition_gate();
    }

    void build_jump_confidence_from_condition_gate() {
        constexpr int bins = 256;
        std::size_t histogram[bins] = {};
        const std::size_t n = H * W;
        for (const double value : condition_gate) {
            int bin = static_cast<int>(value * double(bins));
            if (bin < 0) bin = 0;
            if (bin >= bins) bin = bins - 1;
            ++histogram[bin];
        }

        double total_moment = 0.0;
        for (int bin = 0; bin < bins; ++bin) {
            const double center = (double(bin) + 0.5) / double(bins);
            total_moment += center * double(histogram[bin]);
        }
        std::size_t low_count = 0;
        double low_moment = 0.0;
        double best_variance = -1.0;
        int split_bin = 0;
        for (int bin = 0; bin < bins - 1; ++bin) {
            low_count += histogram[bin];
            low_moment += (double(bin) + 0.5) / double(bins)
                * double(histogram[bin]);
            const std::size_t high_count = n - low_count;
            if (low_count == 0 || high_count == 0) continue;
            const double numerator =
                total_moment * double(low_count)
                - low_moment * double(n);
            const double variance = numerator * numerator /
                (double(low_count) * double(high_count));
            if (variance > best_variance) {
                best_variance = variance;
                split_bin = bin;
            }
        }

        std::size_t high_count = 0;
        double high_moment = 0.0;
        for (int bin = split_bin + 1; bin < bins; ++bin) {
            high_count += histogram[bin];
            high_moment += (double(bin) + 0.5) / double(bins)
                * double(histogram[bin]);
        }
        const double boundary = double(split_bin + 1) / double(bins);
        const double high_mean = high_count > 0
            ? high_moment / double(high_count) : boundary;
        jump_confidence_boundary = boundary;
        jump_confidence_inverse_span = 1.0 /
            std::fmax(high_mean - boundary, 1e-30);
    }

    inline double jump_confidence_at(std::size_t i) const {
        return std::fmin(1.0, std::fmax(
            0.0, (condition_gate[i] - jump_confidence_boundary) *
                jump_confidence_inverse_span));
    }

    // q_spec <- scalar potential whose gradient is the longitudinal Hodge
    // projection of the Otsu-supported, nonnegative-garrote jump bonds.
    void build_jump_potential_spectrum(const double* value,
                                       double threshold) {
        P.run([&](int tid) {
            for (std::size_t y = std::size_t(tid); y < H;
                 y += std::size_t(P.lanes())) {
                const std::size_t yn = y + 1 == H ? 0 : y + 1;
                for (std::size_t x = 0; x < W; ++x) {
                    const std::size_t xn = x + 1 == W ? 0 : x + 1;
                    const std::size_t i = y * W + x;
                    const double gx = value[y * W + xn] - value[i];
                    const double gy = value[yn * W + x] - value[i];
                    const double magnitude2 = gx * gx + gy * gy;
                    const double activation = std::fmax(
                        1.0 - threshold * threshold /
                            std::fmax(magnitude2, 1e-30),
                        0.0);
                    const double weight =
                        jump_confidence_at(i) * activation;
                    bux[i] = weight * gx;
                    buy[i] = weight * gy;
                }
            }
        });
        fwd2d_div(bux, buy, d_spec);
        const double c_u = lam, eta_u = 2.0 * lam;
        P.run([&](int tid) {
            std::size_t lo, hi;
            split(tid, n2(), lo, hi);
            const double* __restrict da = d_spec.a.data();
            const double* __restrict db = d_spec.b.data();
            const double* __restrict su = s_u.data();
            double* __restrict qa = q_spec.a.data();
            double* __restrict qb = q_spec.b.data();
            for (std::size_t r = lo; r < hi; ++r) {
                const double laplacian =
                    (c_u - 1.0 / su[r]) / eta_u;
                const double inverse = std::fabs(laplacian) > 1e-15
                    ? 1.0 / laplacian : 0.0;
                qa[r] = da[r] * inverse;
                qb[r] = db[r] * inverse;
            }
        });
    }

    // fq_spec <- periodic scalar potential whose gradient is the FACR
    // longitudinal projection of the supported jump bonds.
    void build_jump_potential_facr(const double* value,
                                   double threshold) {
        P.run([&](int tid) {
            for (std::size_t y = std::size_t(tid); y < H;
                 y += std::size_t(P.lanes())) {
                const std::size_t yn = y + 1 == H ? 0 : y + 1;
                for (std::size_t x = 0; x < W; ++x) {
                    const std::size_t xn = x + 1 == W ? 0 : x + 1;
                    const std::size_t i = y * W + x;
                    const double gx = value[y * W + xn] - value[i];
                    const double gy = value[yn * W + x] - value[i];
                    const double magnitude2 = gx * gx + gy * gy;
                    const double activation = std::fmax(
                        1.0 - threshold * threshold /
                            std::fmax(magnitude2, 1e-30),
                        0.0);
                    const double weight =
                        jump_confidence_at(i) * activation;
                    bux[i] = weight * gx;
                    buy[i] = weight * gy;
                }
            }
        });
        facr_fwd_div(bux, buy, fd_spec);
        facr_poisson(fd_spec, fq_spec);
    }

    // d_spec <- (I-H_u^virtual_passes) (f_spec-q_spec).
    void build_virtual_oscillation_spectrum(int virtual_passes) {
        const double c_u = lam;
        P.run([&](int tid) {
            std::size_t lo, hi;
            split(tid, n2(), lo, hi);
            const double* __restrict fa = f_spec.a.data();
            const double* __restrict fb = f_spec.b.data();
            const double* __restrict qa = q_spec.a.data();
            const double* __restrict qb = q_spec.b.data();
            const double* __restrict su = s_u.data();
            double* __restrict da = d_spec.a.data();
            double* __restrict db = d_spec.b.data();
            for (std::size_t r = lo; r < hi; ++r) {
                double base = c_u * su[r], power = 1.0;
                int exponent = virtual_passes;
                while (exponent > 0) {
                    if (exponent & 1) power *= base;
                    base *= base;
                    exponent >>= 1;
                }
                const double highpass = 1.0 - power;
                da[r] = highpass * (fa[r] - qa[r]);
                db[r] = highpass * (fb[r] - qb[r]);
            }
        });
    }

    bool split_conditioned_first(const double* image, double* cartoon,
                                 double* texture, double strength) {
        if (facr_active || !(strength >= 0.0)) return false;
        ensure_conditioning_storage();
        const std::size_t n = H * W;
        fwd2d(image, f_spec);
        build_condition_gate_from_source_spec();

        const double c_u = lam, eta_u = 2.0 * lam;
        const double c_v = 1.0 / mu;
        fwd2d_conditioned_source_reflection(
            image, condition_gate, eta_u, strength, d_spec);
        solve_meyer_triangle_conditioned_first(
            d_spec, c_u, eta_u, c_v);
        inv2d(u_spec, u.data());
        inv2d(w_spec, w.data());
        std::memcpy(cartoon, u.data(), n * sizeof(double));
        finish_split_texture(image, texture);
        return true;
    }

    bool split_preconditioned(const double* image, double* cartoon,
                              double* texture, double strength,
                              int virtual_passes, int gate_power) {
        if (facr_active || !(strength >= 0.0) || virtual_passes < 1 ||
            virtual_passes > 64 || gate_power < 1 || gate_power > 64)
            return false;
        ensure_conditioning_storage();
        ensure_visit_storage();
        const std::size_t n = H * W;
        const double c_u = lam, eta_u = 2.0 * lam;

        fwd2d(image, f_spec);
        build_condition_gate_from_source_spec();

        // One multiplier takes virtual_passes early linear cartoon steps.
        // Integer exponentiation avoids both a runtime scan and libm pow in
        // the spectral hot loop.
        P.run([&](int tid) {
            std::size_t lo, hi;
            split(tid, n2(), lo, hi);
            const double* __restrict fa = f_spec.a.data();
            const double* __restrict fb = f_spec.b.data();
            const double* __restrict su = s_u.data();
            double* __restrict qa = q_spec.a.data();
            double* __restrict qb = q_spec.b.data();
            for (std::size_t r = lo; r < hi; ++r) {
                double base = c_u * su[r], power = 1.0;
                int exponent = virtual_passes;
                while (exponent > 0) {
                    if (exponent & 1) power *= base;
                    base *= base;
                    exponent >>= 1;
                }
                qa[r] = fa[r] * power;
                qb[r] = fb[r] * power;
            }
        });
        inv2d(q_spec, w.data());

        // target = f - confidence * (f - H^K f).  The high-certainty gate
        // keeps authentic fronts out of the virtual texture state.
        P.run([&](int tid) {
            for (std::size_t i = std::size_t(tid); i < n;
                 i += std::size_t(P.lanes())) {
                const double base = 1.0 - condition_gate[i];
                double confidence = 1.0;
                for (int p = 0; p < gate_power; ++p) confidence *= base;
                u[i] = image[i] - confidence * (image[i] - w[i]);
            }
        });
        fwd2d(u.data(), q_spec);
        fwd2d_conditioned_source_reflection(
            u.data(), condition_gate, eta_u, strength, d_spec);
        solve_conditioned_source(q_spec, d_spec, c_u, eta_u, u_spec);
        inv2d(u_spec, u.data());

        // Proposed texture is then lifted through one scalar periodic
        // Poisson solve.  Clamping grad(phi) to the radius-mu disk makes the
        // returned divergence constructively G_mu-feasible.
        P.run([&](int tid) {
            for (std::size_t i = std::size_t(tid); i < n;
                 i += std::size_t(P.lanes()))
                vplane[i] = image[i] - u[i];
        });
        fwd2d(vplane.data(), q_spec);
        P.run([&](int tid) {
            std::size_t lo, hi;
            split(tid, n2(), lo, hi);
            const double* __restrict qa = q_spec.a.data();
            const double* __restrict qb = q_spec.b.data();
            const double* __restrict su = s_u.data();
            double* __restrict da = d_spec.a.data();
            double* __restrict db = d_spec.b.data();
            for (std::size_t r = lo; r < hi; ++r) {
                const double laplacian =
                    (c_u - 1.0 / su[r]) / eta_u;
                const double inverse = std::fabs(laplacian) > 1e-15
                    ? 1.0 / laplacian : 0.0;
                da[r] = qa[r] * inverse;
                db[r] = qb[r] * inverse;
            }
        });
        inv2d(d_spec, w.data());
        P.run([&](int tid) {
            for (std::size_t y = std::size_t(tid); y < H;
                 y += std::size_t(P.lanes())) {
                const std::size_t yn = y + 1 == H ? 0 : y + 1;
                for (std::size_t x = 0; x < W; ++x) {
                    const std::size_t xn = x + 1 == W ? 0 : x + 1;
                    const std::size_t i = y * W + x;
                    bux[i] = w[y * W + xn] - w[i];
                    buy[i] = w[yn * W + x] - w[i];
                }
            }
        });

        // The longitudinal lift is minimum-L2, not minimum-L-infinity.
        // Use its capacity frame n=p/|p|, t=J n to take one deterministic
        // null-space route before the disk hit.  In stream form the desired
        // gradient is demand*t and q=-J P_L(demand*t); algebraically the
        // cheaper equivalent is q=P_T(demand*n).  A fractional underloaded
        // reservoir keeps P_T from merely moving overload to the nearest
        // inactive pixel.
        constexpr double slack_fraction = 0.20;
        P.run([&](int tid) {
            for (std::size_t i = std::size_t(tid); i < n;
                 i += std::size_t(P.lanes())) {
                const double px = bux[i], py = buy[i];
                const double magnitude = std::sqrt(px * px + py * py);
                const double inverse = 1.0 / std::fmax(magnitude, 1e-30);
                const double demand = magnitude > mu
                    ? mu - magnitude
                    : slack_fraction * (mu - magnitude);
                const double confidence = 1.0 - condition_gate[i];
                bvx[i] = 2.0 * demand * px * inverse * confidence;
                bvy[i] = 2.0 * demand * py * inverse * confidence;
            }
        });
        fwd2d_div(bvx, bvy, q_spec);
        P.run([&](int tid) {
            std::size_t lo, hi;
            split(tid, n2(), lo, hi);
            const double* __restrict qa = q_spec.a.data();
            const double* __restrict qb = q_spec.b.data();
            const double* __restrict su = s_u.data();
            double* __restrict da = d_spec.a.data();
            double* __restrict db = d_spec.b.data();
            for (std::size_t r = lo; r < hi; ++r) {
                const double laplacian =
                    (c_u - 1.0 / su[r]) / eta_u;
                const double inverse = std::fabs(laplacian) > 1e-15
                    ? 1.0 / laplacian : 0.0;
                da[r] = qa[r] * inverse;
                db[r] = qb[r] * inverse;
            }
        });
        inv2d(d_spec, w.data());
        P.run([&](int tid) {
            for (std::size_t y = std::size_t(tid); y < H;
                 y += std::size_t(P.lanes())) {
                const std::size_t yn = y + 1 == H ? 0 : y + 1;
                for (std::size_t x = 0; x < W; ++x) {
                    const std::size_t xn = x + 1 == W ? 0 : x + 1;
                    const std::size_t i = y * W + x;
                    bvx[i] -= w[y * W + xn] - w[i];
                    bvy[i] -= w[yn * W + x] - w[i];
                }
            }
        });

        // One right-Newton coefficient for squared capacity overload.  The
        // reductions are serial by design, preserving thread-count identity.
        double first = 0.0, second = 0.0, old_energy = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            const double px = bux[i], py = buy[i];
            const double magnitude = std::sqrt(px * px + py * py);
            if (!(magnitude > mu)) continue;
            const double qx = bvx[i], qy = bvy[i];
            const double radial = (px * qx + py * qy) / magnitude;
            const double excess = magnitude - mu;
            const double q2 = qx * qx + qy * qy;
            first += 2.0 * excess * radial;
            second += 2.0 * (
                radial * radial
                + excess / magnitude * (q2 - radial * radial));
            old_energy += excess * excess;
        }
        double route_alpha = second > 1e-30
            ? std::fmin(2.0, std::fmax(0.0, -first / second)) : 0.0;
        double new_energy = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            const double px = bux[i] + route_alpha * bvx[i];
            const double py = buy[i] + route_alpha * bvy[i];
            const double excess = std::fmax(
                std::sqrt(px * px + py * py) - mu, 0.0);
            new_energy += excess * excess;
        }
        if (!(new_energy < old_energy)) route_alpha = 0.0;

        P.run([&](int tid) {
            for (std::size_t i = std::size_t(tid); i < n;
                 i += std::size_t(P.lanes())) {
                double px = bux[i] + route_alpha * bvx[i];
                double py = buy[i] + route_alpha * bvy[i];
                const double magnitude = std::sqrt(px * px + py * py);
                const double scale = std::fmin(
                    1.0, mu / std::fmax(magnitude, 1e-30));
                bux[i] = px * scale;
                buy[i] = py * scale;
            }
        });
        P.run([&](int tid) {
            for (std::size_t y = std::size_t(tid); y < H;
                 y += std::size_t(P.lanes())) {
                const std::size_t yp = (y == 0 ? H : y) - 1;
                for (std::size_t x = 0; x < W; ++x) {
                    const std::size_t xp = (x == 0 ? W : x) - 1;
                    const std::size_t i = y * W + x;
                    const double value =
                        bux[i] - bux[y * W + xp]
                        + buy[i] - buy[yp * W + x];
                    texture[i] = value;
                    cartoon[i] = image[i] - value;
                }
            }
        });
        return true;
    }

    bool split_jump_measure_facr(const double* image, double* cartoon,
                                 double* texture, int virtual_passes) {
        if (!facr_active || solver != 1 || virtual_passes < 1 ||
            virtual_passes > 64)
            return false;
        ensure_jump_measure_storage();
        const std::size_t n = H * W;

        facr_fwd(image, ff_spec);
        build_condition_gate_facr(image);
        build_jump_confidence_from_condition_gate();

        // The first observation is deliberately permissive: it learns a
        // carrier seed at half the Meyer reflected-current threshold.
        build_jump_potential_facr(image, 1.0 / (4.0 * lam));
        const facr_spectrum& first_lowpass =
            build_virtual_lowpass_facr(
                ff_spec, fq_spec, virtual_passes);
        facr_inv_difference(fd_spec, first_lowpass, w.data());

        // Remove the first carrier estimate, then take the same fixed second
        // jump observation as the full spectral operator.
        P.run([&](int tid) {
            std::size_t lo, hi;
            split(tid, n, lo, hi);
            for (std::size_t i = lo; i < hi; ++i)
                u[i] = image[i] - w[i];
        });
        // After removing that seed, admit structure at the full Meyer
        // threshold. Reusing the exploratory threshold here promotes a pure
        // alternating carrier to the structural jump potential.
        build_jump_potential_facr(u.data(), 1.0 / (2.0 * lam));
        const facr_spectrum& resident_lowpass =
            build_virtual_lowpass_facr(
                ff_spec, fq_spec, virtual_passes);

        // Longitudinal lift of the resident oscillation.
        facr_poisson_difference(
            fd_spec, resident_lowpass, fq_spec);
        facr_inv(fq_spec, w.data());
        P.run([&](int tid) {
            for (std::size_t y = std::size_t(tid); y < H;
                 y += std::size_t(P.lanes())) {
                const std::size_t yn = y + 1 == H ? 0 : y + 1;
                for (std::size_t x = 0; x < W; ++x) {
                    const std::size_t xn = x + 1 == W ? 0 : x + 1;
                    const std::size_t i = y * W + x;
                    bux[i] = w[y * W + xn] - w[i];
                    buy[i] = w[yn * W + x] - w[i];
                }
            }
        });

        bool capacity_overload = false;
        const double mu2 = mu * mu;
        for (std::size_t i = 0; i < n && !capacity_overload; ++i)
            capacity_overload =
                bux[i] * bux[i] + buy[i] * buy[i] > mu2;
        if (!capacity_overload) {
            // With an already feasible lift, the transverse proposal would
            // receive route_alpha=0 and the final disk projection would be
            // the identity. Avoid its forward transform, Poisson solve, and
            // inverse transform entirely.
            P.run([&](int tid) {
                for (std::size_t y = std::size_t(tid); y < H;
                     y += std::size_t(P.lanes())) {
                    const std::size_t yp = (y == 0 ? H : y) - 1;
                    for (std::size_t x = 0; x < W; ++x) {
                        const std::size_t xp = (x == 0 ? W : x) - 1;
                        const std::size_t i = y * W + x;
                        const double value =
                            bux[i] - bux[y * W + xp] +
                            buy[i] - buy[yp * W + x];
                        cartoon[i] = image[i] - value;
                        texture[i] = value;
                    }
                }
            });
            return true;
        }

        constexpr double slack_fraction = 0.20;
        P.run([&](int tid) {
            std::size_t lo, hi;
            split(tid, n, lo, hi);
            for (std::size_t i = lo; i < hi; ++i) {
                const double px = bux[i], py = buy[i];
                const double magnitude = std::sqrt(px * px + py * py);
                const double inverse = 1.0 / std::fmax(magnitude, 1e-30);
                const double demand = magnitude > mu
                    ? mu - magnitude
                    : slack_fraction * (mu - magnitude);
                const double confidence = 1.0 - condition_gate[i];
                bvx[i] = 2.0 * demand * px * inverse * confidence;
                bvy[i] = 2.0 * demand * py * inverse * confidence;
            }
        });
        facr_fwd_div(bvx, bvy, fq_spec);
        facr_poisson(fq_spec, fu_spec);
        facr_inv(fu_spec, w.data());
        P.run([&](int tid) {
            for (std::size_t y = std::size_t(tid); y < H;
                 y += std::size_t(P.lanes())) {
                const std::size_t yn = y + 1 == H ? 0 : y + 1;
                for (std::size_t x = 0; x < W; ++x) {
                    const std::size_t xn = x + 1 == W ? 0 : x + 1;
                    const std::size_t i = y * W + x;
                    bvx[i] -= w[y * W + xn] - w[i];
                    bvy[i] -= w[yn * W + x] - w[i];
                }
            }
        });

        double first = 0.0, second = 0.0, old_energy = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            const double px = bux[i], py = buy[i];
            const double magnitude = std::sqrt(px * px + py * py);
            if (!(magnitude > mu)) continue;
            const double qx = bvx[i], qy = bvy[i];
            const double radial = (px * qx + py * qy) / magnitude;
            const double excess = magnitude - mu;
            const double q2 = qx * qx + qy * qy;
            first += 2.0 * excess * radial;
            second += 2.0 * (
                radial * radial +
                excess / magnitude * (q2 - radial * radial));
            old_energy += excess * excess;
        }
        double route_alpha = second > 1e-30
            ? std::fmin(2.0, std::fmax(0.0, -first / second)) : 0.0;
        double new_energy = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            const double px = bux[i] + route_alpha * bvx[i];
            const double py = buy[i] + route_alpha * bvy[i];
            const double excess = std::fmax(
                std::sqrt(px * px + py * py) - mu, 0.0);
            new_energy += excess * excess;
        }
        if (!(new_energy < old_energy)) route_alpha = 0.0;

        P.run([&](int tid) {
            std::size_t lo, hi;
            split(tid, n, lo, hi);
            for (std::size_t i = lo; i < hi; ++i) {
                double px = bux[i] + route_alpha * bvx[i];
                double py = buy[i] + route_alpha * bvy[i];
                const double magnitude = std::sqrt(px * px + py * py);
                const double scale = std::fmin(
                    1.0, mu / std::fmax(magnitude, 1e-30));
                bux[i] = px * scale;
                buy[i] = py * scale;
            }
        });
        P.run([&](int tid) {
            for (std::size_t y = std::size_t(tid); y < H;
                 y += std::size_t(P.lanes())) {
                const std::size_t yp = (y == 0 ? H : y) - 1;
                for (std::size_t x = 0; x < W; ++x) {
                    const std::size_t xp = (x == 0 ? W : x) - 1;
                    const std::size_t i = y * W + x;
                    const double oscillation =
                        bux[i] - bux[y * W + xp] +
                        buy[i] - buy[yp * W + x];
                    cartoon[i] = image[i] - oscillation;
                    texture[i] = oscillation;
                }
            }
        });
        return true;
    }

    bool split_jump_measure(const double* image, double* cartoon,
                            double* texture, int virtual_passes) {
        if (virtual_passes < 1 || virtual_passes > 64)
            return false;
        if (facr_active)
            return split_jump_measure_facr(
                image, cartoon, texture, virtual_passes);
        ensure_jump_measure_storage();
        const std::size_t n = H * W;
        const double c_u = lam, eta_u = 2.0 * lam;

        fwd2d(image, f_spec);
        build_condition_gate_from_source_spec();
        build_jump_confidence_from_condition_gate();

        // First feed-forward jump observation and its virtual oscillation.
        // Exploratory high-recall observation: learn a carrier seed at half
        // the Meyer reflected-current threshold.
        build_jump_potential_spectrum(image, 1.0 / (4.0 * lam));
        build_virtual_oscillation_spectrum(virtual_passes);
        inv2d(d_spec, w.data());

        // The initial oscillation supplies a carrier estimate. Removing it
        // once before rebuilding the jump prevents the crossing carrier from
        // entering the boundary measure. This is a fixed second measurement,
        // not a convergence loop.
        P.run([&](int tid) {
            for (std::size_t i = std::size_t(tid); i < n;
                 i += std::size_t(P.lanes()))
                u[i] = image[i] - w[i];
        });
        // Structural high-precision observation: after carrier removal, use
        // the full Meyer threshold so alternating current is not relabelled
        // as a coherent jump.
        build_jump_potential_spectrum(u.data(), 1.0 / (2.0 * lam));

        // Keep the final material oscillation spectral. The capacity route
        // consumes this spectrum directly, avoiding an inverse+forward round
        // trip through vplane.
        build_virtual_oscillation_spectrum(virtual_passes);

        // Longitudinal Hodge lift of the resident oscillatory spectrum.
        P.run([&](int tid) {
            std::size_t lo, hi;
            split(tid, n2(), lo, hi);
            const double* __restrict da = d_spec.a.data();
            const double* __restrict db = d_spec.b.data();
            const double* __restrict su = s_u.data();
            double* __restrict qa = q_spec.a.data();
            double* __restrict qb = q_spec.b.data();
            for (std::size_t r = lo; r < hi; ++r) {
                const double laplacian =
                    (c_u - 1.0 / su[r]) / eta_u;
                const double inverse = std::fabs(laplacian) > 1e-15
                    ? 1.0 / laplacian : 0.0;
                qa[r] = da[r] * inverse;
                qb[r] = db[r] * inverse;
            }
        });
        inv2d(q_spec, w.data());
        P.run([&](int tid) {
            for (std::size_t y = std::size_t(tid); y < H;
                 y += std::size_t(P.lanes())) {
                const std::size_t yn = y + 1 == H ? 0 : y + 1;
                for (std::size_t x = 0; x < W; ++x) {
                    const std::size_t xn = x + 1 == W ? 0 : x + 1;
                    const std::size_t i = y * W + x;
                    bux[i] = w[y * W + xn] - w[i];
                    buy[i] = w[yn * W + x] - w[i];
                }
            }
        });

        // One transverse capacity route, identical to the validated native
        // preconditioner. The structural statistic governs where reservoir
        // capacity is available; the correction itself is divergence-free.
        constexpr double slack_fraction = 0.20;
        P.run([&](int tid) {
            for (std::size_t i = std::size_t(tid); i < n;
                 i += std::size_t(P.lanes())) {
                const double px = bux[i], py = buy[i];
                const double magnitude = std::sqrt(px * px + py * py);
                const double inverse = 1.0 / std::fmax(magnitude, 1e-30);
                const double demand = magnitude > mu
                    ? mu - magnitude
                    : slack_fraction * (mu - magnitude);
                const double confidence = 1.0 - condition_gate[i];
                bvx[i] = 2.0 * demand * px * inverse * confidence;
                bvy[i] = 2.0 * demand * py * inverse * confidence;
            }
        });
        fwd2d_div(bvx, bvy, q_spec);
        P.run([&](int tid) {
            std::size_t lo, hi;
            split(tid, n2(), lo, hi);
            const double* __restrict qa = q_spec.a.data();
            const double* __restrict qb = q_spec.b.data();
            const double* __restrict su = s_u.data();
            double* __restrict da = d_spec.a.data();
            double* __restrict db = d_spec.b.data();
            for (std::size_t r = lo; r < hi; ++r) {
                const double laplacian =
                    (c_u - 1.0 / su[r]) / eta_u;
                const double inverse = std::fabs(laplacian) > 1e-15
                    ? 1.0 / laplacian : 0.0;
                da[r] = qa[r] * inverse;
                db[r] = qb[r] * inverse;
            }
        });
        inv2d(d_spec, w.data());
        P.run([&](int tid) {
            for (std::size_t y = std::size_t(tid); y < H;
                 y += std::size_t(P.lanes())) {
                const std::size_t yn = y + 1 == H ? 0 : y + 1;
                for (std::size_t x = 0; x < W; ++x) {
                    const std::size_t xn = x + 1 == W ? 0 : x + 1;
                    const std::size_t i = y * W + x;
                    bvx[i] -= w[y * W + xn] - w[i];
                    bvy[i] -= w[yn * W + x] - w[i];
                }
            }
        });

        double first = 0.0, second = 0.0, old_energy = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            const double px = bux[i], py = buy[i];
            const double magnitude = std::sqrt(px * px + py * py);
            if (!(magnitude > mu)) continue;
            const double qx = bvx[i], qy = bvy[i];
            const double radial = (px * qx + py * qy) / magnitude;
            const double excess = magnitude - mu;
            const double q2 = qx * qx + qy * qy;
            first += 2.0 * excess * radial;
            second += 2.0 * (
                radial * radial
                + excess / magnitude * (q2 - radial * radial));
            old_energy += excess * excess;
        }
        double route_alpha = second > 1e-30
            ? std::fmin(2.0, std::fmax(0.0, -first / second)) : 0.0;
        double new_energy = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            const double px = bux[i] + route_alpha * bvx[i];
            const double py = buy[i] + route_alpha * bvy[i];
            const double excess = std::fmax(
                std::sqrt(px * px + py * py) - mu, 0.0);
            new_energy += excess * excess;
        }
        if (!(new_energy < old_energy)) route_alpha = 0.0;

        P.run([&](int tid) {
            for (std::size_t i = std::size_t(tid); i < n;
                 i += std::size_t(P.lanes())) {
                double px = bux[i] + route_alpha * bvx[i];
                double py = buy[i] + route_alpha * bvy[i];
                const double magnitude = std::sqrt(px * px + py * py);
                const double scale = std::fmin(
                    1.0, mu / std::fmax(magnitude, 1e-30));
                bux[i] = px * scale;
                buy[i] = py * scale;
            }
        });
        P.run([&](int tid) {
            for (std::size_t y = std::size_t(tid); y < H;
                 y += std::size_t(P.lanes())) {
                const std::size_t yp = (y == 0 ? H : y) - 1;
                for (std::size_t x = 0; x < W; ++x) {
                    const std::size_t xp = (x == 0 ? W : x) - 1;
                    const std::size_t i = y * W + x;
                    const double oscillation =
                        bux[i] - bux[y * W + xp]
                        + buy[i] - buy[yp * W + x];
                    cartoon[i] = image[i] - oscillation;
                    texture[i] = oscillation;
                }
            }
        });
        return true;
    }

    // All intermediate model states from one run.  Outputs are
    // passes*height*width doubles in pass-major order.
    void split_trace(const double* image, double* cartoon_trace,
                     double* texture_trace) {
        run_split_reduced(image, nullptr, cartoon_trace, texture_trace);
    }

    void split_visit(const double* image, trace_visitor visitor, void* user) {
        ensure_visit_storage();
        run_split_reduced(image, nullptr, nullptr, nullptr, visitor, user);
    }

    // ---- the full decomposition -----------------------------------------

    void decompose(const double* image, double* cartoon, double* texture,
                   double* band_coarse, double* band_mid, double* band_fine) {
        ensure_decompose_storage();
        if (facr_active) {
            decompose_facr(image, cartoon, texture, band_coarse, band_mid,
                           band_fine);
            return;
        }
        const std::size_t n = H * W;
        run_split_reduced_spectral(image, vplane.data());

        // g for every rung is v: spectrum by linear combination, no
        // transform
        {
            double* __restrict va = v_spec.a.data();
            double* __restrict vb = v_spec.b.data();
            const double* __restrict fa = f_spec.a.data();
            const double* __restrict fb = f_spec.b.data();
            const double* __restrict ua = u_spec.a.data();
            const double* __restrict ub = u_spec.b.data();
            const double* __restrict wa = w_spec.a.data();
            const double* __restrict wb = w_spec.b.data();
            const std::size_t m = n2();
            for (std::size_t r = 0; r < m; ++r) {
                va[r] = fa[r] - ua[r] - wa[r];
                vb[r] = fb[r] - ub[r] - wb[r];
            }
        }

        // ladder rungs: independent solves, fresh states, coarse -> fine
        // (u_spec is reused as spectral scratch; u spatial stays intact)
        const std::vector<double>* rung_s[3] = {&s_r0, &s_r1, &s_r2};
        const double rung_mu[3] = {mu, mu / 4.0, mu / 16.0};
        double* rung_out[3] = {cartoon, band_mid, band_fine};  // staging
        for (int rr = 0; rr < 3; ++rr)
            rof_from_spec(v_spec, vplane.data(), 1.0 / rung_mu[rr],
                          10.0 / rung_mu[rr], *rung_s[rr], rung_sweeps,
                          rung_tol, rung_out[rr]);

        // assemble outputs: cartoon/band_mid/band_fine currently hold
        // s0/s1/s2 and are rewritten in place (per-index reads complete
        // before the overwrites of the same slots), so those pointers
        // deliberately carry no restrict qualifier
        {
            const double* __restrict vp = vplane.data();
            const double* __restrict up = u.data();
            for (std::size_t i = 0; i < n; ++i) {
                const double a0 = cartoon[i], a1 = band_mid[i],
                             a2 = band_fine[i];
                texture[i] = vp[i];
                cartoon[i] = up[i] + a0;
                band_mid[i] = a2 - a1;
                band_fine[i] = vp[i] - a2;
                band_coarse[i] = a1 - a0;
            }
        }
    }

    void decompose_facr(const double* image, double* cartoon,
                        double* texture, double* band_coarse,
                        double* band_mid, double* band_fine) {
        const std::size_t n = H * W;
        run_split_reduced_facr(image, vplane.data());
        const std::size_t m = FS * FB;
        for (std::size_t i = 0; i < m; ++i) {
            fv_spec.a[i] = ff_spec.a[i] - fu_spec.a[i] - fw_spec.a[i];
            fv_spec.b[i] = ff_spec.b[i] - fu_spec.b[i] - fw_spec.b[i];
        }
        const tri_factors* rung_t[3] = {&t_r0, &t_r1, &t_r2};
        const double rung_mu[3] = {mu, mu / 4.0, mu / 16.0};
        double* rung_out[3] = {cartoon, band_mid, band_fine};
        for (int rr = 0; rr < 3; ++rr)
            rof_from_facr(fv_spec, vplane.data(), 1.0 / rung_mu[rr],
                           10.0 / rung_mu[rr], *rung_t[rr], rung_sweeps,
                           rung_tol, rung_out[rr]);
        for (std::size_t i = 0; i < n; ++i) {
            const double a0 = cartoon[i], a1 = band_mid[i],
                         a2 = band_fine[i];
            texture[i] = vplane[i];
            cartoon[i] = u[i] + a0;
            band_mid[i] = a2 - a1;
            band_fine[i] = vplane[i] - a2;
            band_coarse[i] = a1 - a0;
        }
    }
};

}  // namespace meyer
