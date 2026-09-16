#pragma once

// MIT joshuah.rainstar@gmail.com 2026
// Internal radix-4 real DIT FFT experiment kernel.
//
// This header keeps the standalone NEON prototype as a reusable kernel object:
// construction owns all planning tables, callers provide input/output/work
// buffers, and the SIMD path is selected through the shared BFFT two-lane
// backend abstraction instead of including one architecture directly.
//
// ---------------------------------------------------------------------------
// The mirrored-lane walk (the SIMD path below)
// ---------------------------------------------------------------------------
// The transform is the usual real-Bruun DIT tree: seed nodes are gathered from
// the input in bit-reversed order, radix-4 merges reduce children modulo the
// real quadratic factors of z^N - 1, and each spectral bin k finally leaves the
// tree as a residue pair (a, b) with X_k = a + b * e^{-i 2 pi k / N}.
//
// The walk through that tree is chosen differently from the obvious one.
// Observe two facts about the node-index space [0, N/2):
//
//   1. Node b and node b + N/4 (top index bit) run *identical* butterfly
//      sequences with *identical* twiddles through every stage below the
//      final merge; they only meet at the top, where they sit in the
//      (q0,q2) / (q1,q3) child quarters of the last radix-4 reduction.
//   2. Their seed reads are input-adjacent: rev(b + N/4) == rev(b) + 1.
//
// So instead of packing SIMD lanes across neighbouring spectral nodes inside
// one block - which forces a deinterleave/interleave permute pair around every
// butterfly plus scalar DC/Nyquist/midpoint special cases - the two lanes of
// every vector are the two mirror trees. Work layout ("slot" = node pair):
//
//   slot s in [0, N/4):  vwork[4s+0..1] = { a(node s), a(node s + N/4) }
//                        vwork[4s+2..3] = { b(node s), b(node s + N/4) }
//
// Consequences:
//   - Seed loads are contiguous V2 loads at rev(b): the intake shuffle stays
//     frontloaded (ordered -> out-of-order reads) but is now vector-width and
//     table-halved. Seed blocks are visited in bit-reversed block order so the
//     gather decays into ~2*SB sequential comb streams instead of full-array
//     sweeps that refetch every input line; large N stops thrashing.
//   - Every merge stage below the top is pure vertical SIMD: no permutes at
//     all, twiddles broadcast with V2_SET1, and the DC/Nyquist/sqrt2 chains
//     vectorize across the mirror lanes instead of running scalar.
//   - The single unavoidable lane crossing happens exactly once, in the
//     terminal merge, where it fuses with the (a,b) -> complex projection.
//     The projection therefore costs no separate pass at any size, and its
//     four output bins per step share one {cos,sin} table vector through the
//     quarter-turn symmetries cos(pi/2 - t) = sin(t), cos(pi - t) = -cos(t).
//
// The inverse runs the same walk backwards and keeps the same single
// lane-crossing point. Standard bins deproject straight into mirrored-lane
// layout inside the terminal split (the reverse of terminal_v, fused with
// (re,im) -> (a,b) deprojection), and every split below it is lane-vertical
// with no per-butterfly permutes. The seed output transport consumes each
// completed vwork block with linear V2 loads; the permutation cost stays on
// the store side, where mirror lanes still land as adjacent time samples
// because rev(b + N/4) = rev(b) + 1. Each inverse primitive is the exact
// algebraic inverse of its forward counterpart, so the distributed 0.5 factors
// compose to exactly 1/N and inverse(forward(x)) == x, matching the bfft
// convention. The in-place inverse merge is clobber-free by the same (i, M-i)
// pairing as the forward: the parent read set of a pair equals its child write
// set.
//
// The scalar path below keeps the plain single-lane walk for reference and
// for BRUUN_LEVEL == 0 builds.
//
// ---------------------------------------------------------------------------
// Normalized (unit-frame) basis  --  the live forward_simd / inverse_simd
// ---------------------------------------------------------------------------
// The residue merge above is written in a raw basis where the twiddle is
// c2 = 2 cos(theta): it grows toward the tree top and forms 1 - c2^2
// cancellations, costing accuracy (~e-12 roundtrip at N = 2^20). The live
// SIMD path instead runs each merge as a Givens rotation with c^2 + s^2 = 1
// exactly (pair_reduce_v2_norm / pair_expand_v2_norm, tables cs_ and
// fterm_cs_). In this normalized basis a leaf residue pair is directly
// (re, -im), so the forward's residue -> complex projection and the inverse's
// complex -> residue deprojection both collapse to a sign flip, and there is
// no per-bin cos/sin at the terminals. The mirrored-lane WALK (seed, lane
// packing, block order, the single terminal lane crossing) is unchanged --
// only the arithmetic primitive differs -- so the layout commentary above
// still applies. Roundtrip is machine precision (~e-15 at every size) because
// inverse_simd is the exact algebraic inverse of forward_simd.

#include "bruun_simd_backend.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>

namespace bruun {

class DIT_RFFT_kernel {
public:
    DIT_RFFT_kernel() noexcept
        : n_(0), half_(0), twiddle_stage_count_(0), radix2_terminal_(false) {}

    bool init(int n) {
        return reset(n);
    }

    bool reset(int n) {
        if (!valid_size(n)) {
            clear();
            return false;
        }

        n_ = n;
        half_ = n >> 1;

        const int logn = ilog2_pow2(n);
        radix2_terminal_ = (logn & 1) != 0;
        const int rev_bits = logn - 1;
        if (!rev_.resize(static_cast<std::size_t>(half_))) {
            clear();
            return false;
        }
        for (int b = 0; b < half_; ++b) {
            rev_[static_cast<std::size_t>(b)] = bitrev_int(b, rev_bits);
        }

        if (!cos_.resize(static_cast<std::size_t>(half_ + 1))) {
            clear();
            return false;
        }
        if (!sin_.resize(static_cast<std::size_t>(half_ + 1))) {
            clear();
            return false;
        }
        for (int k = 0; k <= half_; ++k) {
            const double theta = bruun_tau * static_cast<double>(k) / static_cast<double>(n_);
            cos_[static_cast<std::size_t>(k)] = std::cos(theta);
            sin_[static_cast<std::size_t>(k)] = std::sin(theta);
        }

        twiddle_stage_count_ = 0;
        int twiddle_count = 0;
        for (int s = 4; s <= n_; s <<= 1) {
            ++twiddle_stage_count_;
            const int q = s >> 2;
            if (q > 1) {
                twiddle_count += q - 1;
            }
        }

        if (!twiddle_offset_.resize(static_cast<std::size_t>(twiddle_stage_count_))) {
            clear();
            return false;
        }
        int alloc_count = twiddle_count;
        if (alloc_count == 0) {
            alloc_count = 1;
        }
        if (!twiddle_.resize(static_cast<std::size_t>(alloc_count))) {
            clear();
            return false;
        }

        // Normalized-basis rotation table, parallel to twiddle_: per entry
        // { c, s } = { cos(theta), sin(theta) } with c = 0.5 * twiddle_[t].
        // These drive the stable Givens-rotation merge/expand
        // (pair_reduce_*_norm / pair_expand_*_norm) in the SIMD path. Built
        // inline with the twiddle table so s is the libm sine of the merge
        // angle rather than sqrt(1 - c^2), which cancels catastrophically for
        // the small top-stage angles (~1e-6 relative error at N = 2^20).
        if (!cs_.resize(static_cast<std::size_t>(2 * alloc_count))) {
            clear();
            return false;
        }

        int offset = 0;
        int stage = 0;
        for (int s = 4; s <= n_; s <<= 1) {
            twiddle_offset_[static_cast<std::size_t>(stage)] = offset;
            const int q = s >> 2;
            for (int k = 1; k < q; ++k) {
                const double theta = bruun_tau * static_cast<double>(k) / static_cast<double>(s);
                const double c = std::cos(theta);
                twiddle_[static_cast<std::size_t>(offset)] = 2.0 * c;
                cs_[static_cast<std::size_t>(2 * offset)] = c;
                cs_[static_cast<std::size_t>(2 * offset + 1)] = std::sin(theta);
                ++offset;
            }
            ++stage;
        }

        // Normalized forward terminal table: for each i in [1, n/8) one
        // 6-double record { cA, sA, cBlo, cBhi, sBlo, sBhi }. cA/sA is the
        // shared level-A rotation (node 2i, angle 2*theta_i); cBlo/sBlo and
        // cBhi/sBhi are the two per-lane level-B rotations (bins {i, n/4-i}
        // and {n/2-i, n/4+i}). Read straight from the libm cos_/sin_ tables so
        // V2_LD(rec+2) = {cBlo, cBhi} and V2_LD(rec+4) = {sBlo, sBhi} load as
        // lane-paired vectors; no sqrt(1 - c^2) and no term_tw_ intermediate.
        const int tm = n_ >> 3;
        const int q4 = n_ >> 2;
        int fterm_count = tm > 1 ? 6 * (tm - 1) : 1;
        if (!fterm_cs_.resize(static_cast<std::size_t>(fterm_count))) {
            clear();
            return false;
        }
        for (int i = 1; i < tm; ++i) {
            double* dst = fterm_cs_.data() + 6 * (i - 1);
            dst[0] = cos_[static_cast<std::size_t>(2 * i)];
            dst[1] = sin_[static_cast<std::size_t>(2 * i)];
            dst[2] = cos_[static_cast<std::size_t>(i)];
            dst[3] = cos_[static_cast<std::size_t>(q4 - i)];
            dst[4] = sin_[static_cast<std::size_t>(i)];
            dst[5] = sin_[static_cast<std::size_t>(q4 - i)];
        }
        return true;
    }

    int size() const noexcept { return n_; }
    int bins() const noexcept { return half_ + 1; }
    int work_size() const noexcept { return n_; }        // doubles
    int work_size_f32() const noexcept { return n_; }    // floats

    void forward_scalar(const double* input, complex_t* output, double* work) const {
        forward_standard_scalar_impl(input, output, work);
    }

    // Cleanup local extension: the existing normalized Bruun walk over a
    // lane-valued coefficient field. One traversal carries multiple responses;
    // callers supply arithmetic lane types and fixed input/output/work storage.
    template <typename Lane, typename ComplexLane>
    void forward_lanes(const Lane* input, ComplexLane* output, Lane* work) const {
        forward_standard_scalar_impl_t<Lane>(input, output, work);
    }

    void forward_simd(const double* input, complex_t* output, double* work) const {
#if BRUUN_LEVEL >= 1
        if (n_ >= 16) {
            compute_vwork_norm(input, work);
            terminal_complex_writer<complex_t> w{output};
            if (radix2_terminal_) {
                terminal_radix2_v_norm(work, w);
            } else {
                terminal_v_norm(work, w);
            }
            return;
        }
#endif
        forward_scalar(input, output, work);
    }

    void inverse_scalar(const complex_t* input, double* output, double* work) const {
        inverse_small_scalar_impl(input, output, work);
    }

    void inverse_simd(const complex_t* input, double* output, double* work) const {
#if BRUUN_LEVEL >= 1
        if (n_ >= 16) {
            // NOTE: an odd-size fused three-level inverse terminal
            // (terminal3_inv_v_norm, radix-2 crossing + first radix-4 split in
            // one pass) was implemented and measured 8-21% SLOWER than this
            // two-pass path. The saved work round trip is real but smaller than
            // the cost of de-vectorizing the radix-2 crossing: the palindromic
            // split access reads 4 non-adjacent slots, so the crossing cannot
            // stay lane-paired the way terminal_radix2_inv_v_norm does. Kept as
            // two passes. (The f32 wide path is different -- see
            // inverse_simd_f32_wide -- because it also carries a V2->V4F
            // conversion pass that the fused terminal removes.)
            if (radix2_terminal_) {
                terminal_radix2_inv_v_norm(input, work);
            } else {
                terminal_inv_v_norm(input, work);
            }
            compute_vwork_inv_norm(work, output);
            return;
        }
#endif
        inverse_scalar(input, output, work);
    }

    // ---- f32 (genuine float32 compute) --------------------------------
    // work buffer is `work_size()` floats. Same standard-order complex bins.
    void forward_scalar_f32(const float* input, complex_f32_t* output, float* work) const {
        forward_standard_scalar_impl_t<float>(input, output, work);
    }

    void inverse_scalar_f32(const complex_f32_t* input, float* output, float* work) const {
        inverse_small_scalar_impl_t<float>(input, output, work);
    }

    // f32 SIMD: n >= 16 uses the wide 4-lane bruun_v4f walk (including the
    // odd-log2 radix-2 terminal); smaller sizes fall back to genuine-float
    // scalar. All portable V4F -- same guard as the rest of the level-1 SIMD.
    void forward_simd_f32(const float* input, complex_f32_t* output, float* work) const {
#if BRUUN_LEVEL >= 1
        if (f32_wide_ok()) {
            forward_simd_f32_wide(input, output, work);
            return;
        }
#endif
        forward_scalar_f32(input, output, work);
    }

    void inverse_simd_f32(const complex_f32_t* input, float* output, float* work) const {
#if BRUUN_LEVEL >= 1
        if (f32_wide_ok()) {
            inverse_simd_f32_wide(input, output, work);
            return;
        }
#endif
        inverse_scalar_f32(input, output, work);
    }

private:
    // DIT blocking thresholds are byte-derived so double/f32 paths stay in
    // cache-footprint parity. 32768 bytes was the measured point where a
    // single fused seed block stopped being the best default on the NEON
    // development target; 256 is the cache-blocked stage size in elements.
    static constexpr int kFuseBlockBytes = 32768;
    static constexpr int kBlockedStageSize = 256;
    static constexpr int kRowTileBlockCount = 64;
    static constexpr int kF32MinInverseChain = 4;

    template <typename T>
    static bool should_cache_block(int n) {
        return static_cast<std::size_t>(n) * sizeof(T) >=
               static_cast<std::size_t>(kFuseBlockBytes);
    }

    static bool valid_size(int n) {
        if (!is_power2(n)) {
            return false;
        }
        if (n < 4) {
            return false;
        }
        return true;
    }

    void clear() noexcept {
        n_ = 0;
        half_ = 0;
        twiddle_stage_count_ = 0;
        radix2_terminal_ = false;
        rev_.clear();
        cos_.clear();
        sin_.clear();
        twiddle_.clear();
        twiddle_offset_.clear();
        cs_.clear();
        fterm_cs_.clear();
    }

    // Bruun radix-2 residue reduction: child pair at angle 2*theta splits into
    // parent pairs at theta and pi - theta, c2 = 2*cos(theta).
    static BRUUN_ALWAYS_INLINE void pair_reduce(double ea, double eb, double oa, double ob, double c2,
                                                double& la, double& lb, double& ha, double& hb) {
        const double c = 0.5 * c2;
        const double s2 = std::max(0.0, 1.0 - c * c);
        const double s = std::sqrt(s2);
        const double r = c * oa - s * ob;
        const double i = s * oa + c * ob;
        la = ea + r;
        lb = eb + i;
        ha = ea - r;
        hb = i - eb;
    }

    // Normalized-basis scalar merge taking a precomputed rotation (c, s) from
    // the libm cos_/sin_ tables instead of deriving s = sqrt(1 - c^2) from a
    // 2*cos twiddle. Same result as pair_reduce, without the small-angle sqrt
    // cancellation; used by the SIMD terminals' scalar node specials.
    static BRUUN_ALWAYS_INLINE void pair_reduce_cs(double ea, double eb, double oa, double ob,
                                                   double c, double s,
                                                   double& la, double& lb, double& ha, double& hb) {
        const double r = c * oa - s * ob;
        const double i = s * oa + c * ob;
        la = ea + r;
        lb = eb + i;
        ha = ea - r;
        hb = i - eb;
    }

    // ------------------------------------------------------------------
    // Scalar reference walk (also serves BRUUN_LEVEL == 0 and n == 4).
    // ------------------------------------------------------------------

    void butterfly4_scalar(const double* q0, const double* q1, const double* q2, const double* q3,
                           int i, const double* cs2, const double* cs1, const double* cs1q,
                           double* out) const {
        double h0a, h0b, h0pa, h0pb, h1a, h1b, h1pa, h1pb;
        pair_reduce_cs(q0[2 * i], q0[2 * i + 1], q1[2 * i], q1[2 * i + 1],
                       cs2[0], cs2[1], h0a, h0b, h0pa, h0pb);
        pair_reduce_cs(q2[2 * i], q2[2 * i + 1], q3[2 * i], q3[2 * i + 1],
                       cs2[0], cs2[1], h1a, h1b, h1pa, h1pb);
        pair_reduce_cs(h0a, h0b, h1a, h1b, cs1[0], cs1[1], out[0], out[1], out[2], out[3]);
        pair_reduce_cs(h0pa, h0pb, h1pa, h1pb, cs1q[0], cs1q[1], out[4], out[5], out[6], out[7]);
    }

    void write_butterfly4(double* block, int hs, int q4, int i, const double* out) const {
        block[2 * i] = out[0];
        block[2 * i + 1] = out[1];
        block[2 * (hs - i)] = out[2];
        block[2 * (hs - i) + 1] = out[3];
        block[2 * (q4 - i)] = out[4];
        block[2 * (q4 - i) + 1] = out[5];
        block[2 * (q4 + i)] = out[6];
        block[2 * (q4 + i) + 1] = out[7];
    }

    void merge4_scalar(double* block, int s, const double* cs2, const double* cs1) const {
        const int q4 = s >> 2;
        const int hs = s >> 1;
        double* q0 = block;
        double* q1 = block + q4;
        double* q2 = block + 2 * q4;
        double* q3 = block + 3 * q4;

        const double q0dc = q0[0];
        const double q0ny = q0[1];
        const double q1dc = q1[0];
        const double q1ny = q1[1];
        const double q2dc = q2[0];
        const double q2ny = q2[1];
        const double q3dc = q3[0];
        const double q3ny = q3[1];
        const double h0dc = q0dc + q1dc;
        const double h0ny = q0dc - q1dc;
        const double h1dc = q2dc + q3dc;
        const double h1ny = q2dc - q3dc;
        double b2a0, b2b0, b2a1, b2b1;
        pair_reduce_cs(q0ny, q1ny, q2ny, q3ny,
                       cs1[2 * (q4 / 2 - 1)], cs1[2 * (q4 / 2 - 1) + 1],
                       b2a0, b2b0, b2a1, b2b1);
        block[0] = h0dc + h1dc;
        block[1] = h0dc - h1dc;
        block[2 * q4] = h0ny;
        block[2 * q4 + 1] = h1ny;
        block[2 * (q4 / 2)] = b2a0;
        block[2 * (q4 / 2) + 1] = b2b0;
        block[2 * (hs - q4 / 2)] = b2a1;
        block[2 * (hs - q4 / 2) + 1] = b2b1;

        for (int i = 1; i < q4 / 2 - i; ++i) {
            const int m = q4 / 2 - i;
            double oi[8];
            double om[8];
            butterfly4_scalar(q0, q1, q2, q3, i,
                              cs2 + 2 * (i - 1), cs1 + 2 * (i - 1), cs1 + 2 * (q4 - i - 1), oi);
            butterfly4_scalar(q0, q1, q2, q3, m,
                              cs2 + 2 * (m - 1), cs1 + 2 * (m - 1), cs1 + 2 * (q4 - m - 1), om);
            write_butterfly4(block, hs, q4, i, oi);
            write_butterfly4(block, hs, q4, m, om);
        }

        if (((q4 / 2) & 1) == 0 && q4 / 2 >= 2) {
            const int i = q4 >> 2;
            double oi[8];
            butterfly4_scalar(q0, q1, q2, q3, i,
                              cs2 + 2 * (i - 1), cs1 + 2 * (i - 1), cs1 + 2 * (q4 - i - 1), oi);
            write_butterfly4(block, hs, q4, i, oi);
        }
    }

    BRUUN_ALWAYS_INLINE void seed_merge2_pair(const double* input, double* work, int b) const {
        const int j0 = rev_[static_cast<std::size_t>(b)];
        const int j1 = rev_[static_cast<std::size_t>(b + 1)];
        const double a0 = input[j0];
        const double c0 = input[half_ + j0];
        const double a1 = input[j1];
        const double c1 = input[half_ + j1];
        const double e0 = a0 + c0;
        const double e1 = a1 + c1;
        work[2 * b] = e0 + e1;
        work[2 * b + 1] = e0 - e1;
        work[2 * b + 2] = a0 - c0;
        work[2 * b + 3] = a1 - c1;
    }

    void compute_work_scalar(const double* input, double* work) const {
        BRUUN_ASSERT(n_ >= 4);
        for (int b = 0; b < half_; b += 2) {
            seed_merge2_pair(input, work, b);
        }
        int stage = 2;
        for (int s = 16; s <= n_; s <<= 2, stage += 2) {
            const double* cs2 = cs_.data() + 2 * twiddle_offset_[static_cast<std::size_t>(stage - 1)];
            const double* cs1 = cs_.data() + 2 * twiddle_offset_[static_cast<std::size_t>(stage)];
            for (int off = 0; off < n_; off += s) {
                merge4_scalar(work + off, s, cs2, cs1);
            }
        }
    }

    void forward_split_scalar(const double* input, double* output_re, double* output_im, double* work) const {
        compute_work_scalar(input, work);
        if (radix2_terminal_) {
            terminal_radix2_split_scalar(work, output_re, output_im);
            return;
        }
        output_re[0] = work[0];
        output_im[0] = 0.0;
        output_re[half_] = work[1];
        output_im[half_] = 0.0;
        for (int k = 1; k < half_; ++k) {
            const double a = work[2 * k];
            const double b = work[2 * k + 1];
            output_re[k] = a;
            output_im[k] = -b;
        }
    }

    template <typename Complex>
    void forward_standard_scalar_impl(const double* input, Complex* output, double* work) const {
        compute_work_scalar(input, work);
        if (radix2_terminal_) {
            terminal_radix2_complex_scalar(work, output);
            return;
        }
        output[0].re = work[0];
        output[0].im = 0.0;
        output[half_].re = work[1];
        output[half_].im = 0.0;
        for (int k = 1; k < half_; ++k) {
            const double a = work[2 * k];
            const double b = work[2 * k + 1];
            output[k].re = a;
            output[k].im = -b;
        }
    }

    template <typename Store>
    void terminal_radix2_scalar_impl(const double* work, const Store& store) const {
        const int q = half_ >> 1;
        store(0, work[0] + work[2 * q], 0.0);
        store(half_, work[0] - work[2 * q], 0.0);
        store(q, work[1], -work[2 * q + 1]);
        for (int i = 1; i < q; ++i) {
            const int hi = half_ - i;
            double la, lb, ha, hb;
            const double* cs = cs_.data() + 2 * (
                twiddle_offset_[static_cast<std::size_t>(twiddle_stage_count_ - 1)] + i - 1);
            pair_reduce_cs(work[2 * i], work[2 * i + 1],
                           work[2 * (i + q)], work[2 * (i + q) + 1],
                           cs[0], cs[1], la, lb, ha, hb);
            store(i,
                  la,
                  -lb);
            store(hi,
                  ha,
                  -hb);
        }
    }

    void terminal_radix2_split_scalar(const double* work, double* output_re, double* output_im) const {
        terminal_radix2_scalar_impl(work, [&](int k, double re, double im) {
            output_re[k] = re;
            output_im[k] = im;
        });
    }

    template <typename Complex>
    void terminal_radix2_complex_scalar(const double* work, Complex* output) const {
        terminal_radix2_scalar_impl(work, [&](int k, double re, double im) {
            output[k].re = re;
            output[k].im = im;
        });
    }

    // Exact inverse of pair_reduce. ic = 0.5/c2, w = 1 - c2^2 (precomputed).
    static BRUUN_ALWAYS_INLINE void pair_expand(double la, double lb, double ha, double hb,
                                                double c2, double w,
                                                double& ea, double& eb, double& oa, double& ob) {
        ob = (ha - la) * c2;
        eb = (lb - hb) * c2;
        ea = 0.5 * (la + ha) + eb;
        oa = 0.5 * (lb + hb) + w * ob;
    }

    static BRUUN_ALWAYS_INLINE void pair_expand_norm(double la, double lb, double ha, double hb,
                                                     double c2,
                                                     double& ea, double& eb, double& oa, double& ob) {
        const double c = 0.5 * c2;
        const double s2 = std::max(0.0, 1.0 - c * c);
        const double s = std::sqrt(s2);
        const double r = 0.5 * (la - ha);
        const double i = 0.5 * (lb + hb);
        ea = 0.5 * (la + ha);
        eb = 0.5 * (lb - hb);
        oa = c * r + s * i;
        ob = c * i - s * r;
    }

    static BRUUN_ALWAYS_INLINE void pair_expand_cs(double la, double lb, double ha, double hb,
                                                   double c, double s,
                                                   double& ea, double& eb, double& oa, double& ob) {
        const double r = 0.5 * (la - ha);
        const double i = 0.5 * (lb + hb);
        ea = 0.5 * (la + ha);
        eb = 0.5 * (lb - hb);
        oa = c * r + s * i;
        ob = c * i - s * r;
    }

    // Exact inverse of butterfly4_scalar: parent nodes {i, hs-i, q4-i, q4+i}
    // back to child node i of the four children.
    void inv_butterfly4_scalar(const double* block, int q4, int i,
                               const double* cs2, const double* cs1, double* o) const {
        const int hs = q4 << 1;
        double h0a, h0b, h1a, h1b, h0pa, h0pb, h1pa, h1pb;
        pair_expand_cs(block[2 * i], block[2 * i + 1],
                       block[2 * (hs - i)], block[2 * (hs - i) + 1],
                       cs1[2 * (i - 1)], cs1[2 * (i - 1) + 1], h0a, h0b, h1a, h1b);
        pair_expand_cs(block[2 * (q4 - i)], block[2 * (q4 - i) + 1],
                       block[2 * (q4 + i)], block[2 * (q4 + i) + 1],
                       cs1[2 * (q4 - i - 1)], cs1[2 * (q4 - i - 1) + 1], h0pa, h0pb, h1pa, h1pb);
        pair_expand_cs(h0a, h0b, h0pa, h0pb,
                       cs2[2 * (i - 1)], cs2[2 * (i - 1) + 1], o[0], o[1], o[2], o[3]);
        pair_expand_cs(h1a, h1b, h1pa, h1pb,
                       cs2[2 * (i - 1)], cs2[2 * (i - 1) + 1], o[4], o[5], o[6], o[7]);
    }

    static void write_inv_butterfly4(double* q0, double* q1, double* q2, double* q3,
                                     int i, const double* o) {
        q0[2 * i] = o[0];
        q0[2 * i + 1] = o[1];
        q1[2 * i] = o[2];
        q1[2 * i + 1] = o[3];
        q2[2 * i] = o[4];
        q2[2 * i + 1] = o[5];
        q3[2 * i] = o[6];
        q3[2 * i + 1] = o[7];
    }

    void split4_scalar(double* block, int s, const double* cs2, const double* cs1) const {
        const int q4 = s >> 2;
        const int M = q4 >> 1;
        double* q0 = block;
        double* q1 = block + q4;
        double* q2 = block + 2 * q4;
        double* q3 = block + 3 * q4;

        const double n0a = block[0];
        const double n0b = block[1];
        const double nMa = block[q4];
        const double nMb = block[q4 + 1];
        const double n2a = block[2 * q4];
        const double n2b = block[2 * q4 + 1];
        const double n3a = block[3 * q4];
        const double n3b = block[3 * q4 + 1];
        const double h0dc = 0.5 * (n0a + n0b);
        const double h1dc = 0.5 * (n0a - n0b);
        double ny0, ny1, ny2, ny3;
        pair_expand_cs(nMa, nMb, n3a, n3b,
                       cs1[2 * (M - 1)], cs1[2 * (M - 1) + 1], ny0, ny1, ny2, ny3);
        q0[0] = 0.5 * (h0dc + n2a);
        q0[1] = ny0;
        q1[0] = 0.5 * (h0dc - n2a);
        q1[1] = ny1;
        q2[0] = 0.5 * (h1dc + n2b);
        q2[1] = ny2;
        q3[0] = 0.5 * (h1dc - n2b);
        q3[1] = ny3;

        for (int i = 1; i < M - i; ++i) {
            const int m = M - i;
            double oi[8];
            double om[8];
            inv_butterfly4_scalar(block, q4, i, cs2, cs1, oi);
            inv_butterfly4_scalar(block, q4, m, cs2, cs1, om);
            write_inv_butterfly4(q0, q1, q2, q3, i, oi);
            write_inv_butterfly4(q0, q1, q2, q3, m, om);
        }

        if (M >= 2) {
            const int i = M >> 1;
            double oi[8];
            inv_butterfly4_scalar(block, q4, i, cs2, cs1, oi);
            write_inv_butterfly4(q0, q1, q2, q3, i, oi);
        }
    }

    BRUUN_ALWAYS_INLINE void seed_split2_pair(double* work, double* output, int b) const {
        const int j0 = rev_[static_cast<std::size_t>(b)];
        const int j1 = rev_[static_cast<std::size_t>(b + 1)];
        const double e0 = 0.5 * (work[2 * b] + work[2 * b + 1]);
        const double e1 = 0.5 * (work[2 * b] - work[2 * b + 1]);
        const double d0 = work[2 * b + 2];
        const double d1 = work[2 * b + 3];
        output[j0] = 0.5 * (e0 + d0);
        output[half_ + j0] = 0.5 * (e0 - d0);
        output[j1] = 0.5 * (e1 + d1);
        output[half_ + j1] = 0.5 * (e1 - d1);
    }

    template <typename Complex>
    void inverse_small_scalar_impl(const Complex* input, double* output, double* work) const {
        if (radix2_terminal_) {
            terminal_radix2_inv_scalar(input, work);
            int stage = twiddle_stage_count_ - 2;
            for (int s = n_ >> 1; s >= 16; s >>= 2, stage -= 2) {
                const double* cs2 = cs_.data() + 2 * twiddle_offset_[static_cast<std::size_t>(stage - 1)];
                const double* cs1 = cs_.data() + 2 * twiddle_offset_[static_cast<std::size_t>(stage)];
                for (int off = 0; off < n_; off += s) {
                    split4_scalar(work + off, s, cs2, cs1);
                }
            }
            for (int b = 0; b < half_; b += 2) {
                seed_split2_pair(work, output, b);
            }
            return;
        }
        work[0] = input[0].re;
        work[1] = input[half_].re;
        for (int k = 1; k < half_; ++k) {
            work[2 * k] = input[k].re;
            work[2 * k + 1] = -input[k].im;
        }
        int stage = twiddle_stage_count_ - 1;
        for (int s = n_; s >= 16; s >>= 2, stage -= 2) {
            const double* cs2 = cs_.data() + 2 * twiddle_offset_[static_cast<std::size_t>(stage - 1)];
            const double* cs1 = cs_.data() + 2 * twiddle_offset_[static_cast<std::size_t>(stage)];
            for (int off = 0; off < n_; off += s) {
                split4_scalar(work + off, s, cs2, cs1);
            }
        }
        for (int b = 0; b < half_; b += 2) {
            seed_split2_pair(work, output, b);
        }
    }

    template <typename Complex>
    void terminal_radix2_inv_scalar(const Complex* input, double* work) const {
        const int q = half_ >> 1;
        work[0] = 0.5 * (input[0].re + input[half_].re);
        work[2 * q] = 0.5 * (input[0].re - input[half_].re);
        work[1] = input[q].re;
        work[2 * q + 1] = -input[q].im;
        const double* cs = cs_.data() + 2 * twiddle_offset_[static_cast<std::size_t>(twiddle_stage_count_ - 1)];
        for (int i = 1; i < q; ++i) {
            const int hi = half_ - i;
            const double la = input[i].re;
            const double lb = -input[i].im;
            const double ha = input[hi].re;
            const double hb = -input[hi].im;
            double ea, eb, oa, ob;
            pair_expand_cs(la, lb, ha, hb, cs[2 * (i - 1)], cs[2 * (i - 1) + 1], ea, eb, oa, ob);
            work[2 * i] = ea;
            work[2 * i + 1] = eb;
            work[2 * (i + q)] = oa;
            work[2 * (i + q) + 1] = ob;
        }
    }

    // ==================================================================
    // Templated scalar walk (T = float or double). Genuine T-precision
    // arithmetic on the data; twiddles are read from the double planning
    // tables and cast to T at use (a higher-precision constant is a plain
    // accuracy win for f32). Structurally identical to the double scalar
    // reference above; the double path keeps its own concrete copies so it
    // stays byte-for-byte as tested. Used by the f32 forward/inverse and as
    // the small-n / BRUUN_LEVEL==0 fallback for both types.
    // ==================================================================

    template <typename T>
    static BRUUN_ALWAYS_INLINE void pair_reduce_t(T ea, T eb, T oa, T ob, const double* cs,
                                                  T& la, T& lb, T& ha, T& hb) {
        const T c = static_cast<T>(cs[0]);
        const T s = static_cast<T>(cs[1]);
        const T r = c * oa - s * ob;
        const T i = s * oa + c * ob;
        la = ea + r;
        lb = eb + i;
        ha = ea - r;
        hb = i - eb;
    }

    template <typename T>
    static BRUUN_ALWAYS_INLINE void pair_expand_norm_t(T la, T lb, T ha, T hb, const double* cs,
                                                       T& ea, T& eb, T& oa, T& ob) {
        const T c = static_cast<T>(cs[0]);
        const T s = static_cast<T>(cs[1]);
        const T half = static_cast<T>(0.5);
        const T r = half * (la - ha);
        const T i = half * (lb + hb);
        ea = half * (la + ha);
        eb = half * (lb - hb);
        oa = c * r + s * i;
        ob = c * i - s * r;
    }

    template <typename T>
    void butterfly4_scalar_t(const T* q0, const T* q1, const T* q2, const T* q3,
                             int i, const double* cs2, const double* cs1, const double* cs1q, T* out) const {
        T h0a, h0b, h0pa, h0pb, h1a, h1b, h1pa, h1pb;
        pair_reduce_t<T>(q0[2 * i], q0[2 * i + 1], q1[2 * i], q1[2 * i + 1], cs2, h0a, h0b, h0pa, h0pb);
        pair_reduce_t<T>(q2[2 * i], q2[2 * i + 1], q3[2 * i], q3[2 * i + 1], cs2, h1a, h1b, h1pa, h1pb);
        pair_reduce_t<T>(h0a, h0b, h1a, h1b, cs1, out[0], out[1], out[2], out[3]);
        pair_reduce_t<T>(h0pa, h0pb, h1pa, h1pb, cs1q, out[4], out[5], out[6], out[7]);
    }

    template <typename T>
    static void write_butterfly4_t(T* block, int hs, int q4, int i, const T* out) {
        block[2 * i] = out[0];
        block[2 * i + 1] = out[1];
        block[2 * (hs - i)] = out[2];
        block[2 * (hs - i) + 1] = out[3];
        block[2 * (q4 - i)] = out[4];
        block[2 * (q4 - i) + 1] = out[5];
        block[2 * (q4 + i)] = out[6];
        block[2 * (q4 + i) + 1] = out[7];
    }

    template <typename T>
    void merge4_scalar_t(T* block, int s, const double* cs2, const double* cs1) const {
        const int q4 = s >> 2;
        const int hs = s >> 1;
        T* q0 = block;
        T* q1 = block + q4;
        T* q2 = block + 2 * q4;
        T* q3 = block + 3 * q4;

        const T q0dc = q0[0];
        const T q0ny = q0[1];
        const T q1dc = q1[0];
        const T q1ny = q1[1];
        const T q2dc = q2[0];
        const T q2ny = q2[1];
        const T q3dc = q3[0];
        const T q3ny = q3[1];
        const T h0dc = q0dc + q1dc;
        const T h0ny = q0dc - q1dc;
        const T h1dc = q2dc + q3dc;
        const T h1ny = q2dc - q3dc;
        T b2a0, b2b0, b2a1, b2b1;
        pair_reduce_t<T>(q0ny, q1ny, q2ny, q3ny, cs1 + 2 * (q4 / 2 - 1), b2a0, b2b0, b2a1, b2b1);
        block[0] = h0dc + h1dc;
        block[1] = h0dc - h1dc;
        block[2 * q4] = h0ny;
        block[2 * q4 + 1] = h1ny;
        block[2 * (q4 / 2)] = b2a0;
        block[2 * (q4 / 2) + 1] = b2b0;
        block[2 * (hs - q4 / 2)] = b2a1;
        block[2 * (hs - q4 / 2) + 1] = b2b1;

        for (int i = 1; i < q4 / 2 - i; ++i) {
            const int m = q4 / 2 - i;
            T oi[8];
            T om[8];
            butterfly4_scalar_t<T>(q0, q1, q2, q3, i,
                                   cs2 + 2 * (i - 1), cs1 + 2 * (i - 1), cs1 + 2 * (q4 - i - 1), oi);
            butterfly4_scalar_t<T>(q0, q1, q2, q3, m,
                                   cs2 + 2 * (m - 1), cs1 + 2 * (m - 1), cs1 + 2 * (q4 - m - 1), om);
            write_butterfly4_t<T>(block, hs, q4, i, oi);
            write_butterfly4_t<T>(block, hs, q4, m, om);
        }

        if (((q4 / 2) & 1) == 0 && q4 / 2 >= 2) {
            const int i = q4 >> 2;
            T oi[8];
            butterfly4_scalar_t<T>(q0, q1, q2, q3, i,
                                   cs2 + 2 * (i - 1), cs1 + 2 * (i - 1), cs1 + 2 * (q4 - i - 1), oi);
            write_butterfly4_t<T>(block, hs, q4, i, oi);
        }
    }

    template <typename T>
    BRUUN_ALWAYS_INLINE void seed_merge2_pair_t(const T* input, T* work, int b) const {
        const int j0 = rev_[static_cast<std::size_t>(b)];
        const int j1 = rev_[static_cast<std::size_t>(b + 1)];
        const T a0 = input[j0];
        const T c0 = input[half_ + j0];
        const T a1 = input[j1];
        const T c1 = input[half_ + j1];
        const T e0 = a0 + c0;
        const T e1 = a1 + c1;
        work[2 * b] = e0 + e1;
        work[2 * b + 1] = e0 - e1;
        work[2 * b + 2] = a0 - c0;
        work[2 * b + 3] = a1 - c1;
    }

    template <typename T>
    void compute_work_scalar_t(const T* input, T* work) const {
        BRUUN_ASSERT(n_ >= 4);
        for (int b = 0; b < half_; b += 2) {
            seed_merge2_pair_t<T>(input, work, b);
        }
        int stage = 2;
        for (int s = 16; s <= n_; s <<= 2, stage += 2) {
            const double* cs2 = cs_.data() + 2 * twiddle_offset_[static_cast<std::size_t>(stage - 1)];
            const double* cs1 = cs_.data() + 2 * twiddle_offset_[static_cast<std::size_t>(stage)];
            for (int off = 0; off < n_; off += s) {
                merge4_scalar_t<T>(work + off, s, cs2, cs1);
            }
        }
    }

    template <typename T, typename Complex>
    void forward_standard_scalar_impl_t(const T* input, Complex* output, T* work) const {
        compute_work_scalar_t<T>(input, work);
        if (radix2_terminal_) {
            const int q = half_ >> 1;
            output[0].re = work[0] + work[2 * q];
            output[0].im = static_cast<T>(0);
            output[half_].re = work[0] - work[2 * q];
            output[half_].im = static_cast<T>(0);
            output[q].re = work[1];
            output[q].im = -work[2 * q + 1];
            const double* cs = cs_.data() + 2 * twiddle_offset_[static_cast<std::size_t>(twiddle_stage_count_ - 1)];
            for (int i = 1; i < q; ++i) {
                const int hi = half_ - i;
                T la, lb, ha, hb;
                pair_reduce_t<T>(work[2 * i], work[2 * i + 1],
                                 work[2 * (i + q)], work[2 * (i + q) + 1],
                                 cs + 2 * (i - 1), la, lb, ha, hb);
                output[i].re = la;
                output[i].im = -lb;
                output[hi].re = ha;
                output[hi].im = -hb;
            }
            return;
        }
        output[0].re = work[0];
        output[0].im = static_cast<T>(0);
        output[half_].re = work[1];
        output[half_].im = static_cast<T>(0);
        for (int k = 1; k < half_; ++k) {
            output[k].re = work[2 * k];
            output[k].im = -work[2 * k + 1];
        }
    }

    template <typename T>
    void inv_butterfly4_scalar_t(const T* block, int q4, int i,
                                 const double* cs2, const double* cs1, T* o) const {
        const int hs = q4 << 1;
        T h0a, h0b, h1a, h1b, h0pa, h0pb, h1pa, h1pb;
        pair_expand_norm_t<T>(block[2 * i], block[2 * i + 1],
                              block[2 * (hs - i)], block[2 * (hs - i) + 1],
                              cs1 + 2 * (i - 1), h0a, h0b, h1a, h1b);
        pair_expand_norm_t<T>(block[2 * (q4 - i)], block[2 * (q4 - i) + 1],
                              block[2 * (q4 + i)], block[2 * (q4 + i) + 1],
                              cs1 + 2 * (q4 - i - 1), h0pa, h0pb, h1pa, h1pb);
        pair_expand_norm_t<T>(h0a, h0b, h0pa, h0pb, cs2 + 2 * (i - 1), o[0], o[1], o[2], o[3]);
        pair_expand_norm_t<T>(h1a, h1b, h1pa, h1pb, cs2 + 2 * (i - 1), o[4], o[5], o[6], o[7]);
    }

    template <typename T>
    static void write_inv_butterfly4_t(T* q0, T* q1, T* q2, T* q3, int i, const T* o) {
        q0[2 * i] = o[0];
        q0[2 * i + 1] = o[1];
        q1[2 * i] = o[2];
        q1[2 * i + 1] = o[3];
        q2[2 * i] = o[4];
        q2[2 * i + 1] = o[5];
        q3[2 * i] = o[6];
        q3[2 * i + 1] = o[7];
    }

    template <typename T>
    void split4_scalar_t(T* block, int s, const double* cs2, const double* cs1) const {
        const int q4 = s >> 2;
        const int M = q4 >> 1;
        const T half = static_cast<T>(0.5);
        T* q0 = block;
        T* q1 = block + q4;
        T* q2 = block + 2 * q4;
        T* q3 = block + 3 * q4;

        const T n0a = block[0];
        const T n0b = block[1];
        const T nMa = block[q4];
        const T nMb = block[q4 + 1];
        const T n2a = block[2 * q4];
        const T n2b = block[2 * q4 + 1];
        const T n3a = block[3 * q4];
        const T n3b = block[3 * q4 + 1];
        const T h0dc = half * (n0a + n0b);
        const T h1dc = half * (n0a - n0b);
        T ny0, ny1, ny2, ny3;
        pair_expand_norm_t<T>(nMa, nMb, n3a, n3b, cs1 + 2 * (M - 1), ny0, ny1, ny2, ny3);
        q0[0] = half * (h0dc + n2a);
        q0[1] = ny0;
        q1[0] = half * (h0dc - n2a);
        q1[1] = ny1;
        q2[0] = half * (h1dc + n2b);
        q2[1] = ny2;
        q3[0] = half * (h1dc - n2b);
        q3[1] = ny3;

        for (int i = 1; i < M - i; ++i) {
            const int m = M - i;
            T oi[8];
            T om[8];
            inv_butterfly4_scalar_t<T>(block, q4, i, cs2, cs1, oi);
            inv_butterfly4_scalar_t<T>(block, q4, m, cs2, cs1, om);
            write_inv_butterfly4_t<T>(q0, q1, q2, q3, i, oi);
            write_inv_butterfly4_t<T>(q0, q1, q2, q3, m, om);
        }

        if (M >= 2) {
            const int i = M >> 1;
            T oi[8];
            inv_butterfly4_scalar_t<T>(block, q4, i, cs2, cs1, oi);
            write_inv_butterfly4_t<T>(q0, q1, q2, q3, i, oi);
        }
    }

    template <typename T>
    BRUUN_ALWAYS_INLINE void seed_split2_pair_t(T* work, T* output, int b) const {
        const int j0 = rev_[static_cast<std::size_t>(b)];
        const int j1 = rev_[static_cast<std::size_t>(b + 1)];
        const T half = static_cast<T>(0.5);
        const T e0 = half * (work[2 * b] + work[2 * b + 1]);
        const T e1 = half * (work[2 * b] - work[2 * b + 1]);
        const T d0 = work[2 * b + 2];
        const T d1 = work[2 * b + 3];
        output[j0] = half * (e0 + d0);
        output[half_ + j0] = half * (e0 - d0);
        output[j1] = half * (e1 + d1);
        output[half_ + j1] = half * (e1 - d1);
    }

    template <typename T, typename Complex>
    void inverse_small_scalar_impl_t(const Complex* input, T* output, T* work) const {
        const T half = static_cast<T>(0.5);
        if (radix2_terminal_) {
            const int q = half_ >> 1;
            work[0] = half * (input[0].re + input[half_].re);
            work[2 * q] = half * (input[0].re - input[half_].re);
            work[1] = input[q].re;
            work[2 * q + 1] = -input[q].im;
            const double* cs = cs_.data() + 2 * twiddle_offset_[static_cast<std::size_t>(twiddle_stage_count_ - 1)];
            for (int i = 1; i < q; ++i) {
                const int hi = half_ - i;
                T ea, eb, oa, ob;
                pair_expand_norm_t<T>(input[i].re, -input[i].im, input[hi].re, -input[hi].im,
                                      cs + 2 * (i - 1), ea, eb, oa, ob);
                work[2 * i] = ea;
                work[2 * i + 1] = eb;
                work[2 * (i + q)] = oa;
                work[2 * (i + q) + 1] = ob;
            }
            int stage = twiddle_stage_count_ - 2;
            for (int s = n_ >> 1; s >= 16; s >>= 2, stage -= 2) {
                const double* cs2 = cs_.data() + 2 * twiddle_offset_[static_cast<std::size_t>(stage - 1)];
                const double* cs1 = cs_.data() + 2 * twiddle_offset_[static_cast<std::size_t>(stage)];
                for (int off = 0; off < n_; off += s) {
                    split4_scalar_t<T>(work + off, s, cs2, cs1);
                }
            }
            for (int b = 0; b < half_; b += 2) {
                seed_split2_pair_t<T>(work, output, b);
            }
            return;
        }
        work[0] = input[0].re;
        work[1] = input[half_].re;
        for (int k = 1; k < half_; ++k) {
            work[2 * k] = input[k].re;
            work[2 * k + 1] = -input[k].im;
        }
        int stage = twiddle_stage_count_ - 1;
        for (int s = n_; s >= 16; s >>= 2, stage -= 2) {
            const double* cs2 = cs_.data() + 2 * twiddle_offset_[static_cast<std::size_t>(stage - 1)];
            const double* cs1 = cs_.data() + 2 * twiddle_offset_[static_cast<std::size_t>(stage)];
            for (int off = 0; off < n_; off += s) {
                split4_scalar_t<T>(work + off, s, cs2, cs1);
            }
        }
        for (int b = 0; b < half_; b += 2) {
            seed_split2_pair_t<T>(work, output, b);
        }
    }

#if BRUUN_LEVEL >= 1
    // ------------------------------------------------------------------
    // Mirrored-lane SIMD walk. Lanes of every vector hold node s (lane 0)
    // and node s + N/4 (lane 1); see the header commentary.
    // ------------------------------------------------------------------

    // Normalized-basis merge, SIMD form of the scalar pair_reduce: the odd
    // child (oa, ob) is rotated by (c, s) = (cos, sin) before combining with
    // the even child. Because c^2 + s^2 = 1 exactly (no 2*cos twiddle that
    // grows toward the tree top, no 1 - (2cos)^2 cancellation), residue pairs
    // stay in the local unit frame and the leaf pair is directly (re, -im).
    static BRUUN_ALWAYS_INLINE void pair_reduce_v2_norm(bruun_v2 ea, bruun_v2 eb, bruun_v2 oa, bruun_v2 ob,
                                                        bruun_v2 c, bruun_v2 s,
                                                        bruun_v2& la, bruun_v2& lb, bruun_v2& ha, bruun_v2& hb) {
        const bruun_v2 r = V2_MSUB(V2_MUL(c, oa), s, ob);   // c*oa - s*ob
        const bruun_v2 im = V2_MADD(V2_MUL(s, oa), c, ob);  // s*oa + c*ob
        la = V2_ADD(ea, r);
        lb = V2_ADD(eb, im);
        ha = V2_SUB(ea, r);
        hb = V2_SUB(im, eb);
    }

    static BRUUN_ALWAYS_INLINE double v2_lane0(bruun_v2 v) {
#if defined(BRUUN_NEON_128)
        return vgetq_lane_f64(v, 0);
#elif defined(BRUUN_X86_128)
        return _mm_cvtsd_f64(v);
#else
        double lane[2];
        V2_ST(lane, v);
        return lane[0];
#endif
    }

    static BRUUN_ALWAYS_INLINE double v2_lane1(bruun_v2 v) {
#if defined(BRUUN_NEON_128)
        return vgetq_lane_f64(v, 1);
#elif defined(BRUUN_X86_128)
        return _mm_cvtsd_f64(V2_UNPHI(v, v));
#else
        double lane[2];
        V2_ST(lane, v);
        return lane[1];
#endif
    }

    // Seed: fused half-split + radix-2 merge into standard s=4 blocks, both
    // mirror lanes at once. rev(b + N/4) == rev(b) + 1 makes both input reads
    // contiguous V2 loads; rev(b + 1) == rev(b) + N/8 removes the second
    // table lookup.
    void seed_block_v(const double* input, double* vw, int base, int count) const {
        const int hq = half_ >> 1;
        for (int b = base; b < base + count; b += 2) {
            const int j0 = rev_[static_cast<std::size_t>(b)];
            const int j1 = j0 + hq;
            const bruun_v2 a0 = V2_LD(input + j0);
            const bruun_v2 c0 = V2_LD(input + j0 + half_);
            const bruun_v2 a1 = V2_LD(input + j1);
            const bruun_v2 c1 = V2_LD(input + j1 + half_);
            const bruun_v2 e0 = V2_ADD(a0, c0);
            const bruun_v2 e1 = V2_ADD(a1, c1);
            double* dst = vw + 4 * b;
            V2_ST(dst, V2_ADD(e0, e1));
            V2_ST(dst + 2, V2_SUB(e0, e1));
            V2_ST(dst + 4, V2_SUB(a0, c0));
            V2_ST(dst + 6, V2_SUB(a1, c1));
        }
    }

    static BRUUN_ALWAYS_INLINE void store_butterfly4_v(double* vblk, int M, int i, const bruun_v2* o) {
        V2_ST(vblk + 4 * i, o[0]);
        V2_ST(vblk + 4 * i + 2, o[1]);
        V2_ST(vblk + 4 * (4 * M - i), o[2]);
        V2_ST(vblk + 4 * (4 * M - i) + 2, o[3]);
        V2_ST(vblk + 4 * (2 * M - i), o[4]);
        V2_ST(vblk + 4 * (2 * M - i) + 2, o[5]);
        V2_ST(vblk + 4 * (2 * M + i), o[6]);
        V2_ST(vblk + 4 * (2 * M + i) + 2, o[7]);
    }

    // -------- Normalized-basis SIMD forward (same walk, stable rotation) -----
    // Identical mirrored-lane structure to butterfly4_v / merge4_v / the
    // driver above, but every pair_reduce runs the (c, s) rotation. Twiddle
    // pointers now index the {c, s} table cs_ (2 doubles per entry).

    BRUUN_ALWAYS_INLINE void butterfly4_v_norm(bruun_v2 a0, bruun_v2 b0, bruun_v2 a1, bruun_v2 b1,
                                               bruun_v2 a2, bruun_v2 b2, bruun_v2 a3, bruun_v2 b3,
                                               const double* cs2, const double* cs1, const double* cs1q,
                                               bruun_v2* o) const {
        const bruun_v2 c2 = V2_SET1(cs2[0]);
        const bruun_v2 s2 = V2_SET1(cs2[1]);
        bruun_v2 h0la, h0lb, h0ha, h0hb, h1la, h1lb, h1ha, h1hb;
        pair_reduce_v2_norm(a0, b0, a1, b1, c2, s2, h0la, h0lb, h0ha, h0hb);
        pair_reduce_v2_norm(a2, b2, a3, b3, c2, s2, h1la, h1lb, h1ha, h1hb);
        pair_reduce_v2_norm(h0la, h0lb, h1la, h1lb, V2_SET1(cs1[0]), V2_SET1(cs1[1]),
                            o[0], o[1], o[2], o[3]);
        pair_reduce_v2_norm(h0ha, h0hb, h1ha, h1hb, V2_SET1(cs1q[0]), V2_SET1(cs1q[1]),
                            o[4], o[5], o[6], o[7]);
    }

    void merge4_v_norm(double* vblk, int M, const double* cs2, const double* cs1) const {
        double* c0 = vblk;
        double* c1 = vblk + 4 * M;
        double* c2 = vblk + 8 * M;
        double* c3 = vblk + 12 * M;

        const bruun_v2 dc0 = V2_LD(c0);
        const bruun_v2 ny0 = V2_LD(c0 + 2);
        const bruun_v2 dc1 = V2_LD(c1);
        const bruun_v2 ny1 = V2_LD(c1 + 2);
        const bruun_v2 dc2 = V2_LD(c2);
        const bruun_v2 ny2 = V2_LD(c2 + 2);
        const bruun_v2 dc3 = V2_LD(c3);
        const bruun_v2 ny3 = V2_LD(c3 + 2);
        bruun_v2 ma, mb, mha, mhb;
        pair_reduce_v2_norm(ny0, ny1, ny2, ny3, V2_SET1(cs1[2 * (M - 1)]), V2_SET1(cs1[2 * (M - 1) + 1]),
                            ma, mb, mha, mhb);
        const bruun_v2 h0dc = V2_ADD(dc0, dc1);
        const bruun_v2 h0ny = V2_SUB(dc0, dc1);
        const bruun_v2 h1dc = V2_ADD(dc2, dc3);
        const bruun_v2 h1ny = V2_SUB(dc2, dc3);
        V2_ST(c0, V2_ADD(h0dc, h1dc));
        V2_ST(c0 + 2, V2_SUB(h0dc, h1dc));
        V2_ST(c1, ma);
        V2_ST(c1 + 2, mb);
        V2_ST(c2, h0ny);
        V2_ST(c2 + 2, h1ny);
        V2_ST(c3, mha);
        V2_ST(c3 + 2, mhb);

        for (int i = 1; i < M - i; ++i) {
            const int m = M - i;
            const bruun_v2 a0i = V2_LD(c0 + 4 * i);
            const bruun_v2 b0i = V2_LD(c0 + 4 * i + 2);
            const bruun_v2 a1i = V2_LD(c1 + 4 * i);
            const bruun_v2 b1i = V2_LD(c1 + 4 * i + 2);
            const bruun_v2 a2i = V2_LD(c2 + 4 * i);
            const bruun_v2 b2i = V2_LD(c2 + 4 * i + 2);
            const bruun_v2 a3i = V2_LD(c3 + 4 * i);
            const bruun_v2 b3i = V2_LD(c3 + 4 * i + 2);
            const bruun_v2 a1m = V2_LD(c1 + 4 * m);
            const bruun_v2 b1m = V2_LD(c1 + 4 * m + 2);
            const bruun_v2 a3m = V2_LD(c3 + 4 * m);
            const bruun_v2 b3m = V2_LD(c3 + 4 * m + 2);
            bruun_v2 oi[8];
            butterfly4_v_norm(a0i, b0i, a1i, b1i, a2i, b2i, a3i, b3i,
                              cs2 + 2 * (i - 1), cs1 + 2 * (i - 1), cs1 + 2 * (2 * M - i - 1), oi);
            store_butterfly4_v(vblk, M, i, oi);
            const bruun_v2 a0m = V2_LD(c0 + 4 * m);
            const bruun_v2 b0m = V2_LD(c0 + 4 * m + 2);
            const bruun_v2 a2m = V2_LD(c2 + 4 * m);
            const bruun_v2 b2m = V2_LD(c2 + 4 * m + 2);
            bruun_v2 om[8];
            butterfly4_v_norm(a0m, b0m, a1m, b1m, a2m, b2m, a3m, b3m,
                              cs2 + 2 * (m - 1), cs1 + 2 * (m - 1), cs1 + 2 * (2 * M - m - 1), om);
            store_butterfly4_v(vblk, M, m, om);
        }

        if (M >= 2) {
            const int i = M >> 1;
            const bruun_v2 a0i = V2_LD(c0 + 4 * i);
            const bruun_v2 b0i = V2_LD(c0 + 4 * i + 2);
            const bruun_v2 a1i = V2_LD(c1 + 4 * i);
            const bruun_v2 b1i = V2_LD(c1 + 4 * i + 2);
            const bruun_v2 a2i = V2_LD(c2 + 4 * i);
            const bruun_v2 b2i = V2_LD(c2 + 4 * i + 2);
            const bruun_v2 a3i = V2_LD(c3 + 4 * i);
            const bruun_v2 b3i = V2_LD(c3 + 4 * i + 2);
            bruun_v2 oi[8];
            butterfly4_v_norm(a0i, b0i, a1i, b1i, a2i, b2i, a3i, b3i,
                              cs2 + 2 * (i - 1), cs1 + 2 * (i - 1), cs1 + 2 * (2 * M - i - 1), oi);
            store_butterfly4_v(vblk, M, i, oi);
        }
    }

    void seed_fused_block_v_norm(const double* input, double* vw, int base, int count) const {
        seed_block_v(input, vw, base, count);
        int stage = 2;
        for (int s = 16; (s >> 1) <= count; s <<= 2, stage += 2) {
            const double* cs2 = cs_.data() + 2 * twiddle_offset_[static_cast<std::size_t>(stage - 1)];
            const double* cs1 = cs_.data() + 2 * twiddle_offset_[static_cast<std::size_t>(stage)];
            const int M = s >> 3;
            for (int sub = base; sub < base + count; sub += (s >> 1)) {
                merge4_v_norm(vw + 4 * sub, M, cs2, cs1);
            }
        }
    }

    void compute_vwork_norm(const double* input, double* vw) const {
        BRUUN_ASSERT(n_ >= 16);
        const int q = half_ >> 1;
        const int chain_limit = radix2_terminal_ ? (n_ >> 1) : (n_ >> 2);
        const int s_target = should_cache_block<double>(n_) ? kBlockedStageSize : chain_limit;
        const int block_slots = s_target >> 1;
        const int nb = q / block_slots;
        const int nb_bits = ilog2_pow2(nb);
        for (int p = 0; p < nb; ++p) {
            const int base = bitrev_int(p, nb_bits) * block_slots;
            seed_fused_block_v_norm(input, vw, base, block_slots);
        }

        int s = s_target << 2;
        int stage = ilog2_pow2(s) - 2;
        for (; s <= chain_limit; s <<= 2, stage += 2) {
            const double* cs2 = cs_.data() + 2 * twiddle_offset_[static_cast<std::size_t>(stage - 1)];
            const double* cs1 = cs_.data() + 2 * twiddle_offset_[static_cast<std::size_t>(stage)];
            const int M = s >> 3;
            for (int off = 0; off < n_; off += 2 * s) {
                merge4_v_norm(vw + off, M, cs2, cs1);
            }
        }
    }

    template <typename Complex>
    struct terminal_complex_writer {
        Complex* X;
        BRUUN_ALWAYS_INLINE void store_bin(int k, double re, double im) const {
            X[k].re = re;
            X[k].im = im;
        }
        BRUUN_ALWAYS_INLINE void store_pair(int k0, int k1, bruun_v2 re, bruun_v2 im) const {
            V2_ST(&X[k0].re, V2_UNPLO(re, im));
            V2_ST(&X[k1].re, V2_UNPHI(re, im));
        }
    };

    struct terminal_split_writer {
        double* re_;
        double* im_;
        BRUUN_ALWAYS_INLINE void store_bin(int k, double re, double im) const {
            re_[k] = re;
            im_[k] = im;
        }
        BRUUN_ALWAYS_INLINE void store_pair(int k0, int k1, bruun_v2 re, bruun_v2 im) const {
            re_[k0] = v2_lane0(re);
            im_[k0] = v2_lane0(im);
            re_[k1] = v2_lane1(re);
            im_[k1] = v2_lane1(im);
        }
    };

    // -------- Normalized-basis SIMD terminals --------
    // Same mirrored-lane structure as terminal_v / terminal_radix2_v, but the
    // last merge runs the (c, s) rotation, so each leaf pair is already
    // (re, -im): the residue -> complex projection collapses to a sign flip,
    // and the high child needs no separate per-bin twiddle at all.
    static BRUUN_ALWAYS_INLINE bruun_v2 v2_neg(bruun_v2 x) {
        return V2_SUB(V2_SET1(0.0), x);
    }

    template <typename Writer>
    void terminal_v_norm(const double* vw, const Writer& w) const {
        const int M = half_ >> 2;   // child node count = n/8
        const double* cA = vw;
        const double* cB = vw + 4 * M;

        const bruun_v2 dcA = V2_LD(cA);
        const bruun_v2 nyA = V2_LD(cA + 2);
        const bruun_v2 dcB = V2_LD(cB);
        const bruun_v2 nyB = V2_LD(cB + 2);
        const bruun_v2 sum = V2_ADD(dcA, dcB);   // {h0dc, h1dc}
        const bruun_v2 dif = V2_SUB(dcA, dcB);   // {h0ny, h1ny}
        w.store_bin(0, v2_lane0(sum) + v2_lane1(sum), 0.0);
        w.store_bin(half_, v2_lane0(sum) - v2_lane1(sum), 0.0);
        w.store_bin(2 * M, v2_lane0(dif), -v2_lane1(dif));   // bin n/4
        {
            // node M: angle 2*pi*M/n = pi/4, rotation (cos_[M], sin_[M]).
            double la, lb, ha, hb;
            pair_reduce_cs(v2_lane0(nyA), v2_lane0(nyB), v2_lane1(nyA), v2_lane1(nyB),
                           cos_[static_cast<std::size_t>(M)], sin_[static_cast<std::size_t>(M)],
                           la, lb, ha, hb);
            w.store_bin(M, la, -lb);
            w.store_bin(half_ - M, ha, -hb);
        }

        const double* rec = fterm_cs_.data();
        for (int i = 1; i < M; ++i, rec += 6) {
            const bruun_v2 aA = V2_LD(cA + 4 * i);
            const bruun_v2 bA = V2_LD(cA + 4 * i + 2);
            const bruun_v2 aB = V2_LD(cB + 4 * i);
            const bruun_v2 bB = V2_LD(cB + 4 * i + 2);
            bruun_v2 la, lb, ha, hb;
            pair_reduce_v2_norm(aA, bA, aB, bB, V2_SET1(rec[0]), V2_SET1(rec[1]), la, lb, ha, hb);
            const bruun_v2 ua = V2_UNPLO(la, ha);   // {h0 low, h0 high}
            const bruun_v2 ub = V2_UNPLO(lb, hb);
            const bruun_v2 va = V2_UNPHI(la, ha);   // {h1 low, h1 high}
            const bruun_v2 vb = V2_UNPHI(lb, hb);
            bruun_v2 fa, fb, ga, gb;
            pair_reduce_v2_norm(ua, ub, va, vb, V2_LD(rec + 2), V2_LD(rec + 4), fa, fb, ga, gb);
            // fa/fb lanes = bins {i, n/4 - i}; ga/gb lanes = bins {n/2 - i, n/4 + i}.
            w.store_pair(i, 2 * M - i, fa, v2_neg(fb));
            w.store_pair(half_ - i, 2 * M + i, ga, v2_neg(gb));
        }
    }

    template <typename Writer>
    void terminal_radix2_v_norm(const double* vw, const Writer& w) const {
        const int q = half_ >> 1;

        const bruun_v2 dc = V2_LD(vw);
        const bruun_v2 ny = V2_LD(vw + 2);
        w.store_bin(0, v2_lane0(dc) + v2_lane1(dc), 0.0);
        w.store_bin(half_, v2_lane0(dc) - v2_lane1(dc), 0.0);
        w.store_bin(q, v2_lane0(ny), -v2_lane1(ny));

        int i = 1;
        for (; i + 1 < q; i += 2) {
            const bruun_v2 a0 = V2_LD(vw + 4 * i);
            const bruun_v2 b0 = V2_LD(vw + 4 * i + 2);
            const bruun_v2 a1 = V2_LD(vw + 4 * (i + 1));
            const bruun_v2 b1 = V2_LD(vw + 4 * (i + 1) + 2);
            const bruun_v2 ea = V2_UNPLO(a0, a1);
            const bruun_v2 oa = V2_UNPHI(a0, a1);
            const bruun_v2 eb = V2_UNPLO(b0, b1);
            const bruun_v2 ob = V2_UNPHI(b0, b1);
            // c/s for the two lane nodes {i, i+1} come from the per-bin
            // (cos_[k], sin_[k]) tables; the rotation emits both the
            // low child (bins i, i+1) and its mirror high child (bins n/2-i,
            // n/2-i-1) already as (re, -im).
            const bruun_v2 cv = V2_LD(cos_.data() + i);
            const bruun_v2 sv = V2_LD(sin_.data() + i);
            bruun_v2 la, lb, ha, hb;
            pair_reduce_v2_norm(ea, eb, oa, ob, cv, sv, la, lb, ha, hb);
            w.store_pair(i, i + 1, la, v2_neg(lb));
            w.store_pair(half_ - i, half_ - i - 1, ha, v2_neg(hb));
        }

        if (i < q) {
            const bruun_v2 a = V2_LD(vw + 4 * i);
            const bruun_v2 b = V2_LD(vw + 4 * i + 2);
            double la, lb, ha, hb;
            pair_reduce_cs(v2_lane0(a), v2_lane0(b), v2_lane1(a), v2_lane1(b),
                           cos_[static_cast<std::size_t>(i)], sin_[static_cast<std::size_t>(i)],
                           la, lb, ha, hb);
            const int hi = half_ - i;
            w.store_bin(i, la, -lb);
            w.store_bin(hi, ha, -hb);
        }
    }

#if BRUUN_LEVEL >= 1
    // ==================================================================
    // Wide f32 SIMD (4-lane bruun_v4f) -- block-paired mirrored walk.
    // Portable across the V4F backends (NEON / SSE), same guard as the rest
    // of the level-1 SIMD.
    //
    // Second lane axis (beyond the mirror pair {node b, b+N/4}) is the
    // cA/cB split the radix-4 terminal already combines: paired-slot s bundles
    // double-slot s (in cA=[0,N/8)) and double-slot s+N/8 (in cB=[N/8,N/4)).
    // The 4 v4f lanes are {cA.lo, cA.hi, cB.lo, cB.hi} = nodes
    // {s, s+N/4, s+N/8, s+3N/8}. cA and cB run identical twiddles through
    // every interior stage, so merge4_v4f / split4_v4f are byte-for-byte V4F
    // transcriptions of merge4_v_norm / split4_v_norm; only the seed
    // (contiguous paired V4F loads) and the terminals (split the v4f into
    // cA=low2 / cB=high2) are cA/cB-aware.
    //
    // Odd log2(n) runs the paired interior through N/4, then depairs the
    // final cross-lane merge4 plus radix-2 terminal into the V2-like f32
    // layout.
    // ==================================================================

    static BRUUN_ALWAYS_INLINE void pair_reduce_v4f_norm(bruun_v4f ea, bruun_v4f eb, bruun_v4f oa, bruun_v4f ob,
                                                         bruun_v4f c, bruun_v4f s,
                                                         bruun_v4f& la, bruun_v4f& lb, bruun_v4f& ha, bruun_v4f& hb) {
        const bruun_v4f r = V4F_MSUB(V4F_MUL(c, oa), s, ob);   // c*oa - s*ob
        const bruun_v4f im = V4F_MADD(V4F_MUL(s, oa), c, ob);  // s*oa + c*ob
        la = V4F_ADD(ea, r);
        lb = V4F_ADD(eb, im);
        ha = V4F_SUB(ea, r);
        hb = V4F_SUB(im, eb);
    }

    static BRUUN_ALWAYS_INLINE bruun_v4f v4f_set1cs(const double* p) { return V4F_SET1(static_cast<float>(*p)); }

    static BRUUN_ALWAYS_INLINE void pair_reduce_cs_f32(float ea, float eb, float oa, float ob, float c, float s,
                                                       float& la, float& lb, float& ha, float& hb) {
        const float r = c * oa - s * ob;
        const float im = s * oa + c * ob;
        la = ea + r;
        lb = eb + im;
        ha = ea - r;
        hb = im - eb;
    }

    static BRUUN_ALWAYS_INLINE void pair_expand_cs_f32(float la, float lb, float ha, float hb, float c, float s,
                                                       float& ea, float& eb, float& oa, float& ob) {
        const float r = 0.5f * (la - ha);
        const float im = 0.5f * (lb + hb);
        ea = 0.5f * (la + ha);
        eb = 0.5f * (lb - hb);
        oa = c * r + s * im;
        ob = c * im - s * r;
    }

    BRUUN_ALWAYS_INLINE void butterfly4_v4f(bruun_v4f a0, bruun_v4f b0, bruun_v4f a1, bruun_v4f b1,
                                            bruun_v4f a2, bruun_v4f b2, bruun_v4f a3, bruun_v4f b3,
                                            const double* cs2, const double* cs1, const double* cs1q,
                                            bruun_v4f* o) const {
        const bruun_v4f c2 = v4f_set1cs(cs2);
        const bruun_v4f s2 = v4f_set1cs(cs2 + 1);
        bruun_v4f h0la, h0lb, h0ha, h0hb, h1la, h1lb, h1ha, h1hb;
        pair_reduce_v4f_norm(a0, b0, a1, b1, c2, s2, h0la, h0lb, h0ha, h0hb);
        pair_reduce_v4f_norm(a2, b2, a3, b3, c2, s2, h1la, h1lb, h1ha, h1hb);
        pair_reduce_v4f_norm(h0la, h0lb, h1la, h1lb, v4f_set1cs(cs1), v4f_set1cs(cs1 + 1),
                             o[0], o[1], o[2], o[3]);
        pair_reduce_v4f_norm(h0ha, h0hb, h1ha, h1hb, v4f_set1cs(cs1q), v4f_set1cs(cs1q + 1),
                             o[4], o[5], o[6], o[7]);
    }

    static BRUUN_ALWAYS_INLINE void store_butterfly4_v4f(float* vblk, int M, int i, const bruun_v4f* o) {
        V4F_ST(vblk + 8 * i, o[0]);
        V4F_ST(vblk + 8 * i + 4, o[1]);
        V4F_ST(vblk + 8 * (4 * M - i), o[2]);
        V4F_ST(vblk + 8 * (4 * M - i) + 4, o[3]);
        V4F_ST(vblk + 8 * (2 * M - i), o[4]);
        V4F_ST(vblk + 8 * (2 * M - i) + 4, o[5]);
        V4F_ST(vblk + 8 * (2 * M + i), o[6]);
        V4F_ST(vblk + 8 * (2 * M + i) + 4, o[7]);
    }

    void merge4_v4f(float* vblk, int M, const double* cs2, const double* cs1) const {
        float* c0 = vblk;
        float* c1 = vblk + 8 * M;
        float* c2 = vblk + 16 * M;
        float* c3 = vblk + 24 * M;

        const bruun_v4f dc0 = V4F_LD(c0);
        const bruun_v4f ny0 = V4F_LD(c0 + 4);
        const bruun_v4f dc1 = V4F_LD(c1);
        const bruun_v4f ny1 = V4F_LD(c1 + 4);
        const bruun_v4f dc2 = V4F_LD(c2);
        const bruun_v4f ny2 = V4F_LD(c2 + 4);
        const bruun_v4f dc3 = V4F_LD(c3);
        const bruun_v4f ny3 = V4F_LD(c3 + 4);
        bruun_v4f ma, mb, mha, mhb;
        pair_reduce_v4f_norm(ny0, ny1, ny2, ny3, v4f_set1cs(cs1 + 2 * (M - 1)), v4f_set1cs(cs1 + 2 * (M - 1) + 1),
                             ma, mb, mha, mhb);
        const bruun_v4f h0dc = V4F_ADD(dc0, dc1);
        const bruun_v4f h0ny = V4F_SUB(dc0, dc1);
        const bruun_v4f h1dc = V4F_ADD(dc2, dc3);
        const bruun_v4f h1ny = V4F_SUB(dc2, dc3);
        V4F_ST(c0, V4F_ADD(h0dc, h1dc));
        V4F_ST(c0 + 4, V4F_SUB(h0dc, h1dc));
        V4F_ST(c1, ma);
        V4F_ST(c1 + 4, mb);
        V4F_ST(c2, h0ny);
        V4F_ST(c2 + 4, h1ny);
        V4F_ST(c3, mha);
        V4F_ST(c3 + 4, mhb);

        for (int i = 1; i < M - i; ++i) {
            const int m = M - i;
            const bruun_v4f a0i = V4F_LD(c0 + 8 * i);
            const bruun_v4f b0i = V4F_LD(c0 + 8 * i + 4);
            const bruun_v4f a1i = V4F_LD(c1 + 8 * i);
            const bruun_v4f b1i = V4F_LD(c1 + 8 * i + 4);
            const bruun_v4f a2i = V4F_LD(c2 + 8 * i);
            const bruun_v4f b2i = V4F_LD(c2 + 8 * i + 4);
            const bruun_v4f a3i = V4F_LD(c3 + 8 * i);
            const bruun_v4f b3i = V4F_LD(c3 + 8 * i + 4);
            const bruun_v4f a1m = V4F_LD(c1 + 8 * m);
            const bruun_v4f b1m = V4F_LD(c1 + 8 * m + 4);
            const bruun_v4f a3m = V4F_LD(c3 + 8 * m);
            const bruun_v4f b3m = V4F_LD(c3 + 8 * m + 4);
            bruun_v4f oi[8];
            butterfly4_v4f(a0i, b0i, a1i, b1i, a2i, b2i, a3i, b3i,
                           cs2 + 2 * (i - 1), cs1 + 2 * (i - 1), cs1 + 2 * (2 * M - i - 1), oi);
            store_butterfly4_v4f(vblk, M, i, oi);
            const bruun_v4f a0m = V4F_LD(c0 + 8 * m);
            const bruun_v4f b0m = V4F_LD(c0 + 8 * m + 4);
            const bruun_v4f a2m = V4F_LD(c2 + 8 * m);
            const bruun_v4f b2m = V4F_LD(c2 + 8 * m + 4);
            bruun_v4f om[8];
            butterfly4_v4f(a0m, b0m, a1m, b1m, a2m, b2m, a3m, b3m,
                           cs2 + 2 * (m - 1), cs1 + 2 * (m - 1), cs1 + 2 * (2 * M - m - 1), om);
            store_butterfly4_v4f(vblk, M, m, om);
        }

        if (M >= 2) {
            const int i = M >> 1;
            const bruun_v4f a0i = V4F_LD(c0 + 8 * i);
            const bruun_v4f b0i = V4F_LD(c0 + 8 * i + 4);
            const bruun_v4f a1i = V4F_LD(c1 + 8 * i);
            const bruun_v4f b1i = V4F_LD(c1 + 8 * i + 4);
            const bruun_v4f a2i = V4F_LD(c2 + 8 * i);
            const bruun_v4f b2i = V4F_LD(c2 + 8 * i + 4);
            const bruun_v4f a3i = V4F_LD(c3 + 8 * i);
            const bruun_v4f b3i = V4F_LD(c3 + 8 * i + 4);
            bruun_v4f oi[8];
            butterfly4_v4f(a0i, b0i, a1i, b1i, a2i, b2i, a3i, b3i,
                           cs2 + 2 * (i - 1), cs1 + 2 * (i - 1), cs1 + 2 * (2 * M - i - 1), oi);
            store_butterfly4_v4f(vblk, M, i, oi);
        }
    }

    // Seed a pair of paired-slots (s, s+1): gather the cA mirror pairs (slots
    // s, s+1) into v4f lanes {0,1} and the cB mirror pairs (slots s+eighth,
    // s+eighth+1) into lanes {2,3}, so the seed radix-2 runs on all four lanes
    // at once with contiguous loads.
    BRUUN_ALWAYS_INLINE void seed_paired_pair_f32(const float* input, float* fvw, int s, int eighth) const {
        const int hq = half_ >> 1;
        const int jA = rev_[static_cast<std::size_t>(s)];
        const int jB = rev_[static_cast<std::size_t>(s + eighth)];
        BRUUN_ASSERT(jB == jA + 2);
        (void)jB;
        const bruun_v4f a0 = V4F_LD(input + jA);
        const bruun_v4f c0 = V4F_LD(input + jA + half_);
        const bruun_v4f a1 = V4F_LD(input + jA + hq);
        const bruun_v4f c1 = V4F_LD(input + jA + hq + half_);
        const bruun_v4f e0 = V4F_ADD(a0, c0);
        const bruun_v4f e1 = V4F_ADD(a1, c1);
        V4F_ST(fvw + 8 * s, V4F_ADD(e0, e1));            // paired-slot s   a-comp
        V4F_ST(fvw + 8 * s + 4, V4F_SUB(e0, e1));        // paired-slot s   b-comp
        V4F_ST(fvw + 8 * (s + 1), V4F_SUB(a0, c0));      // paired-slot s+1 a-comp
        V4F_ST(fvw + 8 * (s + 1) + 4, V4F_SUB(a1, c1));  // paired-slot s+1 b-comp
    }

    void seed_fused_block_v4f(const float* input, float* fvw, int base, int count) const {
        const int eighth = half_ >> 2;   // P = number of paired-slots = N/8
        for (int s = base; s < base + count; s += 2) {
            seed_paired_pair_f32(input, fvw, s, eighth);
        }
        int stage = 2;
        for (int s = 16; (s >> 1) <= count; s <<= 2, stage += 2) {
            const double* cs2 = cs_.data() + 2 * twiddle_offset_[static_cast<std::size_t>(stage - 1)];
            const double* cs1 = cs_.data() + 2 * twiddle_offset_[static_cast<std::size_t>(stage)];
            const int M = s >> 3;
            for (int poff = base; poff < base + count; poff += (s >> 1)) {
                merge4_v4f(fvw + 8 * poff, M, cs2, cs1);
            }
        }
    }

    void compute_vwork_v4f(const float* input, float* fvw) const {
        const int eighth = half_ >> 2;   // P = number of paired-slots = N/8
        const int chain_limit = n_ >> 2;   // odd sizes finish the N/2 top merge separately
        const int s_target = should_cache_block<float>(n_) ? kBlockedStageSize : chain_limit;
        const int block_slots = s_target >> 1;
        const int nb = eighth / block_slots;
        const int nb_bits = ilog2_pow2(nb);
        for (int p = 0; p < nb; ++p) {
            const int base = bitrev_int(p, nb_bits) * block_slots;
            seed_fused_block_v4f(input, fvw, base, block_slots);
        }

        int s = s_target << 2;
        int stage = ilog2_pow2(s) - 2;
        for (; s <= chain_limit; s <<= 2, stage += 2) {
            const double* cs2 = cs_.data() + 2 * twiddle_offset_[static_cast<std::size_t>(stage - 1)];
            const double* cs1 = cs_.data() + 2 * twiddle_offset_[static_cast<std::size_t>(stage)];
            const int M = s >> 3;
            for (int poff = 0; poff < eighth; poff += (s >> 1)) {
                merge4_v4f(fvw + 8 * poff, M, cs2, cs1);
            }
        }
    }

    BRUUN_ALWAYS_INLINE void butterfly4_v_norm_f32_v4_inputs(bruun_v4f n0, bruun_v4f n1,
                                                             bruun_v4f n2, bruun_v4f n3,
                                                             const double* cs2,
                                                             const double* cs1, const double* cs1q,
                                                             bruun_v4f* out) const {
        const bruun_v4f a02 = V4F_CATLO(n0, n2);
        const bruun_v4f b02 = V4F_CATHI(n0, n2);
        const bruun_v4f a13 = V4F_CATLO(n1, n3);
        const bruun_v4f b13 = V4F_CATHI(n1, n3);

        bruun_v4f hla, hlb, hha, hhb;
        pair_reduce_v4f_norm(a02, b02, a13, b13,
                             v4f_set1cs(cs2), v4f_set1cs(cs2 + 1),
                             hla, hlb, hha, hhb);

        bruun_v4f la, lb, ha, hb;
        pair_reduce_v4f_norm(hla, hlb, V4F_CATHI(hla, hla), V4F_CATHI(hlb, hlb),
                             v4f_set1cs(cs1), v4f_set1cs(cs1 + 1), la, lb, ha, hb);
        out[0] = V4F_CATLO(la, lb);
        out[1] = V4F_CATLO(ha, hb);

        pair_reduce_v4f_norm(hha, hhb, V4F_CATHI(hha, hha), V4F_CATHI(hhb, hhb),
                             v4f_set1cs(cs1q), v4f_set1cs(cs1q + 1), la, lb, ha, hb);
        out[2] = V4F_CATLO(la, lb);
        out[3] = V4F_CATLO(ha, hb);
    }

    static BRUUN_ALWAYS_INLINE void store_butterfly4_v_norm_f32_v4(float* vblk, int M, int i,
                                                                   const bruun_v4f* out) {
        V4F_ST(vblk + 4 * i, out[0]);
        V4F_ST(vblk + 4 * (4 * M - i), out[1]);
        V4F_ST(vblk + 4 * (2 * M - i), out[2]);
        V4F_ST(vblk + 4 * (2 * M + i), out[3]);
    }

    // Fused V4F->V2 depair + the extra radix-2-level merge for the odd forward:
    // reads the paired-V4F chain output fvw, produces the merge4 inputs in
    // registers (paired-slot i depairs to V2-slots {i, 2M+i}, paired-slot M+i
    // to {M+i, 3M+i} -- the four children of merge butterfly i), runs the merge,
    // and writes the V2 layout to vw. Collapses depair_v4f_to_v2_f32 +
    // merge4_v_norm_f32 into one pass. Source (fvw) and dest (vw) differ, so the
    // in-place palindrome merge4 needs is unnecessary -- a linear i sweep reads
    // every paired-slot once and writes every V2-slot once.
    void depair_merge4_v_norm_f32(const float* fvw, float* vw, int M,
                                  const double* cs2, const double* cs1) const {
        float* c0 = vw;
        float* c1 = vw + 4 * M;
        float* c2 = vw + 8 * M;
        float* c3 = vw + 12 * M;

        // i = 0 special (merge4_v_norm_f32's DC/Nyquist block).
        {
            const bruun_v4f av0 = V4F_LD(fvw);
            const bruun_v4f bv0 = V4F_LD(fvw + 4);
            const bruun_v4f avM = V4F_LD(fvw + 8 * M);
            const bruun_v4f bvM = V4F_LD(fvw + 8 * M + 4);
            float s0[4], sM[4], s2[4], s3[4];
            V4F_ST(s0, V4F_CATLO(av0, bv0));   // V2-slot 0
            V4F_ST(s2, V4F_CATHI(av0, bv0));   // V2-slot 2M
            V4F_ST(sM, V4F_CATLO(avM, bvM));   // V2-slot M
            V4F_ST(s3, V4F_CATHI(avM, bvM));   // V2-slot 3M
            float ma0, mb0, mha0, mhb0, ma1, mb1, mha1, mhb1;
            pair_reduce_cs_f32(s0[2], sM[2], s2[2], s3[2],
                               static_cast<float>(cs1[2 * (M - 1)]),
                               static_cast<float>(cs1[2 * (M - 1) + 1]),
                               ma0, mb0, mha0, mhb0);
            pair_reduce_cs_f32(s0[3], sM[3], s2[3], s3[3],
                               static_cast<float>(cs1[2 * (M - 1)]),
                               static_cast<float>(cs1[2 * (M - 1) + 1]),
                               ma1, mb1, mha1, mhb1);
            const float h0dc0 = s0[0] + sM[0];
            const float h0dc1 = s0[1] + sM[1];
            const float h0ny0 = s0[0] - sM[0];
            const float h0ny1 = s0[1] - sM[1];
            const float h1dc0 = s2[0] + s3[0];
            const float h1dc1 = s2[1] + s3[1];
            const float h1ny0 = s2[0] - s3[0];
            const float h1ny1 = s2[1] - s3[1];
            c0[0] = h0dc0 + h1dc0; c0[1] = h0dc1 + h1dc1; c0[2] = h0dc0 - h1dc0; c0[3] = h0dc1 - h1dc1;
            c1[0] = ma0; c1[1] = ma1; c1[2] = mb0; c1[3] = mb1;
            c2[0] = h0ny0; c2[1] = h0ny1; c2[2] = h1ny0; c2[3] = h1ny1;
            c3[0] = mha0; c3[1] = mha1; c3[2] = mhb0; c3[3] = mhb1;
        }

        for (int i = 1; i < M; ++i) {
            const bruun_v4f avi = V4F_LD(fvw + 8 * i);
            const bruun_v4f bvi = V4F_LD(fvw + 8 * i + 4);
            const bruun_v4f avM = V4F_LD(fvw + 8 * (M + i));
            const bruun_v4f bvM = V4F_LD(fvw + 8 * (M + i) + 4);
            bruun_v4f out[4];
            butterfly4_v_norm_f32_v4_inputs(V4F_CATLO(avi, bvi),   // c0 = slot i
                                            V4F_CATLO(avM, bvM),   // c1 = slot M+i
                                            V4F_CATHI(avi, bvi),   // c2 = slot 2M+i
                                            V4F_CATHI(avM, bvM),   // c3 = slot 3M+i
                                            cs2 + 2 * (i - 1), cs1 + 2 * (i - 1),
                                            cs1 + 2 * (2 * M - i - 1), out);
            store_butterfly4_v_norm_f32_v4(vw, M, i, out);
        }
    }

    void terminal_radix2_v_norm_f32(const float* vw, complex_f32_t* X) const {
        const int q = half_ >> 1;
        X[0].re = vw[0] + vw[1];
        X[0].im = 0.0f;
        X[half_].re = vw[0] - vw[1];
        X[half_].im = 0.0f;
        X[q].re = vw[2];
        X[q].im = -vw[3];

        const bruun_v4f neg = V4F_SET1(-1.0f);
        int i = 1;
        for (; i + 1 < q; i += 2) {
            const bruun_v4f n0 = V4F_LD(vw + 4 * i);
            const bruun_v4f n1 = V4F_LD(vw + 4 * (i + 1));
            const bruun_v4f ea_oa = V4F_ZIPLO(n0, n1);
            const bruun_v4f eb_ob = V4F_ZIPHI(n0, n1);
            const bruun_v4f ea = V4F_CATLO(ea_oa, ea_oa);
            const bruun_v4f oa = V4F_CATHI(ea_oa, ea_oa);
            const bruun_v4f eb = V4F_CATLO(eb_ob, eb_ob);
            const bruun_v4f ob = V4F_CATHI(eb_ob, eb_ob);
            const bruun_v4f cv = V4F_SET4(static_cast<float>(cos_[static_cast<std::size_t>(i)]),
                                          static_cast<float>(cos_[static_cast<std::size_t>(i + 1)]),
                                          static_cast<float>(cos_[static_cast<std::size_t>(i)]),
                                          static_cast<float>(cos_[static_cast<std::size_t>(i + 1)]));
            const bruun_v4f sv = V4F_SET4(static_cast<float>(sin_[static_cast<std::size_t>(i)]),
                                          static_cast<float>(sin_[static_cast<std::size_t>(i + 1)]),
                                          static_cast<float>(sin_[static_cast<std::size_t>(i)]),
                                          static_cast<float>(sin_[static_cast<std::size_t>(i + 1)]));
            bruun_v4f la, lb, ha, hb;
            pair_reduce_v4f_norm(ea, eb, oa, ob, cv, sv, la, lb, ha, hb);
            V4F_ST(&X[i].re, V4F_ZIPLO(la, V4F_MUL(lb, neg)));

            const int hi = half_ - i;
            const bruun_v4f high_pair = V4F_ZIPLO(ha, V4F_MUL(hb, neg));
            V4F_ST(&X[hi - 1].re, V4F_CATLO(V4F_CATHI(high_pair, high_pair), high_pair));
        }

        for (; i < q; ++i) {
            const int hi = half_ - i;
            float la, lb, ha, hb;
            pair_reduce_cs_f32(vw[4 * i], vw[4 * i + 2],
                               vw[4 * i + 1], vw[4 * i + 3],
                               static_cast<float>(cos_[static_cast<std::size_t>(i)]),
                               static_cast<float>(sin_[static_cast<std::size_t>(i)]),
                               la, lb, ha, hb);
            X[i].re = la;
            X[i].im = -lb;
            X[hi].re = ha;
            X[hi].im = -hb;
        }
    }

    // Terminal for the wide path: read each paired-slot, split the v4f into
    // Terminal: the single cA/cB lane crossing, in 128-bit V4F. Each paired
    // slot's a-comp v4f is {aA.lo, aA.hi, aB.lo, aB.hi} (mirror pair per half);
    // the radix-4 terminal combines cA with cB. CATHI aligns aB into the low
    // lanes for level A, ZIPLO does the mirror crossing (UNPLO/UNPHI), CATHI
    // aligns va for level B -- the same shuffle abstraction the DIF uses.
    void terminal_v4f_f32(const float* fvw, complex_f32_t* X) const {
        const int M = half_ >> 2;   // = eighth = P
        const double* cs1 = cs_.data() + 2 * twiddle_offset_[static_cast<std::size_t>(twiddle_stage_count_ - 1)];

        // paired-slot 0 (DC/Nyquist/midpoint specials): a-comp {cA.lo,cA.hi,
        // cB.lo,cB.hi} at fvw[0..3], b-comp at fvw[4..7].
        const float sum0 = fvw[0] + fvw[2];
        const float sum1 = fvw[1] + fvw[3];
        X[0].re = sum0 + sum1;      X[0].im = 0.0f;
        X[half_].re = sum0 - sum1;  X[half_].im = 0.0f;
        X[2 * M].re = fvw[0] - fvw[2];  X[2 * M].im = -(fvw[1] - fvw[3]);
        {
            float la, lb, ha, hb;
            pair_reduce_t<float>(fvw[4], fvw[6], fvw[5], fvw[7], cs1 + 2 * (M - 1), la, lb, ha, hb);
            X[M].re = la;         X[M].im = -lb;
            X[half_ - M].re = ha; X[half_ - M].im = -hb;
        }

        const double* rec = fterm_cs_.data();
        for (int i = 1; i < M; ++i, rec += 6) {
            const bruun_v4f av = V4F_LD(fvw + 8 * i);
            const bruun_v4f bv = V4F_LD(fvw + 8 * i + 4);
            // level A: ea=aA (av lanes 0,1), oa=aB (CATHI -> lanes 0,1)
            bruun_v4f la, lb, ha, hb;
            pair_reduce_v4f_norm(av, bv, V4F_CATHI(av, av), V4F_CATHI(bv, bv),
                                 v4f_set1cs(rec), v4f_set1cs(rec + 1), la, lb, ha, hb);
            // mirror crossing: ZIPLO(la,ha) = {ua | va} in lanes {0,1}|{2,3}
            const bruun_v4f uva = V4F_ZIPLO(la, ha);
            const bruun_v4f uvb = V4F_ZIPLO(lb, hb);
            // level B: ea=ua (lanes 0,1), oa=va (CATHI), per-lane c/s {rec2,rec3}
            const bruun_v4f cB = V4F_SET4(static_cast<float>(rec[2]), static_cast<float>(rec[3]),
                                          static_cast<float>(rec[2]), static_cast<float>(rec[3]));
            const bruun_v4f sB = V4F_SET4(static_cast<float>(rec[4]), static_cast<float>(rec[5]),
                                          static_cast<float>(rec[4]), static_cast<float>(rec[5]));
            bruun_v4f fa, fb, ga, gb;
            pair_reduce_v4f_norm(uva, uvb, V4F_CATHI(uva, uva), V4F_CATHI(uvb, uvb), cB, sB,
                                 fa, fb, ga, gb);
            float tfa[4], tfb[4], tga[4], tgb[4];
            V4F_ST(tfa, fa); V4F_ST(tfb, fb); V4F_ST(tga, ga); V4F_ST(tgb, gb);
            X[i].re = tfa[0];         X[i].im = -tfb[0];
            X[2 * M - i].re = tfa[1]; X[2 * M - i].im = -tfb[1];
            X[half_ - i].re = tga[0]; X[half_ - i].im = -tgb[0];
            X[2 * M + i].re = tga[1]; X[2 * M + i].im = -tgb[1];
        }
    }

    bool f32_wide_ok() const { return n_ >= 16; }

    void forward_simd_f32_wide(const float* input, complex_f32_t* output, float* work) const {
        if (radix2_terminal_) {
            // Odd log2(n): the shared V4F chain lands in the output buffer used
            // as V4F scratch, the depair writes the V2 layout into work, the
            // extra radix-2-level merge runs in place there, and the terminal
            // reads work and writes the bins back into output. Swapping the
            // V4F/V2 buffer roles this way removes the full-array copy the old
            // ordering needed (it left the merged V2 data in output, which the
            // terminal must also write).
            float* v4f_scratch = reinterpret_cast<float*>(output);
            compute_vwork_v4f(input, v4f_scratch);
            const int s = n_ >> 1;
            const int stage = ilog2_pow2(s) - 2;
            const double* cs2 = cs_.data() + 2 * twiddle_offset_[static_cast<std::size_t>(stage - 1)];
            const double* cs1 = cs_.data() + 2 * twiddle_offset_[static_cast<std::size_t>(stage)];
            depair_merge4_v_norm_f32(v4f_scratch, work, s >> 3, cs2, cs1);
            terminal_radix2_v_norm_f32(work, output);
            return;
        }
        compute_vwork_v4f(input, work);
        terminal_v4f_f32(work, output);
    }
#endif  // BRUUN_LEVEL >= 1 (wide f32)

    // ------------------------------------------------------------------
    // Mirrored-lane inverse: the same walk backwards. Standard bins enter
    // through the terminal split (the only lane crossing, fused with
    // deprojection), every split below is lane-vertical, and the seed
    // scatter leaves as contiguous V2 stores at rev(b).
    // ------------------------------------------------------------------

    static BRUUN_ALWAYS_INLINE void store_inv_butterfly4_v(double* vblk, int M, int i, const bruun_v2* o) {
        V2_ST(vblk + 4 * i, o[0]);
        V2_ST(vblk + 4 * i + 2, o[1]);
        V2_ST(vblk + 4 * (M + i), o[2]);
        V2_ST(vblk + 4 * (M + i) + 2, o[3]);
        V2_ST(vblk + 4 * (2 * M + i), o[4]);
        V2_ST(vblk + 4 * (2 * M + i) + 2, o[5]);
        V2_ST(vblk + 4 * (3 * M + i), o[6]);
        V2_ST(vblk + 4 * (3 * M + i) + 2, o[7]);
    }

    // Inverse seed: read one completed mirrored-lane seed block linearly from
    // vwork. Mirror lanes are adjacent time samples, so each destination is a
    // contiguous V2 store at rev(b); block order carries the permutation cost.
    void seed_scatter_block_v(const double* vw, double* output, int base, int count) const {
        const bruun_v2 hv = V2_SET1(0.5);
        const int hq = half_ >> 1;
        for (int b = base; b < base + count; b += 2) {
            const int j0 = rev_[static_cast<std::size_t>(b)];
            const int j1 = j0 + hq;
            const double* src = vw + 4 * b;
            const bruun_v2 va = V2_LD(src);
            const bruun_v2 vb = V2_LD(src + 2);
            const bruun_v2 vc = V2_LD(src + 4);
            const bruun_v2 vd = V2_LD(src + 6);
            const bruun_v2 e0 = V2_MUL(V2_ADD(va, vb), hv);
            const bruun_v2 e1 = V2_MUL(V2_SUB(va, vb), hv);
            V2_ST(output + j0, V2_MUL(V2_ADD(e0, vc), hv));
            V2_ST(output + j0 + half_, V2_MUL(V2_SUB(e0, vc), hv));
            V2_ST(output + j1, V2_MUL(V2_ADD(e1, vd), hv));
            V2_ST(output + j1 + half_, V2_MUL(V2_SUB(e1, vd), hv));
        }
    }

    void seed_scatter_row_tiles_v(const double* vw, double* output,
                                  int block_slots, int nb, int nb_bits) const {
        const bruun_v2 hv = V2_SET1(0.5);
        const int hq = half_ >> 1;
        const int pairs = block_slots >> 1;
        const int row_bits = ilog2_pow2(pairs);

        for (int r = 0; r < pairs; ++r) {
            const int row = bitrev_int(r, row_bits);
            const int row_base = 2 * row * nb;
            double* row0 = output + row_base;
            double* row1 = output + row_base + half_;
            double* row2 = output + row_base + hq;
            double* row3 = output + row_base + hq + half_;
            for (int h = 0; h < nb; ++h) {
                const int col = bitrev_int(h, nb_bits);
                const int b = h * block_slots + 2 * r;
                const double* src = vw + 4 * b;
                const bruun_v2 va = V2_LD(src);
                const bruun_v2 vb = V2_LD(src + 2);
                const bruun_v2 vc = V2_LD(src + 4);
                const bruun_v2 vd = V2_LD(src + 6);
                const bruun_v2 e0 = V2_MUL(V2_ADD(va, vb), hv);
                const bruun_v2 e1 = V2_MUL(V2_SUB(va, vb), hv);
                V2_ST(row0 + 2 * col, V2_MUL(V2_ADD(e0, vc), hv));
                V2_ST(row1 + 2 * col, V2_MUL(V2_SUB(e0, vc), hv));
                V2_ST(row2 + 2 * col, V2_MUL(V2_ADD(e1, vd), hv));
                V2_ST(row3 + 2 * col, V2_MUL(V2_SUB(e1, vd), hv));
            }
        }
    }

    // -------- Normalized-basis SIMD inverse (exact inverse of the forward) --
    // pair_expand_v2_norm undoes pair_reduce_v2_norm: the same (c, s) rotation
    // run backwards, so the whole inverse composes with the forward to exactly
    // 1/N with no separate normalization and no accuracy loss. Deprojection is
    // now trivial - the forward stored (re, -im) = (a, b), so the inverse reads
    // a = re, b = -im with no cos/sin at all.

    static BRUUN_ALWAYS_INLINE void pair_expand_v2_norm(bruun_v2 la, bruun_v2 lb, bruun_v2 ha, bruun_v2 hb,
                                                        bruun_v2 c, bruun_v2 s, bruun_v2 hv,
                                                        bruun_v2& ea, bruun_v2& eb, bruun_v2& oa, bruun_v2& ob) {
        const bruun_v2 r = V2_MUL(V2_SUB(la, ha), hv);   // 0.5*(la - ha)
        const bruun_v2 im = V2_MUL(V2_ADD(lb, hb), hv);  // 0.5*(lb + hb)
        ea = V2_MUL(V2_ADD(la, ha), hv);
        eb = V2_MUL(V2_SUB(lb, hb), hv);
        oa = V2_MADD(V2_MUL(c, r), s, im);   // c*r + s*im
        ob = V2_MSUB(V2_MUL(c, im), s, r);   // c*im - s*r
    }

    BRUUN_ALWAYS_INLINE void inv_butterfly4_v_norm(bruun_v2 nia, bruun_v2 nib, bruun_v2 nha, bruun_v2 nhb,
                                                   bruun_v2 nqla, bruun_v2 nqlb, bruun_v2 nqha, bruun_v2 nqhb,
                                                   const double* cs1i, const double* cs1q, const double* cs2i,
                                                   bruun_v2 hv, bruun_v2* o) const {
        bruun_v2 h0la, h0lb, h1la, h1lb, h0ha, h0hb, h1ha, h1hb;
        pair_expand_v2_norm(nia, nib, nha, nhb, V2_SET1(cs1i[0]), V2_SET1(cs1i[1]), hv,
                            h0la, h0lb, h1la, h1lb);
        pair_expand_v2_norm(nqla, nqlb, nqha, nqhb, V2_SET1(cs1q[0]), V2_SET1(cs1q[1]), hv,
                            h0ha, h0hb, h1ha, h1hb);
        const bruun_v2 c2 = V2_SET1(cs2i[0]);
        const bruun_v2 s2 = V2_SET1(cs2i[1]);
        pair_expand_v2_norm(h0la, h0lb, h0ha, h0hb, c2, s2, hv, o[0], o[1], o[2], o[3]);
        pair_expand_v2_norm(h1la, h1lb, h1ha, h1hb, c2, s2, hv, o[4], o[5], o[6], o[7]);
    }

    void split4_v_norm(double* vblk, int M, const double* cs2, const double* cs1) const {
        const bruun_v2 hv = V2_SET1(0.5);
        double* c0 = vblk;
        double* c1 = vblk + 4 * M;
        double* c2 = vblk + 8 * M;
        double* c3 = vblk + 12 * M;

        const bruun_v2 n0a = V2_LD(c0);
        const bruun_v2 n0b = V2_LD(c0 + 2);
        const bruun_v2 nMa = V2_LD(c1);
        const bruun_v2 nMb = V2_LD(c1 + 2);
        const bruun_v2 n2a = V2_LD(c2);
        const bruun_v2 n2b = V2_LD(c2 + 2);
        const bruun_v2 n3a = V2_LD(c3);
        const bruun_v2 n3b = V2_LD(c3 + 2);
        const bruun_v2 h0dc = V2_MUL(V2_ADD(n0a, n0b), hv);
        const bruun_v2 h1dc = V2_MUL(V2_SUB(n0a, n0b), hv);
        bruun_v2 ny0, ny1, ny2, ny3;
        pair_expand_v2_norm(nMa, nMb, n3a, n3b,
                            V2_SET1(cs1[2 * (M - 1)]), V2_SET1(cs1[2 * (M - 1) + 1]), hv,
                            ny0, ny1, ny2, ny3);
        V2_ST(c0, V2_MUL(V2_ADD(h0dc, n2a), hv));
        V2_ST(c0 + 2, ny0);
        V2_ST(c1, V2_MUL(V2_SUB(h0dc, n2a), hv));
        V2_ST(c1 + 2, ny1);
        V2_ST(c2, V2_MUL(V2_ADD(h1dc, n2b), hv));
        V2_ST(c2 + 2, ny2);
        V2_ST(c3, V2_MUL(V2_SUB(h1dc, n2b), hv));
        V2_ST(c3 + 2, ny3);

        for (int i = 1; i < M - i; ++i) {
            const int m = M - i;
            const bruun_v2 ia = V2_LD(vblk + 4 * i);
            const bruun_v2 ib = V2_LD(vblk + 4 * i + 2);
            const bruun_v2 iha = V2_LD(vblk + 4 * (4 * M - i));
            const bruun_v2 ihb = V2_LD(vblk + 4 * (4 * M - i) + 2);
            const bruun_v2 iqla = V2_LD(vblk + 4 * (2 * M - i));
            const bruun_v2 iqlb = V2_LD(vblk + 4 * (2 * M - i) + 2);
            const bruun_v2 iqha = V2_LD(vblk + 4 * (2 * M + i));
            const bruun_v2 iqhb = V2_LD(vblk + 4 * (2 * M + i) + 2);
            const bruun_v2 mha = V2_LD(vblk + 4 * (4 * M - m));
            const bruun_v2 mhb = V2_LD(vblk + 4 * (4 * M - m) + 2);
            const bruun_v2 mqla = V2_LD(vblk + 4 * (2 * M - m));
            const bruun_v2 mqlb = V2_LD(vblk + 4 * (2 * M - m) + 2);
            bruun_v2 oi[8];
            inv_butterfly4_v_norm(ia, ib, iha, ihb, iqla, iqlb, iqha, iqhb,
                                  cs1 + 2 * (i - 1), cs1 + 2 * (2 * M - i - 1), cs2 + 2 * (i - 1), hv, oi);
            store_inv_butterfly4_v(vblk, M, i, oi);
            const bruun_v2 ma = V2_LD(vblk + 4 * m);
            const bruun_v2 mb = V2_LD(vblk + 4 * m + 2);
            const bruun_v2 mqha = V2_LD(vblk + 4 * (2 * M + m));
            const bruun_v2 mqhb = V2_LD(vblk + 4 * (2 * M + m) + 2);
            bruun_v2 om[8];
            inv_butterfly4_v_norm(ma, mb, mha, mhb, mqla, mqlb, mqha, mqhb,
                                  cs1 + 2 * (m - 1), cs1 + 2 * (2 * M - m - 1), cs2 + 2 * (m - 1), hv, om);
            store_inv_butterfly4_v(vblk, M, m, om);
        }

        if (M >= 2) {
            const int i = M >> 1;
            const bruun_v2 ia = V2_LD(vblk + 4 * i);
            const bruun_v2 ib = V2_LD(vblk + 4 * i + 2);
            const bruun_v2 iha = V2_LD(vblk + 4 * (4 * M - i));
            const bruun_v2 ihb = V2_LD(vblk + 4 * (4 * M - i) + 2);
            const bruun_v2 iqla = V2_LD(vblk + 4 * (2 * M - i));
            const bruun_v2 iqlb = V2_LD(vblk + 4 * (2 * M - i) + 2);
            const bruun_v2 iqha = V2_LD(vblk + 4 * (2 * M + i));
            const bruun_v2 iqhb = V2_LD(vblk + 4 * (2 * M + i) + 2);
            bruun_v2 oi[8];
            inv_butterfly4_v_norm(ia, ib, iha, ihb, iqla, iqlb, iqha, iqhb,
                                  cs1 + 2 * (i - 1), cs1 + 2 * (2 * M - i - 1), cs2 + 2 * (i - 1), hv, oi);
            store_inv_butterfly4_v(vblk, M, i, oi);
        }
    }

    void compute_vwork_inv_norm(double* vw, double* output) const {
        BRUUN_ASSERT(n_ >= 16);
        const int q = half_ >> 1;
        const int chain_limit = radix2_terminal_ ? (n_ >> 1) : (n_ >> 2);
        const int s_target = should_cache_block<double>(n_) ? kBlockedStageSize : chain_limit;
        const int block_slots = s_target >> 1;
        const int nb = q / block_slots;
        const int nb_bits = ilog2_pow2(nb);

        int s = chain_limit;
        int stage = ilog2_pow2(s) - 2;
        for (; s >= (s_target << 2); s >>= 2, stage -= 2) {
            const double* cs2 = cs_.data() + 2 * twiddle_offset_[static_cast<std::size_t>(stage - 1)];
            const double* cs1 = cs_.data() + 2 * twiddle_offset_[static_cast<std::size_t>(stage)];
            const int M = s >> 3;
            for (int off = 0; off < n_; off += 2 * s) {
                split4_v_norm(vw + off, M, cs2, cs1);
            }
        }

        if (nb >= kRowTileBlockCount) {
            for (int p = 0; p < nb; ++p) {
                const int base = p * block_slots;
                int ss = s_target;
                int st = ilog2_pow2(ss) - 2;
                for (; ss >= 16; ss >>= 2, st -= 2) {
                    const double* cs2 = cs_.data() + 2 * twiddle_offset_[static_cast<std::size_t>(st - 1)];
                    const double* cs1 = cs_.data() + 2 * twiddle_offset_[static_cast<std::size_t>(st)];
                    const int M = ss >> 3;
                    for (int sub = base; sub < base + block_slots; sub += (ss >> 1)) {
                        split4_v_norm(vw + 4 * sub, M, cs2, cs1);
                    }
                }
            }
            seed_scatter_row_tiles_v(vw, output, block_slots, nb, nb_bits);
            return;
        }

        for (int p = 0; p < nb; ++p) {
            const int base = bitrev_int(p, nb_bits) * block_slots;
            int ss = s_target;
            int st = ilog2_pow2(ss) - 2;
            for (; ss >= 16; ss >>= 2, st -= 2) {
                const double* cs2 = cs_.data() + 2 * twiddle_offset_[static_cast<std::size_t>(st - 1)];
                const double* cs1 = cs_.data() + 2 * twiddle_offset_[static_cast<std::size_t>(st)];
                const int M = ss >> 3;
                for (int sub = base; sub < base + block_slots; sub += (ss >> 1)) {
                    split4_v_norm(vw + 4 * sub, M, cs2, cs1);
                }
            }
            seed_scatter_block_v(vw, output, base, block_slots);
        }
    }

    // Inverse terminal, normalized basis. Standard bins deproject to (a, b) =
    // (re, -im) with no cos/sin, then two normalized inverse-rotation levels
    // and the single lane crossing reproduce the mirrored-lane child slots.
    template <typename Complex>
    void terminal_inv_v_norm(const Complex* X, double* vw) const {
        const int M = half_ >> 2;   // child node count = n/8
        double* cA = vw;
        double* cB = vw + 4 * M;
        const double* cs1 = cs_.data() + 2 * twiddle_offset_[static_cast<std::size_t>(twiddle_stage_count_ - 1)];
        const bruun_v2 hv = V2_SET1(0.5);

        const double h0dc = 0.5 * (X[0].re + X[half_].re);
        const double h1dc = 0.5 * (X[0].re - X[half_].re);
        const double h0ny = X[2 * M].re;
        const double h1ny = -X[2 * M].im;
        {
            const double la = X[M].re;
            const double lb = -X[M].im;
            const double ha = X[half_ - M].re;
            const double hb = -X[half_ - M].im;
            double ny0, ny1, ny2, ny3;
            pair_expand_cs(la, lb, ha, hb, cs1[2 * (M - 1)], cs1[2 * (M - 1) + 1], ny0, ny1, ny2, ny3);
            V2_ST(cA, V2_SETLH(0.5 * (h0dc + h0ny), 0.5 * (h1dc + h1ny)));   // {C0dc, C2dc}
            V2_ST(cA + 2, V2_SETLH(ny0, ny2));                               // {C0ny, C2ny}
            V2_ST(cB, V2_SETLH(0.5 * (h0dc - h0ny), 0.5 * (h1dc - h1ny)));   // {C1dc, C3dc}
            V2_ST(cB + 2, V2_SETLH(ny1, ny3));                               // {C1ny, C3ny}
        }

        const double* rec = fterm_cs_.data();
        for (int i = 1; i < M; ++i, rec += 6) {
            const bruun_v2 xl0 = V2_LD(&X[i].re);
            const bruun_v2 xl1 = V2_LD(&X[2 * M - i].re);
            const bruun_v2 xh0 = V2_LD(&X[half_ - i].re);
            const bruun_v2 xh1 = V2_LD(&X[2 * M + i].re);
            const bruun_v2 re_l = V2_UNPLO(xl0, xl1);
            const bruun_v2 im_l = V2_UNPHI(xl0, xl1);
            const bruun_v2 re_h = V2_UNPLO(xh0, xh1);
            const bruun_v2 im_h = V2_UNPHI(xh0, xh1);
            // deproject: (a, b) = (re, -im) for bins {i, n/4-i} and {n/2-i, n/4+i}
            const bruun_v2 fa = re_l;
            const bruun_v2 fb = v2_neg(im_l);
            const bruun_v2 ga = re_h;
            const bruun_v2 gb = v2_neg(im_h);
            bruun_v2 ua, ub, va, vb;
            pair_expand_v2_norm(fa, fb, ga, gb, V2_LD(rec + 2), V2_LD(rec + 4), hv, ua, ub, va, vb);
            const bruun_v2 la = V2_UNPLO(ua, va);   // {h0 low, h1 low}
            const bruun_v2 lb = V2_UNPLO(ub, vb);
            const bruun_v2 ha = V2_UNPHI(ua, va);   // {h0 high, h1 high}
            const bruun_v2 hb = V2_UNPHI(ub, vb);
            bruun_v2 aA, bA, aB, bB;
            pair_expand_v2_norm(la, lb, ha, hb, V2_SET1(rec[0]), V2_SET1(rec[1]), hv, aA, bA, aB, bB);
            V2_ST(cA + 4 * i, aA);
            V2_ST(cA + 4 * i + 2, bA);
            V2_ST(cB + 4 * i, aB);
            V2_ST(cB + 4 * i + 2, bB);
        }
    }

    template <typename Complex>
    void terminal_radix2_inv_v_norm(const Complex* X, double* vw) const {
        const int q = half_ >> 1;
        const double* cs = cs_.data() + 2 * twiddle_offset_[static_cast<std::size_t>(twiddle_stage_count_ - 1)];
        const bruun_v2 hv = V2_SET1(0.5);

        V2_ST(vw, V2_SETLH(0.5 * (X[0].re + X[half_].re),
                           0.5 * (X[0].re - X[half_].re)));
        V2_ST(vw + 2, V2_SETLH(X[q].re, -X[q].im));

        int i = 1;
        for (; i + 1 < q; i += 2) {
            const bruun_v2 xl0 = V2_LD(&X[i].re);
            const bruun_v2 xl1 = V2_LD(&X[i + 1].re);
            const bruun_v2 re_l = V2_UNPLO(xl0, xl1);
            const bruun_v2 im_l = V2_UNPHI(xl0, xl1);

            const int hi0 = half_ - i;
            const int hi1 = half_ - i - 1;
            const bruun_v2 xh0 = V2_LD(&X[hi0].re);
            const bruun_v2 xh1 = V2_LD(&X[hi1].re);
            const bruun_v2 re_h = V2_UNPLO(xh0, xh1);
            const bruun_v2 im_h = V2_UNPHI(xh0, xh1);

            // low pair (bins i, i+1) and its mirror high pair both deproject
            // to (re, -im); one rotation (c, s of nodes i, i+1) inverts them.
            const bruun_v2 la = re_l;
            const bruun_v2 lb = v2_neg(im_l);
            const bruun_v2 ha = re_h;
            const bruun_v2 hb = v2_neg(im_h);
            const bruun_v2 cv = V2_LD(cos_.data() + i);
            const bruun_v2 sv = V2_LD(sin_.data() + i);

            bruun_v2 ea, eb, oa, ob;
            pair_expand_v2_norm(la, lb, ha, hb, cv, sv, hv, ea, eb, oa, ob);

            V2_ST(vw + 4 * i, V2_UNPLO(ea, oa));
            V2_ST(vw + 4 * i + 2, V2_UNPLO(eb, ob));
            V2_ST(vw + 4 * (i + 1), V2_UNPHI(ea, oa));
            V2_ST(vw + 4 * (i + 1) + 2, V2_UNPHI(eb, ob));
        }

        if (i < q) {
            const int hi = half_ - i;
            const double la = X[i].re;
            const double lb = -X[i].im;
            const double ha = X[hi].re;
            const double hb = -X[hi].im;
            double ea, eb, oa, ob;
            pair_expand_cs(la, lb, ha, hb, cs[2 * (i - 1)], cs[2 * (i - 1) + 1], ea, eb, oa, ob);
            V2_ST(vw + 4 * i, V2_SETLH(ea, oa));
            V2_ST(vw + 4 * i + 2, V2_SETLH(eb, ob));
        }
    }

#if BRUUN_LEVEL >= 1
    // ---- Wide f32 NEON inverse (exact reverse of the wide forward) ----
    static BRUUN_ALWAYS_INLINE void pair_expand_v4f_norm(bruun_v4f la, bruun_v4f lb, bruun_v4f ha, bruun_v4f hb,
                                                         bruun_v4f c, bruun_v4f s,
                                                         bruun_v4f& ea, bruun_v4f& eb, bruun_v4f& oa, bruun_v4f& ob) {
        const bruun_v4f hv = V4F_SET1(0.5f);
        const bruun_v4f r = V4F_MUL(V4F_SUB(la, ha), hv);
        const bruun_v4f im = V4F_MUL(V4F_ADD(lb, hb), hv);
        ea = V4F_MUL(V4F_ADD(la, ha), hv);
        eb = V4F_MUL(V4F_SUB(lb, hb), hv);
        oa = V4F_MADD(V4F_MUL(c, r), s, im);   // c*r + s*im
        ob = V4F_MSUB(V4F_MUL(c, im), s, r);   // c*im - s*r
    }

    BRUUN_ALWAYS_INLINE void inv_butterfly4_v4f(bruun_v4f nia, bruun_v4f nib, bruun_v4f nha, bruun_v4f nhb,
                                                bruun_v4f nqla, bruun_v4f nqlb, bruun_v4f nqha, bruun_v4f nqhb,
                                                const double* cs1i, const double* cs1q, const double* cs2i,
                                                bruun_v4f* o) const {
        bruun_v4f h0la, h0lb, h1la, h1lb, h0ha, h0hb, h1ha, h1hb;
        pair_expand_v4f_norm(nia, nib, nha, nhb, v4f_set1cs(cs1i), v4f_set1cs(cs1i + 1), h0la, h0lb, h1la, h1lb);
        pair_expand_v4f_norm(nqla, nqlb, nqha, nqhb, v4f_set1cs(cs1q), v4f_set1cs(cs1q + 1), h0ha, h0hb, h1ha, h1hb);
        const bruun_v4f c2 = v4f_set1cs(cs2i);
        const bruun_v4f s2 = v4f_set1cs(cs2i + 1);
        pair_expand_v4f_norm(h0la, h0lb, h0ha, h0hb, c2, s2, o[0], o[1], o[2], o[3]);
        pair_expand_v4f_norm(h1la, h1lb, h1ha, h1hb, c2, s2, o[4], o[5], o[6], o[7]);
    }

    static BRUUN_ALWAYS_INLINE void store_inv_butterfly4_v4f(float* vblk, int M, int i, const bruun_v4f* o) {
        V4F_ST(vblk + 8 * i, o[0]);
        V4F_ST(vblk + 8 * i + 4, o[1]);
        V4F_ST(vblk + 8 * (M + i), o[2]);
        V4F_ST(vblk + 8 * (M + i) + 4, o[3]);
        V4F_ST(vblk + 8 * (2 * M + i), o[4]);
        V4F_ST(vblk + 8 * (2 * M + i) + 4, o[5]);
        V4F_ST(vblk + 8 * (3 * M + i), o[6]);
        V4F_ST(vblk + 8 * (3 * M + i) + 4, o[7]);
    }

    void split4_v4f(float* vblk, int M, const double* cs2, const double* cs1) const {
        const bruun_v4f hv = V4F_SET1(0.5f);
        float* c0 = vblk;
        float* c1 = vblk + 8 * M;
        float* c2 = vblk + 16 * M;
        float* c3 = vblk + 24 * M;

        const bruun_v4f n0a = V4F_LD(c0);
        const bruun_v4f n0b = V4F_LD(c0 + 4);
        const bruun_v4f nMa = V4F_LD(c1);
        const bruun_v4f nMb = V4F_LD(c1 + 4);
        const bruun_v4f n2a = V4F_LD(c2);
        const bruun_v4f n2b = V4F_LD(c2 + 4);
        const bruun_v4f n3a = V4F_LD(c3);
        const bruun_v4f n3b = V4F_LD(c3 + 4);
        const bruun_v4f h0dc = V4F_MUL(V4F_ADD(n0a, n0b), hv);
        const bruun_v4f h1dc = V4F_MUL(V4F_SUB(n0a, n0b), hv);
        bruun_v4f ny0, ny1, ny2, ny3;
        pair_expand_v4f_norm(nMa, nMb, n3a, n3b, v4f_set1cs(cs1 + 2 * (M - 1)), v4f_set1cs(cs1 + 2 * (M - 1) + 1),
                             ny0, ny1, ny2, ny3);
        V4F_ST(c0, V4F_MUL(V4F_ADD(h0dc, n2a), hv));
        V4F_ST(c0 + 4, ny0);
        V4F_ST(c1, V4F_MUL(V4F_SUB(h0dc, n2a), hv));
        V4F_ST(c1 + 4, ny1);
        V4F_ST(c2, V4F_MUL(V4F_ADD(h1dc, n2b), hv));
        V4F_ST(c2 + 4, ny2);
        V4F_ST(c3, V4F_MUL(V4F_SUB(h1dc, n2b), hv));
        V4F_ST(c3 + 4, ny3);

        for (int i = 1; i < M - i; ++i) {
            const int m = M - i;
            const bruun_v4f ia = V4F_LD(vblk + 8 * i);
            const bruun_v4f ib = V4F_LD(vblk + 8 * i + 4);
            const bruun_v4f iha = V4F_LD(vblk + 8 * (4 * M - i));
            const bruun_v4f ihb = V4F_LD(vblk + 8 * (4 * M - i) + 4);
            const bruun_v4f iqla = V4F_LD(vblk + 8 * (2 * M - i));
            const bruun_v4f iqlb = V4F_LD(vblk + 8 * (2 * M - i) + 4);
            const bruun_v4f iqha = V4F_LD(vblk + 8 * (2 * M + i));
            const bruun_v4f iqhb = V4F_LD(vblk + 8 * (2 * M + i) + 4);
            const bruun_v4f mha = V4F_LD(vblk + 8 * (4 * M - m));
            const bruun_v4f mhb = V4F_LD(vblk + 8 * (4 * M - m) + 4);
            const bruun_v4f mqla = V4F_LD(vblk + 8 * (2 * M - m));
            const bruun_v4f mqlb = V4F_LD(vblk + 8 * (2 * M - m) + 4);
            bruun_v4f oi[8];
            inv_butterfly4_v4f(ia, ib, iha, ihb, iqla, iqlb, iqha, iqhb,
                               cs1 + 2 * (i - 1), cs1 + 2 * (2 * M - i - 1), cs2 + 2 * (i - 1), oi);
            store_inv_butterfly4_v4f(vblk, M, i, oi);
            const bruun_v4f ma = V4F_LD(vblk + 8 * m);
            const bruun_v4f mb = V4F_LD(vblk + 8 * m + 4);
            const bruun_v4f mqha = V4F_LD(vblk + 8 * (2 * M + m));
            const bruun_v4f mqhb = V4F_LD(vblk + 8 * (2 * M + m) + 4);
            bruun_v4f om[8];
            inv_butterfly4_v4f(ma, mb, mha, mhb, mqla, mqlb, mqha, mqhb,
                               cs1 + 2 * (m - 1), cs1 + 2 * (2 * M - m - 1), cs2 + 2 * (m - 1), om);
            store_inv_butterfly4_v4f(vblk, M, m, om);
        }

        if (M >= 2) {
            const int i = M >> 1;
            const bruun_v4f ia = V4F_LD(vblk + 8 * i);
            const bruun_v4f ib = V4F_LD(vblk + 8 * i + 4);
            const bruun_v4f iha = V4F_LD(vblk + 8 * (4 * M - i));
            const bruun_v4f ihb = V4F_LD(vblk + 8 * (4 * M - i) + 4);
            const bruun_v4f iqla = V4F_LD(vblk + 8 * (2 * M - i));
            const bruun_v4f iqlb = V4F_LD(vblk + 8 * (2 * M - i) + 4);
            const bruun_v4f iqha = V4F_LD(vblk + 8 * (2 * M + i));
            const bruun_v4f iqhb = V4F_LD(vblk + 8 * (2 * M + i) + 4);
            bruun_v4f oi[8];
            inv_butterfly4_v4f(ia, ib, iha, ihb, iqla, iqlb, iqha, iqhb,
                               cs1 + 2 * (i - 1), cs1 + 2 * (2 * M - i - 1), cs2 + 2 * (i - 1), oi);
            store_inv_butterfly4_v4f(vblk, M, i, oi);
        }
    }

    BRUUN_ALWAYS_INLINE void inv_butterfly4_v_norm_f32_v4_inputs(bruun_v4f ni, bruun_v4f nh,
                                                                 bruun_v4f nql, bruun_v4f nqh,
                                                                 const double* cs1i, const double* cs1q,
                                                                 const double* cs2i, bruun_v4f* out) const {
        bruun_v4f h0la, h0lb, h1la, h1lb;
        pair_expand_v4f_norm(V4F_CATLO(ni, ni), V4F_CATHI(ni, ni),
                             V4F_CATLO(nh, nh), V4F_CATHI(nh, nh),
                             v4f_set1cs(cs1i), v4f_set1cs(cs1i + 1),
                             h0la, h0lb, h1la, h1lb);

        bruun_v4f h0ha, h0hb, h1ha, h1hb;
        pair_expand_v4f_norm(V4F_CATLO(nql, nql), V4F_CATHI(nql, nql),
                             V4F_CATLO(nqh, nqh), V4F_CATHI(nqh, nqh),
                             v4f_set1cs(cs1q), v4f_set1cs(cs1q + 1),
                             h0ha, h0hb, h1ha, h1hb);

        bruun_v4f ea, eb, oa, ob;
        pair_expand_v4f_norm(h0la, h0lb, h0ha, h0hb,
                             v4f_set1cs(cs2i), v4f_set1cs(cs2i + 1),
                             ea, eb, oa, ob);
        out[0] = V4F_CATLO(ea, eb);
        out[1] = V4F_CATLO(oa, ob);

        pair_expand_v4f_norm(h1la, h1lb, h1ha, h1hb,
                             v4f_set1cs(cs2i), v4f_set1cs(cs2i + 1),
                             ea, eb, oa, ob);
        out[2] = V4F_CATLO(ea, eb);
        out[3] = V4F_CATLO(oa, ob);
    }

    BRUUN_ALWAYS_INLINE void inv_butterfly4_v_norm_f32_v4(const float* vblk, int M, int i,
                                                           const double* cs1i, const double* cs1q,
                                                           const double* cs2i, bruun_v4f* out) const {
        inv_butterfly4_v_norm_f32_v4_inputs(V4F_LD(vblk + 4 * i),
                                            V4F_LD(vblk + 4 * (4 * M - i)),
                                            V4F_LD(vblk + 4 * (2 * M - i)),
                                            V4F_LD(vblk + 4 * (2 * M + i)),
                                            cs1i, cs1q, cs2i, out);
    }

    static BRUUN_ALWAYS_INLINE void store_inv_butterfly4_v_norm_f32_v4(float* vblk, int M, int i,
                                                                       const bruun_v4f* out) {
        V4F_ST(vblk + 4 * i, out[0]);
        V4F_ST(vblk + 4 * (M + i), out[1]);
        V4F_ST(vblk + 4 * (2 * M + i), out[2]);
        V4F_ST(vblk + 4 * (3 * M + i), out[3]);
    }

    void terminal_radix2_inv_v_norm_f32(const complex_f32_t* X, float* vw) const {
        const int q = half_ >> 1;
        vw[0] = 0.5f * (X[0].re + X[half_].re);
        vw[1] = 0.5f * (X[0].re - X[half_].re);
        vw[2] = X[q].re;
        vw[3] = -X[q].im;

        for (int i = 1; i < q; ++i) {
            const int hi = half_ - i;
            float ea, eb, oa, ob;
            pair_expand_cs_f32(X[i].re, -X[i].im, X[hi].re, -X[hi].im,
                               static_cast<float>(cos_[static_cast<std::size_t>(i)]),
                               static_cast<float>(sin_[static_cast<std::size_t>(i)]),
                               ea, eb, oa, ob);
            vw[4 * i] = ea;
            vw[4 * i + 1] = oa;
            vw[4 * i + 2] = eb;
            vw[4 * i + 3] = ob;
        }
    }

    // Inverse terminal: standard bins -> paired-slot layout (reverse of
    // terminal_v4f_f32). Inverse rotations in V4F, ZIPLO undoes the mirror
    // crossing, CATLO writes cA/cB back into the low2/high2 lanes.
    void terminal_inv_v4f_f32(const complex_f32_t* X, float* fvw) const {
        const int M = half_ >> 2;
        const double* cs1 = cs_.data() + 2 * twiddle_offset_[static_cast<std::size_t>(twiddle_stage_count_ - 1)];

        {
            const float sum0 = 0.5f * (X[0].re + X[half_].re);
            const float sum1 = 0.5f * (X[0].re - X[half_].re);
            const float d0 = X[2 * M].re;
            const float d1 = -X[2 * M].im;
            fvw[0] = 0.5f * (sum0 + d0);
            fvw[2] = 0.5f * (sum0 - d0);
            fvw[1] = 0.5f * (sum1 + d1);
            fvw[3] = 0.5f * (sum1 - d1);
            float e4, e6, e5, e7;
            pair_expand_norm_t<float>(X[M].re, -X[M].im, X[half_ - M].re, -X[half_ - M].im,
                                      cs1 + 2 * (M - 1), e4, e6, e5, e7);
            fvw[4] = e4;
            fvw[5] = e5;
            fvw[6] = e6;
            fvw[7] = e7;
        }

        const double* rec = fterm_cs_.data();
        for (int i = 1; i < M; ++i, rec += 6) {
            // deproject the 4 bins to (a, b) = (re, -im) in lanes {0,1}
            const bruun_v4f fa = V4F_SET4(X[i].re, X[2 * M - i].re, X[i].re, X[2 * M - i].re);
            const bruun_v4f fb = V4F_SET4(-X[i].im, -X[2 * M - i].im, -X[i].im, -X[2 * M - i].im);
            const bruun_v4f ga = V4F_SET4(X[half_ - i].re, X[2 * M + i].re, X[half_ - i].re, X[2 * M + i].re);
            const bruun_v4f gb = V4F_SET4(-X[half_ - i].im, -X[2 * M + i].im, -X[half_ - i].im, -X[2 * M + i].im);
            const bruun_v4f cB = V4F_SET4(static_cast<float>(rec[2]), static_cast<float>(rec[3]),
                                          static_cast<float>(rec[2]), static_cast<float>(rec[3]));
            const bruun_v4f sB = V4F_SET4(static_cast<float>(rec[4]), static_cast<float>(rec[5]),
                                          static_cast<float>(rec[4]), static_cast<float>(rec[5]));
            // inverse level B -> ua, ub, va, vb (lanes 0,1)
            bruun_v4f ua, ub, va, vb;
            pair_expand_v4f_norm(fa, fb, ga, gb, cB, sB, ua, ub, va, vb);
            // reconstruct {la | ha} = ZIPLO(ua, va) (undoes the forward crossing)
            const bruun_v4f laha = V4F_ZIPLO(ua, va);
            const bruun_v4f lbhb = V4F_ZIPLO(ub, vb);
            // inverse level A -> aA, bA, aB, bB (lanes 0,1)
            bruun_v4f aA, bA, aB, bB;
            pair_expand_v4f_norm(laha, lbhb, V4F_CATHI(laha, laha), V4F_CATHI(lbhb, lbhb),
                                 v4f_set1cs(rec), v4f_set1cs(rec + 1), aA, bA, aB, bB);
            V4F_ST(fvw + 8 * i, V4F_CATLO(aA, aB));       // {aA.lo,aA.hi,aB.lo,aB.hi}
            V4F_ST(fvw + 8 * i + 4, V4F_CATLO(bA, bB));
        }
    }

    // Inverse seed scatter: reverse of seed_paired_pair_f32. cA is lanes {0,1},
    // cB is lanes {2,3}; each is the double seed_scatter_block_v core, scalar.
    BRUUN_ALWAYS_INLINE void seed_scatter_paired_pair_f32(const float* fvw, float* output, int s, int eighth) const {
        const bruun_v4f hv = V4F_SET1(0.5f);
        const int hq = half_ >> 1;
        const float* p = fvw + 8 * s;
        const float* q = fvw + 8 * (s + 1);
        const int j0 = rev_[static_cast<std::size_t>(s)];
        const int jB = rev_[static_cast<std::size_t>(s + eighth)];
        BRUUN_ASSERT(jB == j0 + 2);
        (void)jB;

        const bruun_v4f va = V4F_LD(p);
        const bruun_v4f vb = V4F_LD(p + 4);
        const bruun_v4f vc = V4F_LD(q);
        const bruun_v4f vd = V4F_LD(q + 4);
        const bruun_v4f e0 = V4F_MUL(V4F_ADD(va, vb), hv);
        const bruun_v4f e1 = V4F_MUL(V4F_SUB(va, vb), hv);
        V4F_ST(output + j0,           V4F_MUL(V4F_ADD(e0, vc), hv));
        V4F_ST(output + j0 + half_,   V4F_MUL(V4F_SUB(e0, vc), hv));
        V4F_ST(output + j0 + hq,      V4F_MUL(V4F_ADD(e1, vd), hv));
        V4F_ST(output + j0 + hq + half_, V4F_MUL(V4F_SUB(e1, vd), hv));
    }

    // Collect-a-scatter-then-organize (row tiles), paired-layout port of
    // seed_scatter_row_tiles_v. rev(s+eighth) = rev(s)+2 makes a paired slot's
    // cA/cB outputs contiguous, so each chunk is one V4F store {cA.lo, cA.hi,
    // cB.lo, cB.hi}. Reads stride the vwork (scattered loads); writes land in
    // cache-local output rows -> no scatter to output.
    void seed_scatter_row_tiles_paired_f32(const float* fvw, float* output,
                                           int block_slots, int nb, int nb_bits) const {
        const bruun_v4f hv = V4F_SET1(0.5f);
        const int hq = half_ >> 1;
        const int pairs = block_slots >> 1;
        const int row_bits = ilog2_pow2(pairs);
        for (int r = 0; r < pairs; ++r) {
            const int row = bitrev_int(r, row_bits);
            const int row_base = 4 * row * nb;
            float* row0 = output + row_base;
            float* row1 = output + row_base + half_;
            float* row2 = output + row_base + hq;
            float* row3 = output + row_base + hq + half_;
            for (int h = 0; h < nb; ++h) {
                const int col = bitrev_int(h, nb_bits);
                const int b = h * block_slots + 2 * r;
                const bruun_v4f va = V4F_LD(fvw + 8 * b);
                const bruun_v4f vb = V4F_LD(fvw + 8 * b + 4);
                const bruun_v4f vc = V4F_LD(fvw + 8 * (b + 1));
                const bruun_v4f vd = V4F_LD(fvw + 8 * (b + 1) + 4);
                const bruun_v4f e0 = V4F_MUL(V4F_ADD(va, vb), hv);
                const bruun_v4f e1 = V4F_MUL(V4F_SUB(va, vb), hv);
                V4F_ST(row0 + 4 * col, V4F_MUL(V4F_ADD(e0, vc), hv));
                V4F_ST(row1 + 4 * col, V4F_MUL(V4F_SUB(e0, vc), hv));
                V4F_ST(row2 + 4 * col, V4F_MUL(V4F_ADD(e1, vd), hv));
                V4F_ST(row3 + 4 * col, V4F_MUL(V4F_SUB(e1, vd), hv));
            }
        }
    }

    // Mirrors compute_vwork_inv_norm on the paired layout: full-array split of
    // the top levels, per-block (cache-resident) split of the bottom levels,
    // then organize the output (row tiles for large N, block scatter for small).
    void compute_vwork_inv_v4f(float* fvw, float* output) const {
        const int eighth = half_ >> 2;   // P paired-slots
        int chain_limit = kF32MinInverseChain;
        for (int s = 16; s <= (n_ >> 2); s <<= 2) {
            chain_limit = s;
        }
        const int s_target = should_cache_block<float>(n_) ? kBlockedStageSize : chain_limit;
        const int block_slots = s_target >> 1;   // paired-slots per block
        const int nb = eighth / block_slots;
        const int nb_bits = ilog2_pow2(nb);

        int s = chain_limit;
        int stage = ilog2_pow2(s) - 2;
        for (; s >= (s_target << 2); s >>= 2, stage -= 2) {
            const double* cs2 = cs_.data() + 2 * twiddle_offset_[static_cast<std::size_t>(stage - 1)];
            const double* cs1 = cs_.data() + 2 * twiddle_offset_[static_cast<std::size_t>(stage)];
            const int M = s >> 3;
            for (int poff = 0; poff < eighth; poff += (s >> 1)) {
                split4_v4f(fvw + 8 * poff, M, cs2, cs1);
            }
        }

        if (nb >= kRowTileBlockCount) {
            for (int p = 0; p < nb; ++p) {
                const int base = p * block_slots;
                int ss = s_target;
                int st = ilog2_pow2(ss) - 2;
                for (; ss >= 16; ss >>= 2, st -= 2) {
                    const double* cs2 = cs_.data() + 2 * twiddle_offset_[static_cast<std::size_t>(st - 1)];
                    const double* cs1 = cs_.data() + 2 * twiddle_offset_[static_cast<std::size_t>(st)];
                    const int M = ss >> 3;
                    for (int sub = base; sub < base + block_slots; sub += (ss >> 1)) {
                        split4_v4f(fvw + 8 * sub, M, cs2, cs1);
                    }
                }
            }
            seed_scatter_row_tiles_paired_f32(fvw, output, block_slots, nb, nb_bits);
            return;
        }

        for (int p = 0; p < nb; ++p) {
            const int base = bitrev_int(p, nb_bits) * block_slots;
            int ss = s_target;
            int st = ilog2_pow2(ss) - 2;
            for (; ss >= 16; ss >>= 2, st -= 2) {
                const double* cs2 = cs_.data() + 2 * twiddle_offset_[static_cast<std::size_t>(st - 1)];
                const double* cs1 = cs_.data() + 2 * twiddle_offset_[static_cast<std::size_t>(st)];
                const int M = ss >> 3;
                for (int sub = base; sub < base + block_slots; sub += (ss >> 1)) {
                    split4_v4f(fvw + 8 * sub, M, cs2, cs1);
                }
            }
            for (int sub = base; sub < base + block_slots; sub += 2) {
                seed_scatter_paired_pair_f32(fvw, output, sub, eighth);
            }
        }
    }

    // Fused first radix-4 split + V2->V4F pairing: reads the post-terminal
    // V2-f32 slots from vw (cache-hot), writes the paired-V4F work layout
    // directly, collapsing split4_v_norm_f32 + pair_v2_to_v4f_f32 into one
    // pass. Same palindromic read pattern as split4_v_norm_f32, but on the
    // cache-resident terminal output rather than the scattered input bins.
    void split4_pair_v_norm_f32(const float* vw, float* fvw, int M,
                                const double* cs2, const double* cs1) const {
        // i = 0 special (split4_v_norm_f32's i=0 algebra), packed to paired 0/M.
        {
            const float* c0 = vw;
            const float* c1 = vw + 4 * M;
            const float* c2 = vw + 8 * M;
            const float* c3 = vw + 12 * M;
            const float h0dc0 = 0.5f * (c0[0] + c0[2]);
            const float h0dc1 = 0.5f * (c0[1] + c0[3]);
            const float h1dc0 = 0.5f * (c0[0] - c0[2]);
            const float h1dc1 = 0.5f * (c0[1] - c0[3]);
            const float cM = static_cast<float>(cs1[2 * (M - 1)]);
            const float sM = static_cast<float>(cs1[2 * (M - 1) + 1]);
            float ny00, ny01, ny20, ny21, ny10, ny11, ny30, ny31;
            pair_expand_cs_f32(c1[0], c1[2], c3[0], c3[2], cM, sM, ny00, ny01, ny20, ny21);
            pair_expand_cs_f32(c1[1], c1[3], c3[1], c3[3], cM, sM, ny10, ny11, ny30, ny31);
            V4F_ST(fvw,     V4F_SET4(0.5f * (h0dc0 + c2[0]), 0.5f * (h0dc1 + c2[1]),
                                     0.5f * (h1dc0 + c2[2]), 0.5f * (h1dc1 + c2[3])));
            V4F_ST(fvw + 4, V4F_SET4(ny00, ny10, ny20, ny30));
            V4F_ST(fvw + 8 * M,     V4F_SET4(0.5f * (h0dc0 - c2[0]), 0.5f * (h0dc1 - c2[1]),
                                             0.5f * (h1dc0 - c2[2]), 0.5f * (h1dc1 - c2[3])));
            V4F_ST(fvw + 8 * M + 4, V4F_SET4(ny01, ny11, ny21, ny31));
        }

        for (int i = 1; i < M; ++i) {
            bruun_v4f out[4];
            inv_butterfly4_v_norm_f32_v4_inputs(V4F_LD(vw + 4 * i),
                                                V4F_LD(vw + 4 * (4 * M - i)),
                                                V4F_LD(vw + 4 * (2 * M - i)),
                                                V4F_LD(vw + 4 * (2 * M + i)),
                                                cs1 + 2 * (i - 1), cs1 + 2 * (2 * M - i - 1),
                                                cs2 + 2 * (i - 1), out);
            V4F_ST(fvw + 8 * i,           V4F_CATLO(out[0], out[2]));
            V4F_ST(fvw + 8 * i + 4,       V4F_CATHI(out[0], out[2]));
            V4F_ST(fvw + 8 * (M + i),     V4F_CATLO(out[1], out[3]));
            V4F_ST(fvw + 8 * (M + i) + 4, V4F_CATHI(out[1], out[3]));
        }
    }

    void inverse_simd_f32_wide(const complex_f32_t* input, float* output, float* work) const {
        if (radix2_terminal_) {
            terminal_radix2_inv_v_norm_f32(input, output);
            const int s = n_ >> 1;
            const int stage = ilog2_pow2(s) - 2;
            const double* cs2 = cs_.data() + 2 * twiddle_offset_[static_cast<std::size_t>(stage - 1)];
            const double* cs1 = cs_.data() + 2 * twiddle_offset_[static_cast<std::size_t>(stage)];
            split4_pair_v_norm_f32(output, work, s >> 3, cs2, cs1);
            compute_vwork_inv_v4f(work, output);
            return;
        }
        terminal_inv_v4f_f32(input, work);
        compute_vwork_inv_v4f(work, output);
    }
#endif  // BRUUN_LEVEL >= 1 (wide f32)
#endif  // BRUUN_LEVEL >= 1

    int n_;
    int half_;
    int twiddle_stage_count_;
    bool radix2_terminal_;
    heap_array<int> rev_;
    heap_array<int> twiddle_offset_;
    heap_array<double> cos_;
    heap_array<double> sin_;
    heap_array<double> twiddle_;
    heap_array<double> cs_;
    heap_array<double> fterm_cs_;
};

} // namespace bruun
