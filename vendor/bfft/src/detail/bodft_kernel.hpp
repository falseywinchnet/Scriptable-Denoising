#pragma once

// MIT joshuah.rainstar@gmail.com 2026
// Internal paired radix-4 BODFT kernel (Bruun odd-frequency DFT).
//
// BODFT is a first-class half-bin spectral primitive, not a DCT. It computes the
// native half-bin-shifted transform
//
//     H[k] = sum_{n=0}^{N-1} x[n] * exp(-2*pi*i*(k+1/2)*n/N),   k = 0..N-1
//
// for real input x. Equivalently it is the DFT of x sampled halfway between the
// ordinary FFT bins -- the odd-frequency DFT (ODFT). Because the input is real,
// H obeys H[N-1-k] = conj(H[k]); the kernel emits only the independent lower
// half H[0..N/2-1] as a contiguous packed complex spectrum. There is no
// DC/Nyquist special case and no final reorder.
//
// The decimation is radix-4. For N = 4M, split samples by n = 4q+r. Each child
// is again a length-M BODFT of real data, so each child spectrum obeys
// A_r[M-1-k] = conj(A_r[k]). Pairing each output position k with its partner
// kp = M-1-k inside the radix-4 combine lets the two partners share the three
// child twiddle multiplies (t, t^2, t^3), the same way the Bruun real kernel
// pairs conjugate residues. That halves the twiddle multiplies of the naive
// radix-4 combine.
//
// Scheduling is iterative, not recursive: a build-time radix-4 digit-reversal
// permutation drives a single leaf pass, then log4(N) ping-pong combine passes
// run leaves-up (forward) or top-down (inverse). Forward and inverse are both
// first class in float and double. The forward combine has an explicit 128-bit
// SoA SIMD specialization (V2 for double, V4F for float) that runs on the NEON
// and SSE2 backends through the shared Bruun vector macros; the scalar path
// stays the exact reference for every backend, precision and tail.
//
// Precision: float and double only. The kernel reuses heap_array,
// complex_t/complex_f32_t and the SIMD-backend machinery from the Bruun real
// kernel; only the decimation arithmetic is new.

#include "bruun_simd_backend.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>

#ifndef M_PI
#define M_PI 3.141592653589793238462643383279502884
#endif

namespace bodft {

using bruun::heap_array;
using bruun::complex_t;
using bruun::complex_f32_t;
using bruun::is_power2;
using bruun::ilog2_pow2;

// ---------------------------------------------------------------------------
// Small complex helpers, generic over complex_t (double) and complex_f32_t.
// ---------------------------------------------------------------------------

template <class CT>
static inline CT cadd(CT a, CT b) { return CT{a.re + b.re, a.im + b.im}; }
template <class CT>
static inline CT csub(CT a, CT b) { return CT{a.re - b.re, a.im - b.im}; }
template <class CT>
static inline CT cmul(CT a, CT b) {
    return CT{a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re};
}
template <class CT, class R>
static inline CT cmul_r(CT a, R s) { return CT{a.re * s, a.im * s}; }
template <class CT>
static inline CT cconj(CT a) { return CT{a.re, -a.im}; }
// i * a   (rotate by +90 degrees)
template <class CT>
static inline CT cmuli(CT a) { return CT{-a.im, a.re}; }

// ---------------------------------------------------------------------------
// Portable 4-lane complex deinterleave/reverse for the float combine. These
// lower to SSE shuffles on x86 and the matching NEON ops, so the float path
// follows the same SoA transport flow as the double path without any
// ISA-specific code. Interleave reuses the Bruun V4F_ZIPLO/V4F_ZIPHI macros.
// ---------------------------------------------------------------------------
#if BRUUN_LEVEL >= 1
#  if defined(BRUUN_X86_128)
#    define V4F_UZPRE(A, B) _mm_shuffle_ps((A), (B), _MM_SHUFFLE(2, 0, 2, 0))
#    define V4F_UZPIM(A, B) _mm_shuffle_ps((A), (B), _MM_SHUFFLE(3, 1, 3, 1))
#    define V4F_REV(A)      _mm_shuffle_ps((A), (A), _MM_SHUFFLE(0, 1, 2, 3))
#  elif defined(BRUUN_NEON_128)
#    define V4F_UZPRE(A, B) vuzp1q_f32((A), (B))
#    define V4F_UZPIM(A, B) vuzp2q_f32((A), (B))
#    define V4F_REV(A)      vextq_f32(vrev64q_f32(A), vrev64q_f32(A), 2)
#  endif
#endif


#if BRUUN_LEVEL >= 2
static inline void bodft_uzp4_pd(const complex_t* RESTRICT src, int k,
                                 __m256d& re, __m256d& im) {
    const __m256d a = _mm256_loadu_pd(&src[k].re);
    const __m256d b = _mm256_loadu_pd(&src[k + 2].re);
    const __m256d lo = _mm256_unpacklo_pd(a, b);
    const __m256d hi = _mm256_unpackhi_pd(a, b);
    re = _mm256_permute4x64_pd(lo, 0xD8);
    im = _mm256_permute4x64_pd(hi, 0xD8);
}

static inline void bodft_store4_pd(complex_t* RESTRICT dst, int k,
                                   __m256d re, __m256d im) {
    const __m256d lo = _mm256_unpacklo_pd(re, im);
    const __m256d hi = _mm256_unpackhi_pd(re, im);
    _mm256_storeu_pd(&dst[k].re,     _mm256_permute2f128_pd(lo, hi, 0x20));
    _mm256_storeu_pd(&dst[k + 2].re, _mm256_permute2f128_pd(lo, hi, 0x31));
}

static inline __m256d bodft_rev4_pd(__m256d v) {
    return _mm256_permute4x64_pd(v, 0x1B);
}

static inline int bodft_combine_fwd_avx2_f64(const complex_t* RESTRICT tab,
                                             const complex_t* RESTRICT tab2,
                                             const complex_t* RESTRICT tab3,
                                             const complex_t* RESTRICT c0,
                                             const complex_t* RESTRICT c1,
                                             const complex_t* RESTRICT c2,
                                             const complex_t* RESTRICT c3,
                                             complex_t* RESTRICT out,
                                             int M, int half) {
    const __m256d zero = _mm256_setzero_pd();
    int k = 0;
    for (; k + 4 <= half; k += 4) {
        __m256d tre, tim, t2re, t2im, t3re, t3im;
        __m256d c0re, c0im, c1re, c1im, c2re, c2im, c3re, c3im;
        bodft_uzp4_pd(tab,  k, tre, tim);
        bodft_uzp4_pd(tab2, k, t2re, t2im);
        bodft_uzp4_pd(tab3, k, t3re, t3im);
        bodft_uzp4_pd(c0, k, c0re, c0im);
        bodft_uzp4_pd(c1, k, c1re, c1im);
        bodft_uzp4_pd(c2, k, c2re, c2im);
        bodft_uzp4_pd(c3, k, c3re, c3im);

        const __m256d b0re = c0re, b0im = c0im;
        const __m256d b1re = _mm256_fmsub_pd(c1re, tre, _mm256_mul_pd(c1im, tim));
        const __m256d b1im = _mm256_fmadd_pd(c1re, tim, _mm256_mul_pd(c1im, tre));
        const __m256d b2re = _mm256_fmsub_pd(c2re, t2re, _mm256_mul_pd(c2im, t2im));
        const __m256d b2im = _mm256_fmadd_pd(c2re, t2im, _mm256_mul_pd(c2im, t2re));
        const __m256d b3re = _mm256_fmsub_pd(c3re, t3re, _mm256_mul_pd(c3im, t3im));
        const __m256d b3im = _mm256_fmadd_pd(c3re, t3im, _mm256_mul_pd(c3im, t3re));

        const __m256d e0re = _mm256_add_pd(b0re, b2re), e0im = _mm256_add_pd(b0im, b2im);
        const __m256d e1re = _mm256_sub_pd(b0re, b2re), e1im = _mm256_sub_pd(b0im, b2im);
        const __m256d o0re = _mm256_add_pd(b1re, b3re), o0im = _mm256_add_pd(b1im, b3im);
        const __m256d o1re = _mm256_sub_pd(b1re, b3re), o1im = _mm256_sub_pd(b1im, b3im);

        const __m256d yk_re = _mm256_add_pd(e0re, o0re), yk_im = _mm256_add_pd(e0im, o0im);
        const __m256d ykM_re = _mm256_add_pd(e1re, o1im), ykM_im = _mm256_sub_pd(e1im, o1re);
        const __m256d ykp_re = _mm256_sub_pd(e1re, o1im);
        const __m256d ykp_im = _mm256_sub_pd(zero, _mm256_add_pd(e1im, o1re));
        const __m256d ykpM_re = _mm256_sub_pd(e0re, o0re);
        const __m256d ykpM_im = _mm256_sub_pd(o0im, e0im);

        bodft_store4_pd(out, k, yk_re, yk_im);
        bodft_store4_pd(out, k + M, ykM_re, ykM_im);
        bodft_store4_pd(out, M - 4 - k, bodft_rev4_pd(ykp_re), bodft_rev4_pd(ykp_im));
        bodft_store4_pd(out, M - 4 - k + M, bodft_rev4_pd(ykpM_re), bodft_rev4_pd(ykpM_im));
    }
    return k;
}


static inline void bodft_uzp8_ps(const complex_f32_t* RESTRICT src, int k,
                                 __m256& re, __m256& im) {
    const __m256 a = _mm256_loadu_ps(&src[k].re);
    const __m256 b = _mm256_loadu_ps(&src[k + 4].re);
    const __m256 lo = _mm256_unpacklo_ps(a, b);
    const __m256 hi = _mm256_unpackhi_ps(a, b);
    const __m256 re_pairs = _mm256_shuffle_ps(lo, hi, _MM_SHUFFLE(1, 0, 1, 0));
    const __m256 im_pairs = _mm256_shuffle_ps(lo, hi, _MM_SHUFFLE(3, 2, 3, 2));
    re = _mm256_permutevar8x32_ps(re_pairs, _mm256_setr_epi32(0, 2, 4, 6, 1, 3, 5, 7));
    im = _mm256_permutevar8x32_ps(im_pairs, _mm256_setr_epi32(0, 2, 4, 6, 1, 3, 5, 7));
}

static inline void bodft_store8_ps(complex_f32_t* RESTRICT dst, int k,
                                   __m256 re, __m256 im) {
    const __m256 lo = _mm256_unpacklo_ps(re, im);
    const __m256 hi = _mm256_unpackhi_ps(re, im);
    _mm256_storeu_ps(&dst[k].re,     _mm256_permute2f128_ps(lo, hi, 0x20));
    _mm256_storeu_ps(&dst[k + 4].re, _mm256_permute2f128_ps(lo, hi, 0x31));
}

static inline __m256 bodft_rev8_ps(__m256 v) {
    return _mm256_permutevar8x32_ps(v, _mm256_setr_epi32(7, 6, 5, 4, 3, 2, 1, 0));
}

static inline int bodft_combine_fwd_avx2_f32(const complex_f32_t* RESTRICT tab,
                                             const complex_f32_t* RESTRICT tab2,
                                             const complex_f32_t* RESTRICT tab3,
                                             const complex_f32_t* RESTRICT c0,
                                             const complex_f32_t* RESTRICT c1,
                                             const complex_f32_t* RESTRICT c2,
                                             const complex_f32_t* RESTRICT c3,
                                             complex_f32_t* RESTRICT out,
                                             int M, int half) {
    const __m256 zero = _mm256_setzero_ps();
    int k = 0;
    for (; k + 8 <= half; k += 8) {
        __m256 tre, tim, t2re, t2im, t3re, t3im;
        __m256 c0re, c0im, c1re, c1im, c2re, c2im, c3re, c3im;
        bodft_uzp8_ps(tab,  k, tre, tim);
        bodft_uzp8_ps(tab2, k, t2re, t2im);
        bodft_uzp8_ps(tab3, k, t3re, t3im);
        bodft_uzp8_ps(c0, k, c0re, c0im);
        bodft_uzp8_ps(c1, k, c1re, c1im);
        bodft_uzp8_ps(c2, k, c2re, c2im);
        bodft_uzp8_ps(c3, k, c3re, c3im);

        const __m256 b0re = c0re, b0im = c0im;
        const __m256 b1re = _mm256_fmsub_ps(c1re, tre, _mm256_mul_ps(c1im, tim));
        const __m256 b1im = _mm256_fmadd_ps(c1re, tim, _mm256_mul_ps(c1im, tre));
        const __m256 b2re = _mm256_fmsub_ps(c2re, t2re, _mm256_mul_ps(c2im, t2im));
        const __m256 b2im = _mm256_fmadd_ps(c2re, t2im, _mm256_mul_ps(c2im, t2re));
        const __m256 b3re = _mm256_fmsub_ps(c3re, t3re, _mm256_mul_ps(c3im, t3im));
        const __m256 b3im = _mm256_fmadd_ps(c3re, t3im, _mm256_mul_ps(c3im, t3re));

        const __m256 e0re = _mm256_add_ps(b0re, b2re), e0im = _mm256_add_ps(b0im, b2im);
        const __m256 e1re = _mm256_sub_ps(b0re, b2re), e1im = _mm256_sub_ps(b0im, b2im);
        const __m256 o0re = _mm256_add_ps(b1re, b3re), o0im = _mm256_add_ps(b1im, b3im);
        const __m256 o1re = _mm256_sub_ps(b1re, b3re), o1im = _mm256_sub_ps(b1im, b3im);

        const __m256 yk_re = _mm256_add_ps(e0re, o0re), yk_im = _mm256_add_ps(e0im, o0im);
        const __m256 ykM_re = _mm256_add_ps(e1re, o1im), ykM_im = _mm256_sub_ps(e1im, o1re);
        const __m256 ykp_re = _mm256_sub_ps(e1re, o1im);
        const __m256 ykp_im = _mm256_sub_ps(zero, _mm256_add_ps(e1im, o1re));
        const __m256 ykpM_re = _mm256_sub_ps(e0re, o0re);
        const __m256 ykpM_im = _mm256_sub_ps(o0im, e0im);

        bodft_store8_ps(out, k, yk_re, yk_im);
        bodft_store8_ps(out, k + M, ykM_re, ykM_im);
        bodft_store8_ps(out, M - 8 - k, bodft_rev8_ps(ykp_re), bodft_rev8_ps(ykp_im));
        bodft_store8_ps(out, M - 8 - k + M, bodft_rev8_ps(ykpM_re), bodft_rev8_ps(ykpM_im));
    }
    return k;
}


static inline int bodft_combine_inv_avx2_f32(const complex_f32_t* RESTRICT tab,
                                             const complex_f32_t* RESTRICT tab2,
                                             const complex_f32_t* RESTRICT tab3,
                                             const complex_f32_t* RESTRICT in,
                                             complex_f32_t* RESTRICT c0,
                                             complex_f32_t* RESTRICT c1,
                                             complex_f32_t* RESTRICT c2,
                                             complex_f32_t* RESTRICT c3,
                                             int M, int half) {
    const __m256 halfv = _mm256_set1_ps(0.5f);
    const __m256 zero = _mm256_setzero_ps();
    int k = 0;
    for (; k + 8 <= half; k += 8) {
        __m256 tre, tim, t2re, t2im, t3re, t3im;
        __m256 ykre, ykim, ykMre, ykMim;
        __m256 ykpre, ykpim, ykpMre, ykpMim;
        bodft_uzp8_ps(tab, k, tre, tim);
        bodft_uzp8_ps(tab2, k, t2re, t2im);
        bodft_uzp8_ps(tab3, k, t3re, t3im);
        bodft_uzp8_ps(in, k, ykre, ykim);
        bodft_uzp8_ps(in, k + M, ykMre, ykMim);
        bodft_uzp8_ps(in, M - 8 - k, ykpre, ykpim);
        bodft_uzp8_ps(in, M - 8 - k + M, ykpMre, ykpMim);
        ykpre = bodft_rev8_ps(ykpre);
        ykpim = bodft_rev8_ps(ykpim);
        ykpMre = bodft_rev8_ps(ykpMre);
        ykpMim = bodft_rev8_ps(ykpMim);

        const __m256 e0re = _mm256_mul_ps(halfv, _mm256_add_ps(ykre, ykpMre));
        const __m256 e0im = _mm256_mul_ps(halfv, _mm256_sub_ps(ykim, ykpMim));
        const __m256 o0re = _mm256_mul_ps(halfv, _mm256_sub_ps(ykre, ykpMre));
        const __m256 o0im = _mm256_mul_ps(halfv, _mm256_add_ps(ykim, ykpMim));
        const __m256 e1re = _mm256_mul_ps(halfv, _mm256_add_ps(ykMre, ykpre));
        const __m256 e1im = _mm256_mul_ps(halfv, _mm256_sub_ps(ykMim, ykpim));
        const __m256 diff_re = _mm256_sub_ps(ykMre, ykpre);
        const __m256 diff_im = _mm256_add_ps(ykMim, ykpim);
        const __m256 o1re = _mm256_mul_ps(halfv, _mm256_sub_ps(zero, diff_im));
        const __m256 o1im = _mm256_mul_ps(halfv, diff_re);

        const __m256 b0re = _mm256_mul_ps(halfv, _mm256_add_ps(e0re, e1re));
        const __m256 b0im = _mm256_mul_ps(halfv, _mm256_add_ps(e0im, e1im));
        const __m256 b2re = _mm256_mul_ps(halfv, _mm256_sub_ps(e0re, e1re));
        const __m256 b2im = _mm256_mul_ps(halfv, _mm256_sub_ps(e0im, e1im));
        const __m256 b1re = _mm256_mul_ps(halfv, _mm256_add_ps(o0re, o1re));
        const __m256 b1im = _mm256_mul_ps(halfv, _mm256_add_ps(o0im, o1im));
        const __m256 b3re = _mm256_mul_ps(halfv, _mm256_sub_ps(o0re, o1re));
        const __m256 b3im = _mm256_mul_ps(halfv, _mm256_sub_ps(o0im, o1im));

        const __m256 c1re = _mm256_fmadd_ps(tim, b1im, _mm256_mul_ps(tre, b1re));
        const __m256 c1im = _mm256_fmsub_ps(tre, b1im, _mm256_mul_ps(tim, b1re));
        const __m256 c2re = _mm256_fmadd_ps(t2im, b2im, _mm256_mul_ps(t2re, b2re));
        const __m256 c2im = _mm256_fmsub_ps(t2re, b2im, _mm256_mul_ps(t2im, b2re));
        const __m256 c3re = _mm256_fmadd_ps(t3im, b3im, _mm256_mul_ps(t3re, b3re));
        const __m256 c3im = _mm256_fmsub_ps(t3re, b3im, _mm256_mul_ps(t3im, b3re));

        bodft_store8_ps(c0, k, b0re, b0im);
        bodft_store8_ps(c1, k, c1re, c1im);
        bodft_store8_ps(c2, k, c2re, c2im);
        bodft_store8_ps(c3, k, c3re, c3im);
    }
    return k;
}

static inline int bodft_combine_inv_avx2_f64(const complex_t* RESTRICT tab,
                                             const complex_t* RESTRICT tab2,
                                             const complex_t* RESTRICT tab3,
                                             const complex_t* RESTRICT in,
                                             complex_t* RESTRICT c0,
                                             complex_t* RESTRICT c1,
                                             complex_t* RESTRICT c2,
                                             complex_t* RESTRICT c3,
                                             int M, int half) {
    const __m256d halfv = _mm256_set1_pd(0.5);
    int k = 0;
    for (; k + 4 <= half; k += 4) {
        __m256d tre, tim, t2re, t2im, t3re, t3im;
        __m256d ykre, ykim, ykMre, ykMim;
        __m256d ykpre, ykpim, ykpMre, ykpMim;
        bodft_uzp4_pd(tab, k, tre, tim);
        bodft_uzp4_pd(tab2, k, t2re, t2im);
        bodft_uzp4_pd(tab3, k, t3re, t3im);
        bodft_uzp4_pd(in, k, ykre, ykim);
        bodft_uzp4_pd(in, k + M, ykMre, ykMim);
        bodft_uzp4_pd(in, M - 4 - k, ykpre, ykpim);
        bodft_uzp4_pd(in, M - 4 - k + M, ykpMre, ykpMim);
        ykpre = bodft_rev4_pd(ykpre);
        ykpim = bodft_rev4_pd(ykpim);
        ykpMre = bodft_rev4_pd(ykpMre);
        ykpMim = bodft_rev4_pd(ykpMim);

        const __m256d e0re = _mm256_mul_pd(halfv, _mm256_add_pd(ykre, ykpMre));
        const __m256d e0im = _mm256_mul_pd(halfv, _mm256_sub_pd(ykim, ykpMim));
        const __m256d o0re = _mm256_mul_pd(halfv, _mm256_sub_pd(ykre, ykpMre));
        const __m256d o0im = _mm256_mul_pd(halfv, _mm256_add_pd(ykim, ykpMim));
        const __m256d e1re = _mm256_mul_pd(halfv, _mm256_add_pd(ykMre, ykpre));
        const __m256d e1im = _mm256_mul_pd(halfv, _mm256_sub_pd(ykMim, ykpim));
        const __m256d diff_re = _mm256_sub_pd(ykMre, ykpre);
        const __m256d diff_im = _mm256_add_pd(ykMim, ykpim);
        const __m256d o1re = _mm256_mul_pd(halfv, _mm256_sub_pd(_mm256_setzero_pd(), diff_im));
        const __m256d o1im = _mm256_mul_pd(halfv, diff_re);

        const __m256d b0re = _mm256_mul_pd(halfv, _mm256_add_pd(e0re, e1re));
        const __m256d b0im = _mm256_mul_pd(halfv, _mm256_add_pd(e0im, e1im));
        const __m256d b2re = _mm256_mul_pd(halfv, _mm256_sub_pd(e0re, e1re));
        const __m256d b2im = _mm256_mul_pd(halfv, _mm256_sub_pd(e0im, e1im));
        const __m256d b1re = _mm256_mul_pd(halfv, _mm256_add_pd(o0re, o1re));
        const __m256d b1im = _mm256_mul_pd(halfv, _mm256_add_pd(o0im, o1im));
        const __m256d b3re = _mm256_mul_pd(halfv, _mm256_sub_pd(o0re, o1re));
        const __m256d b3im = _mm256_mul_pd(halfv, _mm256_sub_pd(o0im, o1im));

        const __m256d c1re = _mm256_fmadd_pd(tim, b1im, _mm256_mul_pd(tre, b1re));
        const __m256d c1im = _mm256_fmsub_pd(tre, b1im, _mm256_mul_pd(tim, b1re));
        const __m256d c2re = _mm256_fmadd_pd(t2im, b2im, _mm256_mul_pd(t2re, b2re));
        const __m256d c2im = _mm256_fmsub_pd(t2re, b2im, _mm256_mul_pd(t2im, b2re));
        const __m256d c3re = _mm256_fmadd_pd(t3im, b3im, _mm256_mul_pd(t3re, b3re));
        const __m256d c3im = _mm256_fmsub_pd(t3re, b3im, _mm256_mul_pd(t3im, b3re));

        bodft_store4_pd(c0, k, b0re, b0im);
        bodft_store4_pd(c1, k, c1re, c1im);
        bodft_store4_pd(c2, k, c2re, c2im);
        bodft_store4_pd(c3, k, c3re, c3im);
    }
    return k;
}
#endif

// ---------------------------------------------------------------------------
// Forward double combine, 128-bit SoA SIMD specialization. Two output positions
// k, k+1 per iteration; children and twiddles transpose into real/imag lane
// registers so the three products are pure lanewise FMA. Partner positions kp,
// kp-1 are written by address arithmetic. Returns the (even) count handled.
// ---------------------------------------------------------------------------
#if BRUUN_LEVEL >= 1
static inline int bodft_combine_fwd_simd_f64(const complex_t* RESTRICT tab,
                                             const complex_t* RESTRICT tab2,
                                             const complex_t* RESTRICT tab3,
                                             const complex_t* RESTRICT c0,
                                             const complex_t* RESTRICT c1,
                                             const complex_t* RESTRICT c2,
                                             const complex_t* RESTRICT c3,
                                             complex_t* RESTRICT out,
                                             int M, int half) {
    const bruun_v2 zero = V2_SET1(0.0);
    int k = 0;
    for (; k + 2 <= half; k += 2) {
        bruun_v2 a, b;
        a = V2_LD(&tab[k].re);  b = V2_LD(&tab[k + 1].re);
        const bruun_v2 tre = V2_UNPLO(a, b), tim = V2_UNPHI(a, b);
        a = V2_LD(&tab2[k].re); b = V2_LD(&tab2[k + 1].re);
        const bruun_v2 t2re = V2_UNPLO(a, b), t2im = V2_UNPHI(a, b);
        a = V2_LD(&tab3[k].re); b = V2_LD(&tab3[k + 1].re);
        const bruun_v2 t3re = V2_UNPLO(a, b), t3im = V2_UNPHI(a, b);

        a = V2_LD(&c0[k].re); b = V2_LD(&c0[k + 1].re);
        const bruun_v2 c0re = V2_UNPLO(a, b), c0im = V2_UNPHI(a, b);
        a = V2_LD(&c1[k].re); b = V2_LD(&c1[k + 1].re);
        const bruun_v2 c1re = V2_UNPLO(a, b), c1im = V2_UNPHI(a, b);
        a = V2_LD(&c2[k].re); b = V2_LD(&c2[k + 1].re);
        const bruun_v2 c2re = V2_UNPLO(a, b), c2im = V2_UNPHI(a, b);
        a = V2_LD(&c3[k].re); b = V2_LD(&c3[k + 1].re);
        const bruun_v2 c3re = V2_UNPLO(a, b), c3im = V2_UNPHI(a, b);

        const bruun_v2 b0re = c0re, b0im = c0im;
        const bruun_v2 b1re = V2_MSUB(V2_MUL(c1re, tre), c1im, tim);
        const bruun_v2 b1im = V2_MADD(V2_MUL(c1re, tim), c1im, tre);
        const bruun_v2 b2re = V2_MSUB(V2_MUL(c2re, t2re), c2im, t2im);
        const bruun_v2 b2im = V2_MADD(V2_MUL(c2re, t2im), c2im, t2re);
        const bruun_v2 b3re = V2_MSUB(V2_MUL(c3re, t3re), c3im, t3im);
        const bruun_v2 b3im = V2_MADD(V2_MUL(c3re, t3im), c3im, t3re);

        const bruun_v2 e0re = V2_ADD(b0re, b2re), e0im = V2_ADD(b0im, b2im);
        const bruun_v2 e1re = V2_SUB(b0re, b2re), e1im = V2_SUB(b0im, b2im);
        const bruun_v2 o0re = V2_ADD(b1re, b3re), o0im = V2_ADD(b1im, b3im);
        const bruun_v2 o1re = V2_SUB(b1re, b3re), o1im = V2_SUB(b1im, b3im);

        const bruun_v2 yk_re = V2_ADD(e0re, o0re), yk_im = V2_ADD(e0im, o0im);
        const bruun_v2 ykM_re = V2_ADD(e1re, o1im), ykM_im = V2_SUB(e1im, o1re);
        const bruun_v2 ykp_re = V2_SUB(e1re, o1im);
        const bruun_v2 ykp_im = V2_SUB(zero, V2_ADD(e1im, o1re));
        const bruun_v2 ykpM_re = V2_SUB(e0re, o0re);
        const bruun_v2 ykpM_im = V2_SUB(o0im, e0im);

        V2_ST(&out[k].re,         V2_UNPLO(yk_re, yk_im));
        V2_ST(&out[k + 1].re,     V2_UNPHI(yk_re, yk_im));
        V2_ST(&out[k + M].re,     V2_UNPLO(ykM_re, ykM_im));
        V2_ST(&out[k + 1 + M].re, V2_UNPHI(ykM_re, ykM_im));
        V2_ST(&out[M - 1 - k].re,     V2_UNPLO(ykp_re, ykp_im));
        V2_ST(&out[M - 2 - k].re,     V2_UNPHI(ykp_re, ykp_im));
        V2_ST(&out[M - 1 - k + M].re, V2_UNPLO(ykpM_re, ykpM_im));
        V2_ST(&out[M - 2 - k + M].re, V2_UNPHI(ykpM_re, ykpM_im));
    }
    return k;
}

// Forward float combine: same SoA transport flow, four positions per iteration
// through the 128-bit float vector. Structurally identical to the f64 path.
static inline int bodft_combine_fwd_simd_f32(const complex_f32_t* RESTRICT tab,
                                             const complex_f32_t* RESTRICT tab2,
                                             const complex_f32_t* RESTRICT tab3,
                                             const complex_f32_t* RESTRICT c0,
                                             const complex_f32_t* RESTRICT c1,
                                             const complex_f32_t* RESTRICT c2,
                                             const complex_f32_t* RESTRICT c3,
                                             complex_f32_t* RESTRICT out,
                                             int M, int half) {
    const bruun_v4f zero = V4F_ZERO();
    int k = 0;
    for (; k + 4 <= half; k += 4) {
        bruun_v4f A, B;
        A = V4F_LD(&tab[k].re);  B = V4F_LD(&tab[k + 2].re);
        const bruun_v4f tre = V4F_UZPRE(A, B), tim = V4F_UZPIM(A, B);
        A = V4F_LD(&tab2[k].re); B = V4F_LD(&tab2[k + 2].re);
        const bruun_v4f t2re = V4F_UZPRE(A, B), t2im = V4F_UZPIM(A, B);
        A = V4F_LD(&tab3[k].re); B = V4F_LD(&tab3[k + 2].re);
        const bruun_v4f t3re = V4F_UZPRE(A, B), t3im = V4F_UZPIM(A, B);

        A = V4F_LD(&c0[k].re); B = V4F_LD(&c0[k + 2].re);
        const bruun_v4f c0re = V4F_UZPRE(A, B), c0im = V4F_UZPIM(A, B);
        A = V4F_LD(&c1[k].re); B = V4F_LD(&c1[k + 2].re);
        const bruun_v4f c1re = V4F_UZPRE(A, B), c1im = V4F_UZPIM(A, B);
        A = V4F_LD(&c2[k].re); B = V4F_LD(&c2[k + 2].re);
        const bruun_v4f c2re = V4F_UZPRE(A, B), c2im = V4F_UZPIM(A, B);
        A = V4F_LD(&c3[k].re); B = V4F_LD(&c3[k + 2].re);
        const bruun_v4f c3re = V4F_UZPRE(A, B), c3im = V4F_UZPIM(A, B);

        const bruun_v4f b0re = c0re, b0im = c0im;
        const bruun_v4f b1re = V4F_MSUB(V4F_MUL(c1re, tre), c1im, tim);
        const bruun_v4f b1im = V4F_MADD(V4F_MUL(c1re, tim), c1im, tre);
        const bruun_v4f b2re = V4F_MSUB(V4F_MUL(c2re, t2re), c2im, t2im);
        const bruun_v4f b2im = V4F_MADD(V4F_MUL(c2re, t2im), c2im, t2re);
        const bruun_v4f b3re = V4F_MSUB(V4F_MUL(c3re, t3re), c3im, t3im);
        const bruun_v4f b3im = V4F_MADD(V4F_MUL(c3re, t3im), c3im, t3re);

        const bruun_v4f e0re = V4F_ADD(b0re, b2re), e0im = V4F_ADD(b0im, b2im);
        const bruun_v4f e1re = V4F_SUB(b0re, b2re), e1im = V4F_SUB(b0im, b2im);
        const bruun_v4f o0re = V4F_ADD(b1re, b3re), o0im = V4F_ADD(b1im, b3im);
        const bruun_v4f o1re = V4F_SUB(b1re, b3re), o1im = V4F_SUB(b1im, b3im);

        const bruun_v4f yk_re = V4F_ADD(e0re, o0re), yk_im = V4F_ADD(e0im, o0im);
        const bruun_v4f ykM_re = V4F_ADD(e1re, o1im), ykM_im = V4F_SUB(e1im, o1re);
        const bruun_v4f ykp_re = V4F_SUB(e1re, o1im);
        const bruun_v4f ykp_im = V4F_SUB(zero, V4F_ADD(e1im, o1re));
        const bruun_v4f ykpM_re = V4F_SUB(e0re, o0re);
        const bruun_v4f ykpM_im = V4F_SUB(o0im, e0im);

        V4F_ST(&out[k].re,         V4F_ZIPLO(yk_re, yk_im));
        V4F_ST(&out[k + 2].re,     V4F_ZIPHI(yk_re, yk_im));
        V4F_ST(&out[k + M].re,     V4F_ZIPLO(ykM_re, ykM_im));
        V4F_ST(&out[k + 2 + M].re, V4F_ZIPHI(ykM_re, ykM_im));
        const bruun_v4f pr = V4F_REV(ykp_re),  pi = V4F_REV(ykp_im);
        const bruun_v4f qr = V4F_REV(ykpM_re), qi = V4F_REV(ykpM_im);
        V4F_ST(&out[M - 4 - k].re,     V4F_ZIPLO(pr, pi));
        V4F_ST(&out[M - 2 - k].re,     V4F_ZIPHI(pr, pi));
        V4F_ST(&out[M - 4 - k + M].re, V4F_ZIPLO(qr, qi));
        V4F_ST(&out[M - 2 - k + M].re, V4F_ZIPHI(qr, qi));
    }
    return k;
}
#endif

// One forward radix-4 combine: 4 child blocks c0..c3 (each `half` complex) ->
// one parent block out (2*M complex). SIMD bulk for double/float plus the exact
// scalar tail/reference.
template <class CT, class RT>
static inline void combine_fwd(const CT* RESTRICT tab, const CT* RESTRICT tab2,
                               const CT* RESTRICT tab3, const CT* RESTRICT c0,
                               const CT* RESTRICT c1, const CT* RESTRICT c2,
                               const CT* RESTRICT c3, CT* RESTRICT out,
                               int M, int half) {
    int k = 0;
#if BRUUN_LEVEL >= 2
    if constexpr (sizeof(RT) == 8) {
        k = bodft_combine_fwd_avx2_f64(
            reinterpret_cast<const complex_t*>(tab),
            reinterpret_cast<const complex_t*>(tab2),
            reinterpret_cast<const complex_t*>(tab3),
            reinterpret_cast<const complex_t*>(c0),
            reinterpret_cast<const complex_t*>(c1),
            reinterpret_cast<const complex_t*>(c2),
            reinterpret_cast<const complex_t*>(c3),
            reinterpret_cast<complex_t*>(out), M, half);
    }
#endif
#if BRUUN_LEVEL >= 2
    if constexpr (sizeof(RT) == 4) {
        k = bodft_combine_fwd_avx2_f32(
            reinterpret_cast<const complex_f32_t*>(tab),
            reinterpret_cast<const complex_f32_t*>(tab2),
            reinterpret_cast<const complex_f32_t*>(tab3),
            reinterpret_cast<const complex_f32_t*>(c0),
            reinterpret_cast<const complex_f32_t*>(c1),
            reinterpret_cast<const complex_f32_t*>(c2),
            reinterpret_cast<const complex_f32_t*>(c3),
            reinterpret_cast<complex_f32_t*>(out), M, half);
    }
#endif
#if BRUUN_LEVEL >= 1
    if constexpr (sizeof(RT) == 8) {
        if (k == 0) k = bodft_combine_fwd_simd_f64(
            reinterpret_cast<const complex_t*>(tab),
            reinterpret_cast<const complex_t*>(tab2),
            reinterpret_cast<const complex_t*>(tab3),
            reinterpret_cast<const complex_t*>(c0),
            reinterpret_cast<const complex_t*>(c1),
            reinterpret_cast<const complex_t*>(c2),
            reinterpret_cast<const complex_t*>(c3),
            reinterpret_cast<complex_t*>(out), M, half);
    } else {
        if (k == 0) {
            k = bodft_combine_fwd_simd_f32(
                reinterpret_cast<const complex_f32_t*>(tab),
                reinterpret_cast<const complex_f32_t*>(tab2),
                reinterpret_cast<const complex_f32_t*>(tab3),
                reinterpret_cast<const complex_f32_t*>(c0),
                reinterpret_cast<const complex_f32_t*>(c1),
                reinterpret_cast<const complex_f32_t*>(c2),
                reinterpret_cast<const complex_f32_t*>(c3),
                reinterpret_cast<complex_f32_t*>(out), M, half);
        }
    }
#endif
    for (; k < half; ++k) {
        const CT t = tab[k];
        const CT b0 = c0[k];
        const CT b1 = cmul(t, c1[k]);
        const CT b2 = cmul(tab2[k], c2[k]);
        const CT b3 = cmul(tab3[k], c3[k]);
        const CT e0 = cadd(b0, b2);
        const CT e1 = csub(b0, b2);
        const CT o0 = cadd(b1, b3);
        const CT o1 = csub(b1, b3);
        const int kp = M - 1 - k;
        const CT io1 = cmuli(o1);
        out[k]      = cadd(e0, o0);
        out[k + M]  = csub(e1, io1);
        out[kp]     = cconj(cadd(e1, io1));
        out[kp + M] = cconj(csub(e0, o0));
    }
}

// One inverse radix-4 combine: parent block in (2*M complex) -> 4 child blocks
// c0..c3 (each `half`). Exact algebraic inverse of combine_fwd. Scalar in both
// precisions.
template <class CT, class RT>
static inline void combine_inv(const CT* RESTRICT tab, const CT* RESTRICT tab2,
                               const CT* RESTRICT tab3, const CT* RESTRICT in,
                               CT* RESTRICT c0, CT* RESTRICT c1, CT* RESTRICT c2,
                               CT* RESTRICT c3, int M, int half) {
    int k = 0;
#if BRUUN_LEVEL >= 2
    if constexpr (sizeof(RT) == 8) {
        k = bodft_combine_inv_avx2_f64(
            reinterpret_cast<const complex_t*>(tab),
            reinterpret_cast<const complex_t*>(tab2),
            reinterpret_cast<const complex_t*>(tab3),
            reinterpret_cast<const complex_t*>(in),
            reinterpret_cast<complex_t*>(c0),
            reinterpret_cast<complex_t*>(c1),
            reinterpret_cast<complex_t*>(c2),
            reinterpret_cast<complex_t*>(c3), M, half);
    }
    if constexpr (sizeof(RT) == 4) {
        k = bodft_combine_inv_avx2_f32(
            reinterpret_cast<const complex_f32_t*>(tab),
            reinterpret_cast<const complex_f32_t*>(tab2),
            reinterpret_cast<const complex_f32_t*>(tab3),
            reinterpret_cast<const complex_f32_t*>(in),
            reinterpret_cast<complex_f32_t*>(c0),
            reinterpret_cast<complex_f32_t*>(c1),
            reinterpret_cast<complex_f32_t*>(c2),
            reinterpret_cast<complex_f32_t*>(c3), M, half);
    }
#endif
    const RT h = static_cast<RT>(0.5);
    for (; k < half; ++k) {
        const int kp = M - 1 - k;
        const CT Yk   = in[k];
        const CT YkM  = in[k + M];
        const CT Ykp  = in[kp];
        const CT YkpM = in[kp + M];

        const CT e0 = cmul_r(cadd(Yk, cconj(YkpM)), h);
        const CT o0 = cmul_r(csub(Yk, cconj(YkpM)), h);
        const CT e1 = cmul_r(cadd(YkM, cconj(Ykp)), h);
        const CT o1 = cmul_r(cmuli(csub(YkM, cconj(Ykp))), h);

        const CT b0 = cmul_r(cadd(e0, e1), h);
        const CT b2 = cmul_r(csub(e0, e1), h);
        const CT b1 = cmul_r(cadd(o0, o1), h);
        const CT b3 = cmul_r(csub(o0, o1), h);

        c0[k] = b0;
        c1[k] = cmul(cconj(tab[k]), b1);
        c2[k] = cmul(cconj(tab2[k]), b2);
        c3[k] = cmul(cconj(tab3[k]), b3);
    }
}

// ---------------------------------------------------------------------------
// plan_t: reusable BODFT of a single power-of-two size N (N >= 2). Forward maps
// N real samples to N/2 packed complex bins; inverse maps them back exactly (the
// packed transform is its own exact inverse, no 1/N scaling). CT is the complex
// element type (complex_t or complex_f32_t) and RT its scalar.
// ---------------------------------------------------------------------------
template <class CT, class RT>
class plan_t {
public:
    explicit plan_t(int n) : n_(n) {
        BRUUN_ASSERT(is_power2(n) && n >= 2);
        const int p = ilog2_pow2(n_);
        leaf4_ = ((p & 1) == 0);            // p even -> radix-4 leaves, else radix-2
        leafsize_ = leaf4_ ? 4 : 2;
        leafbins_ = leafsize_ / 2;
        numleaves_ = n_ / leafsize_;
        passes_ = 0;
        for (int s = leafsize_ * 4; s <= n_; s <<= 2) ++passes_;

        const double a4 = -M_PI / 4.0;       // radix-4 leaf root: exp(-i*pi/4)
        root4_ = CT{static_cast<RT>(std::cos(a4)), static_cast<RT>(std::sin(a4))};

        // Per-level twiddle powers t, t^2, t^3 with t_k = exp(-2*pi*i*(k+1/2)/s),
        // k = 0 .. s/8-1; each power built straight from libm at the multiplied
        // angle so the inner loop is three muls with no derivation chain.
        for (int s = 8; s <= n_; s <<= 1) {
            const int lg = ilog2_pow2(s);
            const int len = s / 8;
            tw_[lg].resize(static_cast<std::size_t>(len));
            tw2_[lg].resize(static_cast<std::size_t>(len));
            tw3_[lg].resize(static_cast<std::size_t>(len));
            for (int k = 0; k < len; ++k) {
                const double a = -2.0 * M_PI * (static_cast<double>(k) + 0.5) /
                                 static_cast<double>(s);
                tw_[lg][static_cast<std::size_t>(k)] =
                    CT{static_cast<RT>(std::cos(a)), static_cast<RT>(std::sin(a))};
                tw2_[lg][static_cast<std::size_t>(k)] =
                    CT{static_cast<RT>(std::cos(2.0 * a)), static_cast<RT>(std::sin(2.0 * a))};
                tw3_[lg][static_cast<std::size_t>(k)] =
                    CT{static_cast<RT>(std::cos(3.0 * a)), static_cast<RT>(std::sin(3.0 * a))};
            }
        }

        // Radix-4 digit-reversal permutation: the order in which leaves consume
        // input samples, generated once from the decimation structure.
        perm_.resize(static_cast<std::size_t>(n_));
        int idx = 0;
        gen_perm(0, 1, n_, idx);

        // Two ping-pong work buffers of N/2 complex each.
        scratch_.resize(static_cast<std::size_t>(n_));
    }

    int size() const noexcept { return n_; }
    int bins() const noexcept { return n_ / 2; }

    // Forward: real input (length N) -> packed complex (length N/2).
    void forward(const RT* RESTRICT input, CT* RESTRICT output) const {
        const int bins = n_ / 2;
        if (passes_ == 0) {                 // N == leafsize: leaf is the whole transform
            leaf_forward(input, output);
            return;
        }
        CT* bufA = scratch_.data();
        CT* bufB = scratch_.data() + bins;
        leaf_forward(input, bufA);

        const CT* cur_in = bufA;
        CT* cur_out = bufB;
        for (int s = leafsize_ * 4; s <= n_; s <<= 2) {
            int next_s = s << 2;
            if (next_s > n_) {
                cur_out = output;
            }
            combine_pass_fwd(cur_in, cur_out, s);
            cur_in = cur_out;
            if (cur_out == output) {
                break;
            }
            if (cur_out == bufA) {
                cur_out = bufB;
            } else {
                cur_out = bufA;
            }
        }
    }

    // Inverse: packed complex (length N/2) -> real output (length N).
    void inverse(const CT* RESTRICT input, RT* RESTRICT output) const {
        const int bins = n_ / 2;
        if (passes_ == 0) {
            leaf_inverse(input, output);
            return;
        }
        CT* bufA = scratch_.data();
        CT* bufB = scratch_.data() + bins;

        const CT* cur_in = input;
        CT* cur_out = bufA;
        for (int s = n_; s >= leafsize_ * 4; s >>= 2) {
            combine_pass_inv(cur_in, cur_out, s);
            cur_in = cur_out;
            if (cur_out == bufA) {
                cur_out = bufB;
            } else {
                cur_out = bufA;
            }
        }
        leaf_inverse(cur_in, output);
    }

private:
    void gen_perm(int base, int stride, int N, int& idx) const {
        if (N == 2) {
            perm_[static_cast<std::size_t>(idx++)] = base;
            perm_[static_cast<std::size_t>(idx++)] = base + stride;
            return;
        }
        if (N == 4) {
            for (int j = 0; j < 4; ++j)
                perm_[static_cast<std::size_t>(idx++)] = base + j * stride;
            return;
        }
        for (int r = 0; r < 4; ++r) gen_perm(base + r * stride, stride * 4, N / 4, idx);
    }

    void leaf_forward(const RT* RESTRICT x, CT* RESTRICT dst) const {
        if (leafbins_ == 1) {               // radix-2 leaf: H[0] = x0 - i*x1
            for (int L = 0; L < numleaves_; ++L) {
                const int i0 = perm_[static_cast<std::size_t>(2 * L)];
                const int i1 = perm_[static_cast<std::size_t>(2 * L + 1)];
                dst[L] = CT{x[i0], -x[i1]};
            }
        } else {                            // radix-4 leaf
            const CT t = root4_, t2 = cmul(t, t), t3 = cmul(t2, t);
            for (int L = 0; L < numleaves_; ++L) {
                const int* p = &perm_[static_cast<std::size_t>(4 * L)];
                const CT b0 = CT{x[p[0]], static_cast<RT>(0)};
                const CT b1 = cmul_r(t, x[p[1]]);
                const CT b2 = cmul_r(t2, x[p[2]]);
                const CT b3 = cmul_r(t3, x[p[3]]);
                dst[2 * L]     = cadd(cadd(b0, b1), cadd(b2, b3));
                dst[2 * L + 1] = cadd(csub(b0, b2), cmuli(csub(b3, b1)));
            }
        }
    }

    void leaf_inverse(const CT* RESTRICT src, RT* RESTRICT x) const {
        if (leafbins_ == 1) {
            for (int L = 0; L < numleaves_; ++L) {
                x[perm_[static_cast<std::size_t>(2 * L)]] = src[L].re;
                x[perm_[static_cast<std::size_t>(2 * L + 1)]] = -src[L].im;
            }
        } else {
            const CT t = root4_, t2 = cmul(t, t), t3 = cmul(t2, t);
            const RT h = static_cast<RT>(0.5);
            for (int L = 0; L < numleaves_; ++L) {
                const CT Yk = src[2 * L], YkM = src[2 * L + 1];
                const CT e0 = cmul_r(cadd(Yk, cconj(YkM)), h);
                const CT o0 = cmul_r(csub(Yk, cconj(YkM)), h);
                const CT e1 = cmul_r(cadd(YkM, cconj(Yk)), h);
                const CT o1 = cmul_r(cmuli(csub(YkM, cconj(Yk))), h);
                const CT b0 = cmul_r(cadd(e0, e1), h);
                const CT b2 = cmul_r(csub(e0, e1), h);
                const CT b1 = cmul_r(cadd(o0, o1), h);
                const CT b3 = cmul_r(csub(o0, o1), h);
                const int* p = &perm_[static_cast<std::size_t>(4 * L)];
                x[p[0]] = b0.re;
                x[p[1]] = cmul(cconj(t), b1).re;
                x[p[2]] = cmul(cconj(t2), b2).re;
                x[p[3]] = cmul(cconj(t3), b3).re;
            }
        }
    }

    // Forward combine pass at parent size s: src holds child blocks, dst parents.
    void combine_pass_fwd(const CT* RESTRICT src, CT* RESTRICT dst, int s) const {
        const int lg = ilog2_pow2(s);
        const int M = s / 4;
        const int cb = s / 8;               // child packed length = half
        const CT* tab = tw_[lg].data();
        const CT* tab2 = tw2_[lg].data();
        const CT* tab3 = tw3_[lg].data();
        const int nump = n_ / s;
        const int pblk = s / 2;
        for (int g = 0; g < nump; ++g) {
            const int off = g * pblk;
            combine_fwd<CT, RT>(tab, tab2, tab3, src + off, src + off + cb,
                                src + off + 2 * cb, src + off + 3 * cb,
                                dst + off, M, cb);
        }
    }

    // Inverse combine pass at parent size s: src holds parents, dst child blocks.
    void combine_pass_inv(const CT* RESTRICT src, CT* RESTRICT dst, int s) const {
        const int lg = ilog2_pow2(s);
        const int M = s / 4;
        const int cb = s / 8;
        const CT* tab = tw_[lg].data();
        const CT* tab2 = tw2_[lg].data();
        const CT* tab3 = tw3_[lg].data();
        const int nump = n_ / s;
        const int pblk = s / 2;
        for (int g = 0; g < nump; ++g) {
            const int off = g * pblk;
            combine_inv<CT, RT>(tab, tab2, tab3, src + off, dst + off,
                                dst + off + cb, dst + off + 2 * cb,
                                dst + off + 3 * cb, M, cb);
        }
    }

    int n_;
    bool leaf4_;
    int leafsize_;
    int leafbins_;
    int numleaves_;
    int passes_;
    CT root4_;
    heap_array<CT> tw_[32];
    heap_array<CT> tw2_[32];
    heap_array<CT> tw3_[32];
    mutable heap_array<int> perm_;
    mutable heap_array<CT> scratch_;
};

using plan = plan_t<complex_t, double>;
using plan_f32 = plan_t<complex_f32_t, float>;

} // namespace bodft
