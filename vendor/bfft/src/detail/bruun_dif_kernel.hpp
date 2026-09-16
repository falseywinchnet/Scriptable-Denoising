#pragma once

// MIT joshuah.rainstar@gmail.com 2026
// Internal normalized-basis Bruun RFFT kernel.
//
// BFFT builds this kernel as part of the library. Public users select transform
// size and output layout through include/bfft/bfft.h or include/bfft/bfft.hpp;
// the Makefile chooses safe host compiler switches automatically. Heap-optimized
// native ordering and fused scatter are the default hot path. The library keeps a
// sequential two-phase standard-output pack available internally and uses it only
// when the standard FFT-order output is large enough to win on the active SIMD
// backend.

#include "bruun_simd_backend.hpp"

#include <algorithm>
#include <cstring>

namespace bruun {

#include "MAG_REPRESENT_KERNEL.hpp"

struct FwdOp {
    uint32_t base;
    uint32_t q;
    uint32_t m;
    uint16_t kind;
};

enum FwdOpKind : uint16_t {
    FWD_OP_NORM2 = 1,
    FWD_OP_CODELET_Q8 = 2,
    FWD_OP_CODELET_D3 = 3,
    FWD_OP_BINOMIAL = 4,
    FWD_OP_SPINE_D3 = 5,
    FWD_OP_SPINE_D2 = 6,
    FWD_OP_SPINE_D1 = 7,
    FWD_OP_SPINE_LEAF = 8,
    FWD_OP_DC_NYQUIST = 9,
    FWD_OP_SPINE_NORM2 = 10
};

static_assert(sizeof(FwdOp) == 16, "FwdOp is kept compact for hot schedule streaming");

// Small explicit traversal records keep depth-first segment ownership visible
// to the program instead of relying on recursive calls through inline member
// symbols. The stack bound covers every supported int-sized power-of-two plan.

// ---------------------------------------------------------------------------
// Streaming kernels. Each has an optional 256 block,
// a 2-lane block for the 128-bit backend, and an exact scalar tail.
// ---------------------------------------------------------------------------

static inline void binomial_fwd(double* RESTRICT v, int h) {
    int i = 0;

#if BRUUN_LEVEL >= 2
    for (; i + 3 < h; i += 4) {
        const bruun_v4d a = V4D_LD(v + i);
        const bruun_v4d b = V4D_LD(v + h + i);
        V4D_ST(v + i, V4D_ADD(a, b));
        V4D_ST(v + h + i, V4D_SUB(a, b));
    }
#endif
#if BRUUN_LEVEL >= 1
    for (; i + 1 < h; i += 2) {
        const bruun_v2 a = V2_LD(v + i);
        const bruun_v2 b = V2_LD(v + h + i);
        V2_ST(v + i, V2_ADD(a, b));
        V2_ST(v + h + i, V2_SUB(a, b));
    }
#endif

    for (; i < h; ++i) {
        const double a = v[i];
        const double b = v[h + i];
        v[i] = a + b;
        v[h + i] = a - b;
    }
}

// Fused "copy input + first binomial split": v[i] = in[i] + in[h+i], v[h+i] = in[i] - in[h+i].
static inline void binomial_oop(const double* RESTRICT in, double* RESTRICT v, int h) {
    int i = 0;

#if BRUUN_LEVEL >= 2
    for (; i + 3 < h; i += 4) {
        const bruun_v4d a = V4D_LD(in + i);
        const bruun_v4d b = V4D_LD(in + h + i);
        V4D_ST(v + i, V4D_ADD(a, b));
        V4D_ST(v + h + i, V4D_SUB(a, b));
    }
#endif
#if BRUUN_LEVEL >= 1
    for (; i + 1 < h; i += 2) {
        const bruun_v2 a = V2_LD(in + i);
        const bruun_v2 b = V2_LD(in + h + i);
        V2_ST(v + i, V2_ADD(a, b));
        V2_ST(v + h + i, V2_SUB(a, b));
    }
#endif

    for (; i < h; ++i) {
        const double a = in[i];
        const double b = in[h + i];
        v[i] = a + b;
        v[h + i] = a - b;
    }
}

static inline void binomial_inv(double* RESTRICT v, int h) {
    int i = 0;

#if BRUUN_LEVEL >= 2
    const bruun_v4d half4 = V4D_SET1(0.5);
    for (; i + 3 < h; i += 4) {
        const bruun_v4d a = V4D_LD(v + i);
        const bruun_v4d b = V4D_LD(v + h + i);
        V4D_ST(v + i, V4D_MUL(half4, V4D_ADD(a, b)));
        V4D_ST(v + h + i, V4D_MUL(half4, V4D_SUB(a, b)));
    }
#endif
#if BRUUN_LEVEL >= 1
    const bruun_v2 half2 = V2_SET1(0.5);
    for (; i + 1 < h; i += 2) {
        const bruun_v2 a = V2_LD(v + i);
        const bruun_v2 b = V2_LD(v + h + i);
        V2_ST(v + i, V2_MUL(half2, V2_ADD(a, b)));
        V2_ST(v + h + i, V2_MUL(half2, V2_SUB(a, b)));
    }
#endif

    for (; i < h; ++i) {
        const double a = v[i];
        const double b = v[h + i];
        v[i] = 0.5 * (a + b);
        v[h + i] = 0.5 * (a - b);
    }
}

// One normalized-quadratic split: block [A0|B0|A1|B1] of quarters q.
static inline void norm_q_fwd(double* RESTRICT p, int q, double c_scalar, double s_scalar) {
    double* RESTRICT A0p = p;
    double* RESTRICT B0p = p + q;
    double* RESTRICT A1p = p + 2*q;
    double* RESTRICT B1p = p + 3*q;

    int n = 0;

#if BRUUN_LEVEL >= 2
    {
        const bruun_v4d vc = V4D_SET1(c_scalar);
        const bruun_v4d vs = V4D_SET1(s_scalar);

        for (; n + 3 < q; n += 4) {
            const bruun_v4d A0 = V4D_LD(A0p + n);
            const bruun_v4d B0 = V4D_LD(B0p + n);
            const bruun_v4d A1 = V4D_LD(A1p + n);
            const bruun_v4d B1 = V4D_LD(B1p + n);

            const bruun_v4d R = V4D_MSUB(V4D_MUL(vc, B0), vs, B1);
            const bruun_v4d I = V4D_MADD(V4D_MUL(vs, B0), vc, B1);

            V4D_ST(A0p + n, V4D_ADD(A0, R));
            V4D_ST(B0p + n, V4D_ADD(A1, I));
            V4D_ST(A1p + n, V4D_SUB(A0, R));
            V4D_ST(B1p + n, V4D_SUB(I, A1));
        }
    }
#endif
#if BRUUN_LEVEL >= 1
    {
        const bruun_v2 vc = V2_SET1(c_scalar);
        const bruun_v2 vs = V2_SET1(s_scalar);

        for (; n + 1 < q; n += 2) {
            const bruun_v2 A0 = V2_LD(A0p + n);
            const bruun_v2 B0 = V2_LD(B0p + n);
            const bruun_v2 A1 = V2_LD(A1p + n);
            const bruun_v2 B1 = V2_LD(B1p + n);

            const bruun_v2 R = V2_MSUB(V2_MUL(vc, B0), vs, B1);
            const bruun_v2 I = V2_MADD(V2_MUL(vs, B0), vc, B1);

            V2_ST(A0p + n, V2_ADD(A0, R));
            V2_ST(B0p + n, V2_ADD(A1, I));
            V2_ST(A1p + n, V2_SUB(A0, R));
            V2_ST(B1p + n, V2_SUB(I, A1));
        }
    }
#endif

    for (; n < q; ++n) {
        const double A0 = A0p[n];
        const double B0 = B0p[n];
        const double A1 = A1p[n];
        const double B1 = B1p[n];

        const double R = c_scalar * B0 - s_scalar * B1;
        const double I = s_scalar * B0 + c_scalar * B1;

        A0p[n] = A0 + R;
        B0p[n] = A1 + I;
        A1p[n] = A0 - R;
        B1p[n] = -A1 + I;
    }
}

// Two tree levels in one pass: parent rotation (c,s) plus both child rotations
// (c0,s0), (c1,s1) applied while the data is in registers. Halves the load/store
// traffic of the norm cascade. valid for q2 spine fusion
static inline void norm2_fused(double* RESTRICT p, int q,
                               double c, double s,
                               double c0, double s0,
                               double c1, double s1) {
    const int qh = q >> 1;
    double* RESTRICT A0 = p;
    double* RESTRICT B0 = p + q;
    double* RESTRICT A1 = p + 2*q;
    double* RESTRICT B1 = p + 3*q;

    int n = 0;

#if BRUUN_LEVEL >= 2
    {
        const bruun_v4d vc  = V4D_SET1(c),  vs  = V4D_SET1(s);
        const bruun_v4d vc0 = V4D_SET1(c0), vs0 = V4D_SET1(s0);
        const bruun_v4d vc1 = V4D_SET1(c1), vs1 = V4D_SET1(s1);

        for (; n + 3 < qh; n += 4) {
            const bruun_v4d a0n = V4D_LD(A0 + n);
            const bruun_v4d a0h = V4D_LD(A0 + qh + n);
            const bruun_v4d b0n = V4D_LD(B0 + n);
            const bruun_v4d b0h = V4D_LD(B0 + qh + n);
            const bruun_v4d a1n = V4D_LD(A1 + n);
            const bruun_v4d a1h = V4D_LD(A1 + qh + n);
            const bruun_v4d b1n = V4D_LD(B1 + n);
            const bruun_v4d b1h = V4D_LD(B1 + qh + n);

            const bruun_v4d Rn = V4D_MSUB(V4D_MUL(vc, b0n), vs, b1n);
            const bruun_v4d In = V4D_MADD(V4D_MUL(vs, b0n), vc, b1n);
            const bruun_v4d Rh = V4D_MSUB(V4D_MUL(vc, b0h), vs, b1h);
            const bruun_v4d Ih = V4D_MADD(V4D_MUL(vs, b0h), vc, b1h);

            const bruun_v4d u0 = V4D_ADD(a0n, Rn);
            const bruun_v4d uh = V4D_ADD(a0h, Rh);
            const bruun_v4d w0 = V4D_ADD(a1n, In);
            const bruun_v4d wh = V4D_ADD(a1h, Ih);
            const bruun_v4d v0 = V4D_SUB(a0n, Rn);
            const bruun_v4d vh = V4D_SUB(a0h, Rh);
            const bruun_v4d x0 = V4D_SUB(In, a1n);
            const bruun_v4d xh = V4D_SUB(Ih, a1h);

            const bruun_v4d R0 = V4D_MSUB(V4D_MUL(vc0, uh), vs0, wh);
            const bruun_v4d I0 = V4D_MADD(V4D_MUL(vs0, uh), vc0, wh);
            const bruun_v4d R1 = V4D_MSUB(V4D_MUL(vc1, vh), vs1, xh);
            const bruun_v4d I1 = V4D_MADD(V4D_MUL(vs1, vh), vc1, xh);

            V4D_ST(A0 + n,      V4D_ADD(u0, R0));
            V4D_ST(A0 + qh + n, V4D_ADD(w0, I0));
            V4D_ST(B0 + n,      V4D_SUB(u0, R0));
            V4D_ST(B0 + qh + n, V4D_SUB(I0, w0));
            V4D_ST(A1 + n,      V4D_ADD(v0, R1));
            V4D_ST(A1 + qh + n, V4D_ADD(x0, I1));
            V4D_ST(B1 + n,      V4D_SUB(v0, R1));
            V4D_ST(B1 + qh + n, V4D_SUB(I1, x0));
        }
    }
#endif
#if BRUUN_LEVEL >= 1
    {
        const bruun_v2 vc  = V2_SET1(c),  vs  = V2_SET1(s);
        const bruun_v2 vc0 = V2_SET1(c0), vs0 = V2_SET1(s0);
        const bruun_v2 vc1 = V2_SET1(c1), vs1 = V2_SET1(s1);

        for (; n + 1 < qh; n += 2) {
            const bruun_v2 a0n = V2_LD(A0 + n);
            const bruun_v2 a0h = V2_LD(A0 + qh + n);
            const bruun_v2 b0n = V2_LD(B0 + n);
            const bruun_v2 b0h = V2_LD(B0 + qh + n);
            const bruun_v2 a1n = V2_LD(A1 + n);
            const bruun_v2 a1h = V2_LD(A1 + qh + n);
            const bruun_v2 b1n = V2_LD(B1 + n);
            const bruun_v2 b1h = V2_LD(B1 + qh + n);

            const bruun_v2 Rn = V2_MSUB(V2_MUL(vc, b0n), vs, b1n);
            const bruun_v2 In = V2_MADD(V2_MUL(vs, b0n), vc, b1n);
            const bruun_v2 Rh = V2_MSUB(V2_MUL(vc, b0h), vs, b1h);
            const bruun_v2 Ih = V2_MADD(V2_MUL(vs, b0h), vc, b1h);

            const bruun_v2 u0 = V2_ADD(a0n, Rn);
            const bruun_v2 uh = V2_ADD(a0h, Rh);
            const bruun_v2 w0 = V2_ADD(a1n, In);
            const bruun_v2 wh = V2_ADD(a1h, Ih);
            const bruun_v2 v0 = V2_SUB(a0n, Rn);
            const bruun_v2 vh = V2_SUB(a0h, Rh);
            const bruun_v2 x0 = V2_SUB(In, a1n);
            const bruun_v2 xh = V2_SUB(Ih, a1h);

            const bruun_v2 R0 = V2_MSUB(V2_MUL(vc0, uh), vs0, wh);
            const bruun_v2 I0 = V2_MADD(V2_MUL(vs0, uh), vc0, wh);
            const bruun_v2 R1 = V2_MSUB(V2_MUL(vc1, vh), vs1, xh);
            const bruun_v2 I1 = V2_MADD(V2_MUL(vs1, vh), vc1, xh);

            V2_ST(A0 + n,      V2_ADD(u0, R0));
            V2_ST(A0 + qh + n, V2_ADD(w0, I0));
            V2_ST(B0 + n,      V2_SUB(u0, R0));
            V2_ST(B0 + qh + n, V2_SUB(I0, w0));
            V2_ST(A1 + n,      V2_ADD(v0, R1));
            V2_ST(A1 + qh + n, V2_ADD(x0, I1));
            V2_ST(B1 + n,      V2_SUB(v0, R1));
            V2_ST(B1 + qh + n, V2_SUB(I1, x0));
        }
    }
#endif

    for (; n < qh; ++n) {
        const double a0n = A0[n],      a0h = A0[qh + n];
        const double b0n = B0[n],      b0h = B0[qh + n];
        const double a1n = A1[n],      a1h = A1[qh + n];
        const double b1n = B1[n],      b1h = B1[qh + n];

        const double Rn = c * b0n - s * b1n;
        const double In = s * b0n + c * b1n;
        const double Rh = c * b0h - s * b1h;
        const double Ih = s * b0h + c * b1h;

        const double u0 = a0n + Rn, uh = a0h + Rh;
        const double w0 = a1n + In, wh = a1h + Ih;
        const double v0 = a0n - Rn, vh = a0h - Rh;
        const double x0 = In - a1n, xh = Ih - a1h;

        const double R0 = c0 * uh - s0 * wh;
        const double I0 = s0 * uh + c0 * wh;
        const double R1 = c1 * vh - s1 * xh;
        const double I1 = s1 * vh + c1 * xh;

        A0[n] = u0 + R0;
        A0[qh + n] = w0 + I0;
        B0[n] = u0 - R0;
        B0[qh + n] = I0 - w0;
        A1[n] = v0 + R1;
        A1[qh + n] = x0 + I1;
        B1[n] = v0 - R1;
        B1[qh + n] = I1 - x0;
    }
}

static inline void norm_q_inv(double* RESTRICT p, int q, double c_scalar, double s_scalar) {
    double* RESTRICT C0p = p;
    double* RESTRICT C1p = p + q;
    double* RESTRICT D0p = p + 2*q;
    double* RESTRICT D1p = p + 3*q;

    int n = 0;

#if BRUUN_LEVEL >= 2
    {
        const bruun_v4d half = V4D_SET1(0.5);
        const bruun_v4d vc = V4D_SET1(c_scalar);
        const bruun_v4d vs = V4D_SET1(s_scalar);

        for (; n + 3 < q; n += 4) {
            const bruun_v4d C0v = V4D_LD(C0p + n);
            const bruun_v4d C1v = V4D_LD(C1p + n);
            const bruun_v4d D0v = V4D_LD(D0p + n);
            const bruun_v4d D1v = V4D_LD(D1p + n);

            const bruun_v4d A0 = V4D_MUL(half, V4D_ADD(C0v, D0v));
            const bruun_v4d R  = V4D_MUL(half, V4D_SUB(C0v, D0v));
            const bruun_v4d I  = V4D_MUL(half, V4D_ADD(C1v, D1v));
            const bruun_v4d A1 = V4D_MUL(half, V4D_SUB(C1v, D1v));

            const bruun_v4d B0 = V4D_MADD(V4D_MUL(vc, R), vs, I);
            const bruun_v4d B1 = V4D_MSUB(V4D_MUL(vc, I), vs, R);

            V4D_ST(C0p + n, A0);
            V4D_ST(C1p + n, B0);
            V4D_ST(D0p + n, A1);
            V4D_ST(D1p + n, B1);
        }
    }
#endif
#if BRUUN_LEVEL >= 1
    {
        const bruun_v2 half = V2_SET1(0.5);
        const bruun_v2 vc = V2_SET1(c_scalar);
        const bruun_v2 vs = V2_SET1(s_scalar);

        for (; n + 1 < q; n += 2) {
            const bruun_v2 C0v = V2_LD(C0p + n);
            const bruun_v2 C1v = V2_LD(C1p + n);
            const bruun_v2 D0v = V2_LD(D0p + n);
            const bruun_v2 D1v = V2_LD(D1p + n);

            const bruun_v2 A0 = V2_MUL(half, V2_ADD(C0v, D0v));
            const bruun_v2 R  = V2_MUL(half, V2_SUB(C0v, D0v));
            const bruun_v2 I  = V2_MUL(half, V2_ADD(C1v, D1v));
            const bruun_v2 A1 = V2_MUL(half, V2_SUB(C1v, D1v));

            const bruun_v2 B0 = V2_MADD(V2_MUL(vc, R), vs, I);
            const bruun_v2 B1 = V2_MSUB(V2_MUL(vc, I), vs, R);

            V2_ST(C0p + n, A0);
            V2_ST(C1p + n, B0);
            V2_ST(D0p + n, A1);
            V2_ST(D1p + n, B1);
        }
    }
#endif

    for (; n < q; ++n) {
        const double C0v = C0p[n];
        const double C1v = C1p[n];
        const double D0v = D0p[n];
        const double D1v = D1p[n];

        const double A0 = 0.5 * (C0v + D0v);
        const double R  = 0.5 * (C0v - D0v);
        const double I  = 0.5 * (C1v + D1v);
        const double A1 = 0.5 * (C1v - D1v);

        C0p[n] = A0;
        C1p[n] = c_scalar * R + s_scalar * I;
        D0p[n] = A1;
        D1p[n] = c_scalar * I - s_scalar * R;
    }
}

// Exact inverse of norm2_fused: undo both child rotations and the parent
// rotation in one pass over the block. Caller guarantees q >= 16 so qh >= 8.
static inline void norm2_inv_fused(double* RESTRICT p, int q,
                                   double c, double s,
                                   double c0, double s0,
                                   double c1, double s1) {
    const int qh = q >> 1;
    double* RESTRICT A0 = p;
    double* RESTRICT B0 = p + q;
    double* RESTRICT A1 = p + 2*q;
    double* RESTRICT B1 = p + 3*q;

    int n = 0;

#if BRUUN_LEVEL >= 2
    {
        const bruun_v4d hf  = V4D_SET1(0.5);
        const bruun_v4d vc  = V4D_SET1(c),  vs  = V4D_SET1(s);
        const bruun_v4d vc0 = V4D_SET1(c0), vs0 = V4D_SET1(s0);
        const bruun_v4d vc1 = V4D_SET1(c1), vs1 = V4D_SET1(s1);

        for (; n + 3 < qh; n += 4) {
            const bruun_v4d A0n = V4D_LD(A0 + n);
            const bruun_v4d A0h = V4D_LD(A0 + qh + n);
            const bruun_v4d B0n = V4D_LD(B0 + n);
            const bruun_v4d B0h = V4D_LD(B0 + qh + n);
            const bruun_v4d A1n = V4D_LD(A1 + n);
            const bruun_v4d A1h = V4D_LD(A1 + qh + n);
            const bruun_v4d B1n = V4D_LD(B1 + n);
            const bruun_v4d B1h = V4D_LD(B1 + qh + n);

            const bruun_v4d u0 = V4D_MUL(hf, V4D_ADD(A0n, B0n));
            const bruun_v4d R0 = V4D_MUL(hf, V4D_SUB(A0n, B0n));
            const bruun_v4d I0 = V4D_MUL(hf, V4D_ADD(A0h, B0h));
            const bruun_v4d w0 = V4D_MUL(hf, V4D_SUB(A0h, B0h));
            const bruun_v4d uh = V4D_MADD(V4D_MUL(vc0, R0), vs0, I0);
            const bruun_v4d wh = V4D_MSUB(V4D_MUL(vc0, I0), vs0, R0);

            const bruun_v4d v0 = V4D_MUL(hf, V4D_ADD(A1n, B1n));
            const bruun_v4d R1 = V4D_MUL(hf, V4D_SUB(A1n, B1n));
            const bruun_v4d I1 = V4D_MUL(hf, V4D_ADD(A1h, B1h));
            const bruun_v4d x0 = V4D_MUL(hf, V4D_SUB(A1h, B1h));
            const bruun_v4d vh = V4D_MADD(V4D_MUL(vc1, R1), vs1, I1);
            const bruun_v4d xh = V4D_MSUB(V4D_MUL(vc1, I1), vs1, R1);

            const bruun_v4d a0n = V4D_MUL(hf, V4D_ADD(u0, v0));
            const bruun_v4d Rn  = V4D_MUL(hf, V4D_SUB(u0, v0));
            const bruun_v4d In  = V4D_MUL(hf, V4D_ADD(w0, x0));
            const bruun_v4d a1n = V4D_MUL(hf, V4D_SUB(w0, x0));
            const bruun_v4d a0h = V4D_MUL(hf, V4D_ADD(uh, vh));
            const bruun_v4d Rh  = V4D_MUL(hf, V4D_SUB(uh, vh));
            const bruun_v4d Ih  = V4D_MUL(hf, V4D_ADD(wh, xh));
            const bruun_v4d a1h = V4D_MUL(hf, V4D_SUB(wh, xh));

            V4D_ST(A0 + n,      a0n);
            V4D_ST(A0 + qh + n, a0h);
            V4D_ST(B0 + n,      V4D_MADD(V4D_MUL(vc, Rn), vs, In));
            V4D_ST(B0 + qh + n, V4D_MADD(V4D_MUL(vc, Rh), vs, Ih));
            V4D_ST(A1 + n,      a1n);
            V4D_ST(A1 + qh + n, a1h);
            V4D_ST(B1 + n,      V4D_MSUB(V4D_MUL(vc, In), vs, Rn));
            V4D_ST(B1 + qh + n, V4D_MSUB(V4D_MUL(vc, Ih), vs, Rh));
        }
    }
#endif
#if BRUUN_LEVEL >= 1
    {
        const bruun_v2 hf  = V2_SET1(0.5);
        const bruun_v2 vc  = V2_SET1(c),  vs  = V2_SET1(s);
        const bruun_v2 vc0 = V2_SET1(c0), vs0 = V2_SET1(s0);
        const bruun_v2 vc1 = V2_SET1(c1), vs1 = V2_SET1(s1);

        for (; n + 1 < qh; n += 2) {
            const bruun_v2 A0n = V2_LD(A0 + n);
            const bruun_v2 A0h = V2_LD(A0 + qh + n);
            const bruun_v2 B0n = V2_LD(B0 + n);
            const bruun_v2 B0h = V2_LD(B0 + qh + n);
            const bruun_v2 A1n = V2_LD(A1 + n);
            const bruun_v2 A1h = V2_LD(A1 + qh + n);
            const bruun_v2 B1n = V2_LD(B1 + n);
            const bruun_v2 B1h = V2_LD(B1 + qh + n);

            const bruun_v2 u0 = V2_MUL(hf, V2_ADD(A0n, B0n));
            const bruun_v2 R0 = V2_MUL(hf, V2_SUB(A0n, B0n));
            const bruun_v2 I0 = V2_MUL(hf, V2_ADD(A0h, B0h));
            const bruun_v2 w0 = V2_MUL(hf, V2_SUB(A0h, B0h));
            const bruun_v2 uh = V2_MADD(V2_MUL(vc0, R0), vs0, I0);
            const bruun_v2 wh = V2_MSUB(V2_MUL(vc0, I0), vs0, R0);

            const bruun_v2 v0 = V2_MUL(hf, V2_ADD(A1n, B1n));
            const bruun_v2 R1 = V2_MUL(hf, V2_SUB(A1n, B1n));
            const bruun_v2 I1 = V2_MUL(hf, V2_ADD(A1h, B1h));
            const bruun_v2 x0 = V2_MUL(hf, V2_SUB(A1h, B1h));
            const bruun_v2 vh = V2_MADD(V2_MUL(vc1, R1), vs1, I1);
            const bruun_v2 xh = V2_MSUB(V2_MUL(vc1, I1), vs1, R1);

            const bruun_v2 a0n = V2_MUL(hf, V2_ADD(u0, v0));
            const bruun_v2 Rn  = V2_MUL(hf, V2_SUB(u0, v0));
            const bruun_v2 In  = V2_MUL(hf, V2_ADD(w0, x0));
            const bruun_v2 a1n = V2_MUL(hf, V2_SUB(w0, x0));
            const bruun_v2 a0h = V2_MUL(hf, V2_ADD(uh, vh));
            const bruun_v2 Rh  = V2_MUL(hf, V2_SUB(uh, vh));
            const bruun_v2 Ih  = V2_MUL(hf, V2_ADD(wh, xh));
            const bruun_v2 a1h = V2_MUL(hf, V2_SUB(wh, xh));

            V2_ST(A0 + n,      a0n);
            V2_ST(A0 + qh + n, a0h);
            V2_ST(B0 + n,      V2_MADD(V2_MUL(vc, Rn), vs, In));
            V2_ST(B0 + qh + n, V2_MADD(V2_MUL(vc, Rh), vs, Ih));
            V2_ST(A1 + n,      a1n);
            V2_ST(A1 + qh + n, a1h);
            V2_ST(B1 + n,      V2_MSUB(V2_MUL(vc, In), vs, Rn));
            V2_ST(B1 + qh + n, V2_MSUB(V2_MUL(vc, Ih), vs, Rh));
        }
    }
#endif

    for (; n < qh; ++n) {
        const double A0n = A0[n], A0h = A0[qh + n];
        const double B0n = B0[n], B0h = B0[qh + n];
        const double A1n = A1[n], A1h = A1[qh + n];
        const double B1n = B1[n], B1h = B1[qh + n];

        const double u0 = 0.5 * (A0n + B0n);
        const double R0 = 0.5 * (A0n - B0n);
        const double I0 = 0.5 * (A0h + B0h);
        const double w0 = 0.5 * (A0h - B0h);
        const double uh = c0 * R0 + s0 * I0;
        const double wh = c0 * I0 - s0 * R0;

        const double v0 = 0.5 * (A1n + B1n);
        const double R1 = 0.5 * (A1n - B1n);
        const double I1 = 0.5 * (A1h + B1h);
        const double x0 = 0.5 * (A1h - B1h);
        const double vh = c1 * R1 + s1 * I1;
        const double xh = c1 * I1 - s1 * R1;

        const double a0n = 0.5 * (u0 + v0);
        const double Rn  = 0.5 * (u0 - v0);
        const double In  = 0.5 * (w0 + x0);
        const double a1n = 0.5 * (w0 - x0);
        const double a0h = 0.5 * (uh + vh);
        const double Rh  = 0.5 * (uh - vh);
        const double Ih  = 0.5 * (wh + xh);
        const double a1h = 0.5 * (wh - xh);

        A0[n] = a0n;
        A0[qh + n] = a0h;
        B0[n] = c * Rn + s * In;
        B0[qh + n] = c * Rh + s * Ih;
        A1[n] = a1n;
        A1[qh + n] = a1h;
        B1[n] = c * In - s * Rn;
        B1[qh + n] = c * Ih - s * Rh;
    }
}

// ---------------------------------------------------------------------------
// Float32 streaming kernels. Same arithmetic as the double kernels above with
// twice the scalar lane count on supported SIMD paths: 8-wide AVX2,
// 4-wide SSE2/NEON, exact scalar tail.
// ---------------------------------------------------------------------------

static inline void binomial_fwd_f32(float* RESTRICT v, int h) {
    int i = 0;

#if BRUUN_LEVEL >= 2
    for (; i + 7 < h; i += 8) {
        const __m256 a = _mm256_loadu_ps(v + i);
        const __m256 b = _mm256_loadu_ps(v + h + i);
        _mm256_storeu_ps(v + i, _mm256_add_ps(a, b));
        _mm256_storeu_ps(v + h + i, _mm256_sub_ps(a, b));
    }
#endif
#if BRUUN_LEVEL >= 1
    for (; i + 3 < h; i += 4) {
        const bruun_v4f a = V4F_LD(v + i);
        const bruun_v4f b = V4F_LD(v + h + i);
        V4F_ST(v + i, V4F_ADD(a, b));
        V4F_ST(v + h + i, V4F_SUB(a, b));
    }
#endif

    for (; i < h; ++i) {
        const float a = v[i];
        const float b = v[h + i];
        v[i] = a + b;
        v[h + i] = a - b;
    }
}

static inline void binomial_oop_f32(const float* RESTRICT in, float* RESTRICT v, int h) {
    int i = 0;

#if BRUUN_LEVEL >= 2
    for (; i + 7 < h; i += 8) {
        const __m256 a = _mm256_loadu_ps(in + i);
        const __m256 b = _mm256_loadu_ps(in + h + i);
        _mm256_storeu_ps(v + i, _mm256_add_ps(a, b));
        _mm256_storeu_ps(v + h + i, _mm256_sub_ps(a, b));
    }
#endif
#if BRUUN_LEVEL >= 1
    for (; i + 3 < h; i += 4) {
        const bruun_v4f a = V4F_LD(in + i);
        const bruun_v4f b = V4F_LD(in + h + i);
        V4F_ST(v + i, V4F_ADD(a, b));
        V4F_ST(v + h + i, V4F_SUB(a, b));
    }
#endif

    for (; i < h; ++i) {
        const float a = in[i];
        const float b = in[h + i];
        v[i] = a + b;
        v[h + i] = a - b;
    }
}

static inline void binomial_inv_f32(float* RESTRICT v, int h) {
    int i = 0;

#if BRUUN_LEVEL >= 2
    {
        const __m256 half8 = _mm256_set1_ps(0.5f);
        for (; i + 7 < h; i += 8) {
            const __m256 a = _mm256_loadu_ps(v + i);
            const __m256 b = _mm256_loadu_ps(v + h + i);
            _mm256_storeu_ps(v + i, _mm256_mul_ps(half8, _mm256_add_ps(a, b)));
            _mm256_storeu_ps(v + h + i, _mm256_mul_ps(half8, _mm256_sub_ps(a, b)));
        }
    }
#endif
#if BRUUN_LEVEL >= 1
    {
        const bruun_v4f half4 = V4F_SET1(0.5f);
        for (; i + 3 < h; i += 4) {
            const bruun_v4f a = V4F_LD(v + i);
            const bruun_v4f b = V4F_LD(v + h + i);
            V4F_ST(v + i, V4F_MUL(half4, V4F_ADD(a, b)));
            V4F_ST(v + h + i, V4F_MUL(half4, V4F_SUB(a, b)));
        }
    }
#endif

    for (; i < h; ++i) {
        const float a = v[i];
        const float b = v[h + i];
        v[i] = 0.5f * (a + b);
        v[h + i] = 0.5f * (a - b);
    }
}

static inline void norm_q_fwd_f32(float* RESTRICT p, int q, float c_scalar, float s_scalar) {
    float* RESTRICT A0p = p;
    float* RESTRICT B0p = p + q;
    float* RESTRICT A1p = p + 2*q;
    float* RESTRICT B1p = p + 3*q;

    int n = 0;

#if BRUUN_LEVEL >= 2
    {
        const __m256 vc = _mm256_set1_ps(c_scalar);
        const __m256 vs = _mm256_set1_ps(s_scalar);

        for (; n + 7 < q; n += 8) {
            const __m256 B0 = _mm256_loadu_ps(B0p + n);
            const __m256 B1 = _mm256_loadu_ps(B1p + n);
            const __m256 R = _mm256_fmsub_ps(vc, B0, _mm256_mul_ps(vs, B1));
            const __m256 I = _mm256_fmadd_ps(vs, B0, _mm256_mul_ps(vc, B1));

            const __m256 A0 = _mm256_loadu_ps(A0p + n);
            const __m256 A1 = _mm256_loadu_ps(A1p + n);

            _mm256_storeu_ps(A0p + n, _mm256_add_ps(A0, R));
            _mm256_storeu_ps(B0p + n, _mm256_add_ps(A1, I));
            _mm256_storeu_ps(A1p + n, _mm256_sub_ps(A0, R));
            _mm256_storeu_ps(B1p + n, _mm256_sub_ps(I, A1));
        }
    }
#endif
#if BRUUN_LEVEL >= 1
    {
        const bruun_v4f vc = V4F_SET1(c_scalar);
        const bruun_v4f vs = V4F_SET1(s_scalar);

        for (; n + 3 < q; n += 4) {
            const bruun_v4f B0 = V4F_LD(B0p + n);
            const bruun_v4f B1 = V4F_LD(B1p + n);
            const bruun_v4f R = V4F_MSUB(V4F_MUL(vc, B0), vs, B1);
            const bruun_v4f I = V4F_MADD(V4F_MUL(vs, B0), vc, B1);

            const bruun_v4f A0 = V4F_LD(A0p + n);
            const bruun_v4f A1 = V4F_LD(A1p + n);

            V4F_ST(A0p + n, V4F_ADD(A0, R));
            V4F_ST(B0p + n, V4F_ADD(A1, I));
            V4F_ST(A1p + n, V4F_SUB(A0, R));
            V4F_ST(B1p + n, V4F_SUB(I, A1));
        }
    }
#endif

    for (; n < q; ++n) {
        const float A0 = A0p[n];
        const float B0 = B0p[n];
        const float A1 = A1p[n];
        const float B1 = B1p[n];

        const float R = c_scalar * B0 - s_scalar * B1;
        const float I = s_scalar * B0 + c_scalar * B1;

        A0p[n] = A0 + R;
        B0p[n] = A1 + I;
        A1p[n] = A0 - R;
        B1p[n] = -A1 + I;
    }
}

// Two fused tree levels, float lane-doubled twin of norm2_fused.
// Caller guarantees q >= 16 so qh >= 8.
static inline void norm2_fused_f32(float* RESTRICT p, int q,
                                   float c, float s,
                                   float c0, float s0,
                                   float c1, float s1) {
    const int qh = q >> 1;
    float* RESTRICT A0 = p;
    float* RESTRICT B0 = p + q;
    float* RESTRICT A1 = p + 2*q;
    float* RESTRICT B1 = p + 3*q;

    int n = 0;

#if BRUUN_LEVEL >= 2
    {
        const __m256 vc  = _mm256_set1_ps(c),  vs  = _mm256_set1_ps(s);
        const __m256 vc0 = _mm256_set1_ps(c0), vs0 = _mm256_set1_ps(s0);
        const __m256 vc1 = _mm256_set1_ps(c1), vs1 = _mm256_set1_ps(s1);

        for (; n + 7 < qh; n += 8) {
            const __m256 b0n = _mm256_loadu_ps(B0 + n);
            const __m256 b1n = _mm256_loadu_ps(B1 + n);
            const __m256 b0h = _mm256_loadu_ps(B0 + qh + n);
            const __m256 b1h = _mm256_loadu_ps(B1 + qh + n);

            const __m256 Rn = _mm256_fmsub_ps(vc, b0n, _mm256_mul_ps(vs, b1n));
            const __m256 In = _mm256_fmadd_ps(vs, b0n, _mm256_mul_ps(vc, b1n));
            const __m256 Rh = _mm256_fmsub_ps(vc, b0h, _mm256_mul_ps(vs, b1h));
            const __m256 Ih = _mm256_fmadd_ps(vs, b0h, _mm256_mul_ps(vc, b1h));

            const __m256 a0n = _mm256_loadu_ps(A0 + n);
            const __m256 a0h = _mm256_loadu_ps(A0 + qh + n);
            const __m256 a1n = _mm256_loadu_ps(A1 + n);
            const __m256 a1h = _mm256_loadu_ps(A1 + qh + n);

            const __m256 u0 = _mm256_add_ps(a0n, Rn);
            const __m256 uh = _mm256_add_ps(a0h, Rh);
            const __m256 w0 = _mm256_add_ps(a1n, In);
            const __m256 wh = _mm256_add_ps(a1h, Ih);
            const __m256 v0 = _mm256_sub_ps(a0n, Rn);
            const __m256 vh = _mm256_sub_ps(a0h, Rh);
            const __m256 x0 = _mm256_sub_ps(In, a1n);
            const __m256 xh = _mm256_sub_ps(Ih, a1h);

            const __m256 R0 = _mm256_fmsub_ps(vc0, uh, _mm256_mul_ps(vs0, wh));
            const __m256 I0 = _mm256_fmadd_ps(vs0, uh, _mm256_mul_ps(vc0, wh));
            const __m256 R1 = _mm256_fmsub_ps(vc1, vh, _mm256_mul_ps(vs1, xh));
            const __m256 I1 = _mm256_fmadd_ps(vs1, vh, _mm256_mul_ps(vc1, xh));

            _mm256_storeu_ps(A0 + n,      _mm256_add_ps(u0, R0));
            _mm256_storeu_ps(A0 + qh + n, _mm256_add_ps(w0, I0));
            _mm256_storeu_ps(B0 + n,      _mm256_sub_ps(u0, R0));
            _mm256_storeu_ps(B0 + qh + n, _mm256_sub_ps(I0, w0));
            _mm256_storeu_ps(A1 + n,      _mm256_add_ps(v0, R1));
            _mm256_storeu_ps(A1 + qh + n, _mm256_add_ps(x0, I1));
            _mm256_storeu_ps(B1 + n,      _mm256_sub_ps(v0, R1));
            _mm256_storeu_ps(B1 + qh + n, _mm256_sub_ps(I1, x0));
        }
    }
#endif
#if BRUUN_LEVEL >= 1
    {
        const bruun_v4f vc  = V4F_SET1(c),  vs  = V4F_SET1(s);
        const bruun_v4f vc0 = V4F_SET1(c0), vs0 = V4F_SET1(s0);
        const bruun_v4f vc1 = V4F_SET1(c1), vs1 = V4F_SET1(s1);

        for (; n + 3 < qh; n += 4) {
            const bruun_v4f b0n = V4F_LD(B0 + n);
            const bruun_v4f b1n = V4F_LD(B1 + n);
            const bruun_v4f b0h = V4F_LD(B0 + qh + n);
            const bruun_v4f b1h = V4F_LD(B1 + qh + n);

            const bruun_v4f Rn = V4F_MSUB(V4F_MUL(vc, b0n), vs, b1n);
            const bruun_v4f In = V4F_MADD(V4F_MUL(vs, b0n), vc, b1n);
            const bruun_v4f Rh = V4F_MSUB(V4F_MUL(vc, b0h), vs, b1h);
            const bruun_v4f Ih = V4F_MADD(V4F_MUL(vs, b0h), vc, b1h);

            const bruun_v4f a0n = V4F_LD(A0 + n);
            const bruun_v4f a0h = V4F_LD(A0 + qh + n);
            const bruun_v4f a1n = V4F_LD(A1 + n);
            const bruun_v4f a1h = V4F_LD(A1 + qh + n);

            const bruun_v4f u0 = V4F_ADD(a0n, Rn);
            const bruun_v4f uh = V4F_ADD(a0h, Rh);
            const bruun_v4f w0 = V4F_ADD(a1n, In);
            const bruun_v4f wh = V4F_ADD(a1h, Ih);
            const bruun_v4f v0 = V4F_SUB(a0n, Rn);
            const bruun_v4f vh = V4F_SUB(a0h, Rh);
            const bruun_v4f x0 = V4F_SUB(In, a1n);
            const bruun_v4f xh = V4F_SUB(Ih, a1h);

            const bruun_v4f R0 = V4F_MSUB(V4F_MUL(vc0, uh), vs0, wh);
            const bruun_v4f I0 = V4F_MADD(V4F_MUL(vs0, uh), vc0, wh);
            const bruun_v4f R1 = V4F_MSUB(V4F_MUL(vc1, vh), vs1, xh);
            const bruun_v4f I1 = V4F_MADD(V4F_MUL(vs1, vh), vc1, xh);

            V4F_ST(A0 + n,      V4F_ADD(u0, R0));
            V4F_ST(A0 + qh + n, V4F_ADD(w0, I0));
            V4F_ST(B0 + n,      V4F_SUB(u0, R0));
            V4F_ST(B0 + qh + n, V4F_SUB(I0, w0));
            V4F_ST(A1 + n,      V4F_ADD(v0, R1));
            V4F_ST(A1 + qh + n, V4F_ADD(x0, I1));
            V4F_ST(B1 + n,      V4F_SUB(v0, R1));
            V4F_ST(B1 + qh + n, V4F_SUB(I1, x0));
        }
    }
#endif

    for (; n < qh; ++n) {
        const float a0n = A0[n],      a0h = A0[qh + n];
        const float b0n = B0[n],      b0h = B0[qh + n];
        const float a1n = A1[n],      a1h = A1[qh + n];
        const float b1n = B1[n],      b1h = B1[qh + n];

        const float Rn = c * b0n - s * b1n;
        const float In = s * b0n + c * b1n;
        const float Rh = c * b0h - s * b1h;
        const float Ih = s * b0h + c * b1h;

        const float u0 = a0n + Rn, uh = a0h + Rh;
        const float w0 = a1n + In, wh = a1h + Ih;
        const float v0 = a0n - Rn, vh = a0h - Rh;
        const float x0 = In - a1n, xh = Ih - a1h;

        const float R0 = c0 * uh - s0 * wh;
        const float I0 = s0 * uh + c0 * wh;
        const float R1 = c1 * vh - s1 * xh;
        const float I1 = s1 * vh + c1 * xh;

        A0[n] = u0 + R0;
        A0[qh + n] = w0 + I0;
        B0[n] = u0 - R0;
        B0[qh + n] = I0 - w0;
        A1[n] = v0 + R1;
        A1[qh + n] = x0 + I1;
        B1[n] = v0 - R1;
        B1[qh + n] = I1 - x0;
    }
}

static inline void norm_q_inv_f32(float* RESTRICT p, int q, float c_scalar, float s_scalar) {
    float* RESTRICT C0p = p;
    float* RESTRICT C1p = p + q;
    float* RESTRICT D0p = p + 2*q;
    float* RESTRICT D1p = p + 3*q;

    int n = 0;

#if BRUUN_LEVEL >= 2
    {
        const __m256 half = _mm256_set1_ps(0.5f);
        const __m256 hvc = _mm256_set1_ps(0.5f * c_scalar);
        const __m256 hvs = _mm256_set1_ps(0.5f * s_scalar);

        for (; n + 7 < q; n += 8) {
            const __m256 C0v = _mm256_loadu_ps(C0p + n);
            const __m256 C1v = _mm256_loadu_ps(C1p + n);
            const __m256 D0v = _mm256_loadu_ps(D0p + n);
            const __m256 D1v = _mm256_loadu_ps(D1p + n);

            const __m256 t0 = _mm256_add_ps(C0v, D0v);
            const __m256 r  = _mm256_sub_ps(C0v, D0v);
            const __m256 i  = _mm256_add_ps(C1v, D1v);
            const __m256 t1 = _mm256_sub_ps(C1v, D1v);

            const __m256 A0 = _mm256_mul_ps(half, t0);
            const __m256 A1 = _mm256_mul_ps(half, t1);
            const __m256 B0 = _mm256_fmadd_ps(hvc, r, _mm256_mul_ps(hvs, i));
            const __m256 B1 = _mm256_fmsub_ps(hvc, i, _mm256_mul_ps(hvs, r));

            _mm256_storeu_ps(C0p + n, A0);
            _mm256_storeu_ps(C1p + n, B0);
            _mm256_storeu_ps(D0p + n, A1);
            _mm256_storeu_ps(D1p + n, B1);
        }
    }
#endif
#if BRUUN_LEVEL >= 1
    {
        const bruun_v4f half = V4F_SET1(0.5f);
        const bruun_v4f hvc = V4F_SET1(0.5f * c_scalar);
        const bruun_v4f hvs = V4F_SET1(0.5f * s_scalar);

        for (; n + 3 < q; n += 4) {
            const bruun_v4f C0v = V4F_LD(C0p + n);
            const bruun_v4f C1v = V4F_LD(C1p + n);
            const bruun_v4f D0v = V4F_LD(D0p + n);
            const bruun_v4f D1v = V4F_LD(D1p + n);

            const bruun_v4f t0 = V4F_ADD(C0v, D0v);
            const bruun_v4f r  = V4F_SUB(C0v, D0v);
            const bruun_v4f i  = V4F_ADD(C1v, D1v);
            const bruun_v4f t1 = V4F_SUB(C1v, D1v);

            const bruun_v4f A0 = V4F_MUL(half, t0);
            const bruun_v4f A1 = V4F_MUL(half, t1);
            const bruun_v4f B0 = V4F_MADD(V4F_MUL(hvc, r), hvs, i);
            const bruun_v4f B1 = V4F_MSUB(V4F_MUL(hvc, i), hvs, r);

            V4F_ST(C0p + n, A0);
            V4F_ST(C1p + n, B0);
            V4F_ST(D0p + n, A1);
            V4F_ST(D1p + n, B1);
        }
    }
#endif

    for (; n < q; ++n) {
        const float C0v = C0p[n];
        const float C1v = C1p[n];
        const float D0v = D0p[n];
        const float D1v = D1p[n];

        const float A0 = 0.5f * (C0v + D0v);
        const float R  = 0.5f * (C0v - D0v);
        const float I  = 0.5f * (C1v + D1v);
        const float A1 = 0.5f * (C1v - D1v);

        C0p[n] = A0;
        C1p[n] = c_scalar * R + s_scalar * I;
        D0p[n] = A1;
        D1p[n] = c_scalar * I - s_scalar * R;
    }
}

// Exact inverse of norm2_fused_f32. Caller guarantees q >= 16 so qh >= 8.
static inline void norm2_inv_fused_f32(float* RESTRICT p, int q,
                                       float c, float s,
                                       float c0, float s0,
                                       float c1, float s1) {
    const int qh = q >> 1;
    float* RESTRICT A0 = p;
    float* RESTRICT B0 = p + q;
    float* RESTRICT A1 = p + 2*q;
    float* RESTRICT B1 = p + 3*q;

    int n = 0;

#if BRUUN_LEVEL >= 2
    {
        const __m256 hf  = _mm256_set1_ps(0.5f);
        const __m256 vc  = _mm256_set1_ps(c),  vs  = _mm256_set1_ps(s);
        const __m256 vc0 = _mm256_set1_ps(c0), vs0 = _mm256_set1_ps(s0);
        const __m256 vc1 = _mm256_set1_ps(c1), vs1 = _mm256_set1_ps(s1);

        for (; n + 7 < qh; n += 8) {
            const __m256 A0n = _mm256_loadu_ps(A0 + n);
            const __m256 A0h = _mm256_loadu_ps(A0 + qh + n);
            const __m256 B0n = _mm256_loadu_ps(B0 + n);
            const __m256 B0h = _mm256_loadu_ps(B0 + qh + n);
            const __m256 A1n = _mm256_loadu_ps(A1 + n);
            const __m256 A1h = _mm256_loadu_ps(A1 + qh + n);
            const __m256 B1n = _mm256_loadu_ps(B1 + n);
            const __m256 B1h = _mm256_loadu_ps(B1 + qh + n);

            const __m256 u0 = _mm256_mul_ps(hf, _mm256_add_ps(A0n, B0n));
            const __m256 R0 = _mm256_mul_ps(hf, _mm256_sub_ps(A0n, B0n));
            const __m256 I0 = _mm256_mul_ps(hf, _mm256_add_ps(A0h, B0h));
            const __m256 w0 = _mm256_mul_ps(hf, _mm256_sub_ps(A0h, B0h));
            const __m256 uh = _mm256_fmadd_ps(vc0, R0, _mm256_mul_ps(vs0, I0));
            const __m256 wh = _mm256_fmsub_ps(vc0, I0, _mm256_mul_ps(vs0, R0));

            const __m256 v0 = _mm256_mul_ps(hf, _mm256_add_ps(A1n, B1n));
            const __m256 R1 = _mm256_mul_ps(hf, _mm256_sub_ps(A1n, B1n));
            const __m256 I1 = _mm256_mul_ps(hf, _mm256_add_ps(A1h, B1h));
            const __m256 x0 = _mm256_mul_ps(hf, _mm256_sub_ps(A1h, B1h));
            const __m256 vh = _mm256_fmadd_ps(vc1, R1, _mm256_mul_ps(vs1, I1));
            const __m256 xh = _mm256_fmsub_ps(vc1, I1, _mm256_mul_ps(vs1, R1));

            const __m256 a0n = _mm256_mul_ps(hf, _mm256_add_ps(u0, v0));
            const __m256 Rn  = _mm256_mul_ps(hf, _mm256_sub_ps(u0, v0));
            const __m256 In  = _mm256_mul_ps(hf, _mm256_add_ps(w0, x0));
            const __m256 a1n = _mm256_mul_ps(hf, _mm256_sub_ps(w0, x0));
            const __m256 a0h = _mm256_mul_ps(hf, _mm256_add_ps(uh, vh));
            const __m256 Rh  = _mm256_mul_ps(hf, _mm256_sub_ps(uh, vh));
            const __m256 Ih  = _mm256_mul_ps(hf, _mm256_add_ps(wh, xh));
            const __m256 a1h = _mm256_mul_ps(hf, _mm256_sub_ps(wh, xh));

            _mm256_storeu_ps(A0 + n,      a0n);
            _mm256_storeu_ps(A0 + qh + n, a0h);
            _mm256_storeu_ps(B0 + n,      _mm256_fmadd_ps(vc, Rn, _mm256_mul_ps(vs, In)));
            _mm256_storeu_ps(B0 + qh + n, _mm256_fmadd_ps(vc, Rh, _mm256_mul_ps(vs, Ih)));
            _mm256_storeu_ps(A1 + n,      a1n);
            _mm256_storeu_ps(A1 + qh + n, a1h);
            _mm256_storeu_ps(B1 + n,      _mm256_fmsub_ps(vc, In, _mm256_mul_ps(vs, Rn)));
            _mm256_storeu_ps(B1 + qh + n, _mm256_fmsub_ps(vc, Ih, _mm256_mul_ps(vs, Rh)));
        }
    }
#endif
#if BRUUN_LEVEL >= 1
    {
        const bruun_v4f hf  = V4F_SET1(0.5f);
        const bruun_v4f vc  = V4F_SET1(c),  vs  = V4F_SET1(s);
        const bruun_v4f vc0 = V4F_SET1(c0), vs0 = V4F_SET1(s0);
        const bruun_v4f vc1 = V4F_SET1(c1), vs1 = V4F_SET1(s1);

        for (; n + 3 < qh; n += 4) {
            const bruun_v4f A0n = V4F_LD(A0 + n);
            const bruun_v4f A0h = V4F_LD(A0 + qh + n);
            const bruun_v4f B0n = V4F_LD(B0 + n);
            const bruun_v4f B0h = V4F_LD(B0 + qh + n);
            const bruun_v4f A1n = V4F_LD(A1 + n);
            const bruun_v4f A1h = V4F_LD(A1 + qh + n);
            const bruun_v4f B1n = V4F_LD(B1 + n);
            const bruun_v4f B1h = V4F_LD(B1 + qh + n);

            const bruun_v4f u0 = V4F_MUL(hf, V4F_ADD(A0n, B0n));
            const bruun_v4f R0 = V4F_MUL(hf, V4F_SUB(A0n, B0n));
            const bruun_v4f I0 = V4F_MUL(hf, V4F_ADD(A0h, B0h));
            const bruun_v4f w0 = V4F_MUL(hf, V4F_SUB(A0h, B0h));
            const bruun_v4f uh = V4F_MADD(V4F_MUL(vc0, R0), vs0, I0);
            const bruun_v4f wh = V4F_MSUB(V4F_MUL(vc0, I0), vs0, R0);

            const bruun_v4f v0 = V4F_MUL(hf, V4F_ADD(A1n, B1n));
            const bruun_v4f R1 = V4F_MUL(hf, V4F_SUB(A1n, B1n));
            const bruun_v4f I1 = V4F_MUL(hf, V4F_ADD(A1h, B1h));
            const bruun_v4f x0 = V4F_MUL(hf, V4F_SUB(A1h, B1h));
            const bruun_v4f vh = V4F_MADD(V4F_MUL(vc1, R1), vs1, I1);
            const bruun_v4f xh = V4F_MSUB(V4F_MUL(vc1, I1), vs1, R1);

            const bruun_v4f a0n = V4F_MUL(hf, V4F_ADD(u0, v0));
            const bruun_v4f Rn  = V4F_MUL(hf, V4F_SUB(u0, v0));
            const bruun_v4f In  = V4F_MUL(hf, V4F_ADD(w0, x0));
            const bruun_v4f a1n = V4F_MUL(hf, V4F_SUB(w0, x0));
            const bruun_v4f a0h = V4F_MUL(hf, V4F_ADD(uh, vh));
            const bruun_v4f Rh  = V4F_MUL(hf, V4F_SUB(uh, vh));
            const bruun_v4f Ih  = V4F_MUL(hf, V4F_ADD(wh, xh));
            const bruun_v4f a1h = V4F_MUL(hf, V4F_SUB(wh, xh));

            V4F_ST(A0 + n,      a0n);
            V4F_ST(A0 + qh + n, a0h);
            V4F_ST(B0 + n,      V4F_MADD(V4F_MUL(vc, Rn), vs, In));
            V4F_ST(B0 + qh + n, V4F_MADD(V4F_MUL(vc, Rh), vs, Ih));
            V4F_ST(A1 + n,      a1n);
            V4F_ST(A1 + qh + n, a1h);
            V4F_ST(B1 + n,      V4F_MSUB(V4F_MUL(vc, In), vs, Rn));
            V4F_ST(B1 + qh + n, V4F_MSUB(V4F_MUL(vc, Ih), vs, Rh));
        }
    }
#endif

    for (; n < qh; ++n) {
        const float A0n = A0[n], A0h = A0[qh + n];
        const float B0n = B0[n], B0h = B0[qh + n];
        const float A1n = A1[n], A1h = A1[qh + n];
        const float B1n = B1[n], B1h = B1[qh + n];

        const float u0 = 0.5f * (A0n + B0n);
        const float R0 = 0.5f * (A0n - B0n);
        const float I0 = 0.5f * (A0h + B0h);
        const float w0 = 0.5f * (A0h - B0h);
        const float uh = c0 * R0 + s0 * I0;
        const float wh = c0 * I0 - s0 * R0;

        const float v0 = 0.5f * (A1n + B1n);
        const float R1 = 0.5f * (A1n - B1n);
        const float I1 = 0.5f * (A1h + B1h);
        const float x0 = 0.5f * (A1h - B1h);
        const float vh = c1 * R1 + s1 * I1;
        const float xh = c1 * I1 - s1 * R1;

        const float a0n = 0.5f * (u0 + v0);
        const float Rn  = 0.5f * (u0 - v0);
        const float In  = 0.5f * (w0 + x0);
        const float a1n = 0.5f * (w0 - x0);
        const float a0h = 0.5f * (uh + vh);
        const float Rh  = 0.5f * (uh - vh);
        const float Ih  = 0.5f * (wh + xh);
        const float a1h = 0.5f * (wh - xh);

        A0[n] = a0n;
        A0[qh + n] = a0h;
        B0[n] = c * Rn + s * In;
        B0[qh + n] = c * Rh + s * Ih;
        A1[n] = a1n;
        A1[qh + n] = a1h;
        B1[n] = c * In - s * Rn;
        B1[qh + n] = c * Ih - s * Rh;
    }
}

class DIF_RFFT_kernel {
public:
    DIF_RFFT_kernel() noexcept : N(0), L(0), NB(0), fuse_tail(false) {}

    bool init(int n, bool fuse_tail_arg = true) {
        BRUUN_ASSERT(is_power2(n) && n >= 4);
        N = n;
        L = ilog2_pow2(n);
        NB = n / 2 + 1;
        fuse_tail = fuse_tail_arg && n >= 32;

        if (!IDX.resize(n / 2)) return false;
        if (!OUTIDX.resize(n / 2)) return false;
        if (!C.resize(n / 2)) return false;

        IDX[0] = 0;
        C[0] = 0.0;

        // Build the Bruun angle table directly from the covering-map half-angle
        // recurrence. The old constructor built a full T[N] cosine table and then
        // sampled it:
        //     C[m] = cos(pi * IDX[m] / N), S[m] = sin(pi * IDX[m] / N)
        // That costs O(N) libm cos() calls and dominates huge-N setup. Here the
        // same values are generated by the Bruun tree:
        //     alpha(1) = pi/4
        //     alpha(2m)   = alpha(m)/2
        //     alpha(2m+1) = pi/2 - alpha(m)/2
        // using only sqrt/adds.
        for (int m = 1; m < N / 2; ++m) {
            IDX[m] = bruun_idx_int(m, L);
        }

        if (N >= 4) {
            const double r = std::sqrt(0.5);
            C[1] = r;
        }

        for (int m = 1; 2*m < N / 2; ++m) {
            const double c = C[m];
            const double s = s_twiddle(m);
            // ce = cos(alpha/2) is stable for alpha in (0, pi/2): 1 + c never cancels.
            // se = sin(alpha/2) via sqrt((1 - c)/2) cancels catastrophically as
            // c -> 1 (deep small-angle lineages), costing ~log(N) digits at large N.
            // sin(alpha/2) = sin(alpha) / (2 cos(alpha/2)) has no cancellation.
            const double ce = std::sqrt(0.5 * (1.0 + c));
            const double se = s / (2.0 * ce);

            C[2*m] = ce;

            if (2*m + 1 < N / 2) {
                C[2*m + 1] = se;
            }
        }

        // OUTIDX is the native complex-output slot for each Bruun leaf.
        // Default is ordinary FFTW frequency-bin order.
        OUTIDX[0] = 0;
#if !defined(BRUUN_HEAPOPT_SPECTRUM_ORDER)
        for (int m = 1; m < N / 2; ++m) OUTIDX[m] = IDX[m];
#endif

#if defined(BRUUN_HEAPOPT_SPECTRUM_ORDER)
        // Legal fast-layout constraint:
        //   every Bruun factor node must keep a contiguous interval of final leaves.
        // DFS preorder fragments factor subtrees. This order keeps all heap/factor
        // intervals contiguous but chooses sibling orientations inside each level
        // to reduce adjacent frequency travel.
        if (!NATIVE_POS.assign(N / 2, 0)) return false;
        if (!NATIVE_LEAF.assign(N / 2, 0)) return false;

        heap_array<int> inv_k;
        if (!inv_k.assign(N / 2, 0)) return false;
        for (int m = 1; m < N / 2; ++m) inv_k[IDX[m]] = m;

        heap_array<int> k_order;
        if (!k_order.reserve(N / 2)) return false;
        const int M = N / 2;

        auto cyclic_dist = [M](int a, int b) {
            int d = std::abs(a - b);
            return std::min(d, M - d);
        };

        heap_array<int> prev;
        if (!prev.reserve(N / 2)) return false;
        if (!prev.push_back(M / 2)) return false;
        if (!k_order.push_back(M / 2)) return false;

        while (true) {
            heap_array<int_pair> pairs;
            if (!pairs.reserve(prev.size())) return false;
            for (int k : prev) {
                if ((k & 1) == 0) {
                    if (!pairs.push_back(int_pair{k / 2, M - k / 2})) return false;
                }
            }
            if (pairs.empty()) break;

            // Same lexicographic orientation optimizer as before, but linear-time
            // and linear-memory. The previous prototype stored a full candidate
            // sequence inside each DP state, causing quadratic copying at huge N.
            // This keeps only costs plus backpointers, then reconstructs one level.
            const size_t P = pairs.size();

            heap_array<int> max0, max1;
            heap_array<long long> sum0, sum1;
            heap_array<unsigned char> back0, back1;
            if (!max0.assign(P, 0) || !max1.assign(P, 0)) return false;
            if (!sum0.assign(P, 0) || !sum1.assign(P, 0)) return false;
            if (!back0.assign(P, 0) || !back1.assign(P, 0)) return false;

            auto start_of = [&pairs](size_t i, int o) {
                return o == 0 ? pairs[i].first : pairs[i].second;
            };
            auto end_of = [&pairs](size_t i, int o) {
                return o == 0 ? pairs[i].second : pairs[i].first;
            };
            auto better = [](int ma, long long sa, int mb, long long sb) {
                return ma < mb || (ma == mb && sa < sb);
            };

            for (size_t pi = 1; pi < P; ++pi) {
                for (int o = 0; o < 2; ++o) {
                    const int first = start_of(pi, o);

                    const int j0 = cyclic_dist(end_of(pi - 1, 0), first);
                    const int cand0_max = std::max(max0[pi - 1], j0);
                    const long long cand0_sum = sum0[pi - 1] + j0;

                    const int j1 = cyclic_dist(end_of(pi - 1, 1), first);
                    const int cand1_max = std::max(max1[pi - 1], j1);
                    const long long cand1_sum = sum1[pi - 1] + j1;

                    if (better(cand0_max, cand0_sum, cand1_max, cand1_sum)) {
                        if (o == 0) {
                            max0[pi] = cand0_max;
                            sum0[pi] = cand0_sum;
                            back0[pi] = 0;
                        } else {
                            max1[pi] = cand0_max;
                            sum1[pi] = cand0_sum;
                            back1[pi] = 0;
                        }
                    } else {
                        if (o == 0) {
                            max0[pi] = cand1_max;
                            sum0[pi] = cand1_sum;
                            back0[pi] = 1;
                        } else {
                            max1[pi] = cand1_max;
                            sum1[pi] = cand1_sum;
                            back1[pi] = 1;
                        }
                    }
                }
            }

            int choose =
                better(max0[P - 1], sum0[P - 1], max1[P - 1], sum1[P - 1]) ? 0 : 1;

            heap_array<unsigned char> orient;
            if (!orient.resize(P)) return false;
            for (size_t rr = P; rr-- > 0;) {
                orient[rr] = static_cast<unsigned char>(choose);
                choose = (choose == 0) ? back0[rr] : back1[rr];
            }

            prev.clear();
            for (size_t pi = 0; pi < P; ++pi) {
                const int a = pairs[pi].first;
                const int b = pairs[pi].second;
                if (orient[pi] == 0) {
                    if (!prev.push_back(a)) return false;
                    if (!prev.push_back(b)) return false;
                } else {
                    if (!prev.push_back(b)) return false;
                    if (!prev.push_back(a)) return false;
                }
            }

            for (int k : prev) {
                if (!k_order.push_back(k)) return false;
            }
        }

        int pos = 1;
        for (int k : k_order) {
            const int m = inv_k[k];
            if (m <= 0 || m >= N / 2) continue;
            NATIVE_POS[m] = pos;
            NATIVE_LEAF[pos] = m;
            ++pos;
        }

        for (int m = 1; m < N / 2; ++m) {
            if (NATIVE_POS[m] == 0) {
                NATIVE_POS[m] = pos;
                NATIVE_LEAF[pos] = m;
                ++pos;
            }
        }

        for (int m = 1; m < N / 2; ++m) OUTIDX[m] = NATIVE_POS[m];
#endif

        // Packed per-leaf metadata: one contiguous 96-byte entry per depth-3 leaf
        // block, read as a sequential stream during the transform instead of
        // heap-strided picks from C, S, and IDX.
        if (N >= 32) {
            if (!TW.resize(N / 16)) return false;
            for (int m = 1; m < N / 16; ++m) {
                LeafTw& e = TW[m];
                for (int g = 0; g < 4; ++g) {
                    e.c4[g] = C[4*m + g];
                }
                e.c2[0] = C[2*m];
                e.c2[1] = C[2*m + 1];
                e.c1 = C[m];
                e.s1 = s_twiddle(m);
                for (int j = 0; j < 8; ++j) e.idx[j] = OUTIDX[8*m + j];
            }
        }

        // Inverse bin permutation for sequential standard-order packing:
        // KINV[k] = m such that IDX[m] = k. IDX is a bijection [1, N/2) -> [1, N/2).
        if (!KINV.assign(N / 2, 0)) return false;
        for (int m = 1; m < N / 2; ++m) KINV[IDX[m]] = m;

#if defined(BRUUN_HEAPOPT_SPECTRUM_ORDER)
        // Compose the public-standard and native heap permutations once. This
        // trades two plan-owned int maps for removing dependent map lookups from
        // the native/standard conversion hot loops.
        if (!STANDARD_NATIVE_POS.assign(N / 2, 0)) return false;
        if (!NATIVE_STANDARD_BIN.assign(N / 2, 0)) return false;
        if (N < 32) {
            for (int k = 1; k < N / 2; ++k) {
                STANDARD_NATIVE_POS[k] = k;
                NATIVE_STANDARD_BIN[k] = k;
            }
        } else {
            for (int k = 1; k < N / 2; ++k) {
                STANDARD_NATIVE_POS[k] = NATIVE_POS[KINV[k]];
            }
            for (int pos = 1; pos < N / 2; ++pos) {
                NATIVE_STANDARD_BIN[pos] = IDX[NATIVE_LEAF[pos]];
            }
        }
#endif

        // Float copies of the Bruun angle tables plus packed per-leaf float
        // twiddles for the float32 engine. Cast from the double tables, so plan
        // setup stays free of extra libm calls.
        if (!CF.resize(N / 2)) return false;
        for (int m = 0; m < N / 2; ++m) {
            CF[m] = static_cast<float>(C[m]);
        }
        if (N >= 32) {
            if (!TWF.resize(N / 16)) return false;
            for (int m = 1; m < N / 16; ++m) {
                const LeafTw& e = TW[m];
                LeafTwF& f = TWF[m];
                for (int g = 0; g < 4; ++g) {
                    f.c4[g] = static_cast<float>(e.c4[g]);
                }
                f.c2d[0] = f.c2d[1] = static_cast<float>(e.c2[0]);
                f.c2d[2] = f.c2d[3] = static_cast<float>(e.c2[1]);
                f.c1 = static_cast<float>(e.c1);
                f.s1 = static_cast<float>(e.s1);
            }
        }

        if (!build_forward_schedules()) return false;
        return true;
    }

    int size() const { return N; }
    int bins() const { return NB; }
    int work_size() const { return N; }
    int work_size_f32() const { return N; }
    int native_scratch_size() const { return NB; }

    bool standard_output_uses_two_phase() const {
#if BRUUN_LEVEL >= 2
        return N >= kAvx2TwoPhaseMinN;
#endif
#if BRUUN_LEVEL >= 1
        return N > kSimd128TwoPhaseMinExclusiveN;
#else
        return false;
#endif
    }

    const char* standard_output_policy_name() const {
        if (standard_output_uses_two_phase()) {
            return "two-phase-standard-pack";
        }
        return "fused-scatter-plus-layout-convert";
    }

    // Fast native-output Bruun transform.
    // With BRUUN_HEAPOPT_SPECTRUM_ORDER, X is in heapopt Bruun-native order.
    // Without BRUUN_HEAPOPT_SPECTRUM_ORDER, native order is ordinary FFTW bin order.
    void forward_native(const double* RESTRICT input, complex_t* RESTRICT X, double* RESTRICT work) const {
        forward_native_with_alignment(input, X, work, false);
    }

    void forward_native_aligned_workspace(const double* RESTRICT input, complex_t* RESTRICT X, double* RESTRICT work) const {
        forward_native_with_alignment(input, X, work, true);
    }

    void forward_native_with_alignment(const double* RESTRICT input,
                                       complex_t* RESTRICT X,
                                       double* RESTRICT work,
                                       bool workspace_is_aligned) const {
        if (fuse_tail && N >= 64) {
            forward_recursive(input, work, X, workspace_is_aligned);
            return;
        }

        std::memcpy(work, input, sizeof(double) * N);

        if (fuse_tail) {
            forward_fused_tail(work, X);
        } else {
            forward_residues_inplace(work);
            residues_to_complex(work, X);
        }
    }

    // Standard FFTW-order float forward: depth-first float residue transform,
    // then one sequential pack pass through KINV. Always two-phase; the float
    // residue pass is wide enough that a fused scatter has no window to win.
    void forward_standard_f32(const float* RESTRICT input,
                              complex_f32_t* RESTRICT X,
                              float* RESTRICT work,
                              complex_f32_t* RESTRICT native_tmp) const {
        (void)native_tmp;
        forward_residues_f32(input, work);

        X[0].re = work[0] + work[1];
        X[0].im = 0.0f;
        X[N / 2].re = work[0] - work[1];
        X[N / 2].im = 0.0f;

        const int* RESTRICT kin = KINV.data();
        int k = 1;
        for (; k + 3 < N / 2; k += 4) {
            pack4_residues_to_complex_f32(X + k, work,
                                          kin[k], kin[k + 1], kin[k + 2], kin[k + 3]);
        }
        for (; k < N / 2; ++k) {
            const int m = kin[k];
            X[k].re = work[2*m];
            X[k].im = -work[2*m + 1];
        }
    }

    void forward_native_f32(const float* RESTRICT input,
                            complex_f32_t* RESTRICT X,
                            float* RESTRICT work) const {
        forward_residues_f32(input, work);

        X[0].re = work[0] + work[1];
        X[0].im = 0.0f;
        X[N / 2].re = work[0] - work[1];
        X[N / 2].im = 0.0f;

#if defined(BRUUN_HEAPOPT_SPECTRUM_ORDER)
        if (N >= 32) {
            // Native heapopt order is a layout permutation; keep the output
            // stream linear and pay the permutation on residue reads.
            const int* RESTRICT native_leaf = NATIVE_LEAF.data();
            int pos = 1;
            for (; pos + 3 < N / 2; pos += 4) {
                pack4_residues_to_complex_f32(X + pos, work,
                                              native_leaf[pos],
                                              native_leaf[pos + 1],
                                              native_leaf[pos + 2],
                                              native_leaf[pos + 3]);
            }
            for (; pos < N / 2; ++pos) {
                const int m = native_leaf[pos];
                X[pos].re = work[2*m];
                X[pos].im = -work[2*m + 1];
            }
            return;
        }
#endif
        for (int m = 1; m < N / 2; ++m) {
            const int k = IDX[m];
            X[k].re = work[2*m];
            X[k].im = -work[2*m + 1];
        }
    }

    // Standard FFT-order magnitude-only forward transform. This keeps the
    // residue-domain output in the caller's real work buffer and writes one
    // scalar magnitude per r2c bin, avoiding complex spectrum storage and native
    // scratch when phases are not needed.
    void forward_magnitude(const double* RESTRICT input,
                           double* RESTRICT magnitudes,
                           double* RESTRICT work) const {
        forward_residues(input, work);

        magnitudes[0] = std::fabs(work[0] + work[1]);
        magnitudes[N / 2] = std::fabs(work[0] - work[1]);

        const int* RESTRICT kin = KINV.data();
        for (int k = 1; k < N / 2; ++k) {
            const int m = kin[k];
            const double re = work[2*m];
            const double im = work[2*m + 1];
            magnitudes[k] = std::sqrt(re * re + im * im);
        }
    }

    // Standard FFT-order real-to-polar forward transform. The output has
    // N / 2 + 1 bins in ordinary FFT order k = 0..N/2, not native or
    // residue order. output[k].re is magnitude and output[k].im is phase in
    // radians in [0, 2*pi). Interior phases use the ordinary forward DFT bin
    // convention, so the imaginary component is -work[2*m + 1].
    void forward_mag_phase(const double* RESTRICT input,
                           complex_t* RESTRICT output,
                           double* RESTRICT work) const {
        forward_residues(input, work);

        const double dc = work[0] + work[1];
        output[0].re = std::fabs(dc);
        if (dc < 0.0) {
            output[0].im = M_PI;
        } else {
            output[0].im = 0.0;
        }

        const double ny = work[0] - work[1];
        output[N / 2].re = std::fabs(ny);
        if (ny < 0.0) {
            output[N / 2].im = M_PI;
        } else {
            output[N / 2].im = 0.0;
        }

        const int* RESTRICT kin = KINV.data();
        for (int k = 1; k < N / 2; ++k) {
            const int m = kin[k];
            output[k].re = work[2*m];
            output[k].im = -work[2*m + 1];
        }

        convert_standard_complex_to_mag_phase(output);
    }

    void convert_standard_complex_to_mag_phase(complex_t* RESTRICT output) const {
        for (int k = 1; k < N / 2; ++k) {
            const double re = output[k].re;
            const double im = output[k].im;
            const double mag = std::sqrt(re * re + im * im);
            const double phase = bruun_phase_atan2_mag(im, re, mag);
            output[k].re = mag;
            output[k].im = phase;
        }
    }

    void convert_standard_complex_to_mag_phase_f32(complex_f32_t* RESTRICT output) const {
        int k = 1;
        for (; k < N / 2; ++k) {
            const float re = output[k].re;
            const float im = output[k].im;
            const float mag = std::sqrt(re * re + im * im);
            const float phase = bruun_phase_atan2_mag_f32(im, re, mag);
            output[k].re = mag;
            output[k].im = phase;
        }
    }

    // Single-precision standard FFT-order magnitude-only forward transform.
    void forward_magnitude_f32(const float* RESTRICT input,
                               float* RESTRICT magnitudes,
                               float* RESTRICT work) const {
        forward_residues_f32(input, work);

        magnitudes[0] = std::fabs(work[0] + work[1]);
        magnitudes[N / 2] = std::fabs(work[0] - work[1]);

        const int* RESTRICT kin = KINV.data();
        for (int k = 1; k < N / 2; ++k) {
            const int m = kin[k];
            const float re = work[2*m];
            const float im = work[2*m + 1];
            magnitudes[k] = std::sqrt(re * re + im * im);
        }
    }

    // Single-precision standard FFT-order real-to-polar forward transform.
    // Output bins are ordinary FFT order k = 0..N/2, not native or residue order.
    void forward_mag_phase_f32(const float* RESTRICT input,
                               complex_f32_t* RESTRICT output,
                               float* RESTRICT work) const {
        forward_residues_f32(input, work);

        const float dc = work[0] + work[1];
        output[0].re = std::fabs(dc);
        if (dc < 0.0f) {
            output[0].im = static_cast<float>(M_PI);
        } else {
            output[0].im = 0.0f;
        }

        const float ny = work[0] - work[1];
        output[N / 2].re = std::fabs(ny);
        if (ny < 0.0f) {
            output[N / 2].im = static_cast<float>(M_PI);
        } else {
            output[N / 2].im = 0.0f;
        }

        const int* RESTRICT kin = KINV.data();
        int k = 1;
        for (; k + 3 < N / 2; k += 4) {
            pack4_residues_to_complex_f32(output + k, work,
                                          kin[k], kin[k + 1], kin[k + 2], kin[k + 3]);
        }
        for (; k < N / 2; ++k) {
            const int m = kin[k];
            output[k].re = work[2*m];
            output[k].im = -work[2*m + 1];
        }

        convert_standard_complex_to_mag_phase_f32(output);
    }

    // Standard FFTW-like real -> complex interface, using caller-provided native scratch.
    // X[k] is the ordinary k-th FFT bin on return.
    void forward_standard(const double* RESTRICT input,
                          complex_t* RESTRICT X,
                          double* RESTRICT work,
                          complex_t* RESTRICT native_tmp) const {
        if (standard_output_uses_two_phase()) {
            (void)native_tmp;
            forward_standard_two_phase(input, work, X);
            return;
        }
#if defined(BRUUN_HEAPOPT_SPECTRUM_ORDER)
        forward_native(input, native_tmp, work);
        native_to_standard_complex(native_tmp, X);
#else
        (void)native_tmp;
        forward_native(input, X, work);
#endif
    }

    // Convenience standard-output API. This allocates temporary native spectrum
    // storage when heapopt native order is enabled. Hot loops should prefer
    // forward_standard(..., native_tmp) to reuse that scratch.
    void forward(const double* RESTRICT input, complex_t* RESTRICT X, double* RESTRICT work) const {
#if defined(BRUUN_HEAPOPT_SPECTRUM_ORDER) && !defined(BRUUN_NATIVE_OUTPUT)
        heap_array<complex_t> native_tmp;
        if (!native_tmp.resize(NB)) return;
        forward_standard(input, X, work, native_tmp.data());
#else
        forward_native(input, X, work);
#endif
    }

    // Standard FFTW-like complex -> real inverse interface.
    // This inverse is intentionally simple/unfused; the current competition target is r2c forward.
    void inverse(const complex_t* RESTRICT X, double* RESTRICT out) const {
        complex_to_residues(X, out);
        inverse_residues_inplace(out);
    }

    void inverse_f32(const complex_f32_t* RESTRICT X, float* RESTRICT out) const {
        out[0] = 0.5f * (X[0].re + X[N / 2].re);
        out[1] = 0.5f * (X[0].re - X[N / 2].re);
        int m = 1;
        for (; m + 3 < N / 2; m += 4) {
            unpack4_complex_to_residues_f32(out + 2*m, X,
                                            IDX[m], IDX[m + 1], IDX[m + 2], IDX[m + 3]);
        }
        for (; m < N / 2; ++m) {
            const int k = IDX[m];
            out[2 * m] = X[k].re;
            out[2 * m + 1] = -X[k].im;
        }
        inverse_residues_inplace_f32(out);
    }

    void inverse_mag_phase(const complex_t* RESTRICT X, double* RESTRICT out) const {
        double s;
        double c;
        bruun_table256_poly3_sincos(X[0].im, &s, &c);
        const double dc = X[0].re * c;
        bruun_table256_poly3_sincos(X[N / 2].im, &s, &c);
        const double ny = X[N / 2].re * c;
        out[0] = 0.5 * (dc + ny);
        out[1] = 0.5 * (dc - ny);

        int m = 1;
        for (; m + 3 < N / 2; m += 4) {
            const int k0 = IDX[m];
            const int k1 = IDX[m + 1];
            const int k2 = IDX[m + 2];
            const int k3 = IDX[m + 3];
            double s0;
            double c0;
            double s1;
            double c1;
            double s2;
            double c2;
            double s3;
            double c3;
            bruun_table256_poly3_sincos(X[k0].im, &s0, &c0);
            bruun_table256_poly3_sincos(X[k1].im, &s1, &c1);
            bruun_table256_poly3_sincos(X[k2].im, &s2, &c2);
            bruun_table256_poly3_sincos(X[k3].im, &s3, &c3);
            out[2*m] = X[k0].re * c0;
            out[2*m + 1] = -X[k0].re * s0;
            out[2*m + 2] = X[k1].re * c1;
            out[2*m + 3] = -X[k1].re * s1;
            out[2*m + 4] = X[k2].re * c2;
            out[2*m + 5] = -X[k2].re * s2;
            out[2*m + 6] = X[k3].re * c3;
            out[2*m + 7] = -X[k3].re * s3;
        }
        for (; m < N / 2; ++m) {
            const int k = IDX[m];
            const double mag = X[k].re;
            const double phase = X[k].im;
            bruun_table256_poly3_sincos(phase, &s, &c);
            const double re = mag * c;
            const double im = mag * s;
            out[2*m] = re;
            out[2*m + 1] = -im;
        }
        inverse_residues_inplace(out);
    }

    void inverse_mag_phase_f32(const complex_f32_t* RESTRICT X, float* RESTRICT out) const {
        float s;
        float c;
        bruun_table256_poly3_sincos_f32(X[0].im, &s, &c);
        const float dc = X[0].re * c;
        bruun_table256_poly3_sincos_f32(X[N / 2].im, &s, &c);
        const float ny = X[N / 2].re * c;
        out[0] = 0.5f * (dc + ny);
        out[1] = 0.5f * (dc - ny);

        int m = 1;
        for (; m + 3 < N / 2; m += 4) {
            const int k0 = IDX[m];
            const int k1 = IDX[m + 1];
            const int k2 = IDX[m + 2];
            const int k3 = IDX[m + 3];
            bruun_table256_poly3_sincos_f32(X[k0].im, &s, &c);
            out[2*m] = X[k0].re * c;
            out[2*m + 1] = -X[k0].re * s;
            bruun_table256_poly3_sincos_f32(X[k1].im, &s, &c);
            out[2*m + 2] = X[k1].re * c;
            out[2*m + 3] = -X[k1].re * s;
            bruun_table256_poly3_sincos_f32(X[k2].im, &s, &c);
            out[2*m + 4] = X[k2].re * c;
            out[2*m + 5] = -X[k2].re * s;
            bruun_table256_poly3_sincos_f32(X[k3].im, &s, &c);
            out[2*m + 6] = X[k3].re * c;
            out[2*m + 7] = -X[k3].re * s;
        }
        for (; m < N / 2; ++m) {
            const int k = IDX[m];
            const float mag = X[k].re;
            const float phase = X[k].im;
            bruun_table256_poly3_sincos_f32(phase, &s, &c);
            const float re = mag * c;
            const float im = mag * s;
            out[2*m] = re;
            out[2*m + 1] = -im;
        }
        inverse_residues_inplace_f32(out);
    }

    void inverse_native_f32(const complex_f32_t* RESTRICT Xnative, float* RESTRICT out) const {
#if defined(BRUUN_HEAPOPT_SPECTRUM_ORDER)
        if (N < 32) {
            inverse_f32(Xnative, out);
            return;
        }
        out[0] = 0.5f * (Xnative[0].re + Xnative[N / 2].re);
        out[1] = 0.5f * (Xnative[0].re - Xnative[N / 2].re);
        const int* RESTRICT native_pos = NATIVE_POS.data();
        int m = 1;
        for (; m + 3 < N / 2; m += 4) {
            unpack4_complex_to_residues_f32(out + 2*m, Xnative,
                                            native_pos[m],
                                            native_pos[m + 1],
                                            native_pos[m + 2],
                                            native_pos[m + 3]);
        }
        for (; m < N / 2; ++m) {
            const int pos = native_pos[m];
            out[2 * m] = Xnative[pos].re;
            out[2 * m + 1] = -Xnative[pos].im;
        }
        inverse_residues_inplace_f32(out);
#else
        inverse_f32(Xnative, out);
#endif
    }

    // Convert the transform's native complex spectrum layout to standard FFTW order.
    // With BRUUN_HEAPOPT_SPECTRUM_ORDER, native nontrivial bins are in a
    // block-contiguous, sibling-orientation-optimized Bruun covering order.
    // Without it, native order already is standard FFTW bin order.
    void native_to_standard_complex(const complex_t* RESTRICT nativeX,
                                    complex_t* RESTRICT standardX) const {
#if defined(BRUUN_HEAPOPT_SPECTRUM_ORDER)
        if (N < 32) {
            std::memcpy(standardX, nativeX, sizeof(complex_t) * NB);
            return;
        }
        standardX[0] = nativeX[0];
        standardX[N / 2] = nativeX[N / 2];
        const int* RESTRICT map = STANDARD_NATIVE_POS.data();
        int k = 1;
        for (; k + 3 < N / 2; k += 4) {
            standardX[k] = nativeX[map[k]];
            standardX[k + 1] = nativeX[map[k + 1]];
            standardX[k + 2] = nativeX[map[k + 2]];
            standardX[k + 3] = nativeX[map[k + 3]];
        }
        for (; k < N / 2; ++k) {
            standardX[k] = nativeX[map[k]];
        }
#else
        std::memcpy(standardX, nativeX, sizeof(complex_t) * NB);
#endif
    }

    // Convert standard FFTW-order bins into this plan's native complex layout.
    void standard_to_native_complex(const complex_t* RESTRICT standardX,
                                    complex_t* RESTRICT nativeX) const {
#if defined(BRUUN_HEAPOPT_SPECTRUM_ORDER)
        if (N < 32) {
            std::memcpy(nativeX, standardX, sizeof(complex_t) * NB);
            return;
        }
        nativeX[0] = standardX[0];
        nativeX[N / 2] = standardX[N / 2];
        const int* RESTRICT map = NATIVE_STANDARD_BIN.data();
        int pos = 1;
        for (; pos + 3 < N / 2; pos += 4) {
            nativeX[pos] = standardX[map[pos]];
            nativeX[pos + 1] = standardX[map[pos + 1]];
            nativeX[pos + 2] = standardX[map[pos + 2]];
            nativeX[pos + 3] = standardX[map[pos + 3]];
        }
        for (; pos < N / 2; ++pos) {
            nativeX[pos] = standardX[map[pos]];
        }
#else
        std::memcpy(nativeX, standardX, sizeof(complex_t) * NB);
#endif
    }

    void native_to_standard_complex_f32(const complex_f32_t* RESTRICT nativeX,
                                        complex_f32_t* RESTRICT standardX) const {
#if defined(BRUUN_HEAPOPT_SPECTRUM_ORDER)
        if (N < 32) {
            std::memcpy(standardX, nativeX, sizeof(complex_f32_t) * NB);
            return;
        }
        standardX[0] = nativeX[0];
        standardX[N / 2] = nativeX[N / 2];
        const int* RESTRICT map = STANDARD_NATIVE_POS.data();
        int k = 1;
        for (; k + 3 < N / 2; k += 4) {
            copy4_complex_f32(standardX + k, nativeX,
                              map[k], map[k + 1], map[k + 2], map[k + 3]);
        }
        for (; k < N / 2; ++k) {
            standardX[k] = nativeX[map[k]];
        }
#else
        std::memcpy(standardX, nativeX, sizeof(complex_f32_t) * NB);
#endif
    }

    void standard_to_native_complex_f32(const complex_f32_t* RESTRICT standardX,
                                        complex_f32_t* RESTRICT nativeX) const {
#if defined(BRUUN_HEAPOPT_SPECTRUM_ORDER)
        if (N < 32) {
            std::memcpy(nativeX, standardX, sizeof(complex_f32_t) * NB);
            return;
        }
        nativeX[0] = standardX[0];
        nativeX[N / 2] = standardX[N / 2];
        const int* RESTRICT map = NATIVE_STANDARD_BIN.data();
        int pos = 1;
        for (; pos + 3 < N / 2; pos += 4) {
            copy4_complex_f32(nativeX + pos, standardX,
                              map[pos], map[pos + 1], map[pos + 2], map[pos + 3]);
        }
        for (; pos < N / 2; ++pos) {
            nativeX[pos] = standardX[map[pos]];
        }
#else
        std::memcpy(nativeX, standardX, sizeof(complex_f32_t) * NB);
#endif
    }

    // -----------------------------------------------------------------------
    // Coordinate systems and the call matrix.
    //
    // This plan exposes three representations of the spectrum of a length-N
    // real signal:
    //
    //   1. Residue domain ("residues"): N doubles. v[0], v[1] carry the DC and
    //      Nyquist content (X[0] = v[0] + v[1], X[N/2] = v[0] - v[1]); for each
    //      nontrivial Bruun leaf m in [1, N/2) the pair (v[2m], v[2m+1])
    //      represents the complex bin X[IDX[m]] = v[2m] - i * v[2m+1]. This is
    //      the transform's own CRT coordinate system. It costs no output
    //      permutation at all and is the fastest representation to produce,
    //      to consume, and to multiply filters in.
    //
    //   2. Native spectrum: NB complex bins in the plan's native slot order.
    //      With BRUUN_HEAPOPT_SPECTRUM_ORDER this is the heap-contiguous,
    //      sibling-orientation-optimized Bruun covering order; without it,
    //      native order is ordinary FFTW bin order.
    //
    //   3. Standard spectrum: NB complex bins in ordinary FFTW r2c order.
    //
    // Transform calls:
    //   forward_residues(in, res)        time -> residues        (fastest)
    //   inverse_residues(res)            residues -> time, in place
    //   forward_native(in, X, work)      time -> native spectrum
    //   inverse_native(X, out)           native spectrum -> time
    //   forward_standard(in, X, work, t) time -> standard bins (uses scratch t)
    //   forward(in, X, work)             convenience standard-bin forward
    //   inverse(X, out)                  standard bins -> time
    //   filter_signal(in, RF, out)       time -> time through one residue-domain
    //                                    filter, no bin conversion anywhere
    //
    // Layout converters:
    //   native_to_standard_complex / standard_to_native_complex
    //   residue_filter_from_standard / residue_filter_from_real
    // -----------------------------------------------------------------------

    // Time -> residues. Uses the fused-copy depth-first path when available.
    void forward_residues(const double* RESTRICT input, double* RESTRICT residues) const {
        if (fuse_tail && N >= 64) {
            forward_residues_recursive(input, residues);
            return;
        }
        std::memcpy(residues, input, sizeof(double) * N);
        forward_residues_inplace(residues);
    }

    // Residues -> time, in place.
    void inverse_residues(double* RESTRICT residues_signal) const {
        inverse_residues_inplace(residues_signal);
    }

    // Native spectrum -> time. With BRUUN_HEAPOPT_SPECTRUM_ORDER this reads the
    // heapopt native layout sequentially; otherwise identical to inverse().
    void inverse_native(const complex_t* RESTRICT Xnative, double* RESTRICT out) const {
#if defined(BRUUN_HEAPOPT_SPECTRUM_ORDER)
        if (N < 32) {
            inverse(Xnative, out);
            return;
        }
        out[0] = 0.5 * (Xnative[0].re + Xnative[N / 2].re);
        out[1] = 0.5 * (Xnative[0].re - Xnative[N / 2].re);
        for (int pos = 1; pos < N / 2; ++pos) {
            const int m = NATIVE_LEAF[pos];
            out[2*m] = Xnative[pos].re;
            out[2*m + 1] = -Xnative[pos].im;
        }
        inverse_residues_inplace(out);
#else
        inverse(Xnative, out);
#endif
    }

    // -----------------------------------------------------------------------
    // Residue-domain filters.
    //
    // A filter is an ordinary frequency response H[k] for k = 0..N/2 given in
    // standard FFTW r2c bin order. residue_filter_from_standard() converts it
    // once into a packed residue-domain coefficient array RF of filter_size()
    // doubles, aligned pair-for-pair with the residue vector. After that,
    // apply_residue_filter() is one sequential SIMD pass, and filter_signal()
    // runs time -> residues -> multiply -> time with no bin permutation at any
    // point. H[0] and H[N/2] must be real for a real-output filter; their
    // imaginary parts are ignored.
    //
    // RF layout: RF[0] = (H[0] + H[N/2])/2 and RF[1] = (H[0] - H[N/2])/2 form
    // the symmetric 2x2 acting on (v[0], v[1]); for m in [1, N/2),
    // RF[2m] = Re H[IDX[m]] and RF[2m+1] = Im H[IDX[m]], applied as the
    // conjugate-coordinate complex multiply
    //   y0 = hr*r0 + hi*r1,  y1 = hr*r1 - hi*r0.
    // -----------------------------------------------------------------------

    int filter_size() const { return N; }

    void residue_filter_from_standard(const complex_t* RESTRICT H, double* RESTRICT RF) const {
        RF[0] = 0.5 * (H[0].re + H[N / 2].re);
        RF[1] = 0.5 * (H[0].re - H[N / 2].re);
        for (int m = 1; m < N / 2; ++m) {
            const int k = IDX[m];
            RF[2*m] = H[k].re;
            RF[2*m + 1] = H[k].im;
        }
    }

    // Zero-phase (real) frequency response, Hmag[k] for k = 0..N/2.
    void residue_filter_from_real(const double* RESTRICT Hmag, double* RESTRICT RF) const {
        RF[0] = 0.5 * (Hmag[0] + Hmag[N / 2]);
        RF[1] = 0.5 * (Hmag[0] - Hmag[N / 2]);
        for (int m = 1; m < N / 2; ++m) {
            RF[2*m] = Hmag[IDX[m]];
            RF[2*m + 1] = 0.0;
        }
    }

    // One streaming pass: residues *= filter, entirely in residue coordinates.
    void apply_residue_filter(double* RESTRICT v, const double* RESTRICT RF) const {
        const double v0 = v[0];
        const double v1 = v[1];
        v[0] = RF[0] * v0 + RF[1] * v1;
        v[1] = RF[1] * v0 + RF[0] * v1;

        int j = 2;
        const int end = N;

#if BRUUN_LEVEL >= 2
        {
            const __m256d negodd = _mm256_set_pd(-0.0, 0.0, -0.0, 0.0);
            for (; j + 3 < end; j += 4) {
                const __m256d r = _mm256_loadu_pd(v + j);
                const __m256d h = _mm256_loadu_pd(RF + j);
                const __m256d hd = _mm256_movedup_pd(h);
                const __m256d ho = _mm256_permute_pd(h, 0xF);
                const __m256d rs = _mm256_xor_pd(_mm256_permute_pd(r, 0x5), negodd);
                _mm256_storeu_pd(v + j, _mm256_fmadd_pd(hd, r, _mm256_mul_pd(ho, rs)));
            }
        }
#endif
#if BRUUN_LEVEL >= 1
        for (; j + 1 < end; j += 2) {
            const bruun_v2 r = V2_LD(v + j);
            const bruun_v2 h = V2_LD(RF + j);
            const bruun_v2 rs = V2_NEGHI(V2_UNPLO(V2_DUP1(r), r)); // [r1, -r0]
            V2_ST(v + j, V2_MADD(V2_MUL(V2_DUP0(h), r), V2_DUP1(h), rs));
        }
#endif

        for (; j < end; j += 2) {
            const double r0 = v[j];
            const double r1 = v[j + 1];
            const double hr = RF[j];
            const double hi = RF[j + 1];
            v[j] = hr * r0 + hi * r1;
            v[j + 1] = hr * r1 - hi * r0;
        }
    }

    // time -> residues -> filter -> time, with no bin-order conversion at any
    // point. out must hold N doubles; out may not alias in.
    void filter_signal(const double* RESTRICT in, const double* RESTRICT RF, double* RESTRICT out) const {
        forward_residues(in, out);
        apply_residue_filter(out, RF);
        inverse_residues_inplace(out);
    }

private:
    // Standard-output pack policy thresholds are deliberately backend-specific:
    // AVX2 measured better with two-phase packing from 8192; 128-bit SIMD keeps
    // fused scatter through 1048576 and only switches above it.
    static constexpr int kAvx2TwoPhaseMinN = 8192;
    static constexpr int kSimd128TwoPhaseMinExclusiveN = 1048576;

    int N;
    int L;
    int NB;
    bool fuse_tail;
    heap_array<int> IDX;
    heap_array<int> OUTIDX;
#if defined(BRUUN_HEAPOPT_SPECTRUM_ORDER)
    heap_array<int> NATIVE_POS;
    heap_array<int> NATIVE_LEAF;
    heap_array<int> STANDARD_NATIVE_POS;
    heap_array<int> NATIVE_STANDARD_BIN;
#endif
    heap_array<double> C;

    inline double s_twiddle(int m) const {
        const unsigned u = static_cast<unsigned>(m);
        const unsigned flip = static_cast<unsigned>(-(m > 1)) & 1u;
        return C[u ^ flip];
    }

    struct LeafTw {
        double c4[4];
        double c2[2];
        double c1;
        double s1;
        int32_t idx[8];
    };
    heap_array<LeafTw> TW;

    heap_array<int> KINV;

    struct LeafTwF {
        float c4[4];
        float c2d[4]; // [c2[0], c2[0], c2[1], c2[1]]
        float c1;
        float s1;
    };
    heap_array<LeafTwF> TWF;

    heap_array<float> CF;
    heap_array<FwdOp> FWD_SCHEDULE;
    heap_array<FwdOp> FWD_RES_SCHEDULE;
    heap_array<FwdOp> INV_RES_SCHEDULE;

    inline float sf_twiddle(int m) const {
        const unsigned u = static_cast<unsigned>(m);
        const unsigned flip = static_cast<unsigned>(-(m > 1)) & 1u;
        return CF[u ^ flip];
    }

    bool append_op(heap_array<FwdOp>& ops,
                   uint16_t kind,
                   uint32_t base,
                   uint32_t q,
                   uint32_t m) const {
        FwdOp op{};
        op.base = base;
        op.q = q;
        op.m = m;
        op.kind = kind;
        return ops.push_back(op);
    }

    struct ScheduleSegment {
        uint32_t base;
        int q;
        int m;
    };

    bool append_segment_schedule(heap_array<FwdOp>& ops,
                                 uint32_t root_base,
                                 int root_q,
                                 int root_m,
                                 bool pack_to_complex) const {
        heap_array<ScheduleSegment> stack;
        if (!stack.reserve(96)) return false;
        if (!stack.push_back(ScheduleSegment{root_base, root_q, root_m})) return false;
        while (!stack.empty()) {
            const ScheduleSegment segment = stack.back();
            stack.pop_back();
            const uint32_t base = segment.base;
            const int q = segment.q;
            const int m = segment.m;
            if (q >= 16) {
                if (!append_op(ops, FWD_OP_NORM2, base, static_cast<uint32_t>(q), static_cast<uint32_t>(m))) return false;
                const int qq = q >> 2;
                const int child_m = 4 * m;
                if (!stack.push_back(ScheduleSegment{base + static_cast<uint32_t>(3 * q), qq, child_m + 3})) return false;
                if (!stack.push_back(ScheduleSegment{base + static_cast<uint32_t>(2 * q), qq, child_m + 2})) return false;
                if (!stack.push_back(ScheduleSegment{base + static_cast<uint32_t>(q), qq, child_m + 1})) return false;
                if (!stack.push_back(ScheduleSegment{base, qq, child_m})) return false;
            } else if (q == 8) {
                if (!append_op(ops, FWD_OP_CODELET_Q8, base, static_cast<uint32_t>(q), static_cast<uint32_t>(m))) return false;
            } else {
                if (!append_op(ops, FWD_OP_CODELET_D3, base, static_cast<uint32_t>(q), static_cast<uint32_t>(m))) return false;
            }
            (void)pack_to_complex;
        }
        return true;
    }

    bool copy_schedule(const heap_array<FwdOp>& source, heap_array<FwdOp>& target) {
        if (!target.resize(source.size())) return false;
        for (std::size_t i = 0; i < source.size(); ++i) {
            target[i] = source[i];
        }
        return true;
    }

    bool build_inverse_residue_schedule() {
        const std::size_t source_count = FWD_RES_SCHEDULE.size();
        if (!INV_RES_SCHEDULE.resize(source_count + 1)) return false;
        for (std::size_t i = 0; i < source_count; ++i) {
            INV_RES_SCHEDULE[i] = FWD_RES_SCHEDULE[source_count - 1 - i];
        }
        FwdOp final_op{};
        final_op.base = 0;
        final_op.q = static_cast<uint32_t>(N / 2);
        final_op.m = 0;
        final_op.kind = FWD_OP_BINOMIAL;
        INV_RES_SCHEDULE[source_count] = final_op;
        return true;
    }

    bool build_forward_schedules() {
        heap_array<FwdOp> fused_ops;
        heap_array<FwdOp> residue_ops;
        if (!fused_ops.reserve(N / 4 + 16)) return false;
        if (!residue_ops.reserve(N / 4 + 16)) return false;
        if (N >= 64) {
            for (int h = N / 2; h >= 32; h >>= 1) {
                if (!append_segment_schedule(fused_ops, static_cast<uint32_t>(h), h >> 2, 1, true)) return false;
                if (!append_segment_schedule(residue_ops, static_cast<uint32_t>(h), h >> 2, 1, false)) return false;
                if (!append_op(fused_ops, FWD_OP_BINOMIAL, 0, static_cast<uint32_t>(h >> 1), 0)) return false;
                if (!append_op(residue_ops, FWD_OP_BINOMIAL, 0, static_cast<uint32_t>(h >> 1), 0)) return false;
            }
            if (!append_op(fused_ops, FWD_OP_SPINE_D3, 16, 0, 1)) return false;
            if (!append_op(fused_ops, FWD_OP_BINOMIAL, 0, 8, 0)) return false;
            if (!append_op(fused_ops, FWD_OP_SPINE_D2, 8, 0, 1)) return false;
            if (!append_op(fused_ops, FWD_OP_BINOMIAL, 0, 4, 0)) return false;
            if (!append_op(fused_ops, FWD_OP_SPINE_D1, 4, 0, 1)) return false;
            if (!append_op(fused_ops, FWD_OP_BINOMIAL, 0, 2, 0)) return false;
            if (!append_op(fused_ops, FWD_OP_SPINE_LEAF, 2, 0, 1)) return false;
            if (!append_op(fused_ops, FWD_OP_DC_NYQUIST, 0, 0, 0)) return false;

            if (!append_op(residue_ops, FWD_OP_SPINE_D3, 16, 0, 1)) return false;
            if (!append_op(residue_ops, FWD_OP_BINOMIAL, 0, 8, 0)) return false;
            if (!append_op(residue_ops, FWD_OP_SPINE_NORM2, 8, 2, 1)) return false;
            if (!append_op(residue_ops, FWD_OP_BINOMIAL, 0, 4, 0)) return false;
            if (!append_op(residue_ops, FWD_OP_SPINE_D1, 4, 0, 1)) return false;
            if (!append_op(residue_ops, FWD_OP_BINOMIAL, 0, 2, 0)) return false;
        }
        if (!copy_schedule(fused_ops, FWD_SCHEDULE)) return false;
        if (!copy_schedule(residue_ops, FWD_RES_SCHEDULE)) return false;
        if (!build_inverse_residue_schedule()) return false;
        return true;
    }

    struct FloatSegment {
        float* data;
        int q;
        int m;
    };

    static inline void store4_interleaved_f32(float* RESTRICT dst,
                                             float re0, float im0,
                                             float re1, float im1,
                                             float re2, float im2,
                                             float re3, float im3) {
#if BRUUN_LEVEL >= 1
        const bruun_v4f re = V4F_SET4(re0, re1, re2, re3);
        const bruun_v4f im = V4F_SET4(im0, im1, im2, im3);
        V4F_ST(dst, V4F_ZIPLO(re, im));
        V4F_ST(dst + 4, V4F_ZIPHI(re, im));
#else
        dst[0] = re0;
        dst[1] = im0;
        dst[2] = re1;
        dst[3] = im1;
        dst[4] = re2;
        dst[5] = im2;
        dst[6] = re3;
        dst[7] = im3;
#endif
    }

    static inline void store4_complex_f32(complex_f32_t* RESTRICT dst,
                                         float re0, float im0,
                                         float re1, float im1,
                                         float re2, float im2,
                                         float re3, float im3) {
        store4_interleaved_f32(reinterpret_cast<float*>(dst),
                               re0, im0, re1, im1, re2, im2, re3, im3);
    }

    static inline void pack4_residues_to_complex_f32(complex_f32_t* RESTRICT dst,
                                                     const float* RESTRICT residues,
                                                     int m0, int m1, int m2, int m3) {
        store4_complex_f32(dst,
                           residues[2*m0], -residues[2*m0 + 1],
                           residues[2*m1], -residues[2*m1 + 1],
                           residues[2*m2], -residues[2*m2 + 1],
                           residues[2*m3], -residues[2*m3 + 1]);
    }

    static inline void copy4_complex_f32(complex_f32_t* RESTRICT dst,
                                        const complex_f32_t* RESTRICT src,
                                        int i0, int i1, int i2, int i3) {
        store4_complex_f32(dst,
                           src[i0].re, src[i0].im,
                           src[i1].re, src[i1].im,
                           src[i2].re, src[i2].im,
                           src[i3].re, src[i3].im);
    }

    static inline void unpack4_complex_to_residues_f32(float* RESTRICT dst,
                                                       const complex_f32_t* RESTRICT src,
                                                       int i0, int i1, int i2, int i3) {
        store4_interleaved_f32(dst,
                               src[i0].re, -src[i0].im,
                               src[i1].re, -src[i1].im,
                               src[i2].re, -src[i2].im,
                               src[i3].re, -src[i3].im);
    }

    static inline void norm_q1_fwd_f32(float* RESTRICT p, float c, float s) {
        const float A0 = p[0];
        const float B0 = p[1];
        const float A1 = p[2];
        const float B1 = p[3];

        const float R = c * B0 - s * B1;
        const float I = s * B0 + c * B1;

        p[0] = A0 + R;
        p[1] = A1 + I;
        p[2] = A0 - R;
        p[3] = -A1 + I;
    }

    static inline void norm_q2_fwd_f32(float* RESTRICT p, float c, float s) {
        for (int n = 0; n < 2; ++n) {
            const float A0 = p[n];
            const float B0 = p[2 + n];
            const float A1 = p[4 + n];
            const float B1 = p[6 + n];

            const float R = c * B0 - s * B1;
            const float I = s * B0 + c * B1;

            p[n] = A0 + R;
            p[2 + n] = A1 + I;
            p[4 + n] = A0 - R;
            p[6 + n] = -A1 + I;
        }
    }

    // Residue-writing float leaf codelet: three tree levels on one 16-float
    // block, fused in 128-bit registers. Lane-for-lane twin of the AVX2 double
    // codelet (4 doubles per __m256d there, 4 floats per 128-bit vector here).
    inline void codelet_d3_tw_res_f32(float* RESTRICT p, const LeafTwF& t) const {
#if BRUUN_LEVEL >= 1
        const bruun_v4f A0 = V4F_LD(p);
        const bruun_v4f B0 = V4F_LD(p + 4);
        const bruun_v4f A1 = V4F_LD(p + 8);
        const bruun_v4f B1 = V4F_LD(p + 12);

        const bruun_v4f c1 = V4F_SET1(t.c1);
        const bruun_v4f s1 = V4F_SET1(t.s1);
        const bruun_v4f R1 = V4F_MSUB(V4F_MUL(c1, B0), s1, B1);
        const bruun_v4f I1 = V4F_MADD(V4F_MUL(s1, B0), c1, B1);

        const bruun_v4f c0a = V4F_ADD(A0, R1);
        const bruun_v4f c0b = V4F_ADD(A1, I1);
        const bruun_v4f c1a = V4F_SUB(A0, R1);
        const bruun_v4f c1b = V4F_SUB(I1, A1);

        const bruun_v4f A0v = V4F_CATLO(c0a, c1a);
        const bruun_v4f B0v = V4F_CATHI(c0a, c1a);
        const bruun_v4f A1v = V4F_CATLO(c0b, c1b);
        const bruun_v4f B1v = V4F_CATHI(c0b, c1b);

        const bruun_v4f c2 = V4F_LD(t.c2d);
        const bruun_v4f s2 = V4F_SWAP_HALVES(c2);
        const bruun_v4f R2 = V4F_MSUB(V4F_MUL(c2, B0v), s2, B1v);
        const bruun_v4f I2 = V4F_MADD(V4F_MUL(s2, B0v), c2, B1v);

        const bruun_v4f P = V4F_ADD(A0v, R2);
        const bruun_v4f Q = V4F_ADD(A1v, I2);
        const bruun_v4f M = V4F_SUB(A0v, R2);
        const bruun_v4f W = V4F_SUB(I2, A1v);

        const bruun_v4f pmL = V4F_ZIPLO(P, M);
        const bruun_v4f pmH = V4F_ZIPHI(P, M);
        const bruun_v4f qwL = V4F_ZIPLO(Q, W);
        const bruun_v4f qwH = V4F_ZIPHI(Q, W);
        const bruun_v4f A0w = V4F_CATLO(pmL, pmH); // [P0,M0,P2,M2]
        const bruun_v4f B0w = V4F_CATHI(pmL, pmH); // [P1,M1,P3,M3]
        const bruun_v4f A1w = V4F_CATLO(qwL, qwH);
        const bruun_v4f B1w = V4F_CATHI(qwL, qwH);

        const bruun_v4f c4 = V4F_LD(t.c4);
        const bruun_v4f s4 = V4F_SWAP_PAIRS(c4);
        const bruun_v4f R3 = V4F_MSUB(V4F_MUL(c4, B0w), s4, B1w);
        const bruun_v4f I3 = V4F_MADD(V4F_MUL(s4, B0w), c4, B1w);

        const bruun_v4f E0 = V4F_ADD(A0w, R3); // leaf even residue r0
        const bruun_v4f E1 = V4F_ADD(A1w, I3); // leaf even residue r1
        const bruun_v4f O0 = V4F_SUB(A0w, R3); // leaf odd residue r0
        const bruun_v4f O1 = V4F_SUB(I3, A1w); // leaf odd residue r1

        // 4x4 transpose to leaf-residue order: p[4g..4g+3] = [E0[g], E1[g], O0[g], O1[g]].
        const bruun_v4f eL = V4F_ZIPLO(E0, E1);
        const bruun_v4f eH = V4F_ZIPHI(E0, E1);
        const bruun_v4f oL = V4F_ZIPLO(O0, O1);
        const bruun_v4f oH = V4F_ZIPHI(O0, O1);
        const bruun_v4f t0 = V4F_CATLO(eL, eH); // [E0_0,E1_0,E0_2,E1_2]
        const bruun_v4f t1 = V4F_CATHI(eL, eH); // [E0_1,E1_1,E0_3,E1_3]
        const bruun_v4f t2 = V4F_CATLO(oL, oH);
        const bruun_v4f t3 = V4F_CATHI(oL, oH);

        V4F_ST(p,      V4F_CATLO(t0, t2));
        V4F_ST(p + 4,  V4F_CATLO(t1, t3));
        V4F_ST(p + 8,  V4F_CATHI(t0, t2));
        V4F_ST(p + 12, V4F_CATHI(t1, t3));
#else
        norm_q_fwd_f32(p, 4, t.c1, t.s1);
        norm_q2_fwd_f32(p, t.c2d[0], t.c2d[2]);
        norm_q2_fwd_f32(p + 8, t.c2d[2], t.c2d[0]);
        norm_q1_fwd_f32(p, t.c4[0], t.c4[1]);
        norm_q1_fwd_f32(p + 4, t.c4[1], t.c4[0]);
        norm_q1_fwd_f32(p + 8, t.c4[2], t.c4[3]);
        norm_q1_fwd_f32(p + 12, t.c4[3], t.c4[2]);
#endif
    }

    // Exact inverse of codelet_d3_tw_res_f32.
    inline void codelet_d3_tw_res_inv_f32(float* RESTRICT p, const LeafTwF& t) const {
#if BRUUN_LEVEL >= 1
        const bruun_v4f out0 = V4F_LD(p);
        const bruun_v4f out1 = V4F_LD(p + 4);
        const bruun_v4f out2 = V4F_LD(p + 8);
        const bruun_v4f out3 = V4F_LD(p + 12);

        // Inverse of the forward 4x4 transpose.
        const bruun_v4f aL = V4F_ZIPLO(out0, out1);
        const bruun_v4f aH = V4F_ZIPHI(out0, out1);
        const bruun_v4f bL = V4F_ZIPLO(out2, out3);
        const bruun_v4f bH = V4F_ZIPHI(out2, out3);
        const bruun_v4f t0 = V4F_CATLO(aL, aH);
        const bruun_v4f t1 = V4F_CATHI(aL, aH);
        const bruun_v4f t2 = V4F_CATLO(bL, bH);
        const bruun_v4f t3 = V4F_CATHI(bL, bH);
        const bruun_v4f E0 = V4F_CATLO(t0, t2);
        const bruun_v4f O0 = V4F_CATHI(t0, t2);
        const bruun_v4f E1 = V4F_CATLO(t1, t3);
        const bruun_v4f O1 = V4F_CATHI(t1, t3);

        const bruun_v4f hf = V4F_SET1(0.5f);

        // Level 3 inverse.
        const bruun_v4f A0w = V4F_MUL(hf, V4F_ADD(E0, O0));
        const bruun_v4f R3  = V4F_MUL(hf, V4F_SUB(E0, O0));
        const bruun_v4f I3  = V4F_MUL(hf, V4F_ADD(E1, O1));
        const bruun_v4f A1w = V4F_MUL(hf, V4F_SUB(E1, O1));
        const bruun_v4f c4 = V4F_LD(t.c4);
        const bruun_v4f s4 = V4F_SWAP_PAIRS(c4);
        const bruun_v4f B0w = V4F_MADD(V4F_MUL(c4, R3), s4, I3);
        const bruun_v4f B1w = V4F_MSUB(V4F_MUL(c4, I3), s4, R3);

        // Level 2 inverse.
        const bruun_v4f abL = V4F_ZIPLO(A0w, B0w);
        const bruun_v4f abH = V4F_ZIPHI(A0w, B0w);
        const bruun_v4f cdL = V4F_ZIPLO(A1w, B1w);
        const bruun_v4f cdH = V4F_ZIPHI(A1w, B1w);
        const bruun_v4f P = V4F_CATLO(abL, abH);
        const bruun_v4f M = V4F_CATHI(abL, abH);
        const bruun_v4f Q = V4F_CATLO(cdL, cdH);
        const bruun_v4f W = V4F_CATHI(cdL, cdH);

        const bruun_v4f A0v = V4F_MUL(hf, V4F_ADD(P, M));
        const bruun_v4f R2  = V4F_MUL(hf, V4F_SUB(P, M));
        const bruun_v4f I2  = V4F_MUL(hf, V4F_ADD(Q, W));
        const bruun_v4f A1v = V4F_MUL(hf, V4F_SUB(Q, W));
        const bruun_v4f c2 = V4F_LD(t.c2d);
        const bruun_v4f s2 = V4F_SWAP_HALVES(c2);
        const bruun_v4f B0v = V4F_MADD(V4F_MUL(c2, R2), s2, I2);
        const bruun_v4f B1v = V4F_MSUB(V4F_MUL(c2, I2), s2, R2);

        // Level 1 inverse.
        const bruun_v4f c0a = V4F_CATLO(A0v, B0v);
        const bruun_v4f c1a = V4F_CATHI(A0v, B0v);
        const bruun_v4f c0b = V4F_CATLO(A1v, B1v);
        const bruun_v4f c1b = V4F_CATHI(A1v, B1v);

        const bruun_v4f A0 = V4F_MUL(hf, V4F_ADD(c0a, c1a));
        const bruun_v4f R1 = V4F_MUL(hf, V4F_SUB(c0a, c1a));
        const bruun_v4f I1 = V4F_MUL(hf, V4F_ADD(c0b, c1b));
        const bruun_v4f A1 = V4F_MUL(hf, V4F_SUB(c0b, c1b));
        const bruun_v4f c1v = V4F_SET1(t.c1);
        const bruun_v4f s1v = V4F_SET1(t.s1);
        const bruun_v4f B0 = V4F_MADD(V4F_MUL(c1v, R1), s1v, I1);
        const bruun_v4f B1 = V4F_MSUB(V4F_MUL(c1v, I1), s1v, R1);

        V4F_ST(p,      A0);
        V4F_ST(p + 4,  B0);
        V4F_ST(p + 8,  A1);
        V4F_ST(p + 12, B1);
#else
        norm_q_inv_f32(p,      1, t.c4[0], t.c4[1]);
        norm_q_inv_f32(p + 4,  1, t.c4[1], t.c4[0]);
        norm_q_inv_f32(p + 8,  1, t.c4[2], t.c4[3]);
        norm_q_inv_f32(p + 12, 1, t.c4[3], t.c4[2]);
        norm_q_inv_f32(p,      2, t.c2d[0], t.c2d[2]);
        norm_q_inv_f32(p + 8,  2, t.c2d[2], t.c2d[0]);
        norm_q_inv_f32(p,      4, t.c1,    t.s1);
#endif
    }

    void run_fwd_res_segments_f32(float* RESTRICT root, int root_q, int root_m) const {
        FloatSegment stack[96];
        int stack_size = 0;
        stack[stack_size++] = FloatSegment{root, root_q, root_m};

        while (stack_size > 0) {
            --stack_size;
            const FloatSegment segment = stack[stack_size];
            float* RESTRICT v = segment.data;
            const int q = segment.q;
            const int m = segment.m;

            if (q >= 16) {
                norm2_fused_f32(v, q, CF[m], sf_twiddle(m), CF[2*m], sf_twiddle(2*m), CF[2*m+1], sf_twiddle(2*m+1));
                const int qq = q >> 2;
                const int child_m = 4 * m;

                stack[stack_size++] = FloatSegment{v + 3*q, qq, child_m + 3};
                stack[stack_size++] = FloatSegment{v + 2*q, qq, child_m + 2};
                stack[stack_size++] = FloatSegment{v + q,   qq, child_m + 1};
                stack[stack_size++] = FloatSegment{v,       qq, child_m};
                continue;
            }

            if (q == 8) {
                norm_q_fwd_f32(v, 8, CF[m], sf_twiddle(m));
                codelet_d3_tw_res_f32(v, TWF[2*m]);
                codelet_d3_tw_res_f32(v + 16, TWF[2*m + 1]);
                continue;
            }

            codelet_d3_tw_res_f32(v, TWF[m]);
        }
    }

    void residue_spine_tail_fwd_f32(float* RESTRICT v) const {
        codelet_d3_tw_res_f32(v + 16, TWF[1]);
        binomial_fwd_f32(v, 8);
        norm_q_fwd_f32(v + 8, 2, CF[1], sf_twiddle(1));
        norm_q1_fwd_f32(v + 8, CF[2], sf_twiddle(2));
        norm_q1_fwd_f32(v + 12, CF[3], sf_twiddle(3));
        binomial_fwd_f32(v, 4);
        norm_q1_fwd_f32(v + 4, CF[1], sf_twiddle(1));
        binomial_fwd_f32(v, 2);
    }

    // Fast depth-first float residue forward with fused input copy. Requires N >= 64.
    void forward_residues_recursive_f32(const float* RESTRICT input, float* RESTRICT v) const {
        binomial_oop_f32(input, v, N / 2);

        for (int h = N / 2; h >= 32; h >>= 1) {
            run_fwd_res_segments_f32(v + h, h >> 2, 1);
            binomial_fwd_f32(v, h >> 1);
        }

        residue_spine_tail_fwd_f32(v);
    }

    void forward_stage_f32(float* RESTRICT v, int jj) const {
        const int s = N >> jj;
        const int h = s >> 1;
        const int q = s >> 2;
        const int m_end = 1 << jj;

        binomial_fwd_f32(v, h);

        if (q == 1) {
            for (int m = 1; m < m_end; ++m) norm_q1_fwd_f32(v + m*s, CF[m], sf_twiddle(m));
        } else if (q == 2) {
            for (int m = 1; m < m_end; ++m) norm_q2_fwd_f32(v + m*s, CF[m], sf_twiddle(m));
        } else {
            for (int m = 1; m < m_end; ++m) norm_q_fwd_f32(v + m*s, q, CF[m], sf_twiddle(m));
        }
    }

    void forward_residues_inplace_f32(float* RESTRICT v) const {
        for (int jj = 0; jj < L - 1; ++jj) {
            forward_stage_f32(v, jj);
        }
    }

    // Time -> residues, float. Leaf m lands in (v[2m], v[2m+1]); v[0], v[1]
    // carry DC and Nyquist content exactly as in the double residue domain.
    void forward_residues_f32(const float* RESTRICT input, float* RESTRICT v) const {
        if (fuse_tail && N >= 64) {
            forward_residues_recursive_f32(input, v);
            return;
        }
        std::memcpy(v, input, sizeof(float) * N);
        forward_residues_inplace_f32(v);
    }

    void rec_inv_res_f32(float* RESTRICT v, int q, int m) const {
        if (q >= 16) {
            const int qq = q >> 2;
            rec_inv_res_f32(v,       qq, 4*m);
            rec_inv_res_f32(v + q,   qq, 4*m + 1);
            rec_inv_res_f32(v + 2*q, qq, 4*m + 2);
            rec_inv_res_f32(v + 3*q, qq, 4*m + 3);
            norm2_inv_fused_f32(v, q, CF[m], sf_twiddle(m), CF[2*m], sf_twiddle(2*m), CF[2*m+1], sf_twiddle(2*m+1));
            return;
        }
        if (q == 8) {
            codelet_d3_tw_res_inv_f32(v, TWF[2*m]);
            codelet_d3_tw_res_inv_f32(v + 16, TWF[2*m + 1]);
            norm_q_inv_f32(v, 8, CF[m], sf_twiddle(m));
            return;
        }
        codelet_d3_tw_res_inv_f32(v, TWF[m]);
    }

    void residue_spine_tail_inv_f32(float* RESTRICT v) const {
        binomial_inv_f32(v, 2);
        norm_q_inv_f32(v + 4, 1, CF[1], sf_twiddle(1));
        binomial_inv_f32(v, 4);
        norm_q_inv_f32(v + 8, 1, CF[2], sf_twiddle(2));
        norm_q_inv_f32(v + 12, 1, CF[3], sf_twiddle(3));
        norm_q_inv_f32(v + 8, 2, CF[1], sf_twiddle(1));
        binomial_inv_f32(v, 8);
        codelet_d3_tw_res_inv_f32(v + 16, TWF[1]);
    }

    void run_inv_residue_schedule_f32(float* RESTRICT v) const {
        const FwdOp* RESTRICT ops = INV_RES_SCHEDULE.data();
        const std::size_t op_count = INV_RES_SCHEDULE.size();
        for (std::size_t op_index = 0; op_index < op_count; ++op_index) {
            const FwdOp& op = ops[op_index];
            float* RESTRICT base = v + op.base;
#if defined(__GNUC__) || defined(__clang__)
            const uint32_t next_base = op_index + 1 < op_count ? ops[op_index + 1].base : 0;
            __builtin_prefetch(v + next_base, 1, 1);
#endif
            switch (op.kind) {
            case FWD_OP_NORM2: {
                const int m = static_cast<int>(op.m);
                const int q = static_cast<int>(op.q);
                norm2_inv_fused_f32(base, q, CF[m], sf_twiddle(m), CF[2*m], sf_twiddle(2*m), CF[2*m+1], sf_twiddle(2*m+1));
                break;
            }
            case FWD_OP_CODELET_Q8: {
                const int m = static_cast<int>(op.m);
                codelet_d3_tw_res_inv_f32(base, TWF[2*m]);
                codelet_d3_tw_res_inv_f32(base + 16, TWF[2*m + 1]);
                norm_q_inv_f32(base, 8, CF[m], sf_twiddle(m));
                break;
            }
            case FWD_OP_CODELET_D3:
            case FWD_OP_SPINE_D3:
                codelet_d3_tw_res_inv_f32(base, TWF[op.m]);
                break;
            case FWD_OP_BINOMIAL:
                binomial_inv_f32(base, static_cast<int>(op.q));
                break;
            case FWD_OP_SPINE_D1:
                norm_q_inv_f32(base, 1, CF[1], sf_twiddle(1));
                break;
            case FWD_OP_SPINE_NORM2:
                norm_q_inv_f32(base, 1, CF[2], sf_twiddle(2));
                norm_q_inv_f32(base + 4, 1, CF[3], sf_twiddle(3));
                norm_q_inv_f32(base, 2, CF[1], sf_twiddle(1));
                break;
            default:
                break;
            }
        }
    }

    // Exact reverse of forward_residues_recursive_f32. Requires N >= 64.
    void inverse_residues_recursive_f32(float* RESTRICT v) const {
#if BRUUN_LEVEL == 1
        run_inv_residue_schedule_f32(v);
#else
        residue_spine_tail_inv_f32(v);

        for (int h = 32; h <= N / 2; h <<= 1) {
            binomial_inv_f32(v, h >> 1);
            rec_inv_res_f32(v + h, h >> 2, 1);
        }

        binomial_inv_f32(v, N / 2);
#endif
    }

    void inverse_residues_inplace_f32(float* RESTRICT v) const {
        if (fuse_tail && N >= 64) {
            inverse_residues_recursive_f32(v);
            return;
        }

        for (int jj = L - 2; jj >= 0; --jj) {
            const int s = N >> jj;
            const int q = s >> 2;
            const int m_end = 1 << jj;

            for (int m = m_end - 1; m > 0; --m) {
                norm_q_inv_f32(v + m*s, q, CF[m], sf_twiddle(m));
            }

            binomial_inv_f32(v, s >> 1);
        }
    }

    static inline void norm_q1_fwd(double* RESTRICT p, double c, double s) {
        const double A0 = p[0];
        const double B0 = p[1];
        const double A1 = p[2];
        const double B1 = p[3];

        const double R = c * B0 - s * B1;
        const double I = s * B0 + c * B1;

        p[0] = A0 + R;
        p[1] = A1 + I;
        p[2] = A0 - R;
        p[3] = -A1 + I;
    }

    static inline void norm_q2_fwd(double* RESTRICT p, double c, double s) {
        for (int n = 0; n < 2; ++n) {
            const double A0 = p[n];
            const double B0 = p[2 + n];
            const double A1 = p[4 + n];
            const double B1 = p[6 + n];

            const double R = c * B0 - s * B1;
            const double I = s * B0 + c * B1;

            p[n] = A0 + R;
            p[2 + n] = A1 + I;
            p[4 + n] = A0 - R;
            p[6 + n] = -A1 + I;
        }
    }

    inline void codelet_d1_pack(const double* RESTRICT p, int m, complex_t* RESTRICT X) const {
        const double t0 = C[m];
        const double t1 = s_twiddle(m);
        const double t2 = t0 * p[1] - t1 * p[3];
        const double t3 = t1 * p[1] + t0 * p[3];
        const int k0 = OUTIDX[2*m];
        X[k0].re = p[0] + t2;
        X[k0].im = -(p[2] + t3);
        const int k1 = OUTIDX[2*m + 1];
        X[k1].re = p[0] - t2;
        X[k1].im = p[2] - t3;
    }

    inline void codelet_d2_pack(const double* RESTRICT p, int m, complex_t* RESTRICT X) const {
        const double t0 = C[m];
        const double t1 = s_twiddle(m);
        const double t2 = t0 * p[2] - t1 * p[6];
        const double t3 = t1 * p[2] + t0 * p[6];
        const double t4 = t0 * p[3] - t1 * p[7];
        const double t5 = t1 * p[3] + t0 * p[7];
        const double t6 = C[2*m];
        const double t7 = s_twiddle(2*m);
        const double t8 = t6 * (p[1] + t4) - t7 * (p[5] + t5);
        const double t9 = t7 * (p[1] + t4) + t6 * (p[5] + t5);
        const int k0 = OUTIDX[4*m];
        X[k0].re = (p[0] + t2) + t8;
        X[k0].im = -((p[4] + t3) + t9);
        const int k1 = OUTIDX[4*m + 1];
        X[k1].re = (p[0] + t2) - t8;
        X[k1].im = (p[4] + t3) - t9;
        const double t12 = C[2*m + 1];
        const double t13 = s_twiddle(2*m + 1);
        const double t14 = t12 * (p[1] - t4) - t13 * (t5 - p[5]);
        const double t15 = t13 * (p[1] - t4) + t12 * (t5 - p[5]);
        const int k2 = OUTIDX[4*m + 2];
        X[k2].re = (p[0] - t2) + t14;
        X[k2].im = -((t3 - p[4]) + t15);
        const int k3 = OUTIDX[4*m + 3];
        X[k3].re = (p[0] - t2) - t14;
        X[k3].im = (t3 - p[4]) - t15;
    }

    // ----- depth-3 leaf codelets, all driven by the packed LeafTw stream -----

    // Scalar reference leaf codelet: 16-double block of node m -> 8 spectrum bins.
    inline void codelet_d3_tw_scalar(const double* RESTRICT p, const LeafTw& t, complex_t* RESTRICT X) const {
        double u[4], w[4], v[4], x[4];
        for (int j = 0; j < 4; ++j) {
            const double R = t.c1 * p[4 + j] - t.s1 * p[12 + j];
            const double I = t.s1 * p[4 + j] + t.c1 * p[12 + j];
            u[j] = p[j] + R;
            w[j] = p[8 + j] + I;
            v[j] = p[j] - R;
            x[j] = I - p[8 + j];
        }

        double g[4][4]; // four leaf blocks [A0, B0, A1, B1]
        for (int j = 0; j < 2; ++j) {
            const double R0 = t.c2[0] * u[2 + j] - t.c2[1] * w[2 + j];
            const double I0 = t.c2[1] * u[2 + j] + t.c2[0] * w[2 + j];
            g[0][2*0 + ((j == 0) ? 0 : 1)] = 0; // placeholder, overwritten below
            (void)R0; (void)I0;
        }
        // child 0 = [u | w], child 1 = [v | x]; each splits into two leaf blocks.
        {
            const double R0a = t.c2[0] * u[2] - t.c2[1] * w[2];
            const double I0a = t.c2[1] * u[2] + t.c2[0] * w[2];
            const double R0b = t.c2[0] * u[3] - t.c2[1] * w[3];
            const double I0b = t.c2[1] * u[3] + t.c2[0] * w[3];
            g[0][0] = u[0] + R0a; g[0][1] = u[1] + R0b; g[0][2] = w[0] + I0a; g[0][3] = w[1] + I0b;
            g[1][0] = u[0] - R0a; g[1][1] = u[1] - R0b; g[1][2] = I0a - w[0]; g[1][3] = I0b - w[1];

            const double R1a = t.c2[1] * v[2] - t.c2[0] * x[2];
            const double I1a = t.c2[0] * v[2] + t.c2[1] * x[2];
            const double R1b = t.c2[1] * v[3] - t.c2[0] * x[3];
            const double I1b = t.c2[0] * v[3] + t.c2[1] * x[3];
            g[2][0] = v[0] + R1a; g[2][1] = v[1] + R1b; g[2][2] = x[0] + I1a; g[2][3] = x[1] + I1b;
            g[3][0] = v[0] - R1a; g[3][1] = v[1] - R1b; g[3][2] = I1a - x[0]; g[3][3] = I1b - x[1];
        }

        for (int gi = 0; gi < 4; ++gi) {
            const double c = t.c4[gi];
            const double s = t.c4[gi ^ 1];
            const double R = c * g[gi][1] - s * g[gi][3];
            const double I = s * g[gi][1] + c * g[gi][3];
            const int ke = t.idx[2*gi];
            const int ko = t.idx[2*gi + 1];
            X[ke].re = g[gi][0] + R;
            X[ke].im = -(g[gi][2] + I);
            X[ko].re = g[gi][0] - R;
            X[ko].im = g[gi][2] - I;
        }
    }

#if BRUUN_LEVEL == 1
    // 128-bit leaf codelet. Levels 1 and 2 are naturally 2-lane contiguous;
    // level 3 pairs (re, im) so each spectrum bin is one 128-bit store.
    inline void codelet_d3_tw_v2(const double* RESTRICT p, const LeafTw& t, complex_t* RESTRICT X) const {
        const bruun_v2 c1 = V2_SET1(t.c1);
        const bruun_v2 s1 = V2_SET1(t.s1);

        bruun_v2 u[2], w[2], v[2], x[2];
        for (int j = 0; j < 2; ++j) {
            const bruun_v2 b0 = V2_LD(p + 4 + 2*j);
            const bruun_v2 b1 = V2_LD(p + 12 + 2*j);
            const bruun_v2 R = V2_MSUB(V2_MUL(c1, b0), s1, b1);
            const bruun_v2 I = V2_MADD(V2_MUL(s1, b0), c1, b1);
            const bruun_v2 a0 = V2_LD(p + 2*j);
            const bruun_v2 a1 = V2_LD(p + 8 + 2*j);
            u[j] = V2_ADD(a0, R);
            w[j] = V2_ADD(a1, I);
            v[j] = V2_SUB(a0, R);
            x[j] = V2_SUB(I, a1);
        }

        const bruun_v2 c20 = V2_SET1(t.c2[0]);
        const bruun_v2 s20 = V2_SET1(t.c2[1]);
        const bruun_v2 R0 = V2_MSUB(V2_MUL(c20, u[1]), s20, w[1]);
        const bruun_v2 I0 = V2_MADD(V2_MUL(s20, u[1]), c20, w[1]);
        const bruun_v2 g0a = V2_ADD(u[0], R0);
        const bruun_v2 g0b = V2_ADD(w[0], I0);
        const bruun_v2 g1a = V2_SUB(u[0], R0);
        const bruun_v2 g1b = V2_SUB(I0, w[0]);

        const bruun_v2 c21 = V2_SET1(t.c2[1]);
        const bruun_v2 s21 = V2_SET1(t.c2[0]);
        const bruun_v2 R1 = V2_MSUB(V2_MUL(c21, v[1]), s21, x[1]);
        const bruun_v2 I1 = V2_MADD(V2_MUL(s21, v[1]), c21, x[1]);
        const bruun_v2 g2a = V2_ADD(v[0], R1);
        const bruun_v2 g2b = V2_ADD(x[0], I1);
        const bruun_v2 g3a = V2_SUB(v[0], R1);
        const bruun_v2 g3b = V2_SUB(I1, x[0]);

        const bruun_v2 ga[4] = { g0a, g1a, g2a, g3a };
        const bruun_v2 gb[4] = { g0b, g1b, g2b, g3b };

        for (int gi = 0; gi < 4; ++gi) {
            const bruun_v2 a02 = V2_UNPLO(ga[gi], gb[gi]); // [x0, x2]
            const bruun_v2 b13 = V2_UNPHI(ga[gi], gb[gi]); // [x1, x3]
            const bruun_v2 csv = V2_SETLH(t.c4[gi], t.c4[gi ^ 1]);   // [ c, s]
            const bruun_v2 cs2 = V2_SETLH(-t.c4[gi ^ 1], t.c4[gi]);  // [-s, c]
            const bruun_v2 tv = V2_MADD(V2_MUL(csv, V2_DUP0(b13)), cs2, V2_DUP1(b13)); // [R, I]
            const bruun_v2 ev = V2_NEGHI(V2_ADD(a02, tv)); // [x0+R, -(x2+I)]
            const bruun_v2 od = V2_SUB(a02, tv);           // [x0-R,   x2-I ]
            V2_ST(&X[t.idx[2*gi]].re, ev);
            V2_ST(&X[t.idx[2*gi + 1]].re, od);
        }
    }
#endif

#if BRUUN_LEVEL >= 2
    // 256-bit depth-3 leaf core operating on register inputs: the four quarters
    // of one 16-double leaf block.
    inline void d3_avx2_core(bruun_v4d A0, bruun_v4d B0, bruun_v4d A1, bruun_v4d B1,
                             const LeafTw& t, complex_t* RESTRICT X) const {
        const bruun_v4d c1 = V4D_SET1(t.c1);
        const bruun_v4d s1 = V4D_SET1(t.s1);
        const bruun_v4d R1 = V4D_MSUB(V4D_MUL(c1, B0), s1, B1);
        const bruun_v4d I1 = V4D_MADD(V4D_MUL(s1, B0), c1, B1);

        const bruun_v4d c0a = V4D_ADD(A0, R1);
        const bruun_v4d c0b = V4D_ADD(A1, I1);
        const bruun_v4d c1a = V4D_SUB(A0, R1);
        const bruun_v4d c1b = V4D_SUB(I1, A1);

        const bruun_v4d A0v = V4D_CAT128_LOHI(c0a, c1a);
        const bruun_v4d B0v = V4D_CAT128_HIHI(c0a, c1a);
        const bruun_v4d A1v = V4D_CAT128_LOHI(c0b, c1b);
        const bruun_v4d B1v = V4D_CAT128_HIHI(c0b, c1b);

        const bruun_v4d c2 = V4D_LD2_DUP(t.c2);
        const bruun_v4d s2 = V4D_SWAP_HALVES(c2);
        const bruun_v4d R2 = V4D_MSUB(V4D_MUL(c2, B0v), s2, B1v);
        const bruun_v4d I2 = V4D_MADD(V4D_MUL(s2, B0v), c2, B1v);

        const bruun_v4d P = V4D_ADD(A0v, R2);
        const bruun_v4d Q = V4D_ADD(A1v, I2);
        const bruun_v4d M = V4D_SUB(A0v, R2);
        const bruun_v4d W = V4D_SUB(I2, A1v);

        const bruun_v4d A0w = V4D_UNPLO(P, M);
        const bruun_v4d B0w = V4D_UNPHI(P, M);
        const bruun_v4d A1w = V4D_UNPLO(Q, W);
        const bruun_v4d B1w = V4D_UNPHI(Q, W);

        const bruun_v4d c4 = V4D_LD(t.c4);
        const bruun_v4d s4 = V4D_SWAP_PAIRS(c4);
        const bruun_v4d R3 = V4D_MSUB(V4D_MUL(c4, B0w), s4, B1w);
        const bruun_v4d I3 = V4D_MADD(V4D_MUL(s4, B0w), c4, B1w);

        const bruun_v4d sgn = V4D_NEG_ZERO();
        const bruun_v4d re_e = V4D_ADD(A0w, R3);
        const bruun_v4d re_o = V4D_SUB(A0w, R3);
        const bruun_v4d im_e = V4D_XOR(V4D_ADD(A1w, I3), sgn);
        const bruun_v4d im_o = V4D_SUB(A1w, I3);

        const bruun_v4d pe = V4D_UNPLO(re_e, im_e); // leaves 8m,   8m+4
        const bruun_v4d ph = V4D_UNPHI(re_e, im_e); // leaves 8m+2, 8m+6
        const bruun_v4d qe = V4D_UNPLO(re_o, im_o); // leaves 8m+1, 8m+5
        const bruun_v4d qh = V4D_UNPHI(re_o, im_o); // leaves 8m+3, 8m+7

        const int32_t* RESTRICT idx = t.idx;
        _mm_storeu_pd(&X[idx[0]].re, _mm256_castpd256_pd128(pe));
        _mm_storeu_pd(&X[idx[4]].re, _mm256_extractf128_pd(pe, 1));
        _mm_storeu_pd(&X[idx[2]].re, _mm256_castpd256_pd128(ph));
        _mm_storeu_pd(&X[idx[6]].re, _mm256_extractf128_pd(ph, 1));
        _mm_storeu_pd(&X[idx[1]].re, _mm256_castpd256_pd128(qe));
        _mm_storeu_pd(&X[idx[5]].re, _mm256_extractf128_pd(qe, 1));
        _mm_storeu_pd(&X[idx[3]].re, _mm256_castpd256_pd128(qh));
        _mm_storeu_pd(&X[idx[7]].re, _mm256_extractf128_pd(qh, 1));
    }

    inline void codelet_d3_tw_avx2(const double* RESTRICT p, const LeafTw& t, complex_t* RESTRICT X) const {
        d3_avx2_core(V4D_LD(p), V4D_LD(p + 4),
                     V4D_LD(p + 8), V4D_LD(p + 12), t, X);
    }

#endif


    inline void d3_one(const double* RESTRICT p, int m, complex_t* RESTRICT X) const {
        const LeafTw& t = TW[m];
#if BRUUN_LEVEL >= 2
        codelet_d3_tw_avx2(p, t, X);
#elif BRUUN_LEVEL == 1
        codelet_d3_tw_v2(p, t, X);
#else
        codelet_d3_tw_scalar(p, t, X);
#endif
    }

    // Residue-writing leaf codelet: identical math to the packing codelets,
    // but the 16 results are written back into p in leaf-residue order. This is
    // the engine for the residue-domain interfaces (filtering, custom CRT layouts)
    // and for the optional two-phase pack.
    inline void codelet_d3_tw_res(double* RESTRICT p, const LeafTw& t) const {
#if BRUUN_LEVEL >= 2
        const bruun_v4d A0 = V4D_LD(p);
        const bruun_v4d B0 = V4D_LD(p + 4);
        const bruun_v4d A1 = V4D_LD(p + 8);
        const bruun_v4d B1 = V4D_LD(p + 12);

        const bruun_v4d c1 = V4D_SET1(t.c1);
        const bruun_v4d s1 = V4D_SET1(t.s1);
        const bruun_v4d R1 = V4D_MSUB(V4D_MUL(c1, B0), s1, B1);
        const bruun_v4d I1 = V4D_MADD(V4D_MUL(s1, B0), c1, B1);

        const bruun_v4d c0a = V4D_ADD(A0, R1);
        const bruun_v4d c0b = V4D_ADD(A1, I1);
        const bruun_v4d c1a = V4D_SUB(A0, R1);
        const bruun_v4d c1b = V4D_SUB(I1, A1);

        const bruun_v4d A0v = V4D_CAT128_LOHI(c0a, c1a);
        const bruun_v4d B0v = V4D_CAT128_HIHI(c0a, c1a);
        const bruun_v4d A1v = V4D_CAT128_LOHI(c0b, c1b);
        const bruun_v4d B1v = V4D_CAT128_HIHI(c0b, c1b);

        const bruun_v4d c2 = V4D_LD2_DUP(t.c2);
        const bruun_v4d s2 = V4D_SWAP_HALVES(c2);
        const bruun_v4d R2 = V4D_MSUB(V4D_MUL(c2, B0v), s2, B1v);
        const bruun_v4d I2 = V4D_MADD(V4D_MUL(s2, B0v), c2, B1v);

        const bruun_v4d P = V4D_ADD(A0v, R2);
        const bruun_v4d Q = V4D_ADD(A1v, I2);
        const bruun_v4d M = V4D_SUB(A0v, R2);
        const bruun_v4d W = V4D_SUB(I2, A1v);

        const bruun_v4d A0w = V4D_UNPLO(P, M);
        const bruun_v4d B0w = V4D_UNPHI(P, M);
        const bruun_v4d A1w = V4D_UNPLO(Q, W);
        const bruun_v4d B1w = V4D_UNPHI(Q, W);

        const bruun_v4d c4 = V4D_LD(t.c4);
        const bruun_v4d s4 = V4D_SWAP_PAIRS(c4);
        const bruun_v4d R3 = V4D_MSUB(V4D_MUL(c4, B0w), s4, B1w);
        const bruun_v4d I3 = V4D_MADD(V4D_MUL(s4, B0w), c4, B1w);

        const __m256d E0 = _mm256_add_pd(A0w, R3); // leaf even residue r0
        const __m256d E1 = _mm256_add_pd(A1w, I3); // leaf even residue r1
        const __m256d O0 = _mm256_sub_pd(A0w, R3); // leaf odd residue r0
        const __m256d O1 = _mm256_sub_pd(I3, A1w); // leaf odd residue r1

        // 4x4 transpose to leaf-residue order: p[4g..4g+3] = [E0[g], E1[g], O0[g], O1[g]].
        const __m256d t0 = _mm256_unpacklo_pd(E0, E1);
        const __m256d t1 = _mm256_unpackhi_pd(E0, E1);
        const __m256d t2 = _mm256_unpacklo_pd(O0, O1);
        const __m256d t3 = _mm256_unpackhi_pd(O0, O1);

        _mm256_storeu_pd(p,      _mm256_permute2f128_pd(t0, t2, 0x20));
        _mm256_storeu_pd(p + 4,  _mm256_permute2f128_pd(t1, t3, 0x20));
        _mm256_storeu_pd(p + 8,  _mm256_permute2f128_pd(t0, t2, 0x31));
        _mm256_storeu_pd(p + 12, _mm256_permute2f128_pd(t1, t3, 0x31));
#else
        norm_q_fwd(p, 4, t.c1, t.s1);
        norm_q2_fwd(p, t.c2[0], t.c2[1]);
        norm_q2_fwd(p + 8, t.c2[1], t.c2[0]);
        norm_q1_fwd(p, t.c4[0], t.c4[1]);
        norm_q1_fwd(p + 4, t.c4[1], t.c4[0]);
        norm_q1_fwd(p + 8, t.c4[2], t.c4[3]);
        norm_q1_fwd(p + 12, t.c4[3], t.c4[2]);
#endif
    }

    void run_fwd_residue_schedule(double* RESTRICT v) const {
        const FwdOp* RESTRICT ops = FWD_RES_SCHEDULE.data();
        const std::size_t op_count = FWD_RES_SCHEDULE.size();
        for (std::size_t op_index = 0; op_index < op_count; ++op_index) {
            const FwdOp& op = ops[op_index];
            double* RESTRICT base = v + op.base;
            switch (op.kind) {
            case FWD_OP_NORM2: {
                const int m = static_cast<int>(op.m);
                const int q = static_cast<int>(op.q);
                BRUUN_ASSERT(q >= 16);
                norm2_fused(base, q, C[m], s_twiddle(m), C[2*m], s_twiddle(2*m), C[2*m+1], s_twiddle(2*m+1));
                break;
            }
            case FWD_OP_CODELET_Q8: {
                const int m = static_cast<int>(op.m);
                norm_q_fwd(base, 8, C[m], s_twiddle(m));
                codelet_d3_tw_res(base, TW[2*m]);
                codelet_d3_tw_res(base + 16, TW[2*m + 1]);
                break;
            }
            case FWD_OP_CODELET_D3:
                codelet_d3_tw_res(base, TW[op.m]);
                break;
            case FWD_OP_BINOMIAL:
                binomial_fwd(base, static_cast<int>(op.q));
                break;
            case FWD_OP_SPINE_D3:
                codelet_d3_tw_res(base, TW[1]);
                break;
            case FWD_OP_SPINE_D1:
                norm_q1_fwd(base, C[1], s_twiddle(1));
                break;
            case FWD_OP_SPINE_NORM2:
                norm_q_fwd(base, 2, C[1], s_twiddle(1));
                norm_q1_fwd(base, C[2], s_twiddle(2));
                norm_q1_fwd(base + 4, C[3], s_twiddle(3));
                break;
            default:
                break;
            }
        }
    }

    // Fast depth-first residue forward with fused input copy. Requires N >= 64.
    void forward_residues_recursive(const double* RESTRICT input, double* RESTRICT v) const {
        binomial_oop(input, v, N / 2);
        run_fwd_residue_schedule(v);
    }

    // Exact inverse of codelet_d3_tw_res: three inverse levels on one 16-double
    // leaf block, fused in registers on AVX2, generic norm_q_inv ladder otherwise.
    inline void codelet_d3_tw_res_inv(double* RESTRICT p, const LeafTw& t) const {
#if BRUUN_LEVEL >= 2
        const __m256d out0 = _mm256_loadu_pd(p);
        const __m256d out1 = _mm256_loadu_pd(p + 4);
        const __m256d out2 = _mm256_loadu_pd(p + 8);
        const __m256d out3 = _mm256_loadu_pd(p + 12);

        // Inverse of the forward 4x4 transpose.
        const __m256d t0 = _mm256_unpacklo_pd(out0, out1);
        const __m256d t1 = _mm256_unpackhi_pd(out0, out1);
        const __m256d t2 = _mm256_unpacklo_pd(out2, out3);
        const __m256d t3 = _mm256_unpackhi_pd(out2, out3);
        const __m256d E0 = _mm256_permute2f128_pd(t0, t2, 0x20);
        const __m256d O0 = _mm256_permute2f128_pd(t0, t2, 0x31);
        const __m256d E1 = _mm256_permute2f128_pd(t1, t3, 0x20);
        const __m256d O1 = _mm256_permute2f128_pd(t1, t3, 0x31);

        const __m256d hf = _mm256_set1_pd(0.5);

        // Level 3 inverse.
        const __m256d A0w = _mm256_mul_pd(hf, _mm256_add_pd(E0, O0));
        const __m256d R3  = _mm256_mul_pd(hf, _mm256_sub_pd(E0, O0));
        const __m256d I3  = _mm256_mul_pd(hf, _mm256_add_pd(E1, O1));
        const __m256d A1w = _mm256_mul_pd(hf, _mm256_sub_pd(E1, O1));
        const bruun_v4d c4 = V4D_LD(t.c4);
        const bruun_v4d s4 = V4D_SWAP_PAIRS(c4);
        const __m256d B0w = _mm256_fmadd_pd(c4, R3, _mm256_mul_pd(s4, I3));
        const __m256d B1w = _mm256_fmsub_pd(c4, I3, _mm256_mul_pd(s4, R3));

        // Level 2 inverse.
        const __m256d P = _mm256_unpacklo_pd(A0w, B0w);
        const __m256d M = _mm256_unpackhi_pd(A0w, B0w);
        const __m256d Q = _mm256_unpacklo_pd(A1w, B1w);
        const __m256d W = _mm256_unpackhi_pd(A1w, B1w);

        const __m256d A0v = _mm256_mul_pd(hf, _mm256_add_pd(P, M));
        const __m256d R2  = _mm256_mul_pd(hf, _mm256_sub_pd(P, M));
        const __m256d I2  = _mm256_mul_pd(hf, _mm256_add_pd(Q, W));
        const __m256d A1v = _mm256_mul_pd(hf, _mm256_sub_pd(Q, W));
        const bruun_v4d c2 = V4D_LD2_DUP(t.c2);
        const bruun_v4d s2 = V4D_SWAP_HALVES(c2);
        const __m256d B0v = _mm256_fmadd_pd(c2, R2, _mm256_mul_pd(s2, I2));
        const __m256d B1v = _mm256_fmsub_pd(c2, I2, _mm256_mul_pd(s2, R2));

        // Level 1 inverse.
        const __m256d c0a = _mm256_permute2f128_pd(A0v, B0v, 0x20);
        const __m256d c1a = _mm256_permute2f128_pd(A0v, B0v, 0x31);
        const __m256d c0b = _mm256_permute2f128_pd(A1v, B1v, 0x20);
        const __m256d c1b = _mm256_permute2f128_pd(A1v, B1v, 0x31);

        const __m256d A0 = _mm256_mul_pd(hf, _mm256_add_pd(c0a, c1a));
        const __m256d R1 = _mm256_mul_pd(hf, _mm256_sub_pd(c0a, c1a));
        const __m256d I1 = _mm256_mul_pd(hf, _mm256_add_pd(c0b, c1b));
        const __m256d A1 = _mm256_mul_pd(hf, _mm256_sub_pd(c0b, c1b));
        const __m256d c1v = _mm256_set1_pd(t.c1);
        const __m256d s1v = _mm256_set1_pd(t.s1);
        const __m256d B0 = _mm256_fmadd_pd(c1v, R1, _mm256_mul_pd(s1v, I1));
        const __m256d B1 = _mm256_fmsub_pd(c1v, I1, _mm256_mul_pd(s1v, R1));

        _mm256_storeu_pd(p,      A0);
        _mm256_storeu_pd(p + 4,  B0);
        _mm256_storeu_pd(p + 8,  A1);
        _mm256_storeu_pd(p + 12, B1);
#else
        norm_q_inv(p,      1, t.c4[0], t.c4[1]);
        norm_q_inv(p + 4,  1, t.c4[1], t.c4[0]);
        norm_q_inv(p + 8,  1, t.c4[2], t.c4[3]);
        norm_q_inv(p + 12, 1, t.c4[3], t.c4[2]);
        norm_q_inv(p,      2, t.c2[0], t.c2[1]);
        norm_q_inv(p + 8,  2, t.c2[1], t.c2[0]);
        norm_q_inv(p,      4, t.c1,    t.s1);
#endif
    }

    void run_inv_residue_schedule(double* RESTRICT v) const {
        const FwdOp* RESTRICT ops = INV_RES_SCHEDULE.data();
        const std::size_t op_count = INV_RES_SCHEDULE.size();
        for (std::size_t op_index = 0; op_index < op_count; ++op_index) {
            const FwdOp& op = ops[op_index];
            double* RESTRICT base = v + op.base;
#if defined(__GNUC__) || defined(__clang__)
            const uint32_t next_base = op_index + 1 < op_count ? ops[op_index + 1].base : 0;
            __builtin_prefetch(v + next_base, 1, 1);
#endif
            switch (op.kind) {
            case FWD_OP_NORM2: {
                const int m = static_cast<int>(op.m);
                const int q = static_cast<int>(op.q);
                norm2_inv_fused(base, q, C[m], s_twiddle(m), C[2*m], s_twiddle(2*m), C[2*m+1], s_twiddle(2*m+1));
                break;
            }
            case FWD_OP_CODELET_Q8: {
                const int m = static_cast<int>(op.m);
                codelet_d3_tw_res_inv(base, TW[2*m]);
                codelet_d3_tw_res_inv(base + 16, TW[2*m + 1]);
                norm_q_inv(base, 8, C[m], s_twiddle(m));
                break;
            }
            case FWD_OP_CODELET_D3:
            case FWD_OP_SPINE_D3:
                codelet_d3_tw_res_inv(base, TW[op.m]);
                break;
            case FWD_OP_BINOMIAL:
                binomial_inv(base, static_cast<int>(op.q));
                break;
            case FWD_OP_SPINE_D1:
                norm_q_inv(base, 1, C[1], s_twiddle(1));
                break;
            case FWD_OP_SPINE_NORM2:
                norm_q_inv(base, 1, C[2], s_twiddle(2));
                norm_q_inv(base + 4, 1, C[3], s_twiddle(3));
                norm_q_inv(base, 2, C[1], s_twiddle(1));
                break;
            default:
                break;
            }
        }
    }

    // Fast scheduled residue inverse: exact reverse of forward_residues_recursive.
    // Requires N >= 64.
    void inverse_residues_recursive(double* RESTRICT v) const {
        // Flat scheduled inverse on every backend (BRUUN_LEVEL is 0/1/2). q==8
        // leaves deliberately stay on the shared d3+d3+norm path to keep
        // compiler pressure predictable across narrow and wide targets.
        run_inv_residue_schedule(v);
    }

    // Two-phase standard-output forward: residues land in block order in v, then
    // one pass writes ordinary FFT bins sequentially while reading residue pairs
    // through KINV. Policy uses this only for large standard-layout outputs.
    void forward_standard_two_phase(const double* RESTRICT input, double* RESTRICT v, complex_t* RESTRICT X) const {
        forward_residues_recursive(input, v);

        X[0].re = v[0] + v[1];
        X[0].im = 0.0;
        X[N / 2].re = v[0] - v[1];
        X[N / 2].im = 0.0;

        const int* RESTRICT kin = KINV.data();
        int k = 1;
#if BRUUN_LEVEL >= 1 && (defined(BRUUN_X86_128) || defined(BRUUN_NEON_128))
        for (; k < N / 2; ++k) {
            const bruun_v2 r = V2_LD(v + 2*kin[k]);
            V2_ST(&X[k].re, V2_NEGHI(r));
        }
#else
        for (; k < N / 2; ++k) {
            const int m = kin[k];
            X[k].re = v[2*m];
            X[k].im = -v[2*m + 1];
        }
#endif
    }

    // Fused copy + scheduled depth-first forward. Requires N >= 64.
    void forward_recursive(const double* RESTRICT input, double* RESTRICT work, complex_t* RESTRICT X, bool workspace_is_aligned) const {
        double* RESTRICT v = work;
        if (workspace_is_aligned) {
            v = static_cast<double*>(BRUUN_ASSUME_ALIGNED(work, bruun_cache_alignment));
        }
        binomial_oop(input, v, N / 2);

        const FwdOp* RESTRICT ops = FWD_SCHEDULE.data();
        const std::size_t op_count = FWD_SCHEDULE.size();
        for (std::size_t op_index = 0; op_index < op_count; ++op_index) {
            const FwdOp& op = ops[op_index];
            double* RESTRICT base = v + op.base;
            switch (op.kind) {
            case FWD_OP_NORM2: {
                const int m = static_cast<int>(op.m);
                const int q = static_cast<int>(op.q);
                BRUUN_ASSERT(q >= 16);
                norm2_fused(base, q, C[m], s_twiddle(m), C[2*m], s_twiddle(2*m), C[2*m+1], s_twiddle(2*m+1));
                break;
            }
            case FWD_OP_CODELET_Q8: {
                const int m = static_cast<int>(op.m);
                norm_q_fwd(base, 8, C[m], s_twiddle(m));
                d3_one(base, 2*m, X);
                d3_one(base + 16, 2*m + 1, X);
                break;
            }
            case FWD_OP_CODELET_D3:
                d3_one(base, static_cast<int>(op.m), X);
                break;
            case FWD_OP_BINOMIAL:
                binomial_fwd(base, static_cast<int>(op.q));
                break;
            case FWD_OP_SPINE_D3:
                d3_one(base, 1, X);
                break;
            case FWD_OP_SPINE_D2:
                codelet_d2_pack(base, 1, X);
                break;
            case FWD_OP_SPINE_D1:
                codelet_d1_pack(base, 1, X);
                break;
            case FWD_OP_SPINE_LEAF:
                pack_leaf_node(1, v[2], v[3], X);
                break;
            case FWD_OP_DC_NYQUIST:
                X[0].re = v[0] + v[1];
                X[0].im = 0.0;
                X[N / 2].re = v[0] - v[1];
                X[N / 2].im = 0.0;
                break;
            default:
                break;
            }
        }
    }

    inline void pack_leaf_node(int leaf, double r0, double r1, complex_t* RESTRICT X) const {
        const int k = OUTIDX[leaf];
        X[k].re = r0;
        X[k].im = -r1;
    }

    void forward_stage(double* RESTRICT v, int jj) const {
        const int s = N >> jj;
        const int h = s >> 1;
        const int q = s >> 2;
        const int m_end = 1 << jj;

        binomial_fwd(v, h);

        if (q == 1) {
            for (int m = 1; m < m_end; ++m) norm_q1_fwd(v + m*s, C[m], s_twiddle(m));
        } else if (q == 2) {
            for (int m = 1; m < m_end; ++m) norm_q2_fwd(v + m*s, C[m], s_twiddle(m));
        } else {
            for (int m = 1; m < m_end; ++m) norm_q_fwd(v + m*s, q, C[m], s_twiddle(m));
        }
    }

    void forward_residues_inplace(double* RESTRICT v) const {
        for (int jj = 0; jj < L - 1; ++jj) {
            forward_stage(v, jj);
        }
    }

    void forward_fused_tail(double* RESTRICT v, complex_t* RESTRICT X) const {
        for (int jj = 0; jj <= L - 5; ++jj) {
            forward_stage(v, jj);
        }

        binomial_fwd(v, 8);
        binomial_fwd(v, 4);
        binomial_fwd(v, 2);

        X[0].re = v[0] + v[1];
        X[0].im = 0.0;
        X[N / 2].re = v[0] - v[1];
        X[N / 2].im = 0.0;

        pack_leaf_node(1, v[2], v[3], X);
        codelet_d1_pack(v + 4, 1, X);
        codelet_d2_pack(v + 8, 1, X);

        for (int m = 1; m < N / 16; ++m) {
            d3_one(v + 16*m, m, X);
        }
    }

    void residues_to_complex(const double* RESTRICT v, complex_t* RESTRICT X) const {
        X[0].re = v[0] + v[1];
        X[0].im = 0.0;
        X[N / 2].re = v[0] - v[1];
        X[N / 2].im = 0.0;

        for (int m = 1; m < N / 2; ++m) {
            const int k = IDX[m];
            X[k].re = v[2*m];
            X[k].im = -v[2*m + 1];
        }
    }

    void complex_to_residues(const complex_t* RESTRICT X, double* RESTRICT v) const {
        v[0] = 0.5 * (X[0].re + X[N / 2].re);
        v[1] = 0.5 * (X[0].re - X[N / 2].re);

        for (int m = 1; m < N / 2; ++m) {
            const int k = IDX[m];
            v[2*m] = X[k].re;
            v[2*m + 1] = -X[k].im;
        }
    }

    void inverse_residues_inplace(double* RESTRICT v) const {
        if (fuse_tail && N >= 64) {
            inverse_residues_recursive(v);
            return;
        }

        for (int jj = L - 2; jj >= 0; --jj) {
            const int s = N >> jj;
            const int h = s >> 1;
            const int q = s >> 2;
            const int m_end = 1 << jj;

            for (int m = m_end - 1; m > 0; --m) {
                norm_q_inv(v + m*s, q, C[m], s_twiddle(m));
            }

            binomial_inv(v, h);
        }
    }
};

} // namespace bruun
