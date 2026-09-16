#pragma once

// SIMD certificate for the intrinsic half-leading-edge phase type.
//
// A packet is a set of frequency bins visiting the same dyadic time node.
// The node shares (lo, length, energy); each lane carries its complex prefix
// before the node, the node's complex total V(k), and its incumbent score.
// For every local prefix P,
//
//   |P - V/2| <= sqrt(length * energy) / 2.
//
// Therefore the node can be rejected when
//
//   |before + V/2| + sqrt(length*energy)/2
//       <= sqrt(incumbent * (lo+1)).
//
// Squaring without square roots gives g >= 0 and g^2 >= 4*h2*r2, where
// h2=|before+V/2|^2, r2=length*energy/4, and
// g=incumbent*(lo+1)-h2-r2.  A small roundoff guard only turns borderline
// prunes into splits; it cannot make the certificate optimistic.

#include "bruun_simd_backend.hpp"

#include <cstddef>
#include <cstdint>
#include <limits>

namespace fct {
namespace half_edge {

inline bool prunable_disk_scalar(double center_re, double center_im,
                                 double radius2, double limit2) {
    const double h2 = center_re * center_re + center_im * center_im;
    constexpr double eps = 64.0 * std::numeric_limits<double>::epsilon();
    const double guard1 = eps * (limit2 + h2 + radius2 + 1.0);
    const double g = limit2 - h2 - radius2 - guard1;
    if (!(g > 0.0)) return false;
    const double lhs = g * g;
    const double rhs = 4.0 * h2 * radius2;
    const double guard2 = eps * (lhs + rhs + 1.0);
    return lhs > rhs + guard2;
}

inline bool prunable_scalar(double before_re, double before_im,
                            double total_re, double total_im,
                            double incumbent, double energy,
                            int length, int lo) {
    const double hr = before_re + 0.5 * total_re;
    const double hi = before_im + 0.5 * total_im;
    const double r2 = 0.25 * static_cast<double>(length) * energy;
    const double q = incumbent * static_cast<double>(lo + 1);
    return prunable_disk_scalar(hr, hi, r2, q);
}

// Structure-of-arrays packet kernel.  Bins are the SIMD dimension; the time
// node data are scalar and shared.  ``pruned[i]`` receives 1 only when the
// half-edge disk certifies that lane i cannot beat its incumbent.
inline void prune_same_node_intrinsic(const double* before_re,
                                      const double* before_im,
                                      const double* total_re,
                                      const double* total_im,
                                      const double* incumbent,
                                      std::size_t count,
                                      double energy,
                                      int length,
                                      int lo,
                                      std::uint8_t* pruned) {
    const double r2s = 0.25 * static_cast<double>(length) * energy;
    const double dens = static_cast<double>(lo + 1);
    constexpr double epss =
        64.0 * std::numeric_limits<double>::epsilon();
    std::size_t i = 0;

#if BRUUN_LEVEL >= 1
    const bruun_v2 half = V2_SET1(0.5);
    const bruun_v2 r2 = V2_SET1(r2s);
    const bruun_v2 den = V2_SET1(dens);
    const bruun_v2 eps = V2_SET1(epss);
    const bruun_v2 one = V2_SET1(1.0);
    const bruun_v2 zero = V2_SET1(0.0);
    const bruun_v2 four = V2_SET1(4.0);
    const bruun_v2 neg_one = V2_SET1(-1.0);
    for (; i + 2 <= count; i += 2) {
        const bruun_v2 hr = V2_MADD(V2_LD(before_re + i), half,
                                    V2_LD(total_re + i));
        const bruun_v2 hi = V2_MADD(V2_LD(before_im + i), half,
                                    V2_LD(total_im + i));
        const bruun_v2 h2 = V2_MADD(V2_MUL(hr, hr), hi, hi);
        const bruun_v2 q = V2_MUL(V2_LD(incumbent + i), den);
        const bruun_v2 scale = V2_ADD(V2_ADD(V2_ADD(q, h2), r2), one);
        const bruun_v2 guard1 = V2_MUL(eps, scale);
        const bruun_v2 g = V2_SUB(V2_SUB(V2_SUB(q, h2), r2), guard1);
        const bruun_v2 lhs = V2_MUL(g, g);
        const bruun_v2 rhs = V2_MUL(four, V2_MUL(h2, r2));
        const bruun_v2 guard2 =
            V2_MUL(eps, V2_ADD(V2_ADD(lhs, rhs), one));
        const bruun_v2 rhs_guard = V2_ADD(rhs, guard2);
        const bruun_v2 positive = V2_CMPGT(g, zero);
        const bruun_v2 safe_lhs = V2_SELECT(positive, lhs, neg_one);
        const bruun_v2 pass = V2_CMPGT(safe_lhs, rhs_guard);
        const bruun_v2 bits = V2_SELECT(pass, one, zero);
        double lane[2];
        V2_ST(lane, bits);
        pruned[i] = static_cast<std::uint8_t>(lane[0] != 0.0);
        pruned[i + 1] = static_cast<std::uint8_t>(lane[1] != 0.0);
    }
#endif

    for (; i < count; ++i) {
        pruned[i] = static_cast<std::uint8_t>(prunable_scalar(
            before_re[i], before_im[i], total_re[i], total_im[i],
            incumbent[i], energy, length, lo));
    }
}

// The branchless SoA loop is intentionally kept separate from the scalar
// oracle.  Current AppleClang lowers this to better NEON than the explicit
// two-lane path above (especially the byte-mask egress), and x86 compilers can
// select four AVX2 lanes without another handwritten backend.
inline void prune_same_node_autovec(const double* before_re,
                                    const double* before_im,
                                    const double* total_re,
                                    const double* total_im,
                                    const double* incumbent,
                                    std::size_t count,
                                    double energy,
                                    int length,
                                    int lo,
                                    std::uint8_t* pruned) {
    const double r2 = 0.25 * static_cast<double>(length) * energy;
    const double den = static_cast<double>(lo + 1);
    constexpr double eps =
        64.0 * std::numeric_limits<double>::epsilon();
#if defined(__clang__)
#pragma clang loop vectorize(enable) interleave(enable)
#elif defined(__GNUC__)
#pragma GCC ivdep
#endif
    for (std::size_t i = 0; i < count; ++i) {
        const double hr = before_re[i] + 0.5 * total_re[i];
        const double hi = before_im[i] + 0.5 * total_im[i];
        const double h2 = hr * hr + hi * hi;
        const double q = incumbent[i] * den;
        const double guard1 = eps * (q + h2 + r2 + 1.0);
        const double g = q - h2 - r2 - guard1;
        const double lhs = g * g;
        const double rhs = 4.0 * h2 * r2;
        const double guard2 = eps * (lhs + rhs + 1.0);
        pruned[i] = static_cast<std::uint8_t>(
            g > 0.0 && lhs > rhs + guard2);
    }
}

inline void prune_same_node(const double* before_re,
                            const double* before_im,
                            const double* total_re,
                            const double* total_im,
                            const double* incumbent,
                            std::size_t count,
                            double energy,
                            int length,
                            int lo,
                            std::uint8_t* pruned) {
    prune_same_node_autovec(before_re, before_im, total_re, total_im,
                            incumbent, count, energy, length, lo, pruned);
}

} // namespace half_edge
} // namespace fct
