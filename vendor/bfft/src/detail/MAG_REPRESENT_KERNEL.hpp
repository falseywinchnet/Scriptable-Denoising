#pragma once

// Fast magnitude/phase representation helpers shared by RFFT kernels.

#include <cmath>
#include <cstddef>
#include <cstdint>

template<typename T>
struct bruun_phase_slope_leaf {
    T center;
    T c;
    T s;
};

template<typename T, int Leaves, int Q>
struct bruun_phase_slope_tables;

static BRUUN_ALWAYS_INLINE double bruun_fused_madd(double a, double b, double c) {
#if defined(__FMA__)
    return __builtin_fma(a, b, c);
#else
    return a * b + c;
#endif
}

static BRUUN_ALWAYS_INLINE double bruun_fused_msub(double a, double b, double c, double d) {
#if defined(__FMA__)
    return __builtin_fma(a, b, -(c * d));
#else
    return a * b - c * d;
#endif
}

static BRUUN_ALWAYS_INLINE float bruun_fused_madd(float a, float b, float c) {
#if defined(__FMA__)
    return __builtin_fmaf(a, b, c);
#else
    return a * b + c;
#endif
}

static BRUUN_ALWAYS_INLINE float bruun_fused_msub(float a, float b, float c, float d) {
#if defined(__FMA__)
    return __builtin_fmaf(a, b, -(c * d));
#else
    return a * b - c * d;
#endif
}

#include "generated/bruun_phase_slope_tables.hpp"

template<typename T, int Leaves, int Q>
static BRUUN_ALWAYS_INLINE T bruun_phase_first_octant_slope_linear(T major, T minor, T mag) {
    using tables = bruun_phase_slope_tables<T, Leaves, Q>;

    const T t = minor / major;
    int q = static_cast<int>(t * static_cast<T>(Q));
    if (q < 0) {
        q = 0;
    } else if (q > Q) {
        q = Q;
    }

    const int idx = static_cast<int>(tables::slope_to_leaf[q]);
    const bruun_phase_slope_leaf<T> leaf = tables::leaves[idx];
    const T dot = bruun_fused_madd(major, leaf.c, minor * leaf.s);
    const T cross = bruun_fused_msub(minor, leaf.c, major, leaf.s);
    const T h = cross / (mag + dot);
    return leaf.center + static_cast<T>(2) * h;
}

static BRUUN_ALWAYS_INLINE double bruun_phase_first_octant(double major, double minor, double mag) {
    return bruun_phase_first_octant_slope_linear<double, 256, 512>(major, minor, mag);
}

static BRUUN_ALWAYS_INLINE float bruun_phase_first_octant_f32(float major, float minor, float mag) {
    return bruun_phase_first_octant_slope_linear<float, 128, 512>(major, minor, mag);
}

static BRUUN_ALWAYS_INLINE double bruun_phase_atan2_mag(double y, double x, double mag) {
    const double ax = std::fabs(x);
    const double ay = std::fabs(y);

    if (ay == 0.0) {
        if (ax == 0.0) {
            return 0.0;
        }
        if (x < 0.0) {
            return M_PI;
        }
        return 0.0;
    }

    if (ax == 0.0) {
        if (y < 0.0) {
            return bruun_tau - bruun_pio2;
        }
        return bruun_pio2;
    }

    double alpha = 0.0;
    if (ax >= ay) {
        alpha = bruun_phase_first_octant(ax, ay, mag);
    } else {
        alpha = bruun_pio2 - bruun_phase_first_octant(ay, ax, mag);
    }

    if (x < 0.0) {
        alpha = M_PI - alpha;
    }
    if (y < 0.0) {
        alpha = -alpha;
    }
    if (alpha < 0.0) {
        alpha += bruun_tau;
    }

    return alpha;
}

static BRUUN_ALWAYS_INLINE float bruun_phase_atan2_mag_f32(float y, float x, float mag) {
    const float ax = std::fabs(x);
    const float ay = std::fabs(y);

    constexpr float tau = 6.2831853071795864769f;
    constexpr float pi = 3.14159265358979323846f;
    constexpr float pio2 = 1.57079632679489661923f;

    if (ay == 0.0f) {
        if (ax == 0.0f) {
            return 0.0f;
        }
        if (x < 0.0f) {
            return pi;
        }
        return 0.0f;
    }

    if (ax == 0.0f) {
        if (y < 0.0f) {
            return tau - pio2;
        }
        return pio2;
    }

    float alpha = 0.0f;
    if (ax >= ay) {
        alpha = bruun_phase_first_octant_f32(ax, ay, mag);
    } else {
        alpha = pio2 - bruun_phase_first_octant_f32(ay, ax, mag);
    }

    if (x < 0.0f) {
        alpha = pi - alpha;
    }
    if (y < 0.0f) {
        alpha = -alpha;
    }
    if (alpha < 0.0f) {
        alpha += tau;
    }

    return alpha;
}

static inline double bruun_phase_atan2(double y, double x) {
    const double mag = std::sqrt(x * x + y * y);
    return bruun_phase_atan2_mag(y, x, mag);
}

static inline float bruun_phase_atan2_f32(float y, float x) {
    const float mag = std::sqrt(x * x + y * y);
    return bruun_phase_atan2_mag_f32(y, x, mag);
}

struct bruun_sincos_sample {
    double s;
    double c;
};

#include "generated/bruun_sincos_table256.hpp"

static inline void bruun_table256_poly3_sincos(double phase, double* s_out, double* c_out) {
    constexpr int table_size = 256;
    constexpr double inv_step = static_cast<double>(table_size) / bruun_tau;
    constexpr double step = bruun_tau / static_cast<double>(table_size);

    if (!(phase >= 0.0 && phase < bruun_tau)) {
        *s_out = std::sin(phase);
        *c_out = std::cos(phase);
        return;
    }

    const double qd = phase * inv_step + 0.5;
    const auto qi = static_cast<std::int64_t>(qd);
    const auto idx = static_cast<std::size_t>(qi) & static_cast<std::size_t>(table_size - 1);

    const double r = phase - static_cast<double>(qi) * step;
    const double r2 = r * r;

    const double sr = r * (1.0 - r2 * (1.0 / 6.0));
    const double cr = 1.0 - r2 * 0.5;

    const bruun_sincos_sample sample = bruun_sincos_table256[idx];
    *c_out = sample.c * cr - sample.s * sr;
    *s_out = sample.s * cr + sample.c * sr;
}

static inline void bruun_table256_poly3_sincos_f32(float phase, float* s_out, float* c_out) {
    double s;
    double c;
    bruun_table256_poly3_sincos(static_cast<double>(phase), &s, &c);
    *s_out = static_cast<float>(s);
    *c_out = static_cast<float>(c);
}
