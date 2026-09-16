#pragma once

// Shared Bruun SIMD backend, aligned storage, and small scalar helpers.

#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <new>

#if defined(_WIN32)
#  include <malloc.h>
#endif

// ---------------------------------------------------------------------------
// SIMD backend resolution.
//   BRUUN_LEVEL 0 = scalar, 1 = 128-bit (SSE2/NEON), 2 = AVX2+FMA
// Wider levels reuse the narrower loops as tails where needed.
// ---------------------------------------------------------------------------

#if defined(__FMA__) || defined(_M_FMA)
#  define BRUUN_HAS_FMA 1
#endif

#if defined(__AVX2__) && defined(BRUUN_HAS_FMA)
#  define BRUUN_LEVEL 2
#elif defined(__SSE2__) || defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 2) || ((defined(__aarch64__) || defined(_M_ARM64) || defined(_M_ARM64EC)) && (defined(__ARM_NEON) || defined(__ARM_NEON__) || defined(_M_ARM64) || defined(_M_ARM64EC)))
#  define BRUUN_LEVEL 1
#else
#  define BRUUN_LEVEL 0
#endif

#if BRUUN_LEVEL >= 2 || (BRUUN_LEVEL >= 1 && (defined(__SSE2__) || defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 2)))
#  include <immintrin.h>
#  define BRUUN_X86_128 1
#endif

#if BRUUN_LEVEL >= 1 && (defined(__aarch64__) || defined(_M_ARM64) || defined(_M_ARM64EC)) && (defined(__ARM_NEON) || defined(__ARM_NEON__) || defined(_M_ARM64) || defined(_M_ARM64EC))
#  include <arm_neon.h>
#  define BRUUN_NEON_128 1
#endif

#if BRUUN_LEVEL == 1 && !defined(BRUUN_X86_128) && !defined(BRUUN_NEON_128)
#  undef BRUUN_LEVEL
#  define BRUUN_LEVEL 0
#endif

// 2-lane double vector primitive shared by the SSE2 and NEON paths.
#if defined(BRUUN_X86_128)
typedef __m128d bruun_v2;
#  define V2_LD(p)        _mm_loadu_pd(p)
#  define V2_ST(p, a)     _mm_storeu_pd((p), (a))
#  define V2_ADD(a, b)    _mm_add_pd((a), (b))
#  define V2_SUB(a, b)    _mm_sub_pd((a), (b))
#  define V2_MUL(a, b)    _mm_mul_pd((a), (b))
#  define V2_DIV(a, b)    _mm_div_pd((a), (b))
#  if defined(BRUUN_HAS_FMA)
#    define V2_MADD(a, b, c) _mm_fmadd_pd((b), (c), (a))
#    define V2_MSUB(a, b, c) _mm_fnmadd_pd((b), (c), (a))
#  else
#    define V2_MADD(a, b, c) V2_ADD((a), V2_MUL((b), (c)))
#    define V2_MSUB(a, b, c) V2_SUB((a), V2_MUL((b), (c)))
#  endif
#  define V2_SET1(x)      _mm_set1_pd(x)
#  define V2_SETLH(l, h)  _mm_set_pd((h), (l))
#  define V2_UNPLO(a, b)  _mm_unpacklo_pd((a), (b))
#  define V2_UNPHI(a, b)  _mm_unpackhi_pd((a), (b))
#  define V2_DUP0(a)      _mm_unpacklo_pd((a), (a))
#  define V2_DUP1(a)      _mm_unpackhi_pd((a), (a))
#  define V2_NEGHI(a)     _mm_xor_pd((a), _mm_set_pd(-0.0, 0.0))
#  define V2_SWAP(a)      _mm_shuffle_pd((a), (a), 1)
#  define V2_CMPGT(a, b)   _mm_cmpgt_pd((a), (b))
#  define V2_SELECT(m, t, f) _mm_or_pd(_mm_and_pd((m), (t)), _mm_andnot_pd((m), (f)))
#elif defined(BRUUN_NEON_128)
typedef float64x2_t bruun_v2;
#  define V2_LD(p)        vld1q_f64(p)
#  define V2_ST(p, a)     vst1q_f64((p), (a))
#  define V2_ADD(a, b)    vaddq_f64((a), (b))
#  define V2_SUB(a, b)    vsubq_f64((a), (b))
#  define V2_MUL(a, b)    vmulq_f64((a), (b))
#  define V2_DIV(a, b)    vdivq_f64((a), (b))
#  define V2_MADD(a, b, c) vfmaq_f64((a), (b), (c))
#  define V2_MSUB(a, b, c) vfmsq_f64((a), (b), (c))
#  define V2_SET1(x)      vdupq_n_f64(x)
#  define V2_SETLH(l, h)  vcombine_f64(vdup_n_f64(l), vdup_n_f64(h))
#  define V2_UNPLO(a, b)  vtrn1q_f64((a), (b))
#  define V2_UNPHI(a, b)  vtrn2q_f64((a), (b))
#  define V2_DUP0(a)      vdupq_laneq_f64((a), 0)
#  define V2_DUP1(a)      vdupq_laneq_f64((a), 1)
static inline float64x2_t bruun_neghi(float64x2_t a) {
    const uint64x2_t m = { 0ULL, 0x8000000000000000ULL };
    return vreinterpretq_f64_u64(veorq_u64(vreinterpretq_u64_f64(a), m));
}
#  define V2_NEGHI(a)     bruun_neghi(a)
#  define V2_SWAP(a)      vextq_f64((a), (a), 1)
static inline float64x2_t bruun_v2_cmpgt(float64x2_t a, float64x2_t b) {
    return vreinterpretq_f64_u64(vcgtq_f64(a, b));
}
static inline float64x2_t bruun_v2_select(float64x2_t m, float64x2_t t, float64x2_t f) {
    return vbslq_f64(vreinterpretq_u64_f64(m), t, f);
}
#  define V2_CMPGT(a, b)   bruun_v2_cmpgt((a), (b))
#  define V2_SELECT(m, t, f) bruun_v2_select((m), (t), (f))
#endif

// 4-lane float vector primitive for internal float32 helper loops.
#if defined(BRUUN_X86_128)
typedef __m128 bruun_v4f;
#  define V4F_LD(p)       _mm_loadu_ps(p)
#  define V4F_ST(p, a)    _mm_storeu_ps((p), (a))
#  define V4F_ADD(a, b)   _mm_add_ps((a), (b))
#  define V4F_SUB(a, b)   _mm_sub_ps((a), (b))
#  define V4F_MUL(a, b)   _mm_mul_ps((a), (b))
#  if defined(BRUUN_HAS_FMA)
#    define V4F_MADD(a, b, c) _mm_fmadd_ps((b), (c), (a))
#    define V4F_MSUB(a, b, c) _mm_fnmadd_ps((b), (c), (a))
#  else
#    define V4F_MADD(a, b, c) V4F_ADD((a), V4F_MUL((b), (c)))
#    define V4F_MSUB(a, b, c) V4F_SUB((a), V4F_MUL((b), (c)))
#  endif
#  define V4F_SET1(x)     _mm_set1_ps(x)
#  define V4F_SET4(a,b,c,d) _mm_setr_ps((a), (b), (c), (d))
#  define V4F_ZERO()      _mm_setzero_ps()
#elif defined(BRUUN_NEON_128)
typedef float32x4_t bruun_v4f;
#  define V4F_LD(p)       vld1q_f32(p)
#  define V4F_ST(p, a)    vst1q_f32((p), (a))
#  define V4F_ADD(a, b)   vaddq_f32((a), (b))
#  define V4F_SUB(a, b)   vsubq_f32((a), (b))
#  define V4F_MUL(a, b)   vmulq_f32((a), (b))
#  define V4F_MADD(a, b, c) vfmaq_f32((a), (b), (c))
#  define V4F_MSUB(a, b, c) vfmsq_f32((a), (b), (c))
#  define V4F_SET1(x)     vdupq_n_f32(x)
#  define V4F_SET4(a,b,c,d) vsetq_lane_f32((d), vsetq_lane_f32((c), vsetq_lane_f32((b), vsetq_lane_f32((a), vdupq_n_f32(0.0f), 0), 1), 2), 3)
#  define V4F_ZERO()      vdupq_n_f32(0.0f)
#endif

#if BRUUN_LEVEL >= 2
typedef __m256d bruun_v4d;
#  define V4D_LD(p)       _mm256_loadu_pd(p)
#  define V4D_ST(p, a)    _mm256_storeu_pd((p), (a))
#  define V4D_ADD(a, b)   _mm256_add_pd((a), (b))
#  define V4D_SUB(a, b)   _mm256_sub_pd((a), (b))
#  define V4D_MUL(a, b)   _mm256_mul_pd((a), (b))
#  define V4D_MADD(a, b, c) _mm256_fmadd_pd((b), (c), (a))
#  define V4D_MSUB(a, b, c) _mm256_fnmadd_pd((b), (c), (a))
#  define V4D_SET1(x)     _mm256_set1_pd(x)
#  define V4D_SET4(a,b,c,d) _mm256_setr_pd((a), (b), (c), (d))
#  define V4D_UNPLO(a, b) _mm256_unpacklo_pd((a), (b))
#  define V4D_UNPHI(a, b) _mm256_unpackhi_pd((a), (b))
#  define V4D_SWAP_PAIRS(a) _mm256_permute_pd((a), 0x5)
#  define V4D_SWAP_HALVES(a) _mm256_permute4x64_pd((a), 0x4E)
#  define V4D_CAT128_LOHI(a, b) _mm256_permute2f128_pd((a), (b), 0x20)
#  define V4D_CAT128_HIHI(a, b) _mm256_permute2f128_pd((a), (b), 0x31)
#  define V4D_XOR(a, b)   _mm256_xor_pd((a), (b))
#  define V4D_NEG_ZERO()   _mm256_set1_pd(-0.0)
#  define V4D_LD2_DUP(p)  _mm256_permute4x64_pd(_mm256_castpd128_pd256(_mm_loadu_pd(p)), 0x50)
#endif

#if BRUUN_LEVEL >= 1
#  if defined(BRUUN_X86_128)
#    define V4F_CATLO(a, b)  _mm_movelh_ps((a), (b))
#    define V4F_CATHI(a, b)  _mm_movehl_ps((b), (a))
#    define V4F_ZIPLO(a, b)  _mm_unpacklo_ps((a), (b))
#    define V4F_ZIPHI(a, b)  _mm_unpackhi_ps((a), (b))
#    define V4F_SWAP_PAIRS(a) _mm_shuffle_ps((a), (a), 0xB1)
#    define V4F_SWAP_HALVES(a) _mm_shuffle_ps((a), (a), 0x4E)
#  elif defined(BRUUN_NEON_128)
#    define V4F_CATLO(a, b)  vcombine_f32(vget_low_f32(a), vget_low_f32(b))
#    define V4F_CATHI(a, b)  vcombine_f32(vget_high_f32(a), vget_high_f32(b))
#    define V4F_ZIPLO(a, b)  vzip1q_f32((a), (b))
#    define V4F_ZIPHI(a, b)  vzip2q_f32((a), (b))
#    define V4F_SWAP_PAIRS(a) vrev64q_f32(a)
#    define V4F_SWAP_HALVES(a) vextq_f32((a), (a), 2)
#  endif
#endif

#if defined(_MSC_VER) && !defined(__clang__)
#define RESTRICT __restrict
#elif defined(__GNUC__) || defined(__clang__)
#define RESTRICT __restrict__
#else
#define RESTRICT
#endif

#ifdef NDEBUG
#define BRUUN_ASSERT(cond) ((void)0)
#else
#define BRUUN_ASSERT(cond) assert(cond)
#endif

// BFFT keeps heap-optimized native spectrum ordering enabled internally.
// Applications select public layouts through the API instead of compile flags.
#ifndef BRUUN_HEAPOPT_SPECTRUM_ORDER
#define BRUUN_HEAPOPT_SPECTRUM_ORDER 1
#endif

#ifndef M_PI
#define M_PI 3.141592653589793238462643383279502884
#endif

#if defined(__GNUC__) || defined(__clang__)
#define BRUUN_ASSUME_ALIGNED(ptr, bytes) __builtin_assume_aligned((ptr), (bytes))
#else
#define BRUUN_ASSUME_ALIGNED(ptr, bytes) (ptr)
#endif

#if defined(__GNUC__) || defined(__clang__)
#define BRUUN_ALWAYS_INLINE inline __attribute__((always_inline))
#else
#define BRUUN_ALWAYS_INLINE inline
#endif

namespace bruun {

static constexpr std::size_t bruun_cache_alignment = 64;
static constexpr double bruun_tau = 2.0 * M_PI;
static constexpr double bruun_pio2 = 0.5 * M_PI;

template <typename T>
class heap_array {
public:
    heap_array() noexcept : ptr_(nullptr), cap_(0), len_(0) {}

    heap_array(const heap_array&) = delete;
    heap_array& operator=(const heap_array&) = delete;

    heap_array(heap_array&& other) noexcept
        : ptr_(other.ptr_), cap_(other.cap_), len_(other.len_) {
        other.ptr_ = nullptr;
        other.cap_ = 0;
        other.len_ = 0;
    }

    heap_array& operator=(heap_array&& other) noexcept {
        if (this != &other) {
            release();
            ptr_ = other.ptr_;
            cap_ = other.cap_;
            len_ = other.len_;
            other.ptr_ = nullptr;
            other.cap_ = 0;
            other.len_ = 0;
        }
        return *this;
    }

    ~heap_array() { release(); }

    bool resize(std::size_t size) {
        release();
        if (size == 0) return true;
        if (!alloc_raw(size)) return false;
        for (std::size_t i = 0; i < size; ++i) {
            new (ptr_ + i) T();
        }
        len_ = size;
        return true;
    }

    bool assign(std::size_t size, const T& value) {
        if (!resize(size)) return false;
        for (std::size_t i = 0; i < len_; ++i) {
            ptr_[i] = value;
        }
        return true;
    }

    bool reserve(std::size_t cap) {
        if (cap <= cap_) return true;
        void* raw = aligned_alloc_raw(cap);
        if (!raw) return false;
        T* new_ptr = static_cast<T*>(raw);
        for (std::size_t i = 0; i < len_; ++i) {
            new (new_ptr + i) T(static_cast<T&&>(ptr_[i]));
            ptr_[i].~T();
        }
        free_raw(ptr_);
        ptr_ = new_ptr;
        cap_ = cap;
        return true;
    }

    bool push_back(const T& val) {
        if (len_ >= cap_) {
            std::size_t new_cap = 16;
            if (cap_ != 0) {
                if (cap_ > static_cast<std::size_t>(-1) / 2) {
                    return false;
                }
                new_cap = cap_ * 2;
            }
            if (!reserve(new_cap)) return false;
        }
        new (ptr_ + len_) T(val);
        ++len_;
        return true;
    }

    void pop_back() noexcept {
        BRUUN_ASSERT(len_ > 0);
        --len_;
        ptr_[len_].~T();
    }

    T& back() noexcept { return ptr_[len_ - 1]; }
    const T& back() const noexcept { return ptr_[len_ - 1]; }

    void clear() noexcept {
        for (std::size_t i = len_; i > 0; --i) {
            ptr_[i - 1].~T();
        }
        len_ = 0;
    }

    T* begin() noexcept { return ptr_; }
    T* end() noexcept { return ptr_ + len_; }
    const T* begin() const noexcept { return ptr_; }
    const T* end() const noexcept { return ptr_ + len_; }
    std::size_t size() const noexcept { return len_; }
    bool empty() const noexcept { return len_ == 0; }
    T* data() noexcept { return ptr_; }
    const T* data() const noexcept { return ptr_; }
    T& operator[](std::size_t index) noexcept { return ptr_[index]; }
    const T& operator[](std::size_t index) const noexcept { return ptr_[index]; }

private:
    static void* aligned_alloc_raw(std::size_t count) {
        if (count > static_cast<std::size_t>(-1) / sizeof(T)) {
            return nullptr;
        }
        const std::size_t bytes = sizeof(T) * count;
        std::size_t padded = bytes;
        const std::size_t remainder = padded % bruun_cache_alignment;
        if (remainder != 0) {
            const std::size_t pad = bruun_cache_alignment - remainder;
            if (padded > static_cast<std::size_t>(-1) - pad) {
                return nullptr;
            }
            padded += pad;
        }
        void* raw = nullptr;
#if defined(_WIN32)
        raw = _aligned_malloc(padded, bruun_cache_alignment);
#else
        if (posix_memalign(&raw, bruun_cache_alignment, padded) != 0) raw = nullptr;
#endif
        return raw;
    }

    static void free_raw(void* p) noexcept {
        if (!p) return;
#if defined(_WIN32)
        _aligned_free(p);
#else
        std::free(p);
#endif
    }

    bool alloc_raw(std::size_t count) {
        void* raw = aligned_alloc_raw(count);
        if (!raw) return false;
        ptr_ = static_cast<T*>(raw);
        cap_ = count;
        return true;
    }

    void release() noexcept {
        if (!ptr_) { cap_ = 0; len_ = 0; return; }
        for (std::size_t i = len_; i > 0; --i) {
            ptr_[i - 1].~T();
        }
        free_raw(ptr_);
        ptr_ = nullptr;
        cap_ = 0;
        len_ = 0;
    }

    T* ptr_;
    std::size_t cap_;
    std::size_t len_;
};

struct int_pair {
    int first;
    int second;
};

struct complex_t {
    double re;
    double im;
};

struct complex_f32_t {
    float re;
    float im;
};

static inline const char* simd_backend_name() {
#if BRUUN_LEVEL == 2
    return "avx2-fma-256";
#elif defined(BRUUN_X86_128)
    return "sse2-128";
#elif defined(BRUUN_NEON_128)
    return "neon-128";
#else
    return "scalar";
#endif
}

static inline bool is_power2(int n) {
    return n > 0 && ((n & (n - 1)) == 0);
}

static inline int ilog2_pow2(int n) {
    int l = 0;
    while (n > 1) {
        n >>= 1;
        ++l;
    }
    return l;
}

static inline int graydecode_int(int g) {
    for (int s = 1; s < 32; s <<= 1) g ^= g >> s;
    return g;
}

static inline int bitrev_int(int r, int t) {
    int out = 0;
    for (int i = 0; i < t; ++i) {
        out = (out << 1) | (r & 1);
        r >>= 1;
    }
    return out;
}

static inline int bruun_idx_int(int m, int L) {
#if defined(__GNUC__) || defined(__clang__)
    const int t = 31 - __builtin_clz((unsigned)m);
#else
    int t = 0;
    for (int x = m; x > 1; x >>= 1) ++t;
#endif
    const int r = m ^ (1 << t);
    return (2 * graydecode_int(bitrev_int(r, t)) + 1) << ((L - 2) - t);
}

} // namespace bruun
