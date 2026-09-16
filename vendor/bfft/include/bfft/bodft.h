#ifndef BFFT_BODFT_H
#define BFFT_BODFT_H

/* Public C ABI for the BODFT kernel: the native half-bin-shifted real transform
   (odd-frequency DFT). This header lives alongside <bfft/bfft.h> and reuses its
   complex and status types.

   BODFT is a first-class half-bin spectral primitive. Forward maps N real
   samples to N/2 packed complex bins:

       H[k] = sum_{n=0}^{N-1} x[n] * exp(-2*pi*i*(k+1/2)*n/N),   k = 0..N/2-1.

   The upper half is recovered by H[N-1-k] = conj(H[k]). Inverse maps the N/2
   packed bins back to the N real samples exactly. */

#include <bfft/bfft.h>

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque BODFT plan. Create with bodft_plan_create, destroy with
   bodft_plan_destroy. A plan serves both double and single precision. */
typedef struct bodft_plan bodft_plan;

/* SIMD backend selected by the build target (shared with the BFFT kernel). */
const char* bodft_backend_name(void);

/* Create a BODFT plan for a power-of-two transform size N >= 2. */
bfft_status bodft_plan_create(size_t n, bodft_plan** plan);

/* Destroy a plan. Passing NULL is allowed. */
void bodft_plan_destroy(bodft_plan* plan);

/* Transform size N, and the packed bin count N/2. Return 0 for a NULL plan. */
size_t bodft_plan_size(const bodft_plan* plan);
size_t bodft_plan_bins(const bodft_plan* plan);

/* Double-precision forward and inverse. input/output have N doubles; the packed
   spectrum has bodft_plan_bins(plan) complex values. */
bfft_status bodft_forward(const bodft_plan* plan,
                          const double* input,
                          bfft_complex* output);

/* Numba-compatible forward entry point with the same call shape as
   bfft_forward. work and native_scratch are accepted for drop-in call-site
   compatibility and are ignored. */
bfft_status bodft_forward_numba(const bodft_plan* plan,
                                const double* input,
                                bfft_complex* output,
                                double* work,
                                bfft_complex* native_scratch);

bfft_status bodft_inverse(const bodft_plan* plan,
                          const bfft_complex* input,
                          double* output);

/* Single-precision forward and inverse. */
bfft_status bodft_forward_f32(const bodft_plan* plan,
                              const float* input,
                              bfft_complex_f32* output);

/* Single-precision Numba-compatible forward entry point. */
bfft_status bodft_forward_numba_f32(const bodft_plan* plan,
                                    const float* input,
                                    bfft_complex_f32* output,
                                    float* work,
                                    bfft_complex_f32* native_scratch);

bfft_status bodft_inverse_f32(const bodft_plan* plan,
                              const bfft_complex_f32* input,
                              float* output);

#ifdef __cplusplus
}
#endif

#endif
