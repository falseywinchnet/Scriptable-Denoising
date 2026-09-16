#ifndef BFFT_FCT_H
#define BFFT_FCT_H

/* Public C ABI for the FCT kernel: the Fast Correlated Transform.
   This header lives alongside <bfft/bfft.h> and reuses its complex and
   status types.

   The FCT is FORWARD-ONLY, by construction.  For each standard bin k of a
   real frame of length N it emits the leading-edge correlation

       C[k] = sum_{t=0}^{tau_k - 1} x[t] * exp(-2*pi*i*k*t/N)

   at the globally maximizing slice tau_k in [t_min, N] selected under the
   score |C|^2 / tau (the emitted phase is the arctan(A/B) of the underlying
   sine/cosine correlation pair at that slice).  Bins with no coherent
   leading edge below an optional certified activity floor default to tau = N,
   i.e. the plain FFT bin. The intrinsic phase-disk walk certifies the global
   argmax for every emitted active bin; with the default zero floor every bin
   is exact. The selection is nonlinear and the fixed truncation family it
   optimizes over is exponentially ill-conditioned, so no inverse exists;
   attempts to invert must reconstruct from the plain FFT bins instead.

   Output layout matches bfft_forward: N/2 + 1 packed complex values in
   standard order. */

#include <bfft/bfft.h>

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque FCT plan. Create with fct_plan_create, destroy with
   fct_plan_destroy. A plan owns its scratch: use one plan per thread. */
typedef struct fct_plan fct_plan;

/* Create an FCT plan for a power-of-two frame size N >= 16. */
bfft_status fct_plan_create(size_t n, fct_plan** plan);

/* Create an FCT plan with explicit selection knobs.
   t_min is the smallest feasible support. The intrinsic kernel still returns
   the exact global argmax, now over tau in [t_min, N]. rel is retained for ABI
   compatibility with the former heuristic selector and is ignored.
   act is a certified activity floor in units of mean |x|^2. Bins whose proven
   optimum is at or below the floor emit the full Fourier support. Use act=0
   for an exact global argmax at every bin (the default). */
bfft_status fct_plan_create_ex(size_t n, int t_min, double rel, double act,
                               fct_plan** plan);

/* Destroy a plan. Passing NULL is allowed. */
void fct_plan_destroy(fct_plan* plan);

/* Frame size N and the packed bin count N/2 + 1. Return 0 for NULL plan. */
size_t fct_plan_size(const fct_plan* plan);
size_t fct_plan_bins(const fct_plan* plan);

/* Forward transform. input has N doubles; output has fct_plan_bins(plan)
   complex values; tau (may be NULL) receives the selected slice per bin in
   [1, N]. Not const on the plan: the plan owns the pyramid scratch. */
bfft_status fct_forward(fct_plan* plan,
                        const double* input,
                        bfft_complex* output,
                        int64_t* tau);

/* Complex-IQ forward FCT. input and output both contain N complex values;
   tau contains N shared slice decisions.  The selector operates on the
   complex correlation directly (it is not two independently selected real
   transforms), so wrapped-negative frequencies are handled correctly. */
bfft_status fct_forward_complex(fct_plan* plan,
                                const bfft_complex* input,
                                bfft_complex* output,
                                int64_t* tau);

/* Complex FCT plus the exact first correlation moment at the selected support:

       moment[k] = sum_{t < tau[k]} t * input[t] * exp(-2*pi*i*k*t/N).

   This is the phase jet used by support-aware reassignment, since
   -d arg(C)/d omega = Re(moment/C). Selection is performed only on input;
   the moment is evaluated afterward at that same certified tau. */
bfft_status fct_forward_complex_moment(fct_plan* plan,
                                       const bfft_complex* input,
                                       bfft_complex* output,
                                       int64_t* tau,
                                       bfft_complex* moment);

/* Numba-compatible forward entry point with the same call shape as
   bfft_forward. work and native_scratch are accepted for drop-in call-site
   compatibility and are ignored; the slice indices are not emitted. */
bfft_status fct_forward_numba(fct_plan* plan,
                              const double* input,
                              bfft_complex* output,
                              double* work,
                              bfft_complex* native_scratch);

#ifdef __cplusplus
}
#endif

#endif
