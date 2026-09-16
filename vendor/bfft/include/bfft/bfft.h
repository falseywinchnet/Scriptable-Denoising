#ifndef BFFT_BFFT_H
#define BFFT_BFFT_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define BFFT_VERSION_MAJOR 1
#define BFFT_VERSION_MINOR 0
#define BFFT_VERSION_PATCH 0

/* Opaque transform plan. Create with bfft_plan_create and destroy with
   bfft_plan_destroy. */
typedef struct bfft_plan bfft_plan;

/* Opaque reusable workspace. A workspace owns aligned scratch for one plan,
   keeping transform calls reentrant by letting callers provide separate
   workspace objects for concurrent use. */
typedef struct bfft_workspace bfft_workspace;

/* Complex value used by the C ABI. Real-to-complex output uses N / 2 + 1
   values in standard or native spectrum order. */
typedef struct bfft_complex {
    double re;
    double im;
} bfft_complex;

/* Single-precision complex value used by the float32 API. */
typedef struct bfft_complex_f32 {
    float re;
    float im;
} bfft_complex_f32;

/* Return status for C API calls. */
typedef enum bfft_status {
    BFFT_OK = 0,
    BFFT_ERROR_INVALID_ARGUMENT = 1,
    BFFT_ERROR_ALLOCATION = 2,
    BFFT_ERROR_INTERNAL = 3
} bfft_status;

/* Public spectrum representations. This enum is descriptive; transform calls
   select the layout by function name. */
typedef enum bfft_layout {
    BFFT_LAYOUT_STANDARD = 0,
    BFFT_LAYOUT_NATIVE = 1,
    BFFT_LAYOUT_RESIDUES = 2
} bfft_layout;

/* Library version, selected SIMD backend, and diagnostic strings. */
const char* bfft_version_string(void);
const char* bfft_backend_name(void);
const char* bfft_status_string(bfft_status status);

/* Create a reusable plan for a power-of-two real FFT transform size N >= 4. */
bfft_status bfft_plan_create(size_t n, bfft_plan** plan);

/* Destroy a plan. Passing NULL is allowed. */
void bfft_plan_destroy(bfft_plan* plan);

/* Plan metadata. Size query functions return 0 for a NULL plan. */
size_t bfft_plan_size(const bfft_plan* plan);
size_t bfft_plan_bins(const bfft_plan* plan);
size_t bfft_plan_work_size(const bfft_plan* plan);
size_t bfft_plan_work_size_f32(const bfft_plan* plan);
size_t bfft_plan_native_scratch_size(const bfft_plan* plan);

/* Create and destroy aligned scratch storage for a plan. Passing NULL to
   bfft_workspace_destroy is allowed. */
bfft_status bfft_workspace_create(const bfft_plan* plan, bfft_workspace** workspace);
void bfft_workspace_destroy(bfft_workspace* workspace);

/* Returns the standard-output packing policy for this plan, or "invalid-plan"
   for a NULL plan. */
const char* bfft_plan_standard_policy(const bfft_plan* plan);

/* Standard FFT-order real-to-complex forward transform.
   input has N doubles, output has bfft_plan_bins(plan) complex values, work has
   bfft_plan_work_size(plan) doubles, and native_scratch should have
   bfft_plan_native_scratch_size(plan) complex values. native_scratch may be
   NULL only when bfft_plan_standard_policy returns "two-phase-standard-pack". */
bfft_status bfft_forward(const bfft_plan* plan,
                         const double* input,
                         bfft_complex* output,
                         double* work,
                         bfft_complex* native_scratch);

/* Standard FFT-order forward transform using plan-matched reusable scratch.
   A workspace must not be shared by concurrent transform calls. */
bfft_status bfft_forward_workspace(const bfft_plan* plan,
                                   bfft_workspace* workspace,
                                   const double* input,
                                   bfft_complex* output);

/* Native-order real-to-complex forward transform. Output has
   bfft_plan_bins(plan) complex values and work has bfft_plan_work_size(plan)
   doubles. */
bfft_status bfft_forward_native(const bfft_plan* plan,
                                const double* input,
                                bfft_complex* output,
                                double* work);

/* Native-order forward transform using plan-matched aligned workspace storage. */
bfft_status bfft_forward_native_workspace(const bfft_plan* plan,
                                          bfft_workspace* workspace,
                                          const double* input,
                                          bfft_complex* output);

/* Single-precision standard FFT-order real-to-complex forward transform.
   input has N floats, output has bfft_plan_bins(plan) float complex values,
   and work has bfft_plan_work_size_f32(plan) floats. native_scratch is accepted
   for API symmetry and may be NULL. */
bfft_status bfft_forward_f32(const bfft_plan* plan,
                             const float* input,
                             bfft_complex_f32* output,
                             float* work,
                             bfft_complex_f32* native_scratch);

/* Single-precision native-order real-to-complex forward transform. */
bfft_status bfft_forward_native_f32(const bfft_plan* plan,
                                    const float* input,
                                    bfft_complex_f32* output,
                                    float* work);

/* Standard FFT-order magnitude-only forward transform. input has N doubles,
   magnitudes has bfft_plan_bins(plan) doubles, and work has
   bfft_plan_work_size(plan) doubles. This avoids complex output and native
   scratch when phase is not needed. */
bfft_status bfft_forward_magnitude(const bfft_plan* plan,
                                   const double* input,
                                   double* magnitudes,
                                   double* work);

/* Single-precision standard FFT-order magnitude-only forward transform. */
bfft_status bfft_forward_magnitude_f32(const bfft_plan* plan,
                                       const float* input,
                                       float* magnitudes,
                                       float* work);

/* Standard FFT-order real-to-polar forward transform. input has N doubles,
   output has bfft_plan_bins(plan) complex-pair values, and work has
   bfft_plan_work_size(plan) doubles. Output is in ordinary FFT bin order
   k = 0..N/2, not native or residue order. output[k].re is magnitude and
   output[k].im is phase in radians in [0, 2*pi). Phase is computed from the
   ordinary forward DFT bin convention, so the imaginary part used internally
   for interior bins is -work[2*m + 1]. */
bfft_status bfft_forward_mag_phase(const bfft_plan* plan,
                                   const double* input,
                                   bfft_complex* output,
                                   double* work);

/* Single-precision standard FFT-order real-to-polar forward transform. input
   has N floats, output has bfft_plan_bins(plan) complex-pair values, and work
   has bfft_plan_work_size_f32(plan) floats. Output is in ordinary FFT bin
   order k = 0..N/2, not native or residue order. output[k].re is magnitude
   and output[k].im is phase in radians in [0, 2*pi). */
bfft_status bfft_forward_mag_phase_f32(const bfft_plan* plan,
                                       const float* input,
                                       bfft_complex_f32* output,
                                       float* work);

/* Standard FFT-order inverse transform. input has bfft_plan_bins(plan) complex
   values and output has N doubles. */
bfft_status bfft_inverse(const bfft_plan* plan,
                         const bfft_complex* input,
                         double* output);

/* Standard FFT-order inverse transform using plan-matched reusable scratch.
   This keeps inverse implementations that require work storage allocation-free
   in hot loops. A workspace must not be shared by concurrent calls. */
bfft_status bfft_inverse_workspace(const bfft_plan* plan,
                                   bfft_workspace* workspace,
                                   const bfft_complex* input,
                                   double* output);

/* Single-precision standard FFT-order inverse transform. */
bfft_status bfft_inverse_f32(const bfft_plan* plan,
                             const bfft_complex_f32* input,
                             float* output);

/* Standard FFT-order polar-to-real inverse transform. input has
   bfft_plan_bins(plan) complex-pair values where input[k].re is magnitude and
   input[k].im is phase in radians. output has N doubles. */
bfft_status bfft_inverse_mag_phase(const bfft_plan* plan,
                                   const bfft_complex* input,
                                   double* output);

/* Single-precision standard FFT-order polar-to-real inverse transform. */
bfft_status bfft_inverse_mag_phase_f32(const bfft_plan* plan,
                                       const bfft_complex_f32* input,
                                       float* output);

/* Native-order inverse transform. */
bfft_status bfft_inverse_native(const bfft_plan* plan,
                                const bfft_complex* input,
                                double* output);

/* Single-precision native-order inverse transform. */
bfft_status bfft_inverse_native_f32(const bfft_plan* plan,
                                    const bfft_complex_f32* input,
                                    float* output);

/* Residue-domain transform. input has N doubles and residues has N doubles. */
bfft_status bfft_forward_residues(const bfft_plan* plan,
                                  const double* input,
                                  double* residues);

/* In-place inverse from residue coordinates back to time samples. */
bfft_status bfft_inverse_residues(const bfft_plan* plan,
                                  double* residues_signal);

/* Convert between native spectrum order and standard FFT-order bins. Both
   arrays have bfft_plan_bins(plan) complex values. */
bfft_status bfft_native_to_standard(const bfft_plan* plan,
                                    const bfft_complex* native_input,
                                    bfft_complex* standard_output);

bfft_status bfft_standard_to_native(const bfft_plan* plan,
                                    const bfft_complex* standard_input,
                                    bfft_complex* native_output);

bfft_status bfft_native_to_standard_f32(const bfft_plan* plan,
                                        const bfft_complex_f32* native_input,
                                        bfft_complex_f32* standard_output);

bfft_status bfft_standard_to_native_f32(const bfft_plan* plan,
                                        const bfft_complex_f32* standard_input,
                                        bfft_complex_f32* native_output);

/* Residue-domain filter size in doubles. */
size_t bfft_filter_size(const bfft_plan* plan);

/* Convert a standard FFT-order complex frequency response into a residue-domain
   filter. standard_response has bfft_plan_bins(plan) complex values and
   residue_filter has bfft_filter_size(plan) doubles. */
bfft_status bfft_residue_filter_from_standard(const bfft_plan* plan,
                                              const bfft_complex* standard_response,
                                              double* residue_filter);

/* Convert a real zero-phase response with bfft_plan_bins(plan) doubles into a
   residue-domain filter. */
bfft_status bfft_residue_filter_from_real(const bfft_plan* plan,
                                          const double* real_response,
                                          double* residue_filter);

/* Apply a residue-domain filter in place to a residue vector. Both arrays have
   bfft_filter_size(plan) doubles. */
bfft_status bfft_apply_residue_filter(const bfft_plan* plan,
                                      double* residues,
                                      const double* residue_filter);

/* Transform input to residues, apply residue_filter, and invert to output.
   input and output have N doubles and must not alias. */
bfft_status bfft_filter_signal(const bfft_plan* plan,
                               const double* input,
                               const double* residue_filter,
                               double* output);

#ifdef __cplusplus
}
#endif

#endif
