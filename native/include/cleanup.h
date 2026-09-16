#ifndef CLEANUP_EXPERIMENT_H
#define CLEANUP_EXPERIMENT_H
#include <stdint.h>
#ifdef _WIN32
# ifdef CLEANUP_BUILD
#  define CLEANUP_API __declspec(dllexport)
# else
#  define CLEANUP_API __declspec(dllimport)
# endif
#else
# define CLEANUP_API __attribute__((visibility("default")))
#endif
#ifdef __cplusplus
extern "C" {
#endif
/* Public C ABI 1 (independent of Filter.py's script ABI). All strings are UTF-8.
   An engine owns all channel/filter/STFT state.
   Call destroy only after the host has stopped its audio callbacks.
   process accepts any number of interleaved float frames, in-place is allowed.
   The DLL has a control worker and a loopback command server. Reload is async.
   Loading failures preserve the active generation. Success resets ALL DSP state.
   Python is used on the control worker only; audio calls native Numba code.
   Before first successful load the engine passes input through without latency. */
typedef struct cleanup_engine cleanup_engine;
CLEANUP_API uint32_t cleanup_abi_version(void);
CLEANUP_API cleanup_engine* cleanup_create(const char* filter_path, const char* runtime_dir,
                                           double sample_rate, uint32_t channels);
CLEANUP_API void cleanup_destroy(cleanup_engine*);
CLEANUP_API int cleanup_process(cleanup_engine*, const float* input, float* output, uint32_t frames);
/* Optional early/late input pair for hosts like SDR#. Both have identical timing.
   analysis_input informs VAD/mask; content_input is the signal being filtered. */
CLEANUP_API int cleanup_process_pair(cleanup_engine*, const float* analysis_input,
                                     const float* content_input, float* output, uint32_t frames);
/* Returns 1 when queued, 0 when a reload is already pending, -1 for NULL. */
CLEANUP_API int cleanup_reload(cleanup_engine*);
CLEANUP_API void cleanup_set_bypass(cleanup_engine*, int enabled);
CLEANUP_API void cleanup_set_squelch(cleanup_engine*, int enabled);
CLEANUP_API void cleanup_set_harmonics(cleanup_engine*, int enabled);
/* Stateless native DIP helper for the strict Filter.py compiler. scratch has
   4*N doubles, cosine/sine each N*(N/E). Output has 2*(N/2+1-half_bin). */
CLEANUP_API int32_t cleanup_dip_transport(const double* source,const double* donors,
    const double* weights,double* output,double* scratch,const double* cosine,
    const double* sine,int64_t n,int64_t e,int64_t half_bin);
/* Exact padded two-branch Cleanup recurrence for finite nonnegative masks.
   Matrices are row-major doubles;
   vertical/horizontal have (frames+2*time_pad)*(bins+2*freq_pad) elements.
   scratch has at least max(frames+2*time_pad,bins+2*freq_pad) elements.
   No allocations; returns 0 on success, a negative value for invalid arguments. */
CLEANUP_API int32_t cleanup_recursive_smooth(double* mask,double* vertical,
    double* horizontal,double* scratch,int64_t frames,int64_t bins,int64_t nb,
    int64_t time_pad,int64_t freq_pad,int64_t iterations);
CLEANUP_API void cleanup_set_bandwidth(cleanup_engine*, double hz);
/* JSON status. Returns required bytes including NUL; truncated outputs are NUL terminated. */
CLEANUP_API uint32_t cleanup_status(cleanup_engine*, char* destination, uint32_t capacity);
CLEANUP_API uint16_t cleanup_control_port(cleanup_engine*);
#ifdef __cplusplus
}
#endif
#endif
