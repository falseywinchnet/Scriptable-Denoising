#ifndef BFFT_STFT_H
#define BFFT_STFT_H

#include <bfft/bfft.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct bfft_stft_plan bfft_stft_plan;

typedef enum bfft_stft_transform {
    BFFT_STFT_RFFT = 0,
    BFFT_STFT_ODFT = 1,
    /* Fast Correlated Transform (see <bfft/fct.h>): each frame bin is
       emitted at its maximally correlated leading-edge slice. Frames are
       gathered in natural time order (no fftshift centering: the leading
       edge is a time-domain notion). FORWARD-ONLY: bfft_stft_inverse
       rejects plans created with this transform. */
    BFFT_STFT_FCT = 2
} bfft_stft_transform;

/* Create a fixed-size, streaming STFT/ISTFT plan.
   n is the real block length and must be divisible by hop_length. n_fft must be
   an even power of two >= 4. hop_length must be in [1, n_fft] and divide n_fft.
   If window is NULL, a periodic Hann-compatible window matching numpy.hanning is
   generated. Otherwise window must point to n_fft finite doubles. */
bfft_status bfft_stft_plan_create(size_t n,
                                  size_t n_fft,
                                  size_t hop_length,
                                  const double* window,
                                  bfft_stft_transform transform,
                                  bfft_stft_plan** plan);

void bfft_stft_plan_destroy(bfft_stft_plan* plan);

size_t bfft_stft_plan_n(const bfft_stft_plan* plan);
size_t bfft_stft_plan_n_fft(const bfft_stft_plan* plan);
size_t bfft_stft_plan_hop_length(const bfft_stft_plan* plan);
size_t bfft_stft_plan_bins(const bfft_stft_plan* plan);
size_t bfft_stft_plan_segments(const bfft_stft_plan* plan);
size_t bfft_stft_plan_buffer_length(const bfft_stft_plan* plan);
bfft_stft_transform bfft_stft_plan_transform(const bfft_stft_plan* plan);

/* Fill output with the default analysis window for n_fft samples. */
bfft_status bfft_stft_hann_window(size_t n_fft, double* output);

/* Zero the persistent inverse overlap buffer owned by the plan. */
bfft_status bfft_stft_reset_buffer(bfft_stft_plan* plan);

/* Forward STFT. input has n doubles. output has bins * segments complex values
   in row-major order, so output[bin * segments + segment] is one bin/column. */
bfft_status bfft_stft_forward(bfft_stft_plan* plan,
                              const double* input,
                              bfft_complex* output);

/* Streaming inverse STFT. input has bins * segments complex values in the same
   row-major layout. output has segments * hop_length doubles. The plan's
   internal overlap buffer is updated; call bfft_stft_reset_buffer to start a
   fresh stream. */
bfft_status bfft_stft_inverse(bfft_stft_plan* plan,
                              const bfft_complex* input,
                              double* output);

#ifdef __cplusplus
}
#endif

#endif
