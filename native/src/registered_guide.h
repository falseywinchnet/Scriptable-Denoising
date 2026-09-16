#pragma once
#include <cstddef>
#include <cstdint>
#ifndef CLEANUP_GUIDE_THREADS
#define CLEANUP_GUIDE_THREADS 4
#endif
#ifndef CLEANUP_DYNAMIC_GUIDE
#define CLEANUP_DYNAMIC_GUIDE 1
#endif
#ifndef CLEANUP_PARALLEL_OBSERVATIONS
#define CLEANUP_PARALLEL_OBSERVATIONS 1
#endif
static_assert(CLEANUP_GUIDE_THREADS >= 1 && CLEANUP_GUIDE_THREADS <= 4,
              "Guide execution lanes must be 1..4, including the caller");
extern "C" {
void* cleanup_guide_create(size_t aperture, size_t rows, size_t columns, size_t hop, long origin);
// registration=0 keeps only the central enhanced observation; 1 is the full guide.
void* cleanup_guide_create_mode(size_t aperture, size_t rows, size_t columns, size_t hop, long origin, int registration);
void cleanup_guide_destroy(void* plan);
int cleanup_guide_run(void* plan, const double* audio, size_t samples, double* time_major);
int cleanup_guide_run_stream(void* plan, const double* audio, size_t samples, size_t advance, double* time_major);
size_t cleanup_guide_reused_frames(void* plan);
// Last completed call: raw observations, perceptual views, shared registration.
void cleanup_guide_timings(void* plan,uint64_t* three_nanoseconds);
// Sum of lane work, not wall time: shared reference, moving FFT, cross products,
// ring products, ring inverse, peak search, chart confidence/blend, final warp.
void cleanup_guide_work_timings(void* plan,uint64_t* eight_nanoseconds);
// Profiling only: lane_count triples (views, reference, registration/warp).
size_t cleanup_guide_lane_count(void* plan);
void cleanup_guide_lane_timings(void* plan,uint64_t* lane_triples_nanoseconds);
// Oracle parity entry point: input lattice,row,time; output time,row.
int cleanup_guide_register(void* plan, const double* observations, double* time_major);
}
