# Four-lane registered-guide scheduling

The concurrency ceiling is four active DSP execution lanes **including the audio
caller**. `CLEANUP_GUIDE_THREADS=1..4` selects a smaller build budget. At plan
construction, the count is bounded by `std::thread::hardware_concurrency()`;
an unknown count gives one lane. This is a scheduling budget, not a reservation
or a measurement of currently idle cores. No affinity or Apple-specific API is
required. Old CMake caches requesting more than four are capped at four.

`CLEANUP_DYNAMIC_GUIDE` and `CLEANUP_PARALLEL_OBSERVATIONS` now default on.
Existing caches with explicit OFF retain their comparison configuration; the
Windows cross-build script explicitly enables both. Early ring pruning remains
an independent, default-off transform experiment.

## Work ownership

The caller and up to three persistent workers share one atomic job index. The
same workers are reused across phases; phases do not create nested pools.

- Each observation job owns a whole lattice and uses a private Fourier plan.
  Existing overlapping-aperture reuse and discontinuity checks remain intact.
- Each perceptual preparation job owns a view.
- Reference spectra are still computed once, split across the actual lane count.
- A registration job owns one chart frequency row for one offset view. It keeps
  the original order of accumulation across time-chart centers, and publishes a
  disjoint row in the shared displacement/confidence arrays.
- A final blend/warp job owns sixteen output frequency rows across all views.
  It sums chart contributions and combines views in their original order.

Phase barriers precede reading shared results. Every dispatched worker is joined
before propagating errors. Scratch, Fourier plans and shared arrays are allocated
at generation construction, not on the audio callback. The one-lane path drains
the same queues directly without creating a worker thread.

Each guide plan owns its pool. Channels in a Cleanup engine are processed
sequentially; inactive channel workers sleep. Independent plugin instances do
not share a process-wide pool. The four-lane guarantee applies to one executing
guide, not every application running on the computer.

## Same-input scheduling measurements

M4 Mini, 2048 guide rows, 48 time columns, 2048-sample aperture, 512-sample hop,
24576-sample rolling context advancing by 8192 samples. Calls are interleaved
between implementations in rotating/reversing order. Two warmup blocks are
excluded from timing summaries, but included in numerical comparison.

Initial 32-block, same early-pruning transform comparison:

| Four-lane schedule | Guide median | p95 |
|---|---:|---:|
| Fixed ranges, serial observations | 68.87 ms | 70.34 ms |
| Shared chart/view/warp jobs | 55.50 ms | 57.36 ms |
| Same queue also prepares observations | 51.98 ms | 53.77 ms |

All guide samples were bit-for-bit equal. This is a 24.5% wall-time reduction.
The later same-transform comparisons also preserved every sample:

| Transform / schedule | Lanes | Guide median | Process CPU median |
|---|---:|---:|---:|
| Early pruning, fixed ranges | 4 | 68.59 ms | 202.74 ms |
| Early pruning, shared queue | 1 | 167.20 ms | 167.19 ms |
| Early pruning, shared queue | 2 | 87.29 ms | 171.31 ms |
| Early pruning, shared queue | 4 | 52.36 ms | 204.87 ms |
| Full fused bank, fixed ranges | 4 | 79.73 ms | 238.48 ms |
| Full fused bank, shared queue | 4 | 62.23 ms | 243.95 ms |

The early comparison used 16 blocks; the full fused comparison used 32. Process
CPU includes all lanes. At four lanes the queue reduced latency with a small
increase in aggregate CPU time (about 1% for early pruning, 2.3% for full fused).
This is better scheduling, not proof of globally minimal arithmetic or universal
real-time behavior. In particular, one lane's guide alone almost consumes the
170.67 ms audio interval. Fewer lanes require further work reduction for that
full-width configuration. Counts do not imply dedicated cores.

An initial mixed-transform comparison deliberately retained the tool's exact
parity gate and failed it: full fused versus early pruning differed by up to
4.64e-16. Those are different floating-point transform paths. The passing
evidence compares schedules within each transform; no tolerance was relaxed.

## Complete callback and dominant costs

Full-width Population Cleanup, early-pruning transform, harmonics disabled,
64 paced blocks, 48 kHz, 8192-sample block (170.67 ms deadline):

| Callback | Median | p95 | Maximum |
|---|---:|---:|---:|
| Fixed four lanes, identical stereo | 94.14 ms | 96.01 ms | 96.39 ms |
| Shared four lanes, identical stereo | 77.62 ms | 80.86 ms | 81.33 ms |
| Shared four lanes, independent stereo | 148.58 ms | 152.55 ms | 157.49 ms |

These callback timings come from separate paced runs, not interleaved trials.
All 1,048,576 saved identical-stereo output samples match bit-for-bit between
fixed and queued scheduling. All measured callbacks met the deadline on this
machine; that does not establish a deadline guarantee on other CPUs or under
contention. This benchmark opens no audio device.

Queued identical-stereo callback stage medians:

| Stage | Time |
|---|---:|
| Registration, confidence/blending and warp | 46.67 ms |
| Cleanup filter/mask | 20.71 ms |
| Perceptual views | 6.30 ms |
| Fourier observations | 2.03 ms |
| STFT/synthesis | 0.49 ms |

Guide total is 56.25 ms. Stage medians do not add exactly and exclude some
orchestration overhead. Registration is approximately 60% of callback latency;
Cleanup's filter is approximately 27%.

A separate 32-block guide profile sums elapsed work across lanes. Its fractions
within registration are approximately 46% ring transform/bounds/peak queries,
22% moving-patch FFTs, 19% chart confidence and blending, 5% ring energy products,
4% shared-reference preparation, 4% cross products, and 1.5% final warping.
These are diagnostic fractions of summed lane work, not additive wall times.
The historical `ring_inverse` counter includes the early transform and its
internal pruning/peak queries. Four median lane registration/warp durations
range from 42.51 to 42.56 ms. Remaining mathematical optimization should focus
on the ring queries and moving-patch transforms, not adding lanes or rewriting
the already small final warp.

## Validation and reproduction

- All eight registered-guide tests pass with early pruning and the shared queue:
  oracle agreement, odd/even chart fallback, exact cached/fresh agreement,
  discontinuity invalidation, complete Cleanup projection, transform/gain
  checks, stream/viewer/reload checks, and stereo processing.
- Exact guide equality holds across 1/2/4 lanes and between fixed/queued
  scheduling, including full fused-bank parity at four lanes.
- Windows DLL cross-build succeeds; the shared-queue full fused build passes
  the frozen silence/noise/tone/radio oracle under Wine. This is not an actual
  SDR# load/audio qualification. No running SDR# installation was replaced.

Reproduce default-transform schedule parity and the registered tests:

```
/Users/ultimussecundai/.local/bin/m4build -- sh tools/verify-scheduling-mini.sh
```

For the early-pruning variant:

```
/Users/ultimussecundai/.local/bin/m4build -- env CLEANUP_TEST_EARLY=ON sh tools/verify-scheduling-mini.sh
```

Evidence is in `docs/evidence/four-lane/`: `paired.json`,
`early-scheduling.json`, `fused-scheduling.json`, `guide-tests.log`,
`callback-static.json`, `callback-observations.json`, `independent.json`,
`audio-parity.json`, `profile.json`, and `windows-oracle.json`.
