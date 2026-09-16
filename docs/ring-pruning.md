# Avoiding correlation outputs that cannot change registration

The registered guide does not need seven images as a downstream product. It
needs seven winning displacements and, normally, their competing-peak confidence
weights. This experimental C++ path proves that some unfinished transforms
cannot change those decisions and omits them.

`CLEANUP_PRUNED_RING_FORWARD=ON` selects it for 48-by-48 charts. It remains **off
by default** while being evaluated. Other chart extents keep the existing path.
The kernel is `native/src/ring_pruned.h`; `registered_guide.cpp` consumes its
decisions. No Cleanup mask, frequency coverage, ring, or observation is removed.

## What 16,128 means

`7 * 48 * 48 = 16,128` is the number of candidate correlation values for **one
registration patch pair**, not the number of samples per pixel or work repeated
independently for every spectral bin. Charts overlap at stride 16. At 2048 guide
rows and 48 time columns, there are 128*3 charts and six moving views: 2,304 patch
pairs and 37,158,912 final correlation values per guide callback in the full-map
implementation. A 512-row guide has 576 pairs and 9,289,728 values.

## Symmetry and an upper bound

All radial masks have reflection and quarter-turn symmetry. The common cross
spectrum generally does not: the symmetry that it must have is Hermitian
symmetry, `U[-k] = conj(U[k])`. Treating the output itself as reflection-symmetric
would discard genuine displacement information.

After one transform axis, let

```
A_j[y,k] = (1/N) sum_l U[l,k] R_j[l,k] exp(2*pi*i*l*y/N).
```

Hermitian symmetry gives `A_j[y,-k] = conj(A_j[y,k])`. The unfinished real row is

```
C_j[y,x] = (A_j[y,0].real + A_j[y,N/2].real*(-1)^x
            + 2*Re(sum(k=1..N/2-1) A_j[y,k]*exp(2*pi*i*k*x/N))) / N.
```

The triangle inequality therefore supplies an upper bound for **every** x:

```
B_j[y] = (A_j[y,0].real + abs(A_j[y,N/2].real)
          + 2*sum(k=1..N/2-1) abs(A_j[y,k])) / N.
```

Keep the signed DC term; taking its absolute value would weaken the bound.
Visit rows in decreasing bound order. Once a row's upper bound is below the
best value actually evaluated, its final 1D transform cannot change the winner.
The same bound applies to the competitor search outside the winner's wrapped
7-by-7 square. Rows already evaluated are reused; remaining rows are evaluated
only if their bound could exceed the incumbent competitor.

Ties are not pruned. The implementation adds a conservative absolute/relative
`1e-12` guard for these bounded inputs, and resolves exactly equal computed
values in original row-major order. The inequality is exact in real arithmetic;
this is guarded floating-point code, not an interval-arithmetic implementation
or a proof of bit-identical decisions for every near-tie on every compiler.

The radial masks allow transposing the transform order without changing their
values. The input is explicitly transposed and returned coordinates mapped back;
we do not assume the input equals its transpose. Both orders were tested. For
the sampled speech, completing the other axis first gave tighter row bounds.

## A downstream cancellation: unanimous zero displacement

When every ring's proven winner is `(0,0)`, the weighted mean displacement and
its dispersion are identically zero, for **any** ring confidence and energy:

```
sum_j weight_j * (0,0) = (0,0)
sum_j weight_j * ||(0,0)-(0,0)||^2 = 0.
```

This also holds when the total-weight denominator is clamped at `1e-12`. The
competitors and ring energies cannot affect this chart's registration result,
so they need not be computed. The API explicitly signals that case; unevaluated
competitors are NaN, never fabricated confidence. The caller returns the proven
zero displacement/dispersion before consuming them.

This shortcut currently covers unanimous **zero** displacement. For nonzero
agreement the denominator clamp needs an additional argument before discarding
weights. No tolerance-based agreement, moment centroid, or candidate guess is
used. An identically zero cross spectrum also has an exact immediate answer.

## Native execution

The first axis shares the four-lane forward radix-3/radix-4 transform already
derived for the full bank, with conjugation/normalization for the target product.
Only nonnegative columns are retained. Surviving rows use planned bfft real
inverses. All storage and plans are allocated at construction; no audio-thread
allocation is introduced. The untouched full-map kernel remains a comparison.

Packing four adaptively chosen final rows together was tested and was slower
than completing surviving rows individually. The current prototype keeps the
faster schedule. Avoided final outputs are **not** equivalent to avoided total
work: the first axis, bounds, and searches still cost time.

## Initial measurements

Fixtures comprise 144 speech chart pairs from the frozen oracle plus silence,
four exact shifts (including wrapping), windowed translation, periodic ambiguity,
incoherence, and competing displacements: 153 pairs / 1,071 ring maps. Input
fixtures explicitly enforce the Hermitian contract after NumPy phase
normalization, which can otherwise amplify tiny +/-k roundoff asymmetry.

M4, same compiler/process, including transforms, bounds, and peak searches:

| Method | Mean final values omitted on speech | Median time per pair |
|---|---:|---:|
| Full fused bank + peak searches | 0% | 37.45 us |
| Bounded row search, all competitors retained | 76.14% | 29.94 us |
| Bounded search + proven zero-consensus shortcut | 82.99% | 27.35 us |

Zero consensus occurred on 41/144 speech pairs. Against full evaluation, there
were zero peak-location mismatches or near-tie location differences; peak and
competitor values differed by at most `1.12e-16`. These timings are about 1.37x
faster than the already-fused full-map bank, not a 5.9x runtime improvement.
Ambiguous cases can require many more rows; the algorithm completes them when
the bound is insufficient. There is no universal 83% pruning guarantee.

The same native Windows x64 probe under Wine measured 58.43 us for full
evaluation, 40.63 us for bounded search with all competitors, and 36.46 us with
zero consensus (1.60x faster than full evaluation). Peak locations matched;
maximum peak/competitor differences remained below `1.12e-16`. This verifies
the Windows executable, not an actual SDR# load or native-Windows deadline.

The integrated default-off M4 build passed all eight registered-guide tests.
For the full 2048-row guide, median wall time fell from 60.68 ms to 53.24 ms.
The complete Population callback on identical stereo histories took median
80.23 ms, p95 83.27 ms, maximum 84.97 ms over 256 paced blocks, versus 86.50 ms
median for the fused full bank and a 170.67 ms audio block deadline. Harmonic
extension was disabled for both comparisons. The saved 4,194,304 output samples
differed from the full-bank output by at most `1.64e-20` (RMS `3.75e-23`).
Evidence is in `guide-tests.log`, `guide-profile.json`, `callback.json`,
`audio-parity.json`, and `wine.json` alongside the isolated probe results.

Independent stereo, which cannot reuse the other channel's guide, took median
148.72 ms, p95 152.11 ms, maximum 154.73 ms over 128 paced blocks. The full-bank
comparison was median 163.99 ms and maximum 170.18 ms. All measured pruned blocks
fit the 170.67 ms deadline, with about 15.94 ms margin at the measured maximum.
This remains an M4 observation on the benchmark input, not a universal deadline
guarantee; see `independent.json`.

Evidence: `docs/evidence/ring-pruning/consensus.json` and `report.json`.
Reproduce with `m4build -- sh tools/verify-pruned-mini.sh`. This is a numerical
and throughput experiment, not a claim that the SDR# plugin has loaded it or
that every host meets its audio deadline.

The general setting is an output-pruned FFT. FFTW's
[pruned FFT discussion](https://fftw.org/pruned.html) explains why omitted
outputs alone do not establish runtime savings. Franchetti and Püschel's
[pruned Cooley–Tukey work](https://www.spiral.net/doc/papers/icassp09_PrunedDFT.pdf)
provides a primary reference for pruning transform graphs. Our data-dependent
row certificate and downstream zero-consensus rule are derived above for this
particular registration calculation.

The next implementation moves the certificate inside the first axis:
[early transform-branch rejection](early-ring-pruning.md). It is a separate
default-off build option so the complete-map and row-pruning paths remain
available for numerical and timing comparisons.
