# DIP audit: moving phase within a packet

2026-09-04. Audited `bruun_dip_kernel.hpp`, with its unmodified SHA256
`ac1a377c97ad81881a27bb0b09b2ae058cb8c61040acaf3df58a23682b0c2977`.

**Finding:** DIP has exploitable algebraic freedom inside its packet. A
phase-gauged two-level cell needs three general rotations where the existing
cell needs four, with the same eight contiguous streams, child placement,
real degrees of freedom, and workspace. It is a promising constant-factor
change. This experiment does **not** establish a broadly faster full FFT.

The isolated new cell was 6–10% faster at several medium widths on M4. A
selective complete transform was within about 1% of the faster repository
DIF/DIT kernel at N=2048 in one follow-up run, but remained behind across the
size sweep. All speed observations are qualified by substantial host load.

Implementation, source generator, raw successful/unsuccessful experiments,
and exact commands are in
[`experiments/dip_packet_gauge`](../../experiments/dip_packet_gauge/README.md).
The complete tables are in [RESULTS.md](../../experiments/dip_packet_gauge/RESULTS.md).

## 1. What the current implementation gets right

A packet `(d,e)` has `w=N/e` columns and occupies exactly `2w` doubles:

```
[a[0:w] | b[0:w]]
```

The binary cell rotates the odd half by `theta=pi*d/e`, combines it with the
even half, and conjugates one child. It writes the children into the same
addresses it read. The important properties survive direct inspection:

- Phase is uniform along a column run, permitting broadcast twiddles and
  ordinary contiguous SIMD loads and stores.
- Both children are self-contained. Depth-first descent can finish a subtree
  within cache without a full-array transpose or ping-pong workspace.
- Sibling phases are complementary. Quarter-wave table symmetry supports
  the existing mirrored terminal's shared twiddle loads.
- The DC/Nyquist ridge promotes a new packet without copying its entire row.
- Forward and inverse are real-linear maps with matching conjugate folds.
  Each general cell multiplies squared Euclidean norm by two; a two-level
  general cell multiplies it by four. This is a local statement, not a claim
  that every mixed ridge/packet frontier has one uniform unweighted scaling.
- A subtree owns two frequency combs `{m e +/- d}`. The existing output
  mechanism emits them with a universal local tree-order permutation.

Those are useful, concrete properties of DIP's representation. They are more
informative than treating DIF, DIT, and DIP as names for three black boxes.

## 2. The exact shifted-polynomial identity

For an arbitrary packet, define `z_j=a_j+i b_j` and `P(z)=sum_j z_j z^j`.
Here `i` is notation for a pair of real coordinates; the state remains `2w`
real scalars.

Evaluate the polynomial at

```
eta_m = exp(2 pi i (m + d/e)/w),    m=0,...,w-1.
Y_m   = sum_j (a_j+i b_j) exp(2 pi i d j/N) exp(2 pi i m j/w).
```

Thus the packet's complete descent is exactly:

```
coefficient phase D_d → fixed positive-sign DFT_w → conjugate fold/order.
```

For the low half of `m`, `Y_m` belongs to bin `m e+d`; for the high half,
`conj(Y_m)` belongs to bin `N-(m e+d)`. The stored pair at a final positive
bin is `(Re X, -Im X)`. Both sign and tree placement are essential: discarding
the fold would be a different, incorrect transform.

This identity follows by splitting `P` into its low and high coefficient
halves. At a child root, `eta^(w/2)=+/- exp(i pi d/e)`, producing exactly the
current cell's rotation and plus/minus pair. Conjugating the upper-frequency
child gives its complementary phase. Induction yields the full packet law.
The experiment also checks it on arbitrary packet coefficients, not only
states reachable from one signal.

The useful freedom is **where the phase is paid**. It can stay distributed
through the original diagonal tree, move into a small codelet's coefficients,
or move to the entire packet boundary. Those choices have identical Fourier
semantics but different instruction and memory costs.

This is not a novelty certification. Twiddle-bearing codelets and DIT/DIF
factorizations are established FFT techniques; see Frigo and Johnson,
[The Design and Implementation of FFTW3](https://fftw.org/fftw-paper-ieee.pdf).
The contribution of this audit is the explicit gauge/fold identification for
this DIP representation and its concrete implementation and measurements.

## 3. A cheaper two-level DIP cell

Write `phi=theta/2`. Four complex coefficient streams are `z0,z1,z2,z3`, each
with `q` independent columns. Instead of the existing four variable rotations,
form

```
u0 = z0
u1 = exp(i phi) z1
u2 = exp(2 i phi) z2
u3 = exp(3 i phi) z3
```

Then apply the fixed positive DFT4:

```
A=u0+u2; B=u0-u2; C=u1+u3; D=u1-u3
Y0=A+C; Y2=A-C; Y1=B+iD; Y3=B-iD
```

Store, in the original packet addresses:

```
[Y0 | conj(Y2) | conj(Y3) | Y1]
```

This is exactly the four existing grandchildren
`(d,4e), (2e-d,4e), (e-d,4e), (e+d,4e)`.

| Per column | Existing two-level cell | Gauged cell |
|---|---:|---:|
| General complex rotations | 4 | 3 |
| Real multiplies, conventional count | 16 | 12 |
| Real additions/subtractions, conventional count | 24 | 22 |
| Input and output real scalars | 8 / 8 | 8 / 8 |
| Extra work array / permutation pass | 0 / 0 | 0 / 0 |

The conventional flop count falls from 40 to 34 (15%). FMA instruction counts
and register pressure are different quantities, so this is not a predicted
15% speedup. The implementation computes the third phase from the first two
once per node, outside the column loop; the microbenchmark precomputes it.

The local test checks equivalence with the original two-level cell, inversion
by the original inverse, energy gain, scalar tails, unaligned addresses, and
sentinels. All 275 cell/terminal cases pass under ASan and UBSan.

## 4. What was tried and what happened

1. **Phase-gauged DFT8/16/32 leaves.** Algebraically exact. Their reduced
   rotation count did not yield a consistent complete-transform advantage.
   They use more independently loaded coefficient phases and a different
   SIMD packing than the broadcast/mirrored original terminal.
2. **Existing-cell fusion control.** Fuse the first two terminal levels while
   preserving the four-rotation algebra. Also no reliable broad win.
3. **Three-rotation interior and fused terminal.** Medium-width isolated cells
   showed useful improvement: at q=32/64/128, measured old/new ratios were
   approximately 1.059/1.084/1.105. At q=1 the ratio was 0.919; the smallest
   cell became slower. Full-transform gains were small and variable.
4. **Entire packet unfolded into a fixed-phase FFT.** No added workspace and
   no separate bit-reversal pass; the final read folds and reorders directly
   into comb output. This generic radix-2 prototype was roughly 2.7 times
   slower than DIP on a geometric-mean basis. This does not rule out a better
   specialized implementation. It rejects this particular realization.
5. **Selective cell policy.** Use the new cell with at least eight columns per
   stream and retain the original leaf code. This avoids its measured weak
   regime. Results from the matched follow-up appear below.

| N | Original DIP µs | Selective DIP µs | Faster DIF/DIT µs |
|---:|---:|---:|---:|
| 1024 | 1.165 | 1.106 | 0.982 |
| 2048 | 2.348 | 2.259 | 2.236 |
| 4096 | 5.281 | 5.126 | 4.944 |
| 8192 | 13.187 | 12.694 | 12.393 |
| 32768 | 71.457 | 69.857 | 63.215 |
| 65536 | 162.844 | 161.973 | 136.087 |

Use the raw/generated results for full precision. Setup is excluded, all
buffers are reused, and both endpoints are natural order. No recurrence on
already-transformed data is timed. The M4 was heavily loaded by unrelated
work; the independently compiled original-algorithm clone is an important
noise/code-placement control. An A18 Pro run of the same earlier executable
also found no decisive overall win. No x86 or float claim follows.

## 5. Other audit findings

**The existing direct DIP correctness test could miss NaNs.** Accumulating
`max(error, abs(NaN))` can leave `error` unchanged, making a nonfinite result
look successful. The audit adds explicit finite checks to both the spectrum
and inverse in `tests/correctness.cpp`. The new experiment also checks
finiteness and uses scaled error thresholds, including extreme-amplitude
round trips. This is a test defect; no nonfinite production output was found.

**The baseline plan allocates unused trigonometric-table range.** Reachable
cell indices satisfy `0 <= k < N/4`, yet cos/sin and the paired table are
allocated through `N/2`. The current trigonometric storage is about `16N`
bytes plus the fixed roughly 32 KiB permutation table. Limiting the baseline
to its reachable quarter-wave range could nearly halve that trigonometric
storage. It was not changed here: the gauged prototypes intentionally use
larger angles, and reducing allocation is a separate setup/cache experiment.

**Some historical claims are stronger than the implementation/evidence.**
The header's “no block sizes” description applies to arithmetic descent;
boundary emission explicitly uses `kEgressW=4096`, a 64 KiB packet. Cache
residency also depends on tables and competing working sets. Likewise,
bit-reversal “bit moves” are not a general lower bound on physical memory
traffic across every FFT representation. Count actual reads, writes, and
address calculations for each implementation instead.

**Earlier synthetic performance numbers cannot rank complete FFTs.** The
archived conversation includes synthetic DIT/DIF walks and nonfinite sinks.
The existing paired example also always times A before B despite its comment
about cancelling order bias. The new harness runs actual kernels, rotates
order, uses finite fixed input, and records median/range; the selective run
also retains each round's samples.

## 6. The direction supported by this experiment

Keep the original phase-packet tree and its efficient mirrored leaves; use
phase relocation locally where it eliminates real work without destroying
broadcast SIMD. The three-rotation cell is the strongest candidate produced
here. Its speed case is a constant-factor one, not a new asymptotic bound.

Before promotion, measure the selective policy on an idle M4 with stable
repeated paired runs, inspect generated code for broadcasts/spills, and add a
dedicated wider implementation before judging AVX2. For larger transforms,
measure boundary emission and twiddle traffic separately: changing only a
small arithmetic cell cannot establish that those costs have been removed.

The production DIP/DIF/DIT kernels and dispatch remain unchanged. The
experiment leaves a reproducible, mathematically validated alternative cell,
a selective integration, and negative results that constrain the next search.
