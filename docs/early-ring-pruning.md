# Rejection inside the first transform axis

This continues [bounded row pruning](ring-pruning.md). The previous kernel
completed the first Fourier axis everywhere, then used row bounds to avoid most
of the second axis. The new C++ kernel pauses **inside the first axis** and
rejects whole unfinished transform branches.

Source: `native/src/ring_early.h`. `CLEANUP_EARLY_RING_FORWARD=ON` selects the
experimental integration in `registered_guide.cpp`; it remains off by default.
It takes precedence over the earlier row-pruning option. Other chart sizes
retain the existing fallback. The seven masks, peak/competitor definitions,
ring weights, registration transport, and Python Cleanup operator are retained.

## The branch certificate

Let `S_j[k,z] = U[k,z] R_j[k,z]`, with N=48. Split the first Fourier index as
`k=r+3m`, where r=0,1,2 and m=0,...,15. Write its output index as `y=t+16*l`,
where t=0,...,15 and l=0,1,2. After the three 16-point transforms, define

```
T_j[t,r,z] = exp(2*pi*i*r*t/48)/48
             * sum(m=0..15) S_j[r+3*m,z] exp(2*pi*i*m*t/16).
```

The unfinished three-way merge would produce

```
A_j[t+16*l,z] = sum(r=0..2) T_j[t,r,z] exp(2*pi*i*r*l/3).
```

For any of these three rows and any final x coordinate, the triangle inequality
gives

```
C_j[t+16*l,x] <= G_j[t]
G_j[t] = (1/48) sum(z=0..47) sum(r=0..2) abs(T_j[t,r,z]).
```

Hermitian symmetry lets us form this sum from z=0,...,24, with multiplicity two
on interior columns. Under z reversal the three residue magnitudes permute,
so their sum is unchanged. The input need not have reflection symmetry as a
spatial image.

If `G_j[t]` is below the incumbent peak, none of the three descendant rows can
win. **Do not perform their first-axis merge, form their intermediate rows, or
run their final-axis transforms.** The same certificate is reused against the
competitor incumbent. A branch rejected during the peak search is retained for
possible completion during the competitor search; a smaller competitor threshold
can require it. The proven unanimous-zero shortcut remains available.

As in the previous kernel, comparisons include a `1e-12` guard for bounded
cross-spectrum inputs. Ties are evaluated and equal computed values retain the
original coordinate ordering. The mathematical certificate is exact; this is
guarded floating-point code, not interval arithmetic or a universal bitwise-tie
guarantee across compilers.

## Execution schedule

The short complex transforms still share four ring lanes and the native bfft
radix-four codelet. The new plan retains their intermediate coefficients.
An admitted branch completes its three-way merge and receives tighter per-row
bounds; only surviving rows then finish the second axis.

The selected schedule sorts the 16 branch bounds once per ring. Within an
admitted branch it visits the three rows in descending row-bound order. This
requires less bookkeeping than a priority queue containing both branches and
rows. It can evaluate a few more final rows, but measured runtime improves.
Plans, weights, bounds, intermediate storage, and queue/order storage are all
fixed and created at construction. No audio-thread allocation is added.

## Alternatives actually measured

The feasibility audit pauses the first axis at several radices and checks
each bound against all covered oracle pixels. It uses oracle incumbents to
report **optimistic pruning potential**, not executable timings.

| Completed radix | Optimistic remaining branches omitted, speech |
|---|---:|
| 2 | 0.7% |
| 4 | 9.8% |
| 8 | 24.7% |
| 12 | 38.5% |
| 16 | 48.6% |
| 24 | 61.6% |

Early in the graph, too little phase cancellation has happened to make the
triangle bound selective. A later bound is tighter, but more work is already
paid for. These percentages are branch counts at different depths, not directly
comparable percentages of the total transform arithmetic.

The native 12-by-4 candidate was slower than the previous row-pruning kernel.
The 16-by-3 candidate won. Three bounds were implemented: the triangle sum,
a Cauchy-Schwarz bound over residues, and a weighted energy bound using the
radial masks as positive weights. The cheaper energy bounds were less selective;
none clearly beat the triangle version overall. They remain reproducible through
research compile definitions `CLEANUP_EARLY_Q=12|16`,
`CLEANUP_EARLY_ENERGY_BOUND=0|1|2`, and `CLEANUP_EARLY_HEAP=0|1`.
These are implementation probes, not filter tuning parameters or UI controls.

## Initial native measurements

Same 153 patch pairs as the preceding experiment, including 144 speech pairs;
1,071 ring maps, both full competitor queries and zero-consensus operation:

- No peak-location mismatches or near-tie location differences.
- Maximum peak difference `5.56e-17`; competitor difference `6.94e-17`.
- 48.64% of first-axis three-row branches omitted on the speech sample.
- 81.23% of final values omitted. The prior priority-queue schedule omitted
  82.99%, but incurred more scheduling overhead.
- M4: 24.87 us per pair versus 27.68 us for the previous row-pruning kernel,
  including bounds and searches: approximately 10% less time / 1.11x speedup.

Evidence: `docs/evidence/ring-early/early-flat-0.json`. The other native variants
and feasibility results are retained in that directory. These sample rates of
pruning are not universal guarantees; ambiguous inputs can require completion.

Reproduce with:

```
/Users/ultimussecundai/.local/bin/m4build -- sh tools/verify-early-mini.sh
```

The isolated probe is `research/true_superresolution/ring_early_probe.cpp`;
the mathematical feasibility audit is `early_ring_bounds.py` in that directory.
Actual SDR# load/audio validation is separate from these native tests.

## Integrated result and practical limit

The default early configuration (16-by-3, triangle bound, fixed branch order)
passed all eight registered-guide tests. A repeated isolated probe during that
build measured 27.93 us for the prior row-pruning kernel and 24.93 us for the
early kernel (1.12x). The native Windows executable under Wine measured
57.51 us versus 48.93 us in the same run (1.18x), also with zero tested peak
location differences. Compare methods within each run, not raw times across
different runs or machines.

Full 2048-row Population processing on the M4, harmonics disabled:

| Measurement | Previous row pruning | Early branch pruning |
|---|---:|---:|
| Guide median, 32 blocks | 53.24 ms | 52.83 ms |
| Identical-stereo callback median, 256 paced blocks | 80.23 ms | 79.09 ms |
| Identical-stereo callback p95 | 83.27 ms | 82.54 ms |
| Identical-stereo callback maximum | 84.97 ms | 85.49 ms |
| Independent-stereo callback median, 128 paced blocks | 148.72 ms | 148.97 ms |
| Independent-stereo callback p95 | 152.11 ms | 151.91 ms |
| Independent-stereo callback maximum | 154.73 ms | 154.66 ms |

These are separate paced runs, not paired per-block timing estimates. The
independent-stereo results are effectively unchanged; the measurements do not
establish an end-to-end throughput improvement there. All observed callbacks
met the 170.67 ms deadline. This is a modest improvement in the sampled isolated
kernel, **not** evidence for a large additional whole-pipeline speedup.

The saved 4,194,304 output samples differ from the preceding row-pruning output
by at most `2.17e-19`, with RMS difference `1.55e-22`. Files `probe.json`,
`wine.json`, `guide-tests.log`, `guide-profile.json`, `callback.json`,
`independent.json`, and `audio-parity.json` in `docs/evidence/ring-early/` retain
the measurements. No early-kernel DLL has been staged into the running SDR#
installation. The implementation stays a default-off comparison option.
