# Exact native forward transform for the seven-ring product

This kernel evaluates the same seven complete correlation fields as the former
seven-inverse path. Peak selection, the wrapped 7-by-7 competitor exclusion,
ring energy, ambiguity weights, displacement dispersion and Cleanup are retained.
It is not the rejected moment/candidate estimator experiment.

## Derivation

For one chart pair, the normalized cross spectrum is Hermitian:

```
U[-k] = conj(U[k]),    U[k] = a[k] + i*b[k].
```

Thus `a` is even under simultaneous coordinate reversal and `b` is odd. Each
real radial ring mask is also even: `R_j[-k]=R_j[k]`. Define the common real
Hartley input and its ring-weighted versions:

```
h[k] = a[k] - b[k]
h_j[k] = R_j[k] * h[k]
F_j = forwardDFT2(h_j)
```

Writing `theta=2*pi*k.d/N`, parity cancels `sum b*cos(theta)` and
`sum a*sin(theta)`. Consequently

```
Re(F_j[d]) = sum_k R_j[k] a[k] cos(theta)
Im(F_j[d]) = sum_k R_j[k] b[k] sin(theta)

inverseDFT2(U*R_j)[d] = (Re(F_j[d]) - Im(F_j[d])) / N^2.
```

This is a full finite-grid identity, including DC, Nyquist, all displacements
and every competing peak. It introduces no truncation, projection, new
estimator, changed mask or spatial search restriction.

The transform's real input also gives `F_j[-d]=conj(F_j[d])`. One retained
half-spectrum therefore produces both opposite output pixels:

```
C_j[ d] = (Re(F_j[d]) - Im(F_j[d])) / N^2
C_j[-d] = (Re(F_j[d]) + Im(F_j[d])) / N^2.
```

## Native factorization

`native/src/ring_forward.h` treats the seven ring responses as components of a
lane-valued real coefficient. The same butterfly graph, indices and twiddles
act on those components. Ring weighting is fused into real-input preparation;
there are no seven complex inverse workspaces or seven inverse API calls.

The current CPU schedule carries four responses at a time, with one inactive
lane in the second tile. This was faster than both two- and eight-lane schedules
on the Mini. The transform still calculates seven outputs: it does not claim
seven results for the arithmetic cost of one scalar transform.

The 48-point chart transform factors as `3 * 16`:

1. Real row transforms use bfft's normalized Bruun DIT walk, generalized to
   lane-valued coefficients through `forward_lanes`.
2. Complex column transforms use a direct `4 * 4` complex codelet in bfft,
   replacing the previous pair of real FFT calls and reconstruction for each
   complex transform. Its factorization is

   ```
   A_r[k] = sum(m=0..3) x[4*m+r] W_4^(m*k)
   X[k+4*l] = sum(r=0..3) A_r[k] W_16^(r*k) W_4^(r*l).
   ```

3. The outer radix-three stage completes each 48-point transform.
4. Hartley recombination emits every real correlation pixel with one final
   `1/2304` normalization. The existing peak search accepts these real arrays.

The two bfft extensions are recorded in `vendor/bfft/LOCAL-EXTENSIONS.md`.
Planning tables and all workspace are allocated at guide construction/Reload.
The audio path creates no plans, arrays, threads or Python objects for this bank.

Only 48-by-48 charts use this specialization. Other supported chart geometries
retain the existing exact transform. `CLEANUP_FUSED_RING_FORWARD=OFF` builds the
original path for controlled comparisons. `CLEANUP_RING_LANES` selects 2, 4 or 8
coefficient lanes; this is a build setting, not a DSP control.

## Isolated native results

The probe compares complete fields against the former C++ seven-inverse path,
checks selected pixels against an independent literal DFT, and compares every
ring's argmax and excluded-neighborhood competitor. Inputs include silence,
known translations near wrapped boundaries and random Hermitian phase fields.

| Probe | Seven inverses | Fused forward bank | Speedup |
|---|---:|---:|---:|
| M4 Mini, Clang, four lanes | 95.38 us | 48.26 us | 1.98x |
| Windows x64 executable under Wine, GCC | 105.84 us | 53.16 us | 1.99x |

Maximum whole-field difference was `1.13e-16`; the independent direct-DFT check
was within `1.00e-16`. The tested peak locations had zero mismatches. Competing
peak values differed by at most `2.09e-17`. This is numerical equivalence, not a
claim of bit-identical rounding for every possible input or tie.

These timings cover the entire ring-bank transform, including mask intake and
output assembly. They exclude guide observation construction, moving-patch
forward spectra, peak scans and the Cleanup mask. The complete callback must
be measured separately. Wine measurements are not native-Windows qualification.

Run `tools/verify-guide-mini.sh` via `m4build` for the transform probe and full
registered-guide/oracle tests. The standalone probe source is
`research/true_superresolution/ring_transform_probe.cpp`.

## Complete guide and callback measurements

The full-bank integration passed all eight registered-guide tests. On the M4,
the 2048-row guide's median wall time was 60.68 ms. A paced full Population
callback with identical stereo channels took median 86.50 ms, p95 88.34 ms,
maximum 90.70 ms over 256 blocks, versus a 170.67 ms block deadline. The earlier
bank's corresponding median was 106.90 ms.

Independent stereo took median 163.99 ms, p95 166.92 ms, maximum 170.18 ms over
128 paced blocks. That maximum only barely fits the deadline; it is not useful
headroom or a guarantee for other inputs. Evidence is under
`docs/evidence/ring-fused-*.json`. These are M4 measurements, not an SDR# test.

The subsequent [bounded-query experiment](ring-pruning.md) asks whether the
complete fields need to be produced at all. It preserves the peak decisions by
bounds and remains a separate build option.
