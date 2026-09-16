# Implementation plan: sweep one axis instead of transforming it

Target: `src/detail/meyer_kernel.hpp`, the engine behind `meyer.h`.
Status: designed and proven equivalent; not yet implemented.
Evidence: `experiments/cartoon_stage_study.py`,
`experiments/cartoon_stage_tridiag.py`, `notes/cartoon_stage_rework.md`.

## 1. Why

Every split-Bregman subproblem solves `(c - eta*Laplacian) u = rhs`. The
engine does this with a full 2-D real transform, a pointwise multiply by
`symbol()`, and a full inverse — four such transforms per pass. Because the
Bruun kernel is radix-2, `_meyer_padded` first rounds **each** dimension up to
the next power of two by symmetric reflection.

Measured consequences:

| | |
|---|---|
| transformed area, 1080p / 1440p / 4K | 2.02x / 2.28x / 2.02x the image |
| transformed area, just above a power of two | 3.99x |
| cost per pixel per pass, 512² -> 2048² | 13.63 -> 39.40 ns (memory cliff) |
| distance to the 48-pass answer at the shipped 24 passes | 8.9e-3, still moving |

The last row matters for prioritisation: there is no slack to reclaim from a
convergence test, so the only lever is making each pass cheaper.

## 2. The construction

The operator is separable, but only the *transform* needs to be. Transform
along one axis. For each transformed-axis frequency `k` the remaining equation
along the swept axis is

```
-eta*u[y-1] + (c - eta*lx(k) + 2*eta)*u[y] - eta*u[y+1] = rhs[k, y]
```

a symmetric tridiagonal Toeplitz system. This is the Hockney/FACR
construction: `O(H)` per bin where the column FFT is `O(H log H)`.

**Stability is structural, not conditional.** `lx(k) = 2cos(2*pi*k/W) - 2`
lies in `[-4, 0]`, so the diagonal `d_k = c - eta*lx(k) + 2*eta` lies in
`[c + 2*eta, c + 6*eta]` while the off-diagonal magnitudes sum to `2*eta`.
Since `c > 0`, every row is strictly diagonally dominant for every `k`. Thomas
needs no pivoting and no growth check, at any `(c, eta)` the engine uses.

Three consequences, in order of size:

1. **The swept axis stops being padded.** A tridiagonal solve does not care
   whether its length is a power of two.
2. **Half the transform work disappears**, even at sizes with no padding at
   all, because one direction stops being transformed.
3. **Neumann boundaries in the swept axis become free** — fold the first and
   last rows instead of wrapping. This is defect 7 of
   `viewer/TRANSPORT_CELL_MATH.md`, gone in that axis.

## 3. Axis selection

Which axis to sweep is free, and it is where the win lives. Sweep the
**worse-padded** axis:

```
sweep_height_area = H * next_pow2(W)
sweep_width_area  = next_pow2(H) * W
```

Choose the smaller; fall back to the present path when
`next_pow2(H)*next_pow2(W)` is already `H*W` — see §8.

Sweeping the height keeps the existing row-transform stage untouched, so
**stage 1 should implement height-sweeping only** and treat width-sweeping as
a later transpose-based variant.

| image | now | swept | saved | axis |
|---|---|---|---|---|
| 1920x1080 | 2.02x | 1.07x | 1.90x | width |
| 2560x1440 | 2.28x | 1.42x | 1.60x | height |
| 3840x2160 | 2.02x | 1.07x | 1.90x | width |
| 1025x1025 | 3.99x | 2.00x | 2.00x | height |

## 4. What changes in the engine

**Replaced.** `cols_fwd`, `cols_inv`, and the `HB`-sized half of every
`spectrum`. The column stage's `panel_scatter` / `panel_gather` traffic
disappears with them; the `lane::col` plans are no longer created.

**Kept unchanged.** `fwd2d`'s row stage, `fwd2d_div`, `shrink`, the pool, the
lane row plans, and the whole `run_passes` structure. The row-rfft still
produces the `a` (Re) and `b` (Im) planes, and because the tridiagonal
coefficients are real, the two planes are still swept independently — the same
reason the current column stage can use real FFTs.

**New: `tri_factors`, replacing `symbol`.** Per `(c, eta)` pair, store the
Thomas reciprocal pivots `pivot[y][k]`. The super-diagonal is
`upper[y] = -eta * pivot[y]`, so it is computed on the fly rather than stored:
the table is `H * WB` doubles, the same size as the current interleaved
`symbol` table (`2 * WB * HB`). **No memory regression.**

```
d_k          = c - eta*lx(k) + 2*eta
pivot[0][k]  = 1 / d_k'                       (d' carries the boundary fold)
pivot[y][k]  = 1 / (d_k' - eta^2 * pivot[y-1][k])
```

*Optional refinement.* For a Toeplitz system the pivot recursion converges to
the fixed point of `p = 1/(d - eta^2 p)`, namely

```
p* = (d - sqrt(d^2 - 4*eta^2)) / (2*eta^2)
```

which is real because `d > 2*eta` (§2). So the table can be truncated to the
transient and `p*` used beyond it, shrinking it from `O(H*WB)` to
`O(transient*WB)`. The transient lengthens as `d -> 2*eta`, i.e. for `k = 0`
with small `c`, so truncation must be per-`k` against a tolerance, not global.
Treat this as a later optimisation, not part of stage 1.

**New: `sweep_columns`.** Two streaming passes over `(H, WB)` — a forward
elimination and a back substitution — parallelised over `k`, which is the
*same* parallel axis `cols_fwd` already uses. Thread mapping is unchanged.

## 5. Boundary conditions

Two variants, and they must not be conflated:

- **Periodic (bit-comparable).** Reproduces the current answer exactly. The
  wrapped corners are restored by a Sherman-Morrison rank-one correction:
  perturb the corner coupling away, solve twice against the factored system
  (the second right-hand side is fixed per `(c, eta)` and precomputable), and
  correct. Cost is two sweeps instead of one, still far below a transform.
  **Measured: agrees with the 2-D spectral solve to 1e-15 relative at every
  size and every `(c, eta)` tested.**

- **Neumann (better, and a behaviour change).** Fold the outward coupling into
  the first and last diagonal entries. One sweep, no correction, and it
  removes the wrap artifact along that axis. **Measured: satisfies its own
  operator to 5.4e-16 relative.** This changes output and must ship behind an
  explicit flag, defaulted off, with the padding logic relaxed only for that
  axis once it is on.

## 6. API surface

`bfft_meyer_plan_create` gains nothing required. Add:

```c
/* 0 = current 2-D spectral path; 1 = sweep the worse-padded axis
   (periodic, bit-comparable); 2 = sweep with Neumann boundaries
   (changes output, removes the wrap artifact on the swept axis). */
bfft_status bfft_meyer_plan_set_solver(bfft_meyer_plan* plan, int mode);
int         bfft_meyer_plan_solver(const bfft_meyer_plan* plan);
```

Default 0 through the first release so nothing moves under existing callers.
`_meyer_padded` in `bfft/_core.py` grows a matching `solver=` keyword and, for
modes 1 and 2, pads only the transformed axis. Plans are cached on a key that
already includes every shape-affecting parameter; add `solver` to it.

## 7. Validation

1. `experiments/cartoon_stage_tridiag.py` already asserts periodic-mode
   equivalence to 1e-15 and the Neumann residual to 5.4e-16 in NumPy. Port
   both as C-level tests against `symbol()` on random right-hand sides at
   non-power-of-two sizes.
2. Mode 1 against mode 0 on `tests/meyer_test.py`'s fixtures: cartoon and
   texture must agree to solver tolerance at every pass count.
3. Diagonal dominance is asserted at factor-build time — cheap, and it turns
   §2's argument into a runtime invariant.
4. Threading determinism: mode 1 output must be bit-identical across thread
   counts, matching the property the serial tolerance test already protects.

## 8. Where this must not be used

At sizes that are already powers of two there is no padding to recover, and
the prototype **loses** (0.77x at 1024²) because an `O(H)` sweep in NumPy
cannot outrun a tuned FFT. A compiled sweep should win there too — the ceiling
is 2.29x, since one of two transform directions goes away — but that is a
claim to re-measure after implementation, not to assume. Until then, select by
padding ratio: sweep only when it removes padding.

## 9. Expected payoff

Measured in NumPy, with a Python-level sweep that leaves most of the win on
the table. "Ceiling" is the spectral time against the transforms alone — what
remains once the sweep costs what a compiled streaming pass should.

| image | now | prototype | ceiling |
|---|---|---|---|
| 1080x1920 | 55.8 ms | 2.55x | 5.77x |
| 1025x1025 | 51.8 ms | 2.47x | 5.44x |
| 1440x2560 | 99.0 ms | 1.80x | 3.64x |
| 2048x2048 | 50.6 ms | 1.22x | 2.81x |
| 1024x1024 | 8.8 ms | 0.77x | 2.29x |

## 10. Staging

1. `tri_factors` + `sweep_columns`, periodic, height-swept, behind mode 1,
   padding unchanged. Pure refactor; proves equivalence in place.
2. Relax the height padding for mode 1. This is where the measured win
   arrives.
3. Neumann as mode 2, plus the flag plumbing through `bfft/_core.py`.
4. Width-sweeping via transpose, and the axis-selection policy of §3.
5. The truncated pivot table of §4.

Stages 1 and 2 are independent of the state reduction in
`notes/cartoon_stage_state_algebra.md`; they touch different parts of the
engine and can land in either order.
