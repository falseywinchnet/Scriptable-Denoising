# Implementation plan: an exact-subproblem cartoon stage

Target: a second solver path behind `meyer.h`, alongside the present split
Bregman engine in `src/detail/meyer_kernel.hpp`.
Status: prototyped, measured, and audited; not implemented in C++.
Alternative to — not a replacement for — `include/bfft/meyer_facr_plan.md`.

Evidence: `experiments/cartoon_stage_tautstring.py`,
`experiments/cartoon_stage_isotropy_cost.py`,
`experiments/cartoon_metric_audit.py`,
`notes/cartoon_stage_problem_statement.md`.

## 1. The claim

The shipped stage reaches its answer only through pointwise thresholding: it
can never *decide* that a region is flat, only converge toward it. Measured,
that costs it the jump set. A method whose subproblems are solved **exactly**
decides instead, and the difference is large:

| | split Bregman | this path |
|---|---:|---:|
| iterations to 90% jump-set agreement (cameraman) | 11 | 4 |
| to 95% | 17 | 5 |
| to 98% | 27 | **7** |
| relative error at 64 iterations | 1.9e-3 | **2.9e-4** |
| ms per iteration (NumPy prototype) | 7 | **3** |

On a synthetic cartoon-plus-texture field split Bregman **never reaches 90%
agreement in 64 iterations**; this path reaches it at 29.

It is faster per iteration as well as needing fewer, because it contains no
transform at all.

## 2. What it costs, measured end to end

The functional changes from isotropic to anisotropic TV. That is a real
substitution, so it was scored against the pipeline's own objective with the
cartoon swapped at the module boundary — every downstream consumer (metric,
edge and texture fields, `base_lab`, allocation pressure) saw the substituted
cartoon, and the scoring operator was left unpatched.

| arm | Pikachu PSNR | objective | Chelsea PSNR | objective |
|---|---:|---:|---:|---:|
| isotropic, periodic (shipped) | 26.36 | 3.192e-3 | 32.50 | 8.833e-4 |
| isotropic, Neumann | 26.04 | +6.9% | 32.49 | +0.5% |
| anisotropic, 24 passes | 26.66 | −7.0% | 32.59 | −0.7% |
| anisotropic, 8 passes | **26.78** | **−9.4%** | 32.49 | see §2.1 |

The isotropic-Neumann arm exists to isolate a confound: this path uses free
line ends where the kernel wraps. Neumann alone is neutral to worse, so the
gain is the anisotropy and this path wins against a handicap.

### 2.1 A metric defect found while auditing this

Chelsea's 8-pass arm initially scored −13.3%. It is an artifact.
`native_components` computes `scale = 255/(max-min)` per channel — an
extreme-value statistic — **separately for the target and the
reconstruction**, so the two are stretched by different affine maps before
being decomposed. Measured drift and its effect:

| chelsea arm | scale drift | shipped metric | commensurable |
|---|---:|---:|---:|
| isotropic, periodic | 8.46% | 1.213e-4 | 3.051e-6 |
| anisotropic, 24 | 5.79% | 1.295e-4 | 2.643e-6 |
| anisotropic, 8 | 0.50% | 3.021e-6 | 2.596e-6 |

Under a commensurable metric all three sit within 17% of each other. **Chelsea
is neutral, not a win.** Pikachu's drift is comparable across arms so it is
not differentially biased, and its commensurable cartoon term improves
1.178e-4 → 9.839e-5 → 9.388e-5, slightly better than shipped scoring showed.

This defect is independent of this plan and affects any `cartoon_mse` or
`texture_mse` comparison between arms of differing dynamic range. Worth fixing
on its own account. `rgb_mse` and PSNR are unaffected.

## 3. The construction

**Inner primitive: exact 1-D TV.** For `min_x (1/2)||x-y||^2 + lam*||Dx||_1`,
substitute cumulative sums. With `X = cumsum(x)` and `Y = cumsum(y)`, the dual
constraint `|p_i| <= lam` becomes `|X_i - Y_i| <= lam`, and the objective
becomes the sum of squared increments of `X`. So the solution is the **taut
string** through a tube of radius `lam` around the cumulative sums, pinned at
both ends, and `x = diff(X)`. No iteration, no tolerance.

**Outer: Douglas-Rachford over rows and columns.** Anisotropic 2-D TV splits
as `TVx + TVy`, and the quadratic folds into the first:

```
F(u) = (1/2)||u-g||^2 + lam*TVx(u)     prox = row-wise 1-D TV on
                                         (v + gamma*g)/(1+gamma)
                                         at gamma*lam/(1+gamma)
G(u) = lam*TVy(u)                      prox = column-wise 1-D TV at gamma*lam

u <- prox_F(z);  v <- prox_G(2u - z);  z <- z + v - u
```

**Alternation: unchanged.** The TGFD structure stays exactly as
`run_passes` has it — at pass zero the texture is defined zero so the cartoon
step sees `f`, and thereafter it sees `u + w`; the texture step always sees
`f - u`. `lam` and `mu` keep their meanings. Only the inner solver changes.

## 4. Why this path is *smaller*, not just faster

It has no transform, so it has no spectra, no symbol tables, and no transform
staging. Against the 26 plane-equivalents inventoried in
`notes/cartoon_stage_state_algebra.md`:

| | planes |
|---|---:|
| `u`, `w` | 2 |
| `z_u`, `z_v` (Douglas-Rachford states) | 2 |
| per-thread 1-D scratch | `O(max(H,W))` per thread |
| **total** | **4** |

versus 26 allocated today, or 12 after the reductions in that note. At 2048²
that is 128 MB against 832 MB. Given that the stage is bandwidth-bound above
1024² — 13.63 ns/pixel/pass at 512² rising to 39.40 at 2048² — this is not a
side benefit; it is likely a second speed effect on top of §1.

**And it removes the padding entirely.** No FFT means no power-of-two
constraint, so arbitrary sizes run natively. That is strictly better than
`meyer_facr_plan.md`, which only unpads one axis: 1080p and 4K stop paying
2.02x, and sizes just above a power of two stop paying 3.99x.

## 5. What to implement

**`tv1d_taut_string(y, lam, x, scratch)`.** The prototype rescans forward from
each anchor and is `O(N^2)` worst case; texture-heavy content cost 2.7x more
per iteration than smooth content because of it. **Implement the `O(N)`
amortized form**: maintain upper and lower hull deques and pop from them as
the string bends, rather than rescanning. This is the single most important
implementation difference from the prototype.

**`tv_rows` / `tv_cols`.** Rows are contiguous. For columns, either transpose
into one scratch plane (costs the fifth plane, keeps the inner loop
contiguous) or sweep with stride. Benchmark both; at these sizes transposing
usually wins. Parallelise over lines — the natural axis, and unlike the
current column stage it needs no panel scatter or gather.

**`dr_step`** and the TGFD alternation, both as §3.

## 6. API surface

Add a mode to the existing selector rather than a new entry point, so the two
plans compose:

```c
/* 0 = split Bregman, 2-D spectral (current default)
   1 = split Bregman, one axis swept   (meyer_facr_plan.md)
   3 = Douglas-Rachford over exact 1-D solves (this plan) */
bfft_status bfft_meyer_plan_set_solver(bfft_meyer_plan* plan, int mode);
```

Mode 3 must relax `_meyer_padded` to pass sizes through untouched, and must be
documented as **changing the functional**, not merely the solver. Default
stays 0.

## 7. Validation

1. **Certify the 1-D solver before anything else.** Duality gap
   `<x, x-y> + lam*||Dx||_1` and dual feasibility `|cumsum(y)-cumsum(x)| <=
   lam`. The prototype certifies at 1.5e-14 and 3.8e-14 over 200 random
   problems, and matches an independent reference to 5e-15. Port both checks;
   they are cheap and they are the whole foundation.
   *Note for whoever writes them:* normalize the gap by `(1/2)||y||^2`, not by
   the primal value — on a constant input the primal is pure rounding and the
   ratio becomes meaningless. Two of my three certificate attempts were wrong
   before the solver ever was.
2. Anisotropic 2-D against a slow reference (long-run split Bregman with a
   componentwise shrink) on the same functional and the same boundary
   condition. `div(grad(u))` must match its DCT symbol first, or the two are
   not minimizing the same thing — that mistake voided a whole run here.
3. Jump-set agreement curves reproduced against §1.
4. The end-to-end objective of §2 on more than two images before this is
   considered for a default.

## 8. Risks and open questions

- **The speed case is a prediction, not a result.** The prototype is
  NumPy/Numba against optimized C: 8-9 ms against 3-4 ms for the shipped
  24-pass split at 128². The iteration count, the absence of transforms, and
  the 6x smaller working set are the reasons to expect a compiled version to
  win. None of them is a measurement of a compiled version.
- **Two images is not generality.** Positive on Pikachu, neutral on Chelsea.
- **Fewer passes beat more passes in every arm.** Eight anisotropic passes
  beat twenty-four on both images. Partly explained by 4x faster convergence,
  but it also means the pipeline objective is not monotone in cartoon
  convergence. Understand this before tuning `passes` on its strength.
- **Anisotropic TV has a grid bias**, favouring axis-aligned edges. Nothing in
  the measurements above isolates it, and the downstream metric is exactly the
  kind that might not notice while a viewer would. Look at a Pikachu cartoon
  from this path at full size before committing.

## 9. Staging

1. `tv1d_taut_string` with the `O(N)` hull deques, plus the §7.1 certificate
   as a test. Standalone and useful on its own.
2. `tv_rows` / `tv_cols` with threading over lines.
3. `dr_step` and the alternation behind mode 3, padding still applied, so the
   only variable is the solver.
4. Relax padding for mode 3. This is where the size win arrives.
5. Broader validation per §7.4, then reconsider the default.
