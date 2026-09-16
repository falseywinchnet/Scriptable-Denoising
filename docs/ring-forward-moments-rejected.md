# Rejected estimator substitution: forward ring moments

Status: the user rejected this estimator substitution as outside the requested
exact transform optimization. Retained as a negative control only; see
[the exact native transform](ring-forward-transform.md). The live guide and Cleanup mask are
unchanged by this experiment. This investigates eliminating the seven complete
correlation inverses, not merely putting seven calls behind a batch API.

## What the current inverses supply

For one pair of chart spectra `A,B`, let

```
U[k] = B[k] conj(A[k]) / max(|B[k] conj(A[k])|, 1e-12)
R_j[k] = exp(-(r[k]-c_j)^2 / (2 sigma^2))
c_j = .035 + j*(.46-.035)/6,  sigma = .055,  j = 0..6
S_j[k] = U[k] R_j[k]
C_j[d] = inverseDFT2(S_j)[d]
```

The present code constructs seven real 48-by-48 `C_j` maps, then keeps only:
the maximum's position and value, the largest competing value outside the wrapped
7-by-7 neighborhood, and a separately computed joint spectral energy. Those
values determine ring weights, mean displacement, and displacement dispersion.
The maps themselves are discarded.

## Exact spectral identity: no correlation inverse needed for moments

Let `M=N*N`. The Fourier coefficients of each correlation's **squared** field are

```
sum_d |C_j[d]|^2 exp(-2 pi i q.d/N)
  = (1/M) sum_k S_j[k] conj(S_j[k-q])
  = (1/M) sum_k U[k] conj(U[k-q]) R_j[k] R_j[k-q].
```

Indices wrap on the original chart torus. This follows by substituting the
inverse DFT twice: orthogonality removes all terms except `k-l=q`.
It is a finite discrete identity, with no continuous approximation, phase
unwrapping, Gaussian truncation, changed aperture, or dropped ring.

For `q=(1,0)` and `(0,1)`, these are circular first moments. Their arguments
give the two circular centroids:

```
d_x = -N/(2*pi) arg(moment_x)
d_y = -N/(2*pi) arg(moment_y).
```

For a pure integer circular translation, `U[k]=exp(-2*pi*i*k.d/N)`, every
nonzero ring moment has exactly that translation's phase. The zero moment is
the correlation power by Parseval. Higher `q` values give additional moments
through the same construction, at additional reduction cost.

### Where the work is shared

After the existing chart forward transforms, calculate each of the two complex
neighbor products `U[k] conj(U[k-q])` once. Every ring then contributes only
a precomputed real weight to seven accumulators. Joint spectral energies can
be accumulated in the same traversal. This is O(7*M) work for fixed moment count,
instead of O(7*M*log(M)) for complete correlation maps.

There is further Gaussian structure. With `r=r[k]`, `r'=r[k-q]`,

```
R_j(r) R_j(r') = exp(-(r-r')^2/(4*sigma^2))
                 * exp(-((r+r')/2-c_j)^2/sigma^2).
```

The first factor is common to all rings; the second evaluates one narrower
Gaussian family at their seven centers. All weights are fixed by geometry and
can be prepared at Reload. No exponential belongs in the audio traversal.
This can be an extension of the forward-spectrum output stage; it does not
require seven additional forward FFTs either.

## Crucial distinction: centroid is not maximum

The moment identity is exact, but replacing the current peak estimator by a
centroid changes the registration decision. The trial normalizes moment
magnitude by `sum |U[k] conj(U[k-q])| R_j[k]R_j[k-q]` and uses that phase
coherence as a candidate reliability. This is **not** the existing
competing-peak confidence and is not a probability of signal or noise.

Measured results in `docs/evidence/ring-forward-moments.json`:

* Moment identity error at extents 13, 24, 48: at most 2.65e-16.
* Ideal translations, including near the wrapped boundary: every ring's
  displacement agrees within 1.2e-14 pixel.
* 144 actual speech chart pairs from six sub-hop views: displacement difference
  from the current weighted-peak result is median .294 pixel, p95 1.961,
  maximum 6.171. These are differences from the existing estimator, not errors
  against known speech-warp ground truth.
* A periodic pattern gives high neighbor-phase coherence despite ambiguous
  peaks. Two competing translations pull the centroid between modes.

Consequently the first-moment estimator alone is **not promoted**. Its useful
role is a low-cost proposal, or a component of a richer phase-domain estimator
that explicitly represents ambiguity and multiple displacement modes.

## One shared forward transform plus selected exact ring evaluations

The unweighted correlation can be obtained with one forward transform:

```
C_common[d] = forwardDFT2(U)[-d] / M.
```

This sign reversal alone saves no work versus one inverse; the potential saving
is using **one common map for proposals** rather than seven complete ring maps.
At any proposed displacement `d`, all original ring scores can be evaluated as

```
C_j[d] = (1/M) sum_k R_j[k] Re(U[k] exp(2*pi*i*k.d/N)).
```

The complex Fourier atom is shared across all seven real ring accumulations.
The prototype uses precomputed phase tables and gives the same queried scores
as the seven complete inverse maps to 2.32e-15 maximum absolute error.

A deliberately small proposal set—the eight highest common-map pixels plus
floor/ceiling coordinates of each ring moment—used an average 17.22 locations
per speech chart. It captured 98.71% of the original ring maxima, but only
43.15% of the original competing maxima. That exposes the real unresolved
problem: finding enough global ambiguity evidence, not evaluating scores.
Missing competitors would overstate confidence, so this is not a safe direct
replacement either. The candidate counts are probe settings, not new DSP defaults.

An exact preservation route would need candidate completeness certificates and
an explicit fallback for uncertified charts. A redesigned phase-domain estimator
would instead need separate quality validation against periodic, multimodal,
incoherent and actual speech fields. Neither gate has been met by this probe.

## What a simple linear-basis reduction can and cannot do

The seven current masks have relative singular values
`1, .744, .541, .374, .241, .142, .0718` on the 48-by-48 grid. They are not
nearly one scalar mask. In fact their equally spaced centers factor into a
nonzero common Gaussian times powers of `exp(delta_c*r/sigma^2)` and nonzero
column scales: a Vandermonde family, full rank at seven distinct radii.

This rules out reconstructing all seven arbitrary correlation maps by fixed
scalar mixing of fewer than seven mask-basis maps. It is **not** a lower bound
against shared FFT butterflies, structured filter banks, selective Fourier
evaluation, or a different displacement estimator. The moment construction
works by asking for different, much smaller outputs.

## Literature and reproducibility

Direct Fourier-domain registration has precedent in Balci and Foroosh,
[*Subpixel estimation of shifts directly in the Fourier domain* (2006)]
(https://stars.library.ucf.edu/facultybib2000/5923/). Their phase-difference and
group-delay method is relevant background; this prototype does not implement
their complete estimator or borrow its performance claims.

Guizar-Sicairos, Thurman and Fienup,
[*Efficient subpixel image registration algorithms* (2008)]
(https://labsites.rochester.edu/fienup/wp-content/uploads/2019/07/MGS_OL08_EffRegistration.pdf),
demonstrate selective matrix-DFT evaluation for local peak refinement. That
supports avoiding unnecessary full maps, but does not certify our ring-specific
global competing peaks. The discrete ring-moment derivation above is given
explicitly and tested independently of either paper.

Run from the authoritative checkout:

```
/Users/ultimussecundai/.local/bin/m4build -- /Users/joshuahkuttenkuler/miniforge3/bin/python research/true_superresolution/ring_forward_moments.py
```

The prototype's moment accumulator takes about 26 microseconds per 48-by-48
patch on the Mini. Its NumPy inverse comparator includes Python peak searches;
the resulting ratio is **not a native bfft speedup measurement** and excludes
the forward input transforms and final estimator overhead. No claim of live
audio improvement follows from this research timing.
