# Sharing Fourier work in the registered observation

Keep the seven offsets, 2048-sample aperture, 4094-coordinate observation grid,
seven ring filters, patch windows, registration confidence and Cleanup mask.
This investigation concerns equivalent evaluation of those operations.

## Work now shared

The six registration lanes formerly computed identical reference patch spectra.
They now divide that preparation into disjoint chart ranges, wait for completion,
then all read one common reference cache. Every reference patch is transformed
once per context. The immutable cache belongs to the parent; each lane retains
its own Fourier workspace. Failures join all jobs before the arrays can move.

Real patches have Hermitian spectra. Only the nonnegative columns are needed
for cross-phase products and ring inversion. Ring energy is accumulated from
that half with multiplicity two for interior columns and one for DC and even-
size Nyquist columns. Negative columns are supplied by conjugate symmetry.
This changes floating-point summation order, not the mathematical ring energy.

The real 48-point inverse also needs only the nonnegative half of each 16-point
radix-three residue. DC/Nyquist columns themselves take a real inverse. No
nonzero Gaussian tails, rings or charts are discarded.

The competing-peak scan now retains row maxima from the first argmax. Outside
the seven wrapped rows intersecting the excluded 7-by-7 neighborhood, those
maxima already answer the competitor query. Only the intersecting rows need
rescanning. First-tie selection and the competing value remain exact, including
wrapped boundaries and odd chart sizes. Bilinear confidence comparison also
reuses reflected coordinates per row/column, preserving each sample's arithmetic.

On the M4 profiling build, this reduced summed-lane peak search work from about
111 ms to 19–21 ms and guide wall time from about 95 ms to 83 ms. Summed-lane
times are not callback latency. Whole-program link optimization produced no
material additional gain and is not enabled by default.

`tools/profile_guide.py` measures native registration internals when built with
`CLEANUP_PROFILE_GUIDE=ON`. `tools/profile_filter.py` instruments an in-memory
copy of the actual Numba callback, using the research-only clock helper in
`research/profile_filter_clock.cpp`. Neither instrumentation is distributed or
inserted into the running SDR# filter. `tools/sample_live_budget.py` reads native
control status without changing buttons or receiver settings.

The filter profile also found three triangular time-smoothing passes consuming
about 25 ms of a 40 ms callback-filter budget. `sawtooth` now buffers small tiles
of adjacent bins, visiting contiguous memory while keeping all 15 original taps,
the literal divisor, zero boundaries, addition order and in-place behavior.
This reduced a full-width pass from about 8.4 ms to 2.5 ms in isolation. The
targeted test compares against the literal old loop, with minimum workspace,
partial tiles, short frames and both aliasing modes; outputs are bit-for-bit equal.

## An exact alternative for overlapping windows

Let `N=2048`, `L=2(N-1)=4094`, `theta_r=2*pi*r/L`, `omega=2*pi/N`, and

```
T_s(q) = sum(j=0..N-1) x[s+j] exp(-i*q*j)
T_(s+1)(q) = exp(i*q) * (T_s(q) - x[s] + x[s+N]*exp(-i*q*N))
```

The periodic Hann is a sum of three exponentials. Consequently its complex
observation at every desired center is exactly

```
H_s(r) = 0.5*T_s(theta_r)
       - 0.25*T_s(theta_r-omega) - 0.25*T_s(theta_r+omega)
C_s(r) = real(H_s(r) - 0.5*w[N-1]*x[s+N-1]*(-1)^r) / (N-1)
observation_s(r) = abs(C_s(r))
```

The final correction retains the half weight of the second inverse's endpoint;
the first endpoint contributes zero because the periodic Hann starts at zero.
The three sums can advance together through sorted requested centers. They
share their input samples, outgoing samples, frequency plans and phase factors.
This is a valid route to a sliding/polyphase bank; it is not the incorrect
shortcut of rotating one window's FFT while ignoring its entering/leaving data.

`research/true_superresolution/shared_window_proof.py` checks this identity
against the literal NumPy inverse at multiple apertures and irregular offsets.
It is a derivation test, not a production replacement.

### Cost and numerical conditions

Direct recurrence costs three complex sums per output row per advanced sample.
At full width, that is 6144 complex state updates per input sample. A blocked
or polyphase implementation must beat the existing cached Bluestein transforms,
including its initialization and periodic re-anchoring to bound roundoff drift.
The retained-overlap cache already avoids recomputing unchanged observations.

The Hann modulation spacing is `2*pi/2048`; the observation spacing is
`2*pi/4094`. Thus the three grids are not adjacent bins of one ordinary
4094-point FFT. Rounding the grid to 4096 or shifting the observation centers
would change the representation and is outside this equivalence optimization.

## The ring bank

The [exact native forward ring transform](ring-forward-transform.md) now uses
a real Hartley input, a lane-valued Bruun walk and a direct complex radix-four
codelet to emit all seven complete correlation fields. The earlier
[moment/candidate estimator substitution](ring-forward-moments-rejected.md) was
rejected by the user and is not part of the live algorithm.

For each patch pair the code already shares the forward spectra and normalized
cross spectrum `U`. Its seven answers are `inverse2(U * R_j)` for seven distinct
radial Gaussian masks. FFT linearity permits sharing transforms of common basis
filters, but it does not reduce seven independent masks to one output. A lower-
rank approximation needs a separate experiment because even a small correlation
error can move a peak or alter its competing-peak confidence.

The full forward bank now shares that butterfly traversal. The next experiment
goes further: [bounded row pruning](ring-pruning.md) uses Hermitian symmetry to
rule out unfinished rows before their last transform, and omits confidence
calculations when every ring's proven winning displacement is zero. Its native
prototype omits 83% of final values on the sampled speech and takes about 27 us
against 37 us for the complete fused bank plus search. This remains a separate,
default-off native experiment pending broader integration validation.
