# Optional DIP comb continuation

This page describes the retained `HARMONIC_MODEL="packet"` reference. The newer
`"excitation"` experiment uses measured harmonic spacing, separated envelope and
excitation, and common phase; see [its design and evidence](occupancy-excitation.md).

This experimental entity follows Cleanup and is disabled by default. The
**Harmonics** button and `harmonics on|off|toggle` native commands select it.
There are no sliders. Header constants in `Filter.py` and Reload set the lattice,
decay, upper limit, irregularity, and temporal peak-spacing limits.

## What the implementation transports

At packet level E, a full N-bin Fourier vector X has Q=N/E periodic coordinates
per residue row:

    B[delta,j] = (1/Q) sum_{k mod E = delta} X[k] exp(2 pi i k j/N).

The native helper descends the inverse DIP stages into B, projects a supported
lower member's periodic component, modulates it to a higher member of the same
row, and finishes the forward stages. It constructs the conjugate partner using
the selected transform's symmetry: N-k for RFFT, N-1-k for half-bin ODFT.
The independent test oracle evaluates the finite-Zak sum directly.

For this single-member projection, the resulting target coefficient is
algebraically donor coefficient times complex weight. The explicit intermediate
walk gives later experiments a place to manipulate the periodic components; it
does not manufacture additional observed information.

The donor selection uses measured lower-band prominence and the remaining
post-Cleanup complex amplitude. Absolute sample origin supplies a coherent
carrier phase across blocks. Upper synthesis has a soft receive-band boundary,
a soft high-frequency cutoff, decreasing gain with comb-member distance, and an
added-power budget of 12% of lower-band coefficient power. Existing upper-band
content reduces the added amount. Lower measured coefficients are preserved.

## Erosion happens between temporal peaks

For each comb, the entity tracks contextual lower-band energy. Between two
plausibly spaced peaks it measures the deficit beneath their connecting roof.
The deficit's weighted time centroid positions a soft elliptical erosion region.
Peak separation determines its time radius; a frequency radius spreads the
decay across adjacent upper members. These holes attenuate the synthesized
addition, with depth capped below one. They do not erase measured lower-band
signal. Constant sustained support creates no arbitrary holes.

Deterministic, smoothly interpolated noise adds slight complex irregularity
which increases toward the upper limit. It is keyed to absolute sample time,
so callback size and contextual reprocessing do not redraw unrelated noise.
This is currently coherent random texture, not a claim of a measured blue-noise
power spectrum or a calibrated speech excitation model.

## Fixed state and controls

The entity owns FilterBank slot 1. All scratch, donor maps, weights, energy
history, and mix state are fixed storage. Trigonometric tables are constructed
at Reload. The parent injects a typed native function address before compiling
the explicitly signed Numba methods. No Python dispatcher or heap-allocated
FFT runs in the audio callback.

Script ABI 2 adds a fifth config value: the absolute sample origin of the
contextual buffer. This is independent of public C ABI 1. Successful Reload
resets the parent histories and entity state; failed Reload preserves the
working generation. Switching off ramps out the added contribution, then the
entity returns exact passthrough. Whole-frame squelch resets its mix state.

## Limits and verification

The default packet spacing is fixed by E and the synthesis FFT size. It is not
an estimated voice fundamental. A tonal interferer may also support a donor.
An explicit voice gate and adaptive lattice selection remain experiments.
Synthetic continuation is not evidence that lost high-band speech was recovered.

## Revision 3: join and envelope experiment

The first listening image showed a gap and repeated high-band patterns. The old
code joined at the nominal receiver bandwidth even when the recording rolled off
earlier, then faded in over350Hz. It also used the same strongest packet donor
with inverse-power attenuation across many upper members.

The revised policy keeps the native DIP transport and changes its weights:

- An occupied-band detector examines contextual PRE-Cleanup power. It requires
  a non-negligible broadband median, searches only the upper third of the nominal
  band, and preserves all bins up to the highest supported bin. A valley with
  renewed energy above it does not become a boundary. Sparse isolated tones are
  insufficient evidence to lower the boundary. This is a receiver-edge heuristic.
- The join spans two synthesis bins above that edge. Measured coefficients below
  it remain exact; additional content may now occupy the empty tail inside the
  nominal receive bandwidth. Existing content still reduces the addition.
- The first available higher member of each packet is the primary continuation.
  Its amplitude follows the packet's local energy relative to its contextual
  mean, bounded between0.35 and1.8. This strengthens the measured envelope's
  modulation rather than applying an independent periodic oscillator.
- Higher-member amplitude is multiplied by
  `exp(-1.1*j - 0.18*j*j)`, with j=0 at that first continuation. The successive
  factors are approximately1,0.278,0.054,0.0073 before other weights. Their power
  factors are the squares of those values. The primary gain is0.18 times donor
  confidence, envelope, boundary taper and temporal erosion.
- The existing complex carrier convention, temporal holes and12% added-power
  ceiling remain. Tests cover lower-band preservation, an empty-tail join,
  interior valleys, sparse tones, faster member decay, and RFFT/ODFT phase.

This reduces the repetitive envelope pattern; the fixed750Hz packet lattice is
still present. A source/filter speech model distinguishes glottal excitation and
spectral tilt from the vocal tract's resonances and radiation. Repeating an entire
formant-shaped band does not reproduce that separation. The present exponential
is an experimental prior, not a universal law for human speech: resonances can
raise individual upper components and unvoiced excitation requires other evidence.

Relevant primary sources:

- [Doval, d'Alessandro and Henrich,2003](https://www.isca-archive.org/voqual_2003/doval03_voqual.html):
  source spectral maximum, spectral tilt, and amplitude/phase behavior.
- [Perrotin and McLoughlin](https://arxiv.org/abs/1712.08034):
  separation of vocal-tract response and source spectral tilt; vocal-effort effects.
- [Epps and Holmes,1998](https://www.isca-archive.org/icslp_1998/epps98_icslp.html):
  bandwidth extension combining sinusoidal structure, envelope mapping and
  separately treated unvoiced components.

The browser retains the previous harmonic rendering as a fifth comparison.
Cleanup's denoising decisions are unchanged in this revision. Both mask passes
now publish MAN/ATD, mean evaluated threshold, and selected fraction of the
whole contextual raster to support the next floor/peak-fitting experiment.

`tests/test_harmonics.py` checks native transport against independent finite-Zak
sums for RFFT and ODFT, invalid donors, disabled identity, lower-band preservation,
bounded added power, and soft temporal holes between peaks. These mathematical
and safety properties do not establish perceptual improvement. Listen and score
the addon separately from the guide-driven Cleanup baseline using full-band
clean references degraded to a narrower receive band.
