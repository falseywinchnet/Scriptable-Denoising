# Cleanup: preserve the tuned separation mechanism

The user's reference is Cleanup's ability to separate signal from noise through
its carefully tuned two-pass process. Treating this as a preference for one
sound, or ranking alternatives by noise attenuation alone, misses that objective.

This audit makes one change at a time to registered Cleanup. It does **not**
promote a new filter or modify the accepted browser demo. All research switches
default off in `Filter.py`; native changes only publish diagnostic stage values.
Harmonics and comfort noise are disabled in the audit to isolate separation.

## What the two passes actually do

1. The sorted spectral-distribution score controls row admission and scales the
   contextual MAN/ATD terms. Exponential weights blend those with row statistics.
   Thus the distribution's resemblance to the reference noise shape influences
   both **whether a row is considered** and **where its threshold lies**.
2. The first peak pass selects cells on temporally smoothed magnitudes. Rejected
   cells are then zeroed in the original magnitude field.
3. Contextual statistics are recomputed from that selected field. Its magnitudes
   are temporally smoothed again; row statistics are computed from those rows.
4. The second peak pass operates **in place**. Passing cells become 1; cells that
   fail the threshold or belong to skipped rows retain their smoothed amplitude.
5. The first mask is intersected with this amplitude-valued result using
   `min(first_mask, multiplier * second_result)`. Temporal smoothing and three
   padded recursive time/frequency rounds then shape the gain surface.

Consequently, the second pass is not another independent binary rejection. For
an initially selected cell, with multiplier 1, retained second-pass amplitudes
0.2, 0.8, 1.2 and 2 produce gains 0.2, 0.8, 1 and 1. Replacing every failed cell
with zero destroys that behavior. The coherent magnitude normalization matters:
the second pass contains amplitude values and the threshold blend is exponential.

There is another important population detail: contextual `statistics` excludes
exact zeros when computing its median and median absolute deviation, but includes
all cells in its squared-deviation denominator. First-pass rejection therefore
changes both the surviving population and its occupancy. Row statistics also
retain the original first-n deviation convention. These are properties of the
existing implementation; silently replacing them with textbook variants would
constitute a different experiment.

MAN here is a median absolute deviation, not a directly measured physical noise
floor. Recovering Cleanup means preserving the relationship between distribution,
selection, recomputation and reconstruction, rather than matching a single curve.

## Controlled native-engine ablations

`research/mechanisms/audit.py` processes eight seconds of the radio fixture, then
known clean/noise pairs. Both channels receive the exact same mixture as analysis;
one filters the clean component and the other filters its noise component. This
measures their passage through the same adaptive decision. Seed 71313; 48 kHz;
physical eighth-order LPFs; synthetic measurements exclude context transitions.

### Only making the second pass binary

The first radio mask is identical. Mean in-band gain after combination changes
from 0.09476 to 0.05574. Full-gain cells fall from 9.245% to 5.574% of that domain.
Many retained amplitudes exceed unity, so this difference is not merely a few
small fractional skirts. Final mean gain falls from 0.09622 to 0.05674.

| 3 kHz voiced fixture | Registered reference | Binary second pass |
|---|---:|---:|
| Clean-component power change | −7.51 dB | −8.66 dB |
| Noise-component power change | −14.40 dB | −18.70 dB |
| Clean waveform coherence | 0.753 | 0.626 |
| Gain-compensated distortion | 0.766 | 1.556 |

The greater noise reduction conceals worse signal preservation. Continuous FM
shows the same direction: coherence falls from 0.897 to 0.771. These are fixture
measurements, not a universal auditory score or proof of every perceptual cause.

### Other isolated changes

- Removing row admission doubles radio seed coverage, from 9.57% to 20.34%.
  Weak-chirp retention improves, but substantially more noise passes. This is
  distinct from requiring pauses or classifying a persistent source as noise.
- Removing distribution-dependent scaling of the contextual statistics reduces
  radio seed coverage to 7.90%. Using row statistics alone reduces it to 7.49%.
  The contextual/distribution coupling materially changes actual radio selection.
- Freezing only the second contextual MAN/ATD produces identical aggregate radio
  stage values here and small synthetic changes. This does **not** remove the
  second pass, its changed magnitude field, or its recomputed row statistics.
  It does not establish that contextual recomputation is generally dispensable.
- Expanding the registered guide to full span while keeping the same in-band
  Cleanup decisions produces close results on these fixtures. Guide registration
  does change: old demo guide crops differ by about 6% in relative norm, with
  correlation 0.9974. It is a confound, but does not explain the much larger
  second-pass change in this audit.
- Removing the unsquelched residual floor changes many tiny nonzero gains but
  barely changes radio RMS. This test does not establish its perceptual relevance.

## Why occupancy is a different first-pass selector

Its target contour is

    T = L + sqrt(v + (mu - L)^2),  L <= mu.

The target is at least the local mean. The actual bounded time-tracked contour
can lag that target; the inequality is not a claim about every transient output.
A broad sustained feature raises its own local reference. Under a common additive
offset to L and mu, with variance fixed, the target rises by that same offset:
absolute separation above the surrounding lower population is not the criterion.
This is a local-contrast selector, unlike Cleanup's shared-population comparison.

`research/mechanisms/plateau.py` invokes the actual Filter.py functions on a
time-constant magnitude field with a broad raised region. This is a decision-rule
probe, **not a waveform or a model of a physical vocal tract**. In its interior:

| Magnitude-field perturbation | Cleanup admitted | Occupancy admitted |
|---|---:|---:|
| Flat raised region | 100% | 0% |
| Fixed ripple, standard deviation 0.005 | 100% | 0% |
| Fixed ripple, standard deviation 0.05 | 100% | 32.97% |

This exposes a concrete failure to preserve coherent raised regions. It supports
a mechanism for fragmented admission; it does not prove that this alone accounts
for the user's listening observations.

Restoring amplitude retention only in occupancy's second pass modestly improves
voiced coherence (0.730 → 0.743), but admits more noise (−9.96 → −9.38 dB), and
does not restore Cleanup's first selection. Comfort noise cannot repair those
decisions; keep that harmonic experiment separate.

## What the next algorithm experiment must preserve

Use actual Cleanup as the starting operator: row/context coupling, selected-field
statistics, the in-place second pass, calibrated magnitude units, combination,
and recursion. Change only the population supplying bandwidth-sensitive statistics
first. Test that population change with identical guide data before changing the
guide registration domain too.

An inferred spectral-support weighting is a candidate for that narrow change,
not an implemented solution. It must retain in-band valleys, exclude irrelevant
stopband observations, and leave continuous signal/FM eligible. It must not use
persistence as evidence that a component is noise. First establish an oracle
support control and invariance to appended excluded bins; then measure support
estimation error separately from separation error. Unknown physical passband is
not uniquely identifiable from arbitrary content alone.

## Evidence and reproduction

- `Filter.py`: `CLEANUP_AUDIT_BITS=0`, `CLEANUP_AUDIT_TRACE=False` preserve defaults.
- `research/mechanisms/audit.py`: ten one-change/reference native renders.
- `research/mechanisms/plateau.py`: direct decision-rule probe.
- `research/mechanisms/plot.py`: scientific figure from measured artifacts.
- `docs/evidence/cleanup-mechanism-results.json`: complete measurements and hashes.
- `docs/evidence/cleanup-mechanism-plateau.json`: controlled field measurements.
- `docs/evidence/cleanup-mechanism-legacy.log`: Downloads C++ reference parity;
  six bandwidth/squelch combinations, maximum error 3.06e-16.

Run `m4build -- sh tools/mechanism-audit-mini.sh`; then run plateau.py and plot.py
using the Mini's existing Python runtime. Detailed arrays, frozen scripts and WAVs
are in the Mini build mirror's `build/mechanism-audit/`.

Stage aggregates average over each variant's decision domain. Legacy ablations
share the same in-band domain; the two occupancy variants share the full domain.
Do not compare their raw stage coverage percentages across those two domains.
All ten native renders completed with zero reported faults. This audit does not
revalidate Windows or live SDR# integration and does not assert a listening verdict.
The targeted registered-guide complete-mask/projection parity test also passed
after these additions (`GuideTests.test_02_complete_cleanup_and_projection`).
