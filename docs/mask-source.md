# Restore Cleanup’s After source routing

The live wrapper repair removed the 3 kHz notch, and the user confirmed that the
squelch closed. A subsequent listening report identified severe signal loss
while masking, including with squelch disabled. This is a separate defect.

## Missing behavior in the port

The original Downloads plugin enables `After` by default in
`Cleanup/ProcessorPanel.cs`. Its tooltip specifies that statistics other than
squelch use the processed output of preceding plugins. In
`CleanupNative/CleanupNative.h`, `process_entropy()` first reads the early stream;
when `after` is true, the magnitude field is then replaced with the later stream
before `smooth_and_mask()` and reconstruction.

The port computed both mask passes from early input, while applying that mask to
later audio. SDR# Next's mono demodulator hook precedes AGC and audio filtering.
These streams can have very different levels. The second Cleanup pass preserves
amplitudes below its peak threshold, and its exponential threshold weights also
depend on amplitude. A global scale difference is therefore not innocuous.
The guide's Fourier normalization converts transform units; it does not account
for AGC between two host hooks.

The captured failing stream had a reference-count range reaching 129/129 while
Squelch was off and emitted spectral RMS attenuation was substantial. The raw
guide and content spectra have different transform scales; their raw numeric
values must not be compared as though they were in the same units. The evidence
establishes that the mask was processing pre-AGC audio; it does not independently
attribute every audible defect to that one cause.

## Corrected routing

`MASK_SOURCE = "content"` in Filter.py restores the default After relationship:

- The fixed 512/128 reference evidence and whole-block squelch still use early input.
- The enhanced magnitude guide is built from the content history to be filtered.
  Its row evidence and both mask passes consequently share that field.
- Baseline mode computes early entropy first, then replaces mask magnitudes with
  content magnitudes, as the original After path did.
- The unchanged mask is applied to that content's complex STFT coefficients.
- Optional signed history follows the mask source for floor experiments.

This is a source-routing correction, with no output makeup gain, window change,
threshold adjustment or harmonic synthesis. The guide's local row evidence remains
part of the enhanced-space experiment, rather than a claim of bitwise equality
with the old multi-window implementation.

`MASK_SOURCE = "analysis"` explicitly retains the preceding experiment. Archived
scripts omitting this field retain analysis-source behavior. Reload rebuilds the
generation and caches. Native status reports `mask_source`. Per-channel reference
and guide reuse now independently compare their actual source histories; equal
early input cannot justify reusing a guide for different later content.

## Regression evidence

`tests/test_mask_source.py` checks actual native-engine routing:

1. A NumPy double-inverse calculation matches the published content guide with a
   different, much quieter early stream. Reference evidence still matches the
   early FFT. Distinct later channels cause two guide computations.
2. The complete default Filter receives equal content and early signals differing
   by 1000x. Its channel outputs match exactly in the tested fixture. Reference
   processing remains independent, while the identical content guide is reused.
3. With loud content and zero early input, enabled squelch drains to exact zero.
4. An invalid source declaration is rejected.

The original C++ mask oracle additionally matches baseline After behavior on six
bandwidth/squelch cases with different early/content levels and spectral shape:
maximum central-mask error 1.67e-16. Parent identity/constant-gain tests pass for
RFFT and ODFT, including geometry reload and published pre-inverse magnitudes.
The 27-call native callback audit still reports zero Numba-runtime allocations.
After deployment to the actual SDR# Next host, the user confirmed: **“Speech is
restored.”** A 15-second sample at 48 kHz stereo / 6 kHz receive bandwidth recorded
89 active blocks, 26.317 ms median, 45.864 ms peak, zero faults and zero missed
170.667 ms block deadlines. This is a measured Wine host run, not a guarantee
for arbitrary reception or a native Windows performance qualification.

Evidence is in `docs/evidence/mask-source/`. Reproduce on the configured Mini with
`m4build -- sh -c 'cmake --build build --target CleanupNative -j4 && /Users/joshuahkuttenkuler/miniforge3/bin/python tests/test_mask_source.py'`.
