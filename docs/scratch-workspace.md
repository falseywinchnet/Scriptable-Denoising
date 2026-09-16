# Explicit sorting workspace and retired harmonics

2026-09-16: sort partition storage is now reserved once at Reload and passed
explicitly through the Filter helpers. `SORT_STACK_OFFSET` follows all other
entity storage; each entity owns 192 float64 cells (1536 bytes) for 64 pending
left/right/depth triples. Channels never share this writable workspace.

`sort_values` is a finite-value introsort: process the smaller partition first,
retain the larger partition on the supplied stack, insertion-sort small ranges,
and use an in-place scalar-workspace heapsort on depth exhaustion. `median_select`
uses that heapsort for its depth fallback. No `.sort()` calls remain in the active
Filter.py. Scratch is not created by helper calls. Existing matrix workspaces
remain views into the per-entity storage allocated by C++ at generation creation.

Validation:

- 66 finite-value cases: empty/singleton, random, constant, duplicates, sorted,
  reverse-sorted, and organ-pipe values, through 393216 entries. Exact agreement
  with sorted reference values and intact workspace boundary canaries; independent
  heapsort and median checks pass.
- The real C callback allocation audit reports zero Numba runtime allocations and
  frees in all 27 sampled calls. This does not count C++/managed allocation and is
  not a proof of every error path.
- A paired native-engine bandwidth sweep (3.4, 6, 8, 16, 24 kHz, Nyquist-clamped
  request, then 6 kHz) produced bit-for-bit identical output with the previous
  Numba sort. It also exercised squelch and the then-present harmonic flag.
- This short run did not show a speedup: callback medians were roughly 1–3% slower
  with explicit workspace. The established benefit is elimination of allocator
  traffic while retaining decisions, not a throughput improvement.
- The legacy two-pass mask comparison and all nine registered-guide tests pass.

Evidence: `evidence/scratch-workspace/`. The `native-callback-scratch.json` audit
includes the pre-retirement harmonic chain; `native-callback-retired.json` is the
current Cleanup-only chain. The archived harmonic experiment also uses caller
workspace, but is not compiled by the live Filter.

## Harmonics retirement

The current `CHAIN=(0,)` contains only Cleanup. Harmonic synthesis methods,
envelope/centroid/comfort adjustments, protected-edge heuristic, and lookup-table
construction were removed from the active file. The wrapper exposes only Reload,
Bypass, and Squelch, plus the result label. `research/harmonics/Filter-retired.py`
preserves the previous experiment. Legacy ABI config slot 3 is ignored by the
current Filter. It remains usable only by explicitly loaded archived scripts.

## Investigating the 2.5–3.5 kHz gap/ridge

The user reports that harmonics was OFF when the artifact occurred; retirement
does not explain or fix that observation. The enhanced-magnitude top panel and
final STFT bottom panel have different transforms and amplitude units. A direct
gain measurement needs pre/post spectra on the same grid and same frame centers.

Diagnostic JSON now includes `input`: magnitudes of the content STFT immediately
before Cleanup, in the same emitted frame/bin order as `filtered`. Its scratch
buffer is allocated at Reload. No additional DSP or diagnostic array allocation
occurs per callback. The viewer's two plots are unchanged. Capture with:

```sh
python3 tools/capture_band.py --port 52381 --seconds 5 --output build/band-artifact-live.json
```

The capture records same-grid input/output power and gain around 1.5–4.5 kHz,
squelch count/run evidence, and complete snapshots. A ridge already in `input`
is upstream; a selective depression in output/input is Cleanup's mask. The
large full-height dark-gray time stripe has a separate possible cause: the
viewer explicitly draws missing diagnostic sequences in gray. It must not be
treated as proof of an audio dropout without corresponding callback evidence.
