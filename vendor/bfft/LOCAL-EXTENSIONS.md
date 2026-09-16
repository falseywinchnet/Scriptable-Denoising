# Cleanup-local extensions to the pinned bfft snapshot

`PROVENANCE.json` records the original upstream/local snapshot and original
file hashes. It is retained as the baseline record, not rewritten to describe
these local modifications as upstream contents.

* `src/detail/bruun_dit_kernel.hpp`: public typed `forward_lanes` entry point
  into the existing normalized templated Bruun DIT walk. The arithmetic walk
  is unchanged; Cleanup supplies SIMD coefficient types so independent ring
  responses share traversal and planning tables.
* `src/detail/complex_dit16_kernel.hpp`: a fixed 16-point complex forward codelet,
  factored as radix 4 by radix 4, supporting the same coefficient lane types.
  This avoids synthesizing a complex transform through two separate real calls.

These are internal C++ extensions. No existing bfft C ABI or public transform
behavior is changed. They are used by Cleanup's exact 48-point ring-bank
specialization. Complete-product, literal-DFT and peak/competitor comparisons
live in `research/true_superresolution/ring_transform_probe.cpp` in Cleanup.
