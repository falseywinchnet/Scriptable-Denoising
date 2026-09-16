# Retired harmonic experiment

Harmonic synthesis was removed from the live version on 2026-09-16 at the user's
request. `Filter-retired.py` preserves the last experiment, including the new
caller-owned sorting workspace, for reproducible historical comparisons only.
The live root Filter.py does not import, compile, or run these harmonic methods,
construct their lookup tables, infer their protected edge, or apply their
envelope/centroid/comfort adjustments. The SDR# panel has Reload, Bypass, Squelch.

Native ABI slot `config[3]` and `cleanup_set_harmonics` remain compatible with
archived scripts; the current Filter ignores that slot. Historical listening
artifacts remain unchanged. Explicit historical harmonic renders use this file.
