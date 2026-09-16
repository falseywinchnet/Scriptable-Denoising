#pragma once
#include "cleanup.h"
/* Windows Cleanup.dll bootstrap. Pass NULL for filter/runtime paths to
   cleanup_create to use Filter.py and runtime/ adjacent to the DLL. It loads
   the private _python/ automatically. No Python installation or PATH edits.
   Load by absolute path; keep the DLL loaded until host process exit. Do not
   call create/reload/destroy from DllMain or from a realtime audio callback.
   If create returns NULL, this returns the startup error as UTF-8 (same buffer
   size convention as cleanup_status). Engine compile errors use status JSON.
   One compatible CPython 3.12 interpreter per process; if another plugin embeds
   Python, isolate Cleanup in a process instead of loading a second interpreter. */
#ifdef __cplusplus
extern "C" {
#endif
CLEANUP_API uint32_t cleanup_loader_error(char* destination,uint32_t capacity);
#ifdef __cplusplus
}
#endif
