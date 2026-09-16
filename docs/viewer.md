# Native cross-platform filter viewer

`viewer/main.cpp` implements **CleanupViewer**, a native C++ executable using Dear
ImGui, ImPlot, GLFW, and programmable OpenGL. The viewer itself needs no Python,
DearPyGui, Direct3D, or host-specific GUI bridge. Python's reload worker only
launches the executable; rendering and polling run entirely in the native child.

The same source builds for Windows, macOS, and Linux. Native macOS and the Windows
executable under Wine have passed real rendering and Pause/Resume tests. Linux
and Windows outside Wine still require execution tests. OpenGL is deprecated on
macOS but works in this tested environment; the ImGui renderer backend can be
changed independently of the DSP and diagnostic protocol if needed later.

## Build and launch

```
cmake -S viewer -B build/viewer -DCMAKE_BUILD_TYPE=Release
cmake --build build/viewer --config Release
```

Dependencies are pinned to exact commits in `viewer/CMakeLists.txt`. A compiler,
CMake, and platform graphics development dependencies are required for building.
Third-party license notices are generated alongside the executable.

On this project use the Mini for the native build:

```
/Users/ultimussecundai/.local/bin/m4build -- sh tools/build-viewer-mini.sh
```

Retrieve the binary before the next sync. The existing local MinGW toolchain
builds Windows with `sh tools/build-viewer-cross.sh`. Staging copies
`CleanupViewer.exe` and `VIEWER-LICENSES.txt` beside the plugin DLL.

Set `VIEWER = True` in Filter.py and Reload. `VIEWER_AUTOSTART = True` launches the
native executable for that generation. `runtime/viewer.py::launch` searches the
installed plugin directory and development build directories; an explicit
`CLEANUP_VIEWER_EXECUTABLE` overrides that search. A frame must render before
startup succeeds; missing binaries and failed startup become `viewer_error`.

Manual attachment:

```
CleanupViewer --port 52381
```

A manual viewer follows reloads; an autostarted generation-specific viewer exits
when superseded. The executable reads only the loopback diagnostic socket. No
SDR# accessibility or audio-device access is required. `tools/launch-sdrsharp.sh`
now launches SDR# directly under Wine, with no native-host bridge.

## Display and pause

- Top: analysis magnitude **before filtering**.
- Bottom: final complex-coefficient magnitude **immediately before inversion**.
- Shared initial frequency range; normal ImPlot pan/zoom controls remain available.
- Strip: actual whole-block squelch decision. Existing synthesis overlap tails
  may still sound after a skip; the strip does not represent already-held samples.
- Inferno, fixed -120 to +20 dB display range, with no per-frame peak normalization.
  Different analysis transforms have different magnitude conventions; the GUI
  explicitly identifies that their raw colors are not calibrated power comparisons.
- Pause freezes images, gate, and displayed block status. Processing, polling,
  and rolling buffers continue. Resume catches up to current history.
- History is bounded by duration, 4096 columns, and 2,097,152 values per panel;
  extreme geometry can shorten the displayed history. Missing snapshots leave gaps.
- `VIEWER_GUIDE` can select a fixed state matrix published by a compiled entity.
  Otherwise the top panel is the parent's ordinary analysis STFT. The enhanced
  sub-hop field is currently shown in offline replay, not yet the live algorithm.

The original Streamcleaner approach was a persistent DearPyGui dynamic texture,
updated as a whole RGBA image and shown by a manual render loop. It displayed one
64-column segment resized to 256 by 192; its rolling 24,576-sample audio buffer
was separate. The native viewer keeps the persistent GPU texture approach and
adds bounded image history. Only the GUI thread touches OpenGL/ImGui. A polling
thread publishes a latest-only snapshot; the audio path never waits for the GUI.

## Renderer failure investigation

The Windows DearPyGui 2.3.1 renderer crashed under Wine 11.17 in the first render
call. Texture rendering and `auto_device=True` did not fix it.
`tools/probe_d3d11.py` isolates device creation without opening a GUI:

- Hardware and WARP requests for feature levels 11.0/10.0 returned `0x80004005`.
- Feature level 9.3 device creation succeeded.
- DearPyGui requests 11.0/10.0 and returns an incomplete graphics object on failure.
  Its `show_viewport` still marks the viewport shown, permitting the DX11 new-frame
  call without initialized backend state. This explains the observed null access
  consistently with the probe and upstream source; no symbol-level patch of the
  distributed DearPyGui binary is claimed.

Sources: [DearPyGui graphics initialization](https://github.com/hoffstadt/DearPyGui/blob/v2.3.1/src/mvGraphics_win32.cpp),
[viewport commands](https://github.com/hoffstadt/DearPyGui/blob/v2.3.1/src/dearpygui_commands.h).

The replacement requests a forward-compatible OpenGL core context. The forward
compatibility flag is also needed through Wine on this machine. It can fall back
to a 2.1/GLSL-120 compatibility context; failed context creation exits cleanly.
The tested Windows executable reports OpenGL 4.1 and passes in the same Wine setup.
No Wine registry or system graphics configuration was changed.

## Verification

`tools/native_viewer_smoke.py` feeds an explicitly labeled offline replay through
the real diagnostic protocol. `--viewer` selects the executable and `--wine`
executes the Windows build. The executable's `--self-test` uses the same pause
handler as the button, verifies unchanged displayed pixels while incoming
sequences advance, then verifies Resume. `--capture` saves its framebuffer to PPM.

`tools/viewer_smoke.py` exercises the actual native filter DLL with timed audio,
a reload, and further audio after reload. No audio device is opened. This harness
is not an actual SDR# integration test. The new viewer/DLL still requires staging
and an SDR# restart before claiming that installation has the update.

`runtime/viewer.py` retains the older Python offline replay prototype for research
and fixture generation. It is not the viewer launched by Reload.
