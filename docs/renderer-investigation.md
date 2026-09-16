# Renderer evidence

See [viewer.md](viewer.md) for the diagnosis and native replacement.
The device probe is reproducible with:

```
wine .build/python-win/tools/python.exe tools/probe_d3d11.py
```

`docs/evidence/d3d11-probe.json` contains hardware/WARP results at requested
feature levels. No graphics or registry settings were modified. The replacement
Windows executable uses OpenGL and runs directly under Wine; the former host-side
Python bridge is no longer used by Reload or the SDR# launcher.
