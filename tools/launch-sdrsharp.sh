#!/bin/sh
set -eu
sdrsharp=${CLEANUP_SDRSHARP:-/Users/ultimussecundai/Downloads/sdrsharp-x64-next}
cd "$sdrsharp"
# WASAPI device selection is stored in SDRSharp.config. Avoid MME on this Mac.
export WINEDEBUG=${WINEDEBUG:--all}
export MVK_CONFIG_LOG_LEVEL=${MVK_CONFIG_LOG_LEVEL:-0}
exec wine SDRSharp.exe
