#!/usr/bin/env bash
# Entry point. Arguments are passed straight through to src/infer_seg_onnx.py:
#   ./run.sh video <video> <F0> <F1> <out.mp4> --cls SL,CW --clean
#   ./run.sh probe <video> <frame,frame,...>
#
# Why this wrapper exists: onnxruntime-gpu needs the CUDA / cuDNN shared objects, but a
# pip install scatters them under site-packages/nvidia/*/lib, which is not on the system
# library path. This script finds those directories and prepends them to
# LD_LIBRARY_PATH. It still runs if none are found -- just on CPU, about 37x slower.
#
#   PYTHON=/path/to/python ./run.sh ...        choose the interpreter
#   SEG_ONNX=/path/to/model.onnx ./run.sh ...  choose the model file
#   SEG_EXTRA_LIB=/a/lib:/b/lib ./run.sh ...   add CUDA library paths by hand
set -euo pipefail
PY="${PYTHON:-python3}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SP="$("$PY" -c 'import site; print(":".join(site.getsitepackages()+[site.getusersitepackages()]))' 2>/dev/null || true)"
LIBS=""
if [ -n "$SP" ]; then
  IFS=: read -ra DIRS <<< "$SP"
  for d in "${DIRS[@]}"; do
    [ -d "$d/nvidia" ] || continue
    while IFS= read -r l; do LIBS="$LIBS:$l"; done \
      < <(find "$d/nvidia" -maxdepth 2 -type d -name lib 2>/dev/null | sort)
  done
fi
LIBS="${LIBS#:}"
export LD_LIBRARY_PATH="${LIBS}${SEG_EXTRA_LIB:+:$SEG_EXTRA_LIB}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
exec "$PY" "$HERE/src/infer_seg_onnx.py" "$@"
