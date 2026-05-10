#!/usr/bin/env bash
set -euo pipefail

echo "Python executable: $(command -v python3)"
python3 - <<'PY'
import sys
import sysconfig

print("Python version:", sys.version.replace("\n", " "))
print("Python include:", sysconfig.get_path("include"))
print("Python libdir:", sysconfig.get_config_var("LIBDIR"))
print("Python library:", sysconfig.get_config_var("LIBRARY"))
PY

python_include="$(python3 -c 'import sysconfig; print(sysconfig.get_path("include"))')"
if [[ -f "${python_include}/Python.h" ]]; then
  echo "[OK] ${python_include}/Python.h"
else
  echo "[MISS] ${python_include}/Python.h"
fi

if [[ -d /usr/include/pybind11 ]]; then
  echo "[OK] /usr/include/pybind11"
else
  echo "[MISS] /usr/include/pybind11"
fi

if command -v trtexec >/dev/null 2>&1; then
  trtexec --help 2>&1 | grep -m 1 'TensorRT' || true
else
  echo "[MISS] trtexec"
fi

if command -v cmake >/dev/null 2>&1; then
  cmake --version | head -n 1
else
  echo "[MISS] cmake"
fi

if command -v nvcc >/dev/null 2>&1; then
  nvcc --version | tail -n 1
else
  echo "[MISS] nvcc"
fi
