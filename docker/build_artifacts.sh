#!/usr/bin/env bash
set -euo pipefail

cd /workspace

model_name="${1:-${FASTBEV_MODEL_NAME:-resnet18}}"
precision="${2:-${FASTBEV_PRECISION:-fp16}}"

required_dirs=(model)
for dir in "${required_dirs[@]}"; do
  if [[ ! -d "$dir" ]]; then
    echo "[MISS] /workspace/$dir"
    echo "Mount or copy $dir before building artifacts."
    exit 1
  fi
done

find_first_dir_with_file() {
  local filename="$1"
  shift
  for dir in "$@"; do
    if [[ -f "$dir/$filename" ]]; then
      echo "$dir"
      return 0
    fi
  done
  find /usr /opt -name "$filename" -printf '%h\n' 2>/dev/null | head -n 1
}

find_first_exe_dir() {
  local exe="$1"
  local path
  path="$(command -v "$exe" || true)"
  if [[ -n "$path" ]]; then
    dirname "$path"
  fi
}

export TensorRT_Lib="${TensorRT_Lib:-$(find_first_dir_with_file libnvinfer.so /opt/tritonserver/lib /usr/lib/x86_64-linux-gnu /usr/local/tensorrt/lib)}"
export TensorRT_Inc="${TensorRT_Inc:-$(find_first_dir_with_file NvInfer.h /usr/include/x86_64-linux-gnu /usr/include /usr/local/tensorrt/include)}"
export TensorRT_Bin="${TensorRT_Bin:-$(find_first_exe_dir trtexec)}"
export CUDA_Lib="${CUDA_Lib:-$(find_first_dir_with_file libcudart.so /usr/local/cuda/lib64 /usr/lib/x86_64-linux-gnu)}"
export CUDA_Inc="${CUDA_Inc:-$(find_first_dir_with_file cuda_runtime.h /usr/local/cuda/include /usr/include)}"
export CUDA_Bin="${CUDA_Bin:-$(find_first_exe_dir nvcc)}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export CUDNN_Lib="${CUDNN_Lib:-${CUDA_Lib}}"
export DEBUG_MODEL="$model_name"
export DEBUG_PRECISION="$precision"
export DEBUG_DATA="${DEBUG_DATA:-example-data}"
export USE_Python=ON

if [[ -z "${TensorRT_Lib}" || -z "${TensorRT_Inc}" || -z "${TensorRT_Bin}" ]]; then
  echo "Could not auto-detect TensorRT lib/include/bin. Set TensorRT_Lib, TensorRT_Inc, TensorRT_Bin."
  exit 1
fi

if [[ -z "${CUDA_Lib}" || -z "${CUDA_Inc}" || -z "${CUDA_Bin}" ]]; then
  echo "Could not auto-detect CUDA lib/include/bin. Set CUDA_Lib, CUDA_Inc, CUDA_Bin."
  exit 1
fi

export PATH="${TensorRT_Bin}:${CUDA_Bin}:${PATH}"
export LD_LIBRARY_PATH="/workspace/build:${TensorRT_Lib}:${CUDA_Lib}:${CUDNN_Lib}:${LD_LIBRARY_PATH:-}"

python_include="$(python3 -c 'import sysconfig; print(sysconfig.get_path("include"))')"
if [[ ! -f "${python_include}/Python.h" ]]; then
  cat <<EOF
[MISS] ${python_include}/Python.h

The pybind extension needs Python development headers.
Rebuild the artifact-builder image after Dockerfile.artifacts installs python3-dev:
  docker compose build --no-cache artifact-builder
EOF
  exit 1
fi
export Python_Inc="${Python_Inc:-${python_include}}"
export Python_Lib="${Python_Lib:-$(python3 -c 'import sysconfig; print(sysconfig.get_config_var("LIBDIR"))')}"
export Python_Soname="${Python_Soname:-$(python3 - <<'PY'
import re
import sysconfig

library = sysconfig.get_config_var("LIBRARY") or ""
print(re.sub(r"\.a$", ".so", library))
PY
)}"

if [[ ! -d /usr/include/pybind11 && ! -d /workspace/dependencies/pybind11/include/pybind11 ]]; then
  cat <<'EOF'
[MISS] pybind11 headers

Rebuild the artifact-builder image after Dockerfile.artifacts installs pybind11-dev:
  docker compose build --no-cache artifact-builder
EOF
  exit 1
fi

if [[ -z "${CUDASM:-}" ]]; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    compute_cap="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n 1 | tr -d '.' || true)"
    export CUDASM="${compute_cap:-80}"
  else
    export CUDASM=80
  fi
fi

echo "=========================================================="
echo "MODEL:      ${DEBUG_MODEL}"
echo "PRECISION:  ${DEBUG_PRECISION}"
echo "TensorRT:   ${TensorRT_Lib}"
echo "CUDA:       ${CUDA_HOME}"
echo "CUDASM:     ${CUDASM}"
echo "=========================================================="

mkdir -p "model/${DEBUG_MODEL}/build"

bash tool/build_trt_engine.sh

mkdir -p build
cd build
rm -rf CMakeCache.txt CMakeFiles Makefile cmake_install.cmake
cmake ..
make -j"$(nproc)" fastbev_native

echo
echo "Artifacts built:"
ls -lh fastbev_native*.so libfastbev_core.so || true
ls -lh "../model/${DEBUG_MODEL}/build/"*.plan || true
