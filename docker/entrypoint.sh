#!/usr/bin/env bash
set -euo pipefail

cd /workspace

export FASTBEV_MODEL_NAME="${FASTBEV_MODEL_NAME:-resnet18}"
export FASTBEV_PRECISION="${FASTBEV_PRECISION:-fp16}"
export FASTBEV_MODEL_DIR="${FASTBEV_MODEL_DIR:-/workspace/model}"
export FASTBEV_NATIVE_PYTHONPATH="${FASTBEV_NATIVE_PYTHONPATH:-/workspace/build}"
export TRITON_MODEL_NAME="${TRITON_MODEL_NAME:-fastbev_pipeline}"
export TRITON_URL="${TRITON_URL:-localhost:8000}"
export PYTHONPATH="/workspace/service:${FASTBEV_NATIVE_PYTHONPATH}:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="/workspace/build:/opt/tritonserver/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"

if [[ "${1:-serve}" != "serve" ]]; then
  exec "$@"
fi

missing=0
for file in \
  "${FASTBEV_MODEL_DIR}/${FASTBEV_MODEL_NAME}/build/fastbev_pre_trt.plan" \
  "${FASTBEV_MODEL_DIR}/${FASTBEV_MODEL_NAME}/build/fastbev_post_trt_decode.plan"; do
  if [[ ! -f "$file" ]]; then
    echo "[MISS] $file"
    missing=1
  fi
done

if ! compgen -G "${FASTBEV_NATIVE_PYTHONPATH}/fastbev_native*.so" >/dev/null; then
  echo "[MISS] ${FASTBEV_NATIVE_PYTHONPATH}/fastbev_native*.so"
  missing=1
fi

if [[ ! -f "/workspace/build/libfastbev_core.so" ]]; then
  echo "[MISS] /workspace/build/libfastbev_core.so"
  missing=1
fi

if [[ "$missing" == "1" ]]; then
  cat <<'EOF'

Artifacts are missing. Build them first:

  docker compose run --rm fastbev docker/build_artifacts.sh resnet18 fp16

If build_artifacts.sh fails, make sure these folders exist on the host:
  dependencies/
  libraries/
  model/

EOF
  exit 1
fi

tritonserver \
  --model-repository=/workspace/models \
  --http-port=8000 \
  --grpc-port=8001 \
  --metrics-port=8002 \
  --log-info=true &

triton_pid=$!
trap 'kill "$triton_pid" 2>/dev/null || true' EXIT

python3 - <<'PY'
import time
import urllib.error
import urllib.request

deadline = time.time() + 180
last_error = ""
while time.time() < deadline:
    try:
        with urllib.request.urlopen("http://127.0.0.1:8000/v2/health/ready", timeout=2) as response:
            if response.status == 200:
                print("Triton is ready.")
                raise SystemExit(0)
    except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
        last_error = str(exc)
    time.sleep(1)

raise SystemExit(f"Triton did not become ready: {last_error}")
PY

exec uvicorn fastbev_service.app:app --host 0.0.0.0 --port 8080
