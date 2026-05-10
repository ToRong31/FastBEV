#!/usr/bin/env bash
set -euo pipefail

cd /workspace

export FASTBEV_MODEL_NAME="${FASTBEV_MODEL_NAME:-resnet18}"
export FASTBEV_PRECISION="${FASTBEV_PRECISION:-fp16}"
export MODAL_VOLUME_NAME="${MODAL_VOLUME_NAME:-fastbev-artifacts}"
export FASTBEV_MODAL_GPU="${FASTBEV_MODAL_GPU:-A10}"
export TRT_HARDWARE_COMPATIBILITY="${TRT_HARDWARE_COMPATIBILITY:-ampere+}"
export FASTBEV_BUILD_ARTIFACTS_ON_MODAL="${FASTBEV_BUILD_ARTIFACTS_ON_MODAL:-1}"
export FASTBEV_MODAL_ARTIFACT_FORCE_REBUILD="${FASTBEV_MODAL_ARTIFACT_FORCE_REBUILD:-1}"

if [[ -z "${MODAL_TOKEN_ID:-}" && -f /run/secrets/modal_token_id ]]; then
  export MODAL_TOKEN_ID="$(cat /run/secrets/modal_token_id)"
fi
if [[ -z "${MODAL_TOKEN_SECRET:-}" && -f /run/secrets/modal_token_secret ]]; then
  export MODAL_TOKEN_SECRET="$(cat /run/secrets/modal_token_secret)"
fi

if [[ -z "${MODAL_TOKEN_ID:-}" || -z "${MODAL_TOKEN_SECRET:-}" ]]; then
  cat <<'EOF'
Missing Modal credentials.

Set these in `.env` or pass them as environment variables:
  MODAL_TOKEN_ID=...
  MODAL_TOKEN_SECRET=...

Then run:
  docker compose --profile modal run --rm modal-deploy
EOF
  exit 1
fi

echo "Verifying Modal account..."
modal token info | grep -E '^(Name|Workspace|User|Created at):' || true
echo "Modal GPU target: ${FASTBEV_MODAL_GPU}"

model_build_dir="model/${FASTBEV_MODEL_NAME}/build"
required_files=(
  "build/libfastbev_core.so"
  "${model_build_dir}/fastbev_pre_trt.plan"
  "${model_build_dir}/fastbev_post_trt_decode.plan"
)

if ! compgen -G "build/fastbev_native*.so" >/dev/null; then
  echo "[MISS] build/fastbev_native*.so"
  exit 1
fi

for file in "${required_files[@]}"; do
  if [[ ! -f "$file" ]]; then
    echo "[MISS] $file"
    exit 1
  fi
done

echo "Creating Modal volume if needed: ${MODAL_VOLUME_NAME}"
modal volume create "${MODAL_VOLUME_NAME}" >/dev/null 2>&1 || true

if [[ "${FASTBEV_BUILD_ARTIFACTS_ON_MODAL}" == "1" || "${FASTBEV_BUILD_ARTIFACTS_ON_MODAL,,}" == "true" ]]; then
  echo "Building runtime artifacts on Modal GPU before deploy..."
  modal run modal_artifacts.py \
    --model-name "${FASTBEV_MODEL_NAME}" \
    --precision "${FASTBEV_PRECISION}" \
    --hardware-compatibility "${TRT_HARDWARE_COMPATIBILITY}" \
    --force-rebuild "${FASTBEV_MODAL_ARTIFACT_FORCE_REBUILD}"
else
  echo "Uploading native runtime libraries..."
  for file in build/*.so; do
    if [[ -f "$file" ]]; then
      modal volume put --force "${MODAL_VOLUME_NAME}" "$file" "/artifacts/build/$(basename "$file")"
    fi
  done

  echo "Uploading TensorRT engines..."
  modal volume put --force "${MODAL_VOLUME_NAME}" \
    "${model_build_dir}/fastbev_pre_trt.plan" \
    "/artifacts/model/${FASTBEV_MODEL_NAME}/build/fastbev_pre_trt.plan"
  modal volume put --force "${MODAL_VOLUME_NAME}" \
    "${model_build_dir}/fastbev_post_trt_decode.plan" \
    "/artifacts/model/${FASTBEV_MODEL_NAME}/build/fastbev_post_trt_decode.plan"

  for meta in \
    "${model_build_dir}/fastbev_pre_trt.plan.meta" \
    "${model_build_dir}/fastbev_post_trt_decode.plan.meta"; do
    if [[ -f "$meta" ]]; then
      modal volume put --force "${MODAL_VOLUME_NAME}" "$meta" "/artifacts/model/${FASTBEV_MODEL_NAME}/build/$(basename "$meta")"
    fi
  done

  echo "Uploading Triton model repository..."
  modal volume put --force "${MODAL_VOLUME_NAME}" \
    models/fastbev_pipeline/config.pbtxt \
    /triton_repo/fastbev_pipeline/config.pbtxt
  modal volume put --force "${MODAL_VOLUME_NAME}" \
    models/fastbev_pipeline/1/model.py \
    /triton_repo/fastbev_pipeline/1/model.py
fi

echo "Verifying Modal volume artifacts..."
for remote in \
  "/artifacts/build/fastbev_native.so" \
  "/artifacts/build/libfastbev_core.so" \
  "/artifacts/model/${FASTBEV_MODEL_NAME}/build/fastbev_pre_trt.plan" \
  "/artifacts/model/${FASTBEV_MODEL_NAME}/build/fastbev_post_trt_decode.plan" \
  "/triton_repo/fastbev_pipeline/config.pbtxt" \
  "/triton_repo/fastbev_pipeline/1/model.py"; do
  if ! modal volume get "${MODAL_VOLUME_NAME}" "$remote" - >/dev/null; then
    echo "[MISS] Modal volume ${MODAL_VOLUME_NAME}:${remote}"
    exit 1
  fi
done

echo "Deploying Modal app..."
modal deploy modal_app.py

cat <<'EOF'

Deploy submitted. Copy the Modal web URL printed above and call:

  python examples/infer_api.py --url https://<modal-url>/infer --data-dir example-data --output modal_output.json

EOF
