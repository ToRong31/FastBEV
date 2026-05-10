from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import modal


APP_NAME = "fastbev-artifact-builder"
VOLUME_NAME = os.environ.get("MODAL_VOLUME_NAME", "fastbev-artifacts")
MODELS_MOUNT = "/models"
DEFAULT_MODEL_NAME = os.environ.get("FASTBEV_MODEL_NAME", "resnet18")
DEFAULT_PRECISION = os.environ.get("FASTBEV_PRECISION", "fp16")
DEFAULT_GPU = os.environ.get("FASTBEV_MODAL_GPU", "A10")
DEFAULT_HARDWARE_COMPATIBILITY = os.environ.get("TRT_HARDWARE_COMPATIBILITY", "ampere+")


app = modal.App(APP_NAME)
artifacts = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

image = (
    modal.Image.from_registry("nvcr.io/nvidia/tensorrt:23.08-py3")
    .run_commands(
        "apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends "
        "bash build-essential ca-certificates cmake git libopencv-dev libprotobuf-dev "
        "protobuf-compiler pybind11-dev python3-dev && rm -rf /var/lib/apt/lists/*",
        "ln -sf $(command -v python3) /usr/local/bin/python",
    )
    .pip_install("requests>=2.31", "pillow>=10.0")
    .add_local_file("CMakeLists.txt", remote_path="/workspace/CMakeLists.txt", copy=True)
    .add_local_dir("src", remote_path="/workspace/src", copy=True)
    .add_local_dir("tool", remote_path="/workspace/tool", copy=True)
    .add_local_dir("docker", remote_path="/workspace/docker", copy=True)
    .add_local_dir("model", remote_path="/workspace/model", copy=True)
    .add_local_dir("models", remote_path="/workspace/models", copy=True)
)


def _copy_file(src: str | Path, dst: str | Path) -> None:
    src_path = Path(src)
    dst_path = Path(dst)
    if not src_path.exists():
        raise FileNotFoundError(src_path)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_path, dst_path)


def _truthy(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return value.strip().lower() in {"1", "true", "yes", "on"}


@app.function(image=image, gpu=DEFAULT_GPU, volumes={MODELS_MOUNT: artifacts}, timeout=3600)
def build_artifacts(
    model_name: str = DEFAULT_MODEL_NAME,
    precision: str = DEFAULT_PRECISION,
    hardware_compatibility: str = DEFAULT_HARDWARE_COMPATIBILITY,
    force_rebuild: bool = True,
) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "FASTBEV_MODEL_NAME": model_name,
            "FASTBEV_PRECISION": precision,
            "TRT_HARDWARE_COMPATIBILITY": hardware_compatibility,
            "TRT_FORCE_REBUILD": "1" if force_rebuild else "0",
        }
    )

    subprocess.run(["bash", "docker/build_artifacts.sh", model_name, precision], cwd="/workspace", env=env, check=True)

    model_build = Path("/workspace/model") / model_name / "build"
    volume_model_build = Path(MODELS_MOUNT) / "artifacts" / "model" / model_name / "build"
    volume_build = Path(MODELS_MOUNT) / "artifacts" / "build"
    volume_triton_repo = Path(MODELS_MOUNT) / "triton_repo" / "fastbev_pipeline"

    for artifact in Path("/workspace/build").glob("*.so"):
        _copy_file(artifact, volume_build / artifact.name)

    for filename in (
        "fastbev_pre_trt.plan",
        "fastbev_pre_trt.plan.meta",
        "fastbev_post_trt_decode.plan",
        "fastbev_post_trt_decode.plan.meta",
    ):
        _copy_file(model_build / filename, volume_model_build / filename)

    _copy_file("/workspace/models/fastbev_pipeline/config.pbtxt", volume_triton_repo / "config.pbtxt")
    _copy_file("/workspace/models/fastbev_pipeline/1/model.py", volume_triton_repo / "1" / "model.py")
    artifacts.commit()

    return {
        "model": model_name,
        "precision": precision,
        "hardware_compatibility": hardware_compatibility,
        "volume": VOLUME_NAME,
    }


@app.local_entrypoint()
def main(
    model_name: str = DEFAULT_MODEL_NAME,
    precision: str = DEFAULT_PRECISION,
    hardware_compatibility: str = DEFAULT_HARDWARE_COMPATIBILITY,
    force_rebuild: str = "1",
) -> None:
    result = build_artifacts.remote(model_name, precision, hardware_compatibility, _truthy(force_rebuild))
    print(result)
