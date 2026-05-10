from __future__ import annotations

import glob
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import modal


APP_NAME = "fastbev-triton-fastapi"
VOLUME_NAME = os.environ.get("MODAL_VOLUME_NAME", "fastbev-artifacts")
MODELS_MOUNT = "/models"
DEFAULT_TRITON_REPO = f"{MODELS_MOUNT}/triton_repo"
FALLBACK_TRITON_REPO = "/opt/fastbev_triton_repo"
DEFAULT_MODEL_NAME = os.environ.get("FASTBEV_MODEL_NAME", "resnet18")
DEFAULT_PRECISION = os.environ.get("FASTBEV_PRECISION", "fp16")
DEFAULT_GPU = os.environ.get("FASTBEV_MODAL_GPU", "A10")


app = modal.App(APP_NAME)
artifacts = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

image = (
    modal.Image.from_registry("nvcr.io/nvidia/tritonserver:23.08-py3")
    .run_commands(
        "apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends libopencv-dev && rm -rf /var/lib/apt/lists/*",
        "ln -sf $(command -v python3) /usr/local/bin/python",
        "python -m pip --version",
    )
    .pip_install(
        "fastapi[standard]>=0.115",
        "numpy>=1.26",
        "pillow>=10.0",
        "python-multipart>=0.0.9",
        "tritonclient[grpc,http]==2.37.0.9383150",
    )
    .env(
        {
            "TRITON_URL": "localhost:8001",
            "TRITON_PROTOCOL": "grpc",
            "TRITON_MODEL_NAME": "fastbev_pipeline",
            "FASTBEV_MODEL_DIR": f"{MODELS_MOUNT}/artifacts/model",
            "FASTBEV_MODEL_NAME": DEFAULT_MODEL_NAME,
            "FASTBEV_PRECISION": DEFAULT_PRECISION,
            "FASTBEV_DEVICE_ID": "0",
            "FASTBEV_NATIVE_PYTHONPATH": f"{MODELS_MOUNT}/artifacts/build:/opt/CUDA-FastBEV/build",
            "LD_LIBRARY_PATH": f"{MODELS_MOUNT}/artifacts/build:/opt/CUDA-FastBEV/build:/usr/local/cuda/lib64:/opt/tritonserver/lib",
        }
    )
    .add_local_dir("service/fastbev_service", remote_path="/root/fastbev_service")
    .add_local_dir("models", remote_path=FALLBACK_TRITON_REPO)
)

_triton_process: subprocess.Popen | None = None


def _model_repo_path() -> str:
    if Path(DEFAULT_TRITON_REPO, "fastbev_pipeline", "config.pbtxt").exists():
        return DEFAULT_TRITON_REPO
    return FALLBACK_TRITON_REPO


def _require_runtime_artifacts() -> None:
    model_name = os.environ.get("FASTBEV_MODEL_NAME", DEFAULT_MODEL_NAME)
    model_root = Path(os.environ.get("FASTBEV_MODEL_DIR", f"{MODELS_MOUNT}/artifacts/model")) / model_name
    required = [
        model_root / "build" / "fastbev_pre_trt.plan",
        model_root / "build" / "fastbev_post_trt_decode.plan",
    ]

    missing = [str(path) for path in required if not path.exists()]
    native_paths = os.environ.get("FASTBEV_NATIVE_PYTHONPATH", "").split(os.pathsep)
    native_matches = []
    for path in native_paths:
        native_matches.extend(glob.glob(str(Path(path) / "fastbev_native*.so")))
    if not native_matches:
        missing.append("fastbev_native*.so in FASTBEV_NATIVE_PYTHONPATH")

    if missing:
        raise RuntimeError(
            "FastBEV artifacts are missing from the Modal volume or image: " + ", ".join(missing)
        )


def _sync_model_py() -> None:
    """Always overwrite volume model.py with the image's baked-in version.

    This prevents stale __pycache__ or old volume files from overriding fixes.
    The image version (FALLBACK_TRITON_REPO) is built fresh on every `modal deploy`.
    """
    import shutil

    src = Path(FALLBACK_TRITON_REPO, "fastbev_pipeline", "1", "model.py")
    dst = Path(DEFAULT_TRITON_REPO, "fastbev_pipeline", "1", "model.py")
    pycache = dst.parent / "__pycache__"

    if src.exists() and dst.exists():
        # Remove stale bytecode cache first
        if pycache.exists():
            shutil.rmtree(pycache, ignore_errors=True)
            print(f"[startup] removed {pycache}", flush=True)
        shutil.copy2(src, dst)
        print(f"[startup] synced model.py from image → volume", flush=True)


def _start_triton(timeout_s: int = 120) -> None:
    global _triton_process
    if _triton_process is not None and _triton_process.poll() is None:
        return

    _require_runtime_artifacts()
    library_paths = [
        f"{MODELS_MOUNT}/artifacts/build",
        "/opt/CUDA-FastBEV/build",
        "/usr/local/cuda/lib64",
        "/opt/tritonserver/lib",
    ]
    current_ld_library_path = os.environ.get("LD_LIBRARY_PATH", "")
    os.environ["LD_LIBRARY_PATH"] = os.pathsep.join(
        [*library_paths, *[p for p in current_ld_library_path.split(os.pathsep) if p]]
    )
    model_repo = _model_repo_path()
    command = [
        "tritonserver",
        f"--model-repository={model_repo}",
        "--http-port=8000",
        "--grpc-port=8001",
        "--metrics-port=8002",
        "--log-info=true",
    ]
    _triton_process = subprocess.Popen(command)

    ready_url = "http://127.0.0.1:8000/v2/health/ready"
    deadline = time.monotonic() + timeout_s
    last_error = ""
    while time.monotonic() < deadline:
        if _triton_process.poll() is not None:
            raise RuntimeError(f"tritonserver exited with code {_triton_process.returncode}")
        try:
            with urllib.request.urlopen(ready_url, timeout=2) as response:
                if response.status == 200:
                    return
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            last_error = str(exc)
        time.sleep(1)

    raise RuntimeError(f"Triton did not become ready within {timeout_s}s: {last_error}")


@app.function(
    image=image,
    gpu=DEFAULT_GPU,
    volumes={MODELS_MOUNT: artifacts},
    timeout=1800,
    scaledown_window=300,
)
@modal.concurrent(max_inputs=8)
@modal.asgi_app()
def fastapi_app():
    # Set up library paths for fastbev_native (CUDA/TensorRT .so files)
    library_paths = [
        f"{MODELS_MOUNT}/artifacts/build",
        "/opt/CUDA-FastBEV/build",
        "/usr/local/cuda/lib64",
        "/opt/tritonserver/lib",
    ]
    current_ld = os.environ.get("LD_LIBRARY_PATH", "")
    os.environ["LD_LIBRARY_PATH"] = os.pathsep.join(
        [*library_paths, *[p for p in current_ld.split(os.pathsep) if p]]
    )

    # Add native Python module paths
    native_paths = os.environ.get("FASTBEV_NATIVE_PYTHONPATH", "").split(os.pathsep)
    for path in reversed([p for p in native_paths if p]):
        if path not in sys.path:
            sys.path.insert(0, path)

    if "/root" not in sys.path:
        sys.path.insert(0, "/root")

    print("[startup] Native direct mode — Triton bypassed", flush=True)
    from fastbev_service import create_app

    return create_app()
