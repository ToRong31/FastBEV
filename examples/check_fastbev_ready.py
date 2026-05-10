from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "service"))

from fastbev_service.tensor import load_tensor_file


CAMERAS = {
    "front": "0-FRONT.jpg",
    "front_right": "1-FRONT_RIGHT.jpg",
    "front_left": "2-FRONT_LEFT.jpg",
    "back": "3-BACK.jpg",
    "back_left": "4-BACK_LEFT.jpg",
    "back_right": "5-BACK_RIGHT.jpg",
}

GEOMETRY = {
    "valid_c_idx": ("valid_c_idx.tensor", "float32", (6, 160000)),
    "x": ("x.tensor", "int64", (6, 160000)),
    "y": ("y.tensor", "int64", (6, 160000)),
}

ENGINE_FILES = (
    "fastbev_pre_trt.plan",
    "fastbev_post_trt_decode.plan",
)

ONNX_FILES = (
    "fastbev_pre_trt.onnx",
    "fastbev_post_trt_decode.onnx",
)


def ok(message: str) -> None:
    print(f"[OK] {message}")


def fail(message: str) -> None:
    print(f"[MISS] {message}")


def check_data(data_dir: Path) -> bool:
    ready = True
    for camera, filename in CAMERAS.items():
        path = data_dir / filename
        if not path.exists():
            fail(f"{camera}: missing {path}")
            ready = False
            continue
        image = Image.open(path)
        if image.size == (1600, 900) and image.mode == "RGB":
            ok(f"{camera}: {path} {image.size[0]}x{image.size[1]} {image.mode}")
        else:
            fail(f"{camera}: expected 1600x900 RGB, got {image.size[0]}x{image.size[1]} {image.mode}")
            ready = False

    for name, (filename, dtype, shape) in GEOMETRY.items():
        path = data_dir / filename
        if not path.exists():
            fail(f"{name}: missing {path}")
            ready = False
            continue
        tensor = load_tensor_file(path)
        if str(tensor.dtype) == dtype and tuple(tensor.shape) == shape:
            ok(f"{name}: {path} {tensor.dtype} {tuple(tensor.shape)}")
        else:
            fail(f"{name}: expected {dtype} {shape}, got {tensor.dtype} {tuple(tensor.shape)}")
            ready = False
    return ready


def check_model(model_root: Path) -> bool:
    ready = True
    build_dir = model_root / "build"

    for filename in ONNX_FILES:
        path = model_root / filename
        if path.exists():
            ok(f"onnx: {path}")
        else:
            fail(f"onnx: missing {path}")
            ready = False

    for filename in ENGINE_FILES:
        path = build_dir / filename
        if path.exists():
            ok(f"engine: {path}")
        else:
            fail(f"engine: missing {path}")
            ready = False
    return ready


def check_native_build(build_dir: Path) -> bool:
    ready = True
    for pattern in ("fastbev_native*.so", "libfastbev_core.so"):
        matches = list(build_dir.glob(pattern))
        if matches:
            ok(f"native: {matches[0]}")
        else:
            fail(f"native: missing {build_dir / pattern}")
            ready = False
    return ready


def check_tools() -> bool:
    ready = True
    for tool in ("cmake", "nvcc", "trtexec"):
        path = shutil.which(tool)
        if path:
            ok(f"tool: {tool} -> {path}")
        else:
            fail(f"tool: {tool} not found in PATH")
            ready = False
    return ready


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check FastBEV sample data, model engines, native build, and tools.")
    parser.add_argument("--data-dir", default="example-data")
    parser.add_argument("--model-name", default="resnet18")
    parser.add_argument("--model-dir", default="model")
    parser.add_argument("--build-dir", default="build")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_ready = check_data(Path(args.data_dir))
    model_ready = check_model(Path(args.model_dir) / args.model_name)
    native_ready = check_native_build(Path(args.build_dir))
    tools_ready = check_tools()

    print()
    if data_ready and model_ready and native_ready:
        print("Ready to run Triton + FastAPI.")
        return 0

    print("Not ready yet.")
    if not model_ready:
        print("- Build TensorRT engines with `bash tool/build_trt_engine.sh` in a TensorRT environment.")
    if not native_ready:
        print("- Build native module with `USE_Python=ON` and `make -j fastbev_native`.")
    if not tools_ready:
        print("- Ensure cmake, nvcc, and trtexec are available in PATH inside the runtime environment.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
