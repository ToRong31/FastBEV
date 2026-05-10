from __future__ import annotations

import os
import sys
import threading
from dataclasses import dataclass
from io import BytesIO

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from PIL import Image, UnidentifiedImageError

from .tensor import TensorFormatError, TensorSpec, load_tensor, validate_tensor


CAMERA_ORDER = ("front", "front_right", "front_left", "back", "back_left", "back_right")
IMAGE_WIDTH = 1600
IMAGE_HEIGHT = 900
IMAGE_CHANNELS = 3
VALID_POINTS = 160000


@dataclass(frozen=True)
class NativeSettings:
    model_dir: str = os.environ.get("FASTBEV_MODEL_DIR", "/models/artifacts/model")
    model_name: str = os.environ.get("FASTBEV_MODEL_NAME", "resnet18")
    precision: str = os.environ.get("FASTBEV_PRECISION", "fp16")
    device_id: int = int(os.environ.get("FASTBEV_DEVICE_ID", "0"))


class NativeInferenceError(RuntimeError):
    pass


TENSOR_SPECS = {
    "valid_c_idx": TensorSpec("valid_c_idx", np.dtype("<f4"), (6, VALID_POINTS)),
    "x": TensorSpec("x", np.dtype("<i8"), (6, VALID_POINTS)),
    "y": TensorSpec("y", np.dtype("<i8"), (6, VALID_POINTS)),
}


async def _read_camera(camera_name: str, upload: UploadFile) -> np.ndarray:
    payload = await upload.read()
    try:
        image = Image.open(BytesIO(payload))
        image.load()
    except UnidentifiedImageError as exc:
        raise HTTPException(status_code=400, detail=f"{camera_name} is not a readable image") from exc

    if image.size != (IMAGE_WIDTH, IMAGE_HEIGHT):
        raise HTTPException(
            status_code=400,
            detail=f"{camera_name} must be {IMAGE_WIDTH}x{IMAGE_HEIGHT}, got {image.size[0]}x{image.size[1]}",
        )
    if image.mode != "RGB":
        raise HTTPException(status_code=400, detail=f"{camera_name} must have 3 RGB channels, got mode {image.mode}")

    array = np.asarray(image, dtype=np.uint8)
    if array.shape != (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS):
        raise HTTPException(status_code=400, detail=f"{camera_name} decoded to unexpected shape {array.shape}")
    return np.ascontiguousarray(array)


async def _read_geometry(name: str, upload: UploadFile) -> np.ndarray:
    payload = await upload.read()
    try:
        tensor = load_tensor(payload)
        return validate_tensor(tensor, TENSOR_SPECS[name])
    except (TensorFormatError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


class _NativeRuntime:
    """Lazy-initialized wrapper around fastbev_native.FastBEVRuntime.

    Thread-safe: uses a lock because the C++ runtime reuses internal buffers.
    """

    def __init__(self, settings: NativeSettings):
        self._settings = settings
        self._lock = threading.Lock()
        self._runtime = None

    def _ensure_initialized(self):
        if self._runtime is not None:
            return
        # Add native library paths
        native_paths = os.environ.get(
            "FASTBEV_NATIVE_PYTHONPATH",
            "/opt/CUDA-FastBEV/build:/models/artifacts/build",
        )
        for path in reversed([p for p in native_paths.split(os.pathsep) if p]):
            if path not in sys.path:
                sys.path.insert(0, path)

        import fastbev_native

        self._runtime = fastbev_native.FastBEVRuntime(
            model_dir=self._settings.model_dir,
            model_name=self._settings.model_name,
            precision=self._settings.precision,
            device_id=self._settings.device_id,
        )
        print(
            f"[native] FastBEVRuntime initialized: "
            f"model_dir={self._settings.model_dir} "
            f"model_name={self._settings.model_name} "
            f"precision={self._settings.precision} "
            f"device_id={self._settings.device_id}",
            flush=True,
        )

    def infer(
        self,
        images: np.ndarray,
        valid_c_idx: np.ndarray,
        valid_x: np.ndarray,
        valid_y: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        self._ensure_initialized()

        # Ensure contiguous arrays with correct dtypes
        images = np.ascontiguousarray(images, dtype=np.uint8)
        valid_c_idx = np.ascontiguousarray(valid_c_idx, dtype=np.float32)
        valid_x = np.ascontiguousarray(valid_x, dtype=np.int64)
        valid_y = np.ascontiguousarray(valid_y, dtype=np.int64)

        image_sum = int(images.astype(np.uint64).sum())
        valid_sum = float(valid_c_idx.astype(np.float64).sum())
        x_sample = valid_x.reshape(-1)[:4].tolist()
        y_sample = valid_y.reshape(-1)[:4].tolist()
        print(
            f"[native] infer input "
            f"images={images.shape} valid_c_idx={valid_c_idx.shape} "
            f"image_sum={image_sum} valid_sum={valid_sum:.6f} "
            f"x0={x_sample} y0={y_sample}",
            flush=True,
        )

        with self._lock:
            self._runtime.infer.__self__  # ensure runtime is still valid
            # Call update first (sets geometry), then forward (runs inference)
            boxes, labels = self._runtime.infer(images, valid_c_idx, valid_x, valid_y)

        boxes = np.ascontiguousarray(boxes, dtype=np.float32)
        labels = np.ascontiguousarray(labels, dtype=np.int32)
        print(
            f"[native] infer output boxes={boxes.shape} labels={labels.shape}",
            flush=True,
        )
        return boxes, labels


def _format_response(boxes: np.ndarray, labels: np.ndarray) -> dict:
    detections = []
    for box, label in zip(boxes, labels):
        detections.append(
            {
                "x": float(box[0]),
                "y": float(box[1]),
                "z": float(box[2]),
                "w": float(box[3]),
                "l": float(box[4]),
                "h": float(box[5]),
                "yaw": float(box[6]),
                "score": float(box[7]),
                "label": int(label),
            }
        )
    return {"boxes": detections, "count": len(detections)}


def create_app(settings: NativeSettings | None = None) -> FastAPI:
    native_settings = settings or NativeSettings()
    runtime = _NativeRuntime(native_settings)
    api = FastAPI(title="FastBEV Inference Service")

    @api.get("/health")
    async def health() -> dict:
        return {
            "status": "ok",
            "mode": "native_direct",
            "model_dir": native_settings.model_dir,
            "model_name": native_settings.model_name,
        }

    @api.post("/infer")
    async def infer(
        front: UploadFile = File(...),
        front_right: UploadFile = File(...),
        front_left: UploadFile = File(...),
        back: UploadFile = File(...),
        back_left: UploadFile = File(...),
        back_right: UploadFile = File(...),
        valid_c_idx: UploadFile = File(...),
        x: UploadFile = File(...),
        y: UploadFile = File(...),
    ) -> dict:
        camera_uploads = {
            "front": front,
            "front_right": front_right,
            "front_left": front_left,
            "back": back,
            "back_left": back_left,
            "back_right": back_right,
        }
        image_arrays = [await _read_camera(camera, camera_uploads[camera]) for camera in CAMERA_ORDER]
        images = np.ascontiguousarray(np.stack(image_arrays, axis=0), dtype=np.uint8)

        valid_c_idx_array = await _read_geometry("valid_c_idx", valid_c_idx)
        x_array = await _read_geometry("x", x)
        y_array = await _read_geometry("y", y)

        try:
            boxes, labels = await run_in_threadpool(
                runtime.infer,
                images,
                valid_c_idx_array,
                x_array,
                y_array,
            )
        except Exception as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return _format_response(boxes, labels)

    return api


app = create_app()
