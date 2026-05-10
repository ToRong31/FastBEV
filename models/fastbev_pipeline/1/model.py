import json
import os
import sys
from typing import Any, Dict

import numpy as np
import triton_python_backend_utils as pb_utils


def _config_parameter(model_config: Dict[str, Any], name: str, default: str) -> str:
    value = model_config.get("parameters", {}).get(name, {})
    return value.get("string_value", default)


def _prepend_native_paths() -> None:
    search_path = os.environ.get(
        "FASTBEV_NATIVE_PYTHONPATH",
        "/opt/CUDA-FastBEV/build:/models/artifacts/build",
    )
    for path in reversed([p for p in search_path.split(os.pathsep) if p]):
        if path not in sys.path:
            sys.path.insert(0, path)


def _input_as_contiguous(request, name: str, dtype: np.dtype, shape: tuple[int, ...]) -> tuple[np.ndarray, tuple[int, ...]]:
    # CRITICAL FIX: Store the Tensor object in a variable FIRST.
    # Writing get_input_tensor_by_name(...).as_numpy() as a chain means the
    # Tensor is a temporary (refcount=1) that Python's GC frees IMMEDIATELY
    # after .as_numpy() returns — releasing the underlying memory buffer.
    # Any subsequent .copy() then reads freed memory → zeros or 0xFF garbage.
    # Storing in `tensor` keeps the buffer alive until after np.ascontiguousarray().
    tensor = pb_utils.get_input_tensor_by_name(request, name)
    view = tensor.as_numpy()
    if tuple(view.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}, got {tuple(view.shape)}")
    dtype = np.dtype(dtype)
    # Copy while `tensor` is still alive and its buffer is valid.
    array = np.ascontiguousarray(view, dtype=dtype)
    del tensor  # Explicitly release after copy is done.
    return array, tuple(int(stride) for stride in array.strides)


class TritonPythonModel:
    def initialize(self, args):
        _prepend_native_paths()
        import fastbev_native

        model_config = json.loads(args["model_config"])
        model_dir = os.environ.get(
            "FASTBEV_MODEL_DIR",
            _config_parameter(model_config, "model_dir", "/models/artifacts/model"),
        )
        model_name = os.environ.get(
            "FASTBEV_MODEL_NAME",
            _config_parameter(model_config, "model_name", "resnet18"),
        )
        precision = os.environ.get(
            "FASTBEV_PRECISION",
            _config_parameter(model_config, "precision", "fp16"),
        )
        device_id = int(args.get("model_instance_device_id", "0"))
        if device_id < 0:
            device_id = int(os.environ.get("FASTBEV_DEVICE_ID", "0"))

        self.runtime = fastbev_native.FastBEVRuntime(
            model_dir=model_dir,
            model_name=model_name,
            precision=precision,
            device_id=device_id,
        )

    def execute(self, requests):
        responses = []
        for request in requests:
            try:
                images, images_strides = _input_as_contiguous(request, "IMAGES", np.uint8, (6, 900, 1600, 3))
                valid_c_idx, valid_c_idx_strides = _input_as_contiguous(
                    request, "VALID_C_IDX", np.float32, (6, 160000)
                )
                valid_x, valid_x_strides = _input_as_contiguous(request, "VALID_X", np.int64, (6, 160000))
                valid_y, valid_y_strides = _input_as_contiguous(request, "VALID_Y", np.int64, (6, 160000))

                image_sum = int(images.astype(np.uint64).sum())
                valid_sum = float(valid_c_idx.astype(np.float64).sum())
                x_sample = valid_x.reshape(-1)[:4].tolist()
                y_sample = valid_y.reshape(-1)[:4].tolist()
                print(
                    "fastbev_pipeline input "
                    f"images={images.shape} valid_c_idx={valid_c_idx.shape} "
                    f"image_sum={image_sum} valid_sum={valid_sum:.6f} "
                    f"x0={x_sample} y0={y_sample} "
                    f"strides images={images_strides} valid={valid_c_idx_strides} "
                    f"x={valid_x_strides} y={valid_y_strides}",
                    flush=True,
                )

                boxes, labels = self.runtime.infer(
                    images,
                    valid_c_idx,
                    valid_x,
                    valid_y,
                )
                print(
                    "fastbev_pipeline output "
                    f"boxes={getattr(boxes, 'shape', None)} labels={getattr(labels, 'shape', None)}",
                    flush=True,
                )

                output_tensors = [
                    pb_utils.Tensor("BOXES", np.ascontiguousarray(boxes, dtype=np.float32)),
                    pb_utils.Tensor("LABELS", np.ascontiguousarray(labels, dtype=np.int32)),
                ]
                responses.append(pb_utils.InferenceResponse(output_tensors=output_tensors))
            except Exception as exc:
                responses.append(pb_utils.InferenceResponse(error=pb_utils.TritonError(str(exc))))
        return responses

    def finalize(self):
        self.runtime = None
