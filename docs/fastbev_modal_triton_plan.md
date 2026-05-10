# FastBEV Modal + Triton Inference Service

This service wraps the existing FastBEV CUDA/TensorRT runtime as:

```
FastAPI multipart upload -> Triton HTTP -> fastbev_pipeline Python backend -> fastbev_native -> fastbev::Core
```

The native runtime loads TensorRT engines once when the Triton Python backend initializes. Each runtime instance owns a CUDA stream and serializes `infer()` calls because the current `fastbev::Core` implementation reuses resident buffers.

## Runtime Contract

FastAPI `POST /infer` expects multipart fields:

- Image files: `front`, `front_right`, `front_left`, `back`, `back_left`, `back_right`
- Geometry tensor files: `valid_c_idx`, `x`, `y`

Image requirements:

- RGB
- `1600x900`
- Uploaded in camera-name fields, then stacked as `[front, front_right, front_left, back, back_left, back_right]`

Tensor requirements:

- `.tensor` format written by `ptq/lean/tensor.py`
- `valid_c_idx`: `float32`, shape `[6, 160000]`
- `x`: `int64`, shape `[6, 160000]`
- `y`: `int64`, shape `[6, 160000]`

Triton model inputs:

- `IMAGES`: `UINT8`, shape `[6, 900, 1600, 3]`
- `VALID_C_IDX`: `FP32`, shape `[6, 160000]`
- `VALID_X`: `INT64`, shape `[6, 160000]`
- `VALID_Y`: `INT64`, shape `[6, 160000]`

Triton model outputs:

- `BOXES`: `FP32`, shape `[-1, 8]`, `[x, y, z, w, l, h, yaw, score]`
- `LABELS`: `INT32`, shape `[-1]`

FastAPI response:

```json
{
  "boxes": [
    {
      "x": 0.0,
      "y": 0.0,
      "z": 0.0,
      "w": 0.0,
      "l": 0.0,
      "h": 0.0,
      "yaw": 0.0,
      "score": 0.0,
      "label": 0
    }
  ],
  "count": 1
}
```

## Files Added

- `src/python.cpp`: pybind11 module exposing `fastbev_native.FastBEVRuntime`
- `models/fastbev_pipeline/config.pbtxt`: Triton Python backend config
- `models/fastbev_pipeline/1/model.py`: Triton backend that calls `fastbev_native`
- `service/fastbev_service/`: FastAPI app and `.tensor` parser
- `modal_app.py`: Modal GPU ASGI deployment that starts Triton in-container

## Native Build

The checkout still requires the upstream runtime dependencies described in `README.md` for the full demo binary. The service build uses TensorRT, CUDA, cuDNN, protobuf, OpenCV headers, and pybind11 headers. `libspconv.so` is not linked by the service build because the runtime source under `src/` does not call spconv.

Build the Python module in an environment compatible with the GPU and TensorRT runtime that will serve it:

```bash
export USE_Python=ON
. tool/environment.sh
mkdir -p build
cd build
cmake ..
make -j fastbev_native
```

The build must produce at least:

```text
build/fastbev_native.so
build/libfastbev_core.so
```

Also keep any non-system shared libraries needed at runtime, especially the spconv library used by this repo.

## Required Modal Volume Layout

Create the volume:

```bash
modal volume create fastbev-artifacts
```

Upload artifacts so the mounted volume looks like this inside Modal:

```text
/models
|-- artifacts
|   |-- build
|   |   |-- fastbev_native.so
|   |   |-- libfastbev_core.so
|   |   `-- <other required runtime .so files>
|   `-- model
|       `-- resnet18
|           `-- build
|               |-- fastbev_pre_trt.plan
|               `-- fastbev_post_trt_decode.plan
`-- triton_repo
    `-- fastbev_pipeline
        |-- config.pbtxt
        `-- 1
            `-- model.py
```

Example upload commands:

```bash
modal volume put fastbev-artifacts build/fastbev_native.so /artifacts/build/fastbev_native.so
modal volume put fastbev-artifacts build/libfastbev_core.so /artifacts/build/libfastbev_core.so
modal volume put fastbev-artifacts model/resnet18/build/fastbev_pre_trt.plan /artifacts/model/resnet18/build/fastbev_pre_trt.plan
modal volume put fastbev-artifacts model/resnet18/build/fastbev_post_trt_decode.plan /artifacts/model/resnet18/build/fastbev_post_trt_decode.plan
modal volume put fastbev-artifacts models/fastbev_pipeline /triton_repo/fastbev_pipeline
```

TensorRT `.plan` files are not portable across arbitrary CUDA/TensorRT/GPU targets. Build them in a container stack compatible with `nvcr.io/nvidia/tritonserver:23.08-py3` and the Modal GPU target. The deployment defaults to `gpu="L40S"`. Use A10 only if the engines and native libraries are rebuilt and validated for A10.

## Local FastAPI Run

Start Triton separately, then run the proxy:

```bash
pip install -r service/requirements.txt
set TRITON_URL=localhost:8000
uvicorn fastbev_service.app:app --app-dir service --host 0.0.0.0 --port 8080
```

Example request:

```bash
curl -X POST http://localhost:8080/infer \
  -F front=@example-data/0-FRONT.jpg \
  -F front_right=@example-data/1-FRONT_RIGHT.jpg \
  -F front_left=@example-data/2-FRONT_LEFT.jpg \
  -F back=@example-data/3-BACK.jpg \
  -F back_left=@example-data/4-BACK_LEFT.jpg \
  -F back_right=@example-data/5-BACK_RIGHT.jpg \
  -F valid_c_idx=@example-data/valid_c_idx.tensor \
  -F x=@example-data/x.tensor \
  -F y=@example-data/y.tensor
```

## Modal Deploy

Install the Modal CLI locally, populate `fastbev-artifacts`, then deploy:

```bash
modal deploy modal_app.py
```

`modal_app.py` mounts the volume at `/models`, starts:

```bash
tritonserver --model-repository=/models/triton_repo --http-port=8000
```

and serves the FastAPI app via `@modal.asgi_app`.

If `/models/triton_repo/fastbev_pipeline` is absent, the deployment falls back to the repo template baked into the Modal image at `/opt/fastbev_triton_repo`. The TensorRT engines and native `.so` files must still exist in `/models/artifacts`.

## Verification Targets

- Unit-test `.tensor` parsing against `ptq/lean/tensor.py`
- Native smoke test: call `fastbev_native.FastBEVRuntime(...).infer(...)` on sample inputs
- Triton smoke test: call `fastbev_pipeline` directly and validate output shapes
- FastAPI smoke test: multipart `/infer` returns JSON boxes
- Modal smoke test: `modal deploy`, then call the public endpoint
- Failure tests: missing camera, wrong image dimensions, wrong tensor dtype/shape, missing Triton, missing artifacts
