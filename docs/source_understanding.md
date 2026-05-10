# CUDA-FastBEV — Hiểu Source Code

> Tài liệu này ghi lại toàn bộ pipeline từ FastAPI → Triton → CUDA runtime,  
> bao gồm các bug đã gặp và cách fix.

---

## 1. Kiến trúc tổng quan (Mới - Direct Native Mode)

```text
Client (HTTP multipart)
    │
    ▼
FastAPI  (service/fastbev_service/app.py)
    │  Đọc 6 ảnh JPEG + 3 tensor files
    │  Validate shape/dtype và tạo numpy array
    │  (Bypass hoàn toàn Triton Server)
    ▼
fastbev_native (pybind11, load trực tiếp từ FastAPI)
    │  core_->update(valid_c_idx, valid_x, valid_y, stream)
    │  core_->forward(image_ptrs, stream)
    ▼
fastbev::Core  (src/fastbev/fastbev.cpp)
    ├── Normalization  (src/fastbev/normalization.cu)
    │     CUDA kernel: resize + crop + normalize ảnh → fp16 planar
    ├── Backbone  (src/fastbev/fastbev_pre.cpp)
    │     TensorRT engine: fastbev_pre_trt.plan
    │     Input: [6, 3, 256, 704] fp16 → Output: camera feature maps
    ├── VTransform  (src/fastbev/vtransform.cu)
    │     CUDA kernel: project camera features → BEV volume
    │     Dùng valid_c_idx, valid_x, valid_y
    └── Fuse Head + TransBBox  (src/fastbev/fastbev_post.cpp)
          TensorRT engine: fastbev_post_trt_decode.plan
          Output: BoundingBox detections
```

---

## 2. Các file chính

### FastAPI Layer

| File | Chức năng |
|------|-----------|
| `service/fastbev_service/app.py` | FastAPI app, endpoint `/health` và `/infer` |
| `service/fastbev_service/tensor.py` | Parser cho custom `.tensor` binary format |

**`app.py` flow (Mới):**
1. Đọc 6 UploadFile ảnh JPEG → validate kích thước (1600×900, RGB) → stack thành `(6, 900, 1600, 3) uint8`
2. Đọc 3 file `.tensor`: `valid_c_idx (float32, 6×160000)`, `x (int64, 6×160000)`, `y (int64, 6×160000)`
3. Khởi tạo `fastbev_native.FastBEVRuntime` (Lazy-loaded, có thread lock).
4. Gọi trực tiếp `runtime.infer(images, valid_c_idx, x, y)` xuống C++ Runtime → nhận về `(boxes, labels)`.
5. Format thành JSON và trả về cho client.

**`.tensor` format** (custom binary, `tensor.py`):
```
[4B magic=0x33FF1101][4B ndim][4B dtype_code]
[ndim × 4B dims]
[raw payload bytes]
```

### Triton Python Backend

| File | Chức năng |
|------|-----------|
| `models/fastbev_pipeline/config.pbtxt` | Khai báo input/output shape, dtype cho Triton |
| `models/fastbev_pipeline/1/model.py` | Backend gọi `fastbev_native` |

**config.pbtxt inputs:**
```
IMAGES     : UINT8  [6, 900, 1600, 3]
VALID_C_IDX: FP32   [6, 160000]
VALID_X    : INT64  [6, 160000]
VALID_Y    : INT64  [6, 160000]
```

### CUDA/TensorRT Runtime

| File | Chức năng |
|------|-----------|
| `src/python.cpp` | pybind11 wrapper: `FastBEVRuntime` class |
| `src/fastbev/fastbev.cpp` | `Core::forward()` orchestration |
| `src/fastbev/normalization.cu` | CUDA kernel: resize + normalize ảnh → fp16 planar |
| `src/fastbev/vtransform.cu` | CUDA kernel: view transform (camera → BEV) |
| `src/fastbev/fastbev_pre.cpp` | TensorRT backbone inference |
| `src/fastbev/fastbev_post.cpp` | TensorRT post-process + decode bbox |
| `src/fastbev/postprecess.cpp` | NMS, score threshold |

**Normalization** (per camera):
```
input: (1600×900) RGB uint8
resize_lim = 0.44 → resized = (704, 396)
crop center → (704, 256)
normalize: (pixel - mean) / std
output: fp16 planar [3, 256, 704]
```
Mean: `[123.675, 116.28, 103.53]`, Std: `[58.395, 57.12, 57.375]`

**VTransform** (BEV pooling):
```
valid_c_idx[cam, point] == 1.0 → active point
Lấy feature từ camera_feature[cam, :, y, x]
Đặt vào output_feature[:, point]
Output shape: [1, C, 200, 200, 4] (BEV volume)
```

### Modal Deployment

| File | Chức năng |
|------|-----------|
| `modal_app.py` | Deploy FastAPI + Triton server trên Modal GPU |
| `modal_artifacts.py` | Upload artifacts lên Modal Volume |
| `Dockerfile.modal-deploy` | Docker image cho Modal |

**modal_app.py flow:**
1. Tạo Modal image từ `nvcr.io/nvidia/tritonserver:23.08-py3`
2. `@modal.asgi_app`: khi cold-start → `_start_triton()` → `create_app()`
3. Triton chạy ở localhost:8001 (gRPC) trong cùng container

---

## 3. Bug đã gặp và fix

### Bug: Triton gRPC buffer corruption (data mismatch)

**Triệu chứng** (từ log deploy thực tế):
```
fastapi triton input  image_sum=2782947298  valid_sum=174011.000000  x0=[230, 231, 231, 231]  y0=[30, 34, 38, 43]
fastbev_pipeline input  image_sum=3317760000  valid_sum=0.000000     x0=[230, 230, 230, 230]  y0=[30, 30, 30, 30]
fastbev_pipeline output boxes=(0, 8) labels=(0,)   ← không detect được gì
```

Dữ liệu gửi vào FastAPI đúng nhưng Triton nhận được:
- `valid_sum` 174011 → **0** (toàn bộ valid_c_idx bị zero)
- `image_sum` tăng lên 3,317,760,000 = 6×900×1600×3×255 (ảnh bị ghi đè bởi 0xFF)
- `x0/y0` bị "flatten" (mất sự biến đổi giữa các camera)

**Root Cause** — `models/fastbev_pipeline/1/model.py`, hàm `_input_as_contiguous`:

```python
# ❌ CODE CŨ - BUG
raw_buffer = ctypes.c_ubyte_array.from_address(int(view.ctypes.data))
array = np.frombuffer(raw_buffer, dtype=dtype, ...).reshape(shape).copy()
```

Khi Triton dùng **gRPC + `FORCE_CPU_ONLY_INPUT_TENSORS=yes`**:
- `view = pb_utils.get_input_tensor_by_name(...).as_numpy()` → view tạm thời vào Triton's internal buffer
- `ctypes.from_address()` lưu **raw pointer** đến vùng nhớ đó
- Triton **reclaim/overwrite** buffer đó sau khi `as_numpy()` return
- → ctypes đọc vào memory đã bị xóa/ghi đè → **silent data corruption**

**Fix** — `models/fastbev_pipeline/1/model.py`:

```python
# ✅ CODE MỚI - ĐÃ FIX
def _input_as_contiguous(request, name, dtype, shape):
    view = pb_utils.get_input_tensor_by_name(request, name).as_numpy()
    if tuple(view.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}, got {tuple(view.shape)}")

    dtype = np.dtype(dtype)
    # Gọi .copy() NGAY LẬP TỨC để sở hữu data trước khi Triton thu hồi buffer
    owned = view.copy()
    if owned.dtype != dtype:
        owned = owned.view(dtype).reshape(shape)
    array = np.ascontiguousarray(owned)
    return array, tuple(int(stride) for stride in array.strides)
```

**Nguyên tắc**: Khi làm việc với `pb_utils.get_input_tensor_by_name(...).as_numpy()` trong Triton Python backend, **luôn `.copy()` ngay lập tức** — không giữ view, không dùng `ctypes.from_address()` vào buffer đó.

---

## 4. Data flow chi tiết (valid_c_idx)

```
example-data/valid_c_idx.tensor
    │  .tensor format (magic + header + raw float32)
    ▼
tensor.py::load_tensor()         → np.ndarray float32 (6, 160000)
    ▼
app.py::_read_geometry()         → validate dtype=float32, shape=(6,160000)
    ▼
_call_triton() → InferInput VALID_C_IDX FP32 (6,160000)  [gRPC]
    ▼
model.py::_input_as_contiguous() → view.copy() → np.ascontiguousarray
    ▼
FastBEVRuntime.infer(valid_c_idx=...) [pybind11]
    ▼
core_->update(valid_c_idx_ptr, valid_x_ptr, valid_y_ptr, stream)
    ▼
vtransform_->update()  →  cudaMemcpyAsync(valid_index_device_, ...)
    ▼
compute_volum_kernel<<<...>>>  (valid_index[cam,pt] == 1.0 → copy feature)
```

---

## 5. Camera order

FastAPI nhận fields: `front, front_right, front_left, back, back_left, back_right`  
Stack theo thứ tự: `CAMERA_ORDER = ("front", "front_right", "front_left", "back", "back_left", "back_right")`  
→ Triton nhận `IMAGES[0]` = front, `IMAGES[1]` = front_right, v.v.

**Quan trọng**: `valid_c_idx`, `valid_x`, `valid_y` phải được tạo ra với cùng thứ tự camera này.

---

## 6. Cần chú ý khi deploy

1. **TensorRT `.plan` files** không portable — phải build trên cùng GPU/CUDA/TRT version với môi trường runtime
2. **`FORCE_CPU_ONLY_INPUT_TENSORS=yes`** trong config.pbtxt là cần thiết vì backend Python không nhận CUDA tensors trực tiếp
3. **Cold start** trên Modal mất ~11s (load Triton + TRT engines) — được cache khi `scaledown_window=300`
4. **`@modal.concurrent(max_inputs=8)`** nhưng `FastBEVRuntime` có `std::mutex` → serialized inference
5. **Fallback**: nếu Volume thiếu triton_repo, dùng baked-in `/opt/fastbev_triton_repo` từ image
