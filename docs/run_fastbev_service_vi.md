# Chạy FastBEV Service Và Gọi API

Tài liệu này chỉ cách chạy service đã thêm trong repo:

```text
Client multipart -> FastAPI /infer -> Triton HTTP -> fastbev_pipeline -> fastbev_native -> CUDA/TensorRT FastBEV
```

## 1. Cần Chuẩn Bị

Repo này không commit các thư mục lớn sau vì đang bị `.gitignore`:

- `model/`
- `example-data/`
- `dependencies/`
- `libraries/`
- `build/`

Muốn chạy inference thật cần có:

- CUDA + TensorRT + cuDNN đúng môi trường build.
- Source dependency theo README gốc nếu muốn build binary demo đầy đủ: `dependencies/stb`, `dependencies/pybind11`, `libraries/cuOSD`.
- Data mẫu có 6 ảnh và 3 tensor:
  - `example-data/0-FRONT.jpg`
  - `example-data/1-FRONT_RIGHT.jpg`
  - `example-data/2-FRONT_LEFT.jpg`
  - `example-data/3-BACK.jpg`
  - `example-data/4-BACK_LEFT.jpg`
  - `example-data/5-BACK_RIGHT.jpg`
  - `example-data/valid_c_idx.tensor`
  - `example-data/x.tensor`
  - `example-data/y.tensor`
- TensorRT engine:
  - `model/resnet18/build/fastbev_pre_trt.plan`
  - `model/resnet18/build/fastbev_post_trt_decode.plan`

Lưu ý: file `.plan` của TensorRT phụ thuộc CUDA/TensorRT/GPU. Build engine trong môi trường tương thích với nơi chạy service.

Kiểm tra nhanh repo đã đủ input/model chưa:

```bash
python -m pip install -r examples/requirements.txt
python examples/check_fastbev_ready.py --model-name resnet18 --data-dir example-data
```

Nếu script báo thiếu:

- `engine`: mới có ONNX, cần build TensorRT `.plan`.
- `native`: chưa build `fastbev_native.so` và `libfastbev_core.so`.
- `trtexec`: TensorRT chưa nằm trong `PATH`.

## 2. Build Native Python Module

Trên máy/container Linux có CUDA + TensorRT. Nếu đang dùng Windows, nên chạy phần này trong WSL2 hoặc container Linux có GPU passthrough.

```bash
export USE_Python=ON
. tool/environment.sh
mkdir -p build
cd build
cmake ..
make -j fastbev_native
```

Sau build cần có:

```text
build/fastbev_native.so
build/libfastbev_core.so
```

Nếu dùng Linux local, thêm runtime library path:

```bash
export LD_LIBRARY_PATH=$PWD/build:$LD_LIBRARY_PATH
export PYTHONPATH=$PWD/build:$PYTHONPATH
```

## 3. Chạy Local Bằng Triton + FastAPI

### 3.1. Tạo Triton model repository

Có thể dùng trực tiếp thư mục template trong repo:

```bash
export FASTBEV_MODEL_DIR=$PWD/model
export FASTBEV_MODEL_NAME=resnet18
export FASTBEV_PRECISION=fp16
export FASTBEV_NATIVE_PYTHONPATH=$PWD/build

tritonserver --model-repository=$PWD/models --http-port=8000 --grpc-port=8001 --metrics-port=8002
```

Triton model `fastbev_pipeline` sẽ load:

```text
$FASTBEV_MODEL_DIR/$FASTBEV_MODEL_NAME/build/fastbev_pre_trt.plan
$FASTBEV_MODEL_DIR/$FASTBEV_MODEL_NAME/build/fastbev_post_trt_decode.plan
```

### 3.2. Chạy FastAPI proxy

Mở terminal khác:

```bash
python -m pip install -r service/requirements.txt
export TRITON_URL=localhost:8000
export TRITON_MODEL_NAME=fastbev_pipeline
uvicorn fastbev_service.app:app --app-dir service --host 0.0.0.0 --port 8080
```

Kiểm tra health:

```bash
curl http://localhost:8080/health
```

Kết quả mẫu:

```json
{"status":"ok","triton_url":"localhost:8000","model":"fastbev_pipeline"}
```

## 4. Gọi API Bằng Python Client Mẫu

Cài thêm dependency cho client mẫu:

```bash
python -m pip install -r examples/requirements.txt
```

Gọi với data mặc định trong `example-data/`:

```bash
python examples/infer_api.py --url http://localhost:8080/infer --data-dir example-data
```

Ghi output ra file:

```bash
python examples/infer_api.py \
  --url http://localhost:8080/infer \
  --data-dir example-data \
  --output output.json
```

Nếu file nằm ở đường dẫn khác, truyền từng input:

```bash
python examples/infer_api.py \
  --url http://localhost:8080/infer \
  --front data/0-FRONT.jpg \
  --front-right data/1-FRONT_RIGHT.jpg \
  --front-left data/2-FRONT_LEFT.jpg \
  --back data/3-BACK.jpg \
  --back-left data/4-BACK_LEFT.jpg \
  --back-right data/5-BACK_RIGHT.jpg \
  --valid-c-idx data/valid_c_idx.tensor \
  --x data/x.tensor \
  --y data/y.tensor
```

Output JSON dạng:

```json
{
  "boxes": [
    {
      "x": 12.3,
      "y": 4.5,
      "z": -1.0,
      "w": 1.8,
      "l": 4.2,
      "h": 1.6,
      "yaw": 0.12,
      "score": 0.91,
      "label": 0
    }
  ],
  "count": 1
}
```

## 5. Gọi API Bằng curl

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

## 6. Chạy Trên Modal

### 6.1. Cấu hình Modal token

Repo không lưu Modal token/API key. Modal CLI đọc credential từ máy của bạn.

Cài Modal CLI:

```bash
python -m pip install modal
```

Cách dễ nhất:

```bash
modal setup
```

Hoặc tạo token qua browser:

```bash
modal token new
```

Nếu đã có token id/secret, set thủ công:

```bash
modal token set --token-id <MODAL_TOKEN_ID> --token-secret <MODAL_TOKEN_SECRET>
```

Modal sẽ lưu credential trong file `.modal.toml` ở home directory của user, không nằm trong repo này. Có thể kiểm tra:

```bash
modal token info
modal config show
```

Nếu chạy trong CI/server không muốn lưu file config, dùng biến môi trường:

```bash
export MODAL_TOKEN_ID=<MODAL_TOKEN_ID>
export MODAL_TOKEN_SECRET=<MODAL_TOKEN_SECRET>
```

### 6.2. Deploy service

Tạo volume:

```bash
modal volume create fastbev-artifacts
```

Upload native libs, engine và Triton repo:

```bash
modal volume put fastbev-artifacts build/fastbev_native.so /artifacts/build/fastbev_native.so
modal volume put fastbev-artifacts build/libfastbev_core.so /artifacts/build/libfastbev_core.so
modal volume put fastbev-artifacts model/resnet18/build/fastbev_pre_trt.plan /artifacts/model/resnet18/build/fastbev_pre_trt.plan
modal volume put fastbev-artifacts model/resnet18/build/fastbev_post_trt_decode.plan /artifacts/model/resnet18/build/fastbev_post_trt_decode.plan
modal volume put fastbev-artifacts models/fastbev_pipeline /triton_repo/fastbev_pipeline
```

Nếu `fastbev_native.so` còn phụ thuộc `.so` ngoài hệ thống, upload thêm vào `/artifacts/build/`.

Deploy:

```bash
modal deploy modal_app.py
```

Sau khi deploy, Modal CLI sẽ in ra URL public của ASGI app. Gọi bằng client mẫu:

```bash
python examples/infer_api.py \
  --url https://<modal-app-url>/infer \
  --data-dir example-data \
  --output modal_output.json
```

Hiện endpoint `/infer` chưa yêu cầu API key riêng. Ai có URL đều có thể gọi. Nếu cần khóa endpoint, thêm xác thực Bearer token trong FastAPI hoặc dùng cơ chế proxy auth/secret của Modal.

## 7. Lỗi Hay Gặp

- `Missing required FastBEV artifact`: thiếu `.plan` hoặc đường dẫn `FASTBEV_MODEL_DIR/FASTBEV_MODEL_NAME` sai.
- `fastbev_native*.so in FASTBEV_NATIVE_PYTHONPATH`: chưa build/upload native module.
- `Triton inference failed`: Triton chưa chạy, model load lỗi, hoặc native runtime lỗi.
- `must be 1600x900`: ảnh upload sai kích thước.
- `must be dtype float32/int64` hoặc `must have shape (6, 160000)`: file `.tensor` sai loại hoặc sai shape.

## 8. Kiểm Tra Nhanh Không Cần GPU

Chỉ kiểm tra parser `.tensor`:

```bash
python -m unittest tests.test_tensor_parser
```

Kiểm tra FastAPI import được:

```bash
PYTHONPATH=service python -c "from fastbev_service import create_app; app = create_app(); print(app.title)"
```

## 9. Chạy Bằng Docker

Docker package đã có các file:

- `Dockerfile`
- `Dockerfile.artifacts`
- `docker-compose.yml`
- `.env.example`
- `docker/entrypoint.sh`
- `docker/build_artifacts.sh`

Yêu cầu host:

- Linux hoặc WSL2 Linux.
- NVIDIA driver.
- NVIDIA Container Toolkit.
- Docker Compose v2.
- Folder `model/` và `example-data/` ở repo root.
- Nếu muốn build artifact trong container: cần thêm `dependencies/` và `libraries/`.

### 9.1. Build Docker image

Không bắt buộc phải có `.env`. `docker-compose.yml` đã có default:

```text
FASTBEV_MODEL_NAME=resnet18
FASTBEV_PRECISION=fp16
```

Nếu muốn đổi model/precision, copy file mẫu:

```bash
cp .env.example .env
```

rồi sửa `.env`. Không commit `.env` nếu có token thật.

```bash
docker compose build
```

Image mặc định không copy `model/`, `example-data/`, `build/` vào image để tránh image quá lớn. Các folder này được mount qua `docker-compose.yml`.

Dockerfile đang dùng NVIDIA NGC tag `23.08-py3` cho TensorRT/Triton vì code repo này dùng TensorRT 8 API. Không đổi lên `24.xx` nếu chưa port `src/common/tensorrt.cpp` sang TensorRT 10 API.

### 9.2. Build `.plan` và native `.so` trong container

Nếu chưa có:

```text
model/resnet18/build/fastbev_pre_trt.plan
model/resnet18/build/fastbev_post_trt_decode.plan
build/fastbev_native.so
build/libfastbev_core.so
```

thì chạy:

```bash
docker compose run --rm artifact-builder
```

Script này sẽ:

1. Dùng ONNX trong `model/resnet18/` để build TensorRT `.plan`.
2. Build `fastbev_native.so`.
3. Ghi output vào folder host `model/resnet18/build/` và `build/`.

Nếu fail vì thiếu dependency:

```text
[MISS] /workspace/dependencies
[MISS] /workspace/libraries
```

thì cần tải/copy đúng folder theo README gốc trước. Với service native `fastbev_native`, `libspconv.so` không còn bắt buộc vì source runtime FastBEV trong `src/` không gọi spconv; dependency đó là link thừa từ CMake cũ.

Nếu muốn truyền model/precision khác:

```bash
docker compose run --rm artifact-builder docker/build_artifacts.sh resnet18int8 int8
```

### 9.3. Chạy service

Sau khi artifact đã đủ:

```bash
docker compose --profile local up fastbev-local
```

Service sẽ mở:

- FastAPI: `http://localhost:8080`
- Triton HTTP: `http://localhost:8000`
- Triton gRPC: `localhost:8001`
- Triton metrics: `http://localhost:8002`

Kiểm tra:

```bash
curl http://localhost:8080/health
```

### 9.4. Gọi API từ host

```bash
python examples/infer_api.py \
  --url http://localhost:8080/infer \
  --data-dir example-data \
  --output output.json
```

### 9.5. Chạy lệnh debug trong container

```bash
docker compose --profile local run --rm fastbev-local bash
```

Check readiness trong container:

```bash
python examples/check_fastbev_ready.py --model-name resnet18 --data-dir example-data
```

### 9.6. Lưu ý về TensorRT engine

Không nên build `.plan` trên GPU khác rồi đem chạy tùy tiện. TensorRT engine phụ thuộc GPU/CUDA/TensorRT. Nếu đổi GPU hoặc đổi base image TensorRT, hãy build lại:

```bash
rm -rf model/resnet18/build
docker compose run --rm artifact-builder docker/build_artifacts.sh resnet18 fp16
```

## 10. Docker Compose Build/Up Để Deploy Lên Modal

Flow mặc định của `docker-compose.yml` là deploy Modal, không chạy API local. Container `modal-deploy` chỉ làm 3 việc:

1. Verify Modal account.
2. Upload artifacts lên Modal Volume.
3. Chạy `modal deploy modal_app.py`.

### 10.1. Chuẩn bị `.env`

Copy file mẫu:

```bash
cp .env.example .env
```

Sửa `.env`:

```env
FASTBEV_MODEL_NAME=resnet18
FASTBEV_PRECISION=fp16
MODAL_VOLUME_NAME=fastbev-artifacts
FASTBEV_MODAL_GPU=A10
MODAL_TOKEN_ID=<token-id>
MODAL_TOKEN_SECRET=<token-secret>
```

Docker Compose truyền token vào container qua `secrets`, nên `docker compose config` chỉ nên hiện tên biến môi trường, không nên hiện giá trị token. Token vẫn là secret; nếu lỡ paste token vào terminal/log/chat, hãy revoke/rotate token trên Modal rồi tạo token mới.

`FASTBEV_MODAL_GPU` phải khớp engine TensorRT. Log Docker builder của bạn đang có `CUDASM=86`, nên dùng `A10`. Không dùng `T4` nếu không rebuild engine cho sm75.

Modal token lấy bằng dashboard hoặc CLI. Nếu đã cài Modal CLI ngoài host:

```bash
modal token new
modal token info
```

Nếu không cài Modal CLI ngoài host thì vẫn được, nhưng bạn phải lấy `MODAL_TOKEN_ID` và `MODAL_TOKEN_SECRET` từ Modal dashboard rồi điền vào `.env`.

### 10.2. Build images

Lệnh này build các image cần cho Modal deploy:

```bash
docker compose build
```

### 10.3. Deploy lên Modal bằng Docker Compose

Lệnh này sẽ tự chạy theo thứ tự:

1. `artifact-builder`: build `.plan` và `.so` nếu thiếu.
2. `modal-deploy`: upload artifacts và deploy lên Modal.

```bash
docker compose up
```

Sau bước `artifact-builder` phải có:

```text
build/fastbev_native.so
build/libfastbev_core.so
model/resnet18/build/fastbev_pre_trt.plan
model/resnet18/build/fastbev_post_trt_decode.plan
```

Sau khi deploy xong, Modal CLI trong container sẽ in ra web URL. Dùng URL đó để gọi API:

```bash
python examples/infer_api.py \
  --url https://<modal-url>/infer \
  --data-dir example-data \
  --output modal_output.json
```

Lưu ý: Docker ở đây chỉ là công cụ build/deploy. Service chạy thật nằm trên Modal GPU, không chạy local.

Nếu muốn chạy lại deploy nhưng không muốn rebuild artifact, vẫn dùng:

```bash
docker compose up modal-deploy
```

Compose vẫn kiểm tra dependency `artifact-builder`; script build artifact sẽ skip `.plan` nếu đã tồn tại.
