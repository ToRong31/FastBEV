# Hướng dẫn Build & Deploy CUDA-FastBEV lên Modal

Tài liệu này hướng dẫn chi tiết cách build artifacts (TensorRT engines, pybind11 `.so`), cập nhật code và deploy API Inference cho dự án FastBEV lên Modal. 

Kiến trúc hiện tại sử dụng **Direct Native Mode**: FastAPI gọi trực tiếp thư viện C++ `fastbev_native` (bypass hoàn toàn Triton Server) để tăng tốc độ và tránh lỗi mất mát dữ liệu do Python Garbage Collection.

---

## Yêu cầu chuẩn bị

1. **Docker & Docker Compose**: Đảm bảo Docker Desktop hoặc Docker Engine đang chạy trên máy (Hỗ trợ GPU nếu build trên máy có GPU Nvidia).
2. **Modal Account**: Bạn cần có tài khoản tại [modal.com](https://modal.com/).
3. **Modal Token**: Đã được thiết lập sẵn trong file `.env` (hoặc thông qua lệnh `modal token new`).

---

## Quy trình 1: Build lại môi trường & Deploy Code (Thường xuyên sử dụng nhất)

Quy trình này sử dụng khi bạn chỉnh sửa các file Python như `service/fastbev_service/app.py`, `modal_app.py` hoặc muốn cập nhật thay đổi cấu hình mà **không sửa** C++ source code hay cần build lại engine.

### Bước 1: Rebuild Docker Image cho Modal Deploy
Vì một số thư viện và cấu trúc được bake cứng vào image ở bước build, bạn cần build lại image để cập nhật code mới nhất từ Windows vào Docker. Mở **PowerShell** và chạy:

```powershell
docker compose build modal-deploy
```

### Bước 2: Khởi động Bash trong Container
Để chạy các lệnh `modal` với đầy đủ môi trường đã setup, truy cập vào Bash của container:

```powershell
docker compose run --rm --entrypoint bash modal-deploy
```

### Bước 3: Xác thực Modal Token (trong Container Bash)
Do bạn ghi đè entrypoint mặc định thành bash, bạn phải tự set Token từ thư mục docker secret:

```bash
export MODAL_TOKEN_ID=$(cat /run/secrets/modal_token_id)
export MODAL_TOKEN_SECRET=$(cat /run/secrets/modal_token_secret)

# Kiểm tra xem đã kết nối đúng tài khoản Modal chưa:
modal token info
```

### Bước 4: Deploy Ứng dụng lên Modal
Vẫn trong Bash của container, tiến hành deploy FastAPI app lên Modal:

```bash
modal deploy /workspace/modal_app.py
```

*Đợi khoảng 10-15 giây, Modal sẽ trả về đường link endpoint của bạn. Ví dụ: `https://<workspace>--fastbev-triton-fastapi-fastapi-app.modal.run`*

### Bước 5: Thoát container
Sau khi deploy xong, bạn có thể thoát bằng lệnh `exit`.

---

## Quy trình 2: Test API Inference

Sau khi có endpoint deploy thành công ở Bước 4, mở **PowerShell trên Windows** để gửi dữ liệu test (`example-data`) lên cloud và nhận dự đoán bounding box.

```powershell
# Chạy script test
python examples/infer_api.py `
  --url https://<workspace>--fastbev-triton-fastapi-fastapi-app.modal.run/infer `
  --data-dir example-data `
  --output modal_output.json
```
*(Lưu ý: Nhớ thay `https://<workspace>--...` bằng URL thực tế nhận được ở Bước 4).*

**Log mong đợi (Thành công):**
* Bạn sẽ nhận được `modal_output.json` chứa tọa độ 3D Boxes.
* Nếu check log trên Modal Dashboard, bạn sẽ thấy `[startup] Native direct mode — Triton bypassed` và số lượng box detect được ví dụ `boxes=(66, 8)`.

---

## Phụ lục: Build lại Artifacts (C++ / TensorRT) (Ít dùng)

Chỉ sử dụng khi bạn thay đổi C++ Source Code (`src/`), cập nhật file trọng số ONNX, hoặc thay đổi thông số TensorRT.

1. Bật container chuyên dụng để build Artifacts:
   ```powershell
   docker compose up --build artifact-builder
   ```
2. Chờ hệ thống tự động: 
   * Compile code C++ qua CMake.
   * Parse mô hình và build `fastbev_pre_trt.plan` và `fastbev_post_trt_decode.plan`.
3. Artifacts mới sẽ tự động được sync (push) lên `Volume` của Modal (`fastbev-artifacts`).
4. Sau đó thực hiện lại **Quy trình 1** để deploy ứng dụng sử dụng bộ artifact mới này.
