---
title: Traffic Sign Detection
emoji: 🚦
colorFrom: orange
colorTo: yellow
sdk: streamlit
sdk_version: 1.41.0
app_file: app.py
pinned: false
license: mit
---

# 🚦 Traffic Sign Detection - Streamlit App

Ứng dụng demo **nhận diện biển báo giao thông Việt Nam** (dữ liệu Zalo AI Challenge), hỗ trợ 3 lựa chọn kiến trúc model: **YOLOv8**, **Faster R-CNN**, **RT-DETR**.

> ⚠️ **Lưu ý quan trọng:** Các model thật (huấn luyện trên 7 lớp biển báo) vẫn đang trong quá trình training. Vì vậy, hiện tại **bất kể bạn chọn model nào ở Sidebar**, app sẽ tạm dùng model `yolov8n.pt` mặc định của `ultralytics` (huấn luyện trên bộ COCO 80 lớp) để chạy thử toàn bộ luồng xử lý (upload → suy luận → vẽ bounding box). Xem mục [Nâng cấp lên model thật](#-nâng-cấp-lên-model-thật-sau-này) để biết cách thay bằng `best.pt` khi huấn luyện xong.

## 📁 Cấu trúc file

```text
zalo-traffic-app/
├── app.py              # Toàn bộ logic UI + Inference (Streamlit)
├── requirements.txt    # Thư viện Python cần cài
├── packages.txt        # Thư viện hệ thống (apt) - an toàn bổ sung cho OpenCV
├── README.md           # File này
└── docs/               # Tài liệu phân tích & thiết kế của đồ án
```

## 🏷️ 7 lớp biển báo mục tiêu

| ID | Tên lớp |
|----|---------|
| 0 | Cấm ngược chiều |
| 1 | Cấm dừng và đỗ |
| 2 | Cấm rẽ |
| 3 | Giới hạn tốc độ |
| 4 | Cấm còn lại |
| 5 | Nguy hiểm |
| 6 | Hiệu lệnh |

## 🖥️ Chạy thử Local

### Yêu cầu
- Python 3.9 - 3.11
- pip

### Các bước

1. **Mở terminal tại thư mục `zalo-traffic-app`:**

```bash
cd "zalo-traffic-app"
```

2. **Tạo môi trường ảo (khuyến nghị):**

```bash
# Windows (PowerShell)
python -m venv venv
venv\Scripts\Activate.ps1

# Linux / macOS
python -m venv venv
source venv/bin/activate
```

3. **Cài đặt thư viện:**

```bash
pip install -r requirements.txt
```

4. **Chạy ứng dụng:**

```bash
streamlit run app.py
```

Ứng dụng sẽ tự mở tại `http://localhost:8501`. Lần chạy đầu tiên, `ultralytics` sẽ tự động tải file `yolov8n.pt` (~6MB) về máy — cần có kết nối Internet.

---

## 🚀 Hướng dẫn Deploy lên Streamlit Community Cloud (Step by Step)

### Bước 1: Đẩy code lên GitHub

Streamlit Community Cloud lấy code trực tiếp từ một repository GitHub, nên trước tiên bạn cần có repo chứa `app.py` và `requirements.txt`.

1. Tạo một repository mới trên [github.com](https://github.com) (VD: `zalo-traffic-sign-app`). Có thể để **Public** hoặc **Private** (Streamlit Cloud dùng tài khoản Free vẫn kết nối được cả 2 loại).
2. Từ thư mục `zalo-traffic-app` trên máy, khởi tạo git và push code lên:

```bash
git init
git add .
git commit -m "Initial commit: Traffic Sign Detection Streamlit app"
git branch -M main
git remote add origin https://github.com/<username>/<ten-repo>.git
git push -u origin main
```

> 💡 Đảm bảo các file `app.py`, `requirements.txt`, `packages.txt` nằm ở **thư mục gốc (root)** của repo — Streamlit Cloud mặc định tìm chúng ở đó.

### Bước 2: Đăng ký / Đăng nhập Streamlit Community Cloud

1. Truy cập [share.streamlit.io](https://share.streamlit.io).
2. Chọn **Continue with GitHub** và đăng nhập bằng tài khoản GitHub vừa dùng ở Bước 1.
3. Cấp quyền (Authorize) cho Streamlit truy cập vào repository của bạn.

### Bước 3: Tạo App mới (Create app)

1. Ở góc trên bên phải workspace, bấm **"Create app"**.
2. Khi được hỏi *"Do you already have an app?"*, chọn **"Yup, I have an app"**.
3. Điền thông tin:
   - **Repository:** `<username>/<ten-repo>`
   - **Branch:** `main`
   - **Main file path:** `app.py`
4. (Tuỳ chọn) Ở ô **"App URL"**, đặt một subdomain riêng, VD: `zalo-traffic-sign` → URL sẽ là `https://zalo-traffic-sign.streamlit.app`.

### Bước 4: (Tuỳ chọn) Cấu hình nâng cao trước khi Deploy

Bấm **"Advanced settings"** nếu cần:
- **Python version:** chọn `3.10` hoặc `3.11` để đảm bảo tương thích tốt với `ultralytics`/`torch`.
- **Secrets:** nếu sau này bạn cần token riêng tư để tải model từ một HF Hub repo **private**, dán vào đây theo định dạng:

```toml
HF_TOKEN = "hf_xxxxxxxxxxxxxxxxxxxx"
```

### Bước 5: Deploy!

Bấm nút **"Deploy"**. Streamlit Cloud sẽ:
1. Clone repo của bạn.
2. Cài hệ thống các gói trong `packages.txt` (VD: `libgl1`).
3. Cài các thư viện Python trong `requirements.txt` (bao gồm `ultralytics`, `torch`,...).
4. Khởi chạy `streamlit run app.py`.

Quá trình này thường mất **1-5 phút** (lâu hơn nếu là lần build đầu tiên do phải tải `torch`). Bạn có thể theo dõi log real-time ngay trên trang deploy.

### Bước 6: Kiểm tra App

Khi thấy dòng chữ **"Your app is live!"**, mở URL app (VD: `https://zalo-traffic-sign.streamlit.app`) và thử:
1. Upload một ảnh bất kỳ.
2. Chờ vài giây để model xử lý (lần đầu sẽ hơi lâu do `ultralytics` tự tải `yolov8n.pt`).
3. Xem ảnh kết quả với bounding box + confidence score.

### Bước 7: Cập nhật App sau này

Mỗi khi bạn `git push` code mới lên nhánh `main`, Streamlit Community Cloud sẽ **tự động rebuild và redeploy** app — không cần thao tác gì thêm.

---

## 🔄 Nâng cấp lên model thật sau này

Khi đã huấn luyện xong model thật (`best.pt`) cho một trong 3 kiến trúc và upload lên một **Hugging Face Hub Model Repository**:

1. Mở file `app.py`, tìm khối comment:

```python
# ============================================================================
# [TƯƠNG LAI] TẢI MODEL THẬT (best.pt) TỪ HUGGING FACE HUB
# ============================================================================
```

2. Bỏ comment hàm `load_model_from_huggingface(repo_id, filename)`.
3. Thêm key `"hf_repo_id"` vào từng model trong `MODEL_OPTIONS`, ví dụ:

```python
"YOLOv8 (Tối ưu Tốc độ)": {
    ...
    "hf_repo_id": "your-username/traffic-sign-yolov8s-p2",
},
```

4. Sửa hàm `get_active_model()` để gọi `load_model_from_huggingface(...)` thay vì `load_placeholder_model()`.
5. `git commit` + `git push` — Streamlit Cloud sẽ tự redeploy với model thật.

Vì nhãn lớp trong hàm `draw_detections()` được đọc **động** từ `result.names` (lấy trực tiếp từ model đang chạy), bạn **không cần sửa** logic vẽ bounding box hay UI — khi model mới đã được train đúng 7 lớp biển báo, nhãn hiển thị sẽ tự động đúng theo `data.yaml`/`categories` lúc train.

---

## 🐛 Xử lý sự cố (Troubleshooting)

### Lỗi `ImportError: libGL.so.1: cannot open shared object file`

Đây là lỗi kinh điển khi deploy ứng dụng dùng OpenCV lên Streamlit Cloud (máy chủ Linux không có sẵn thư viện đồ họa GUI). App này đã được thiết kế để **tránh hoàn toàn** lỗi này:

- `requirements.txt` dùng `opencv-python-headless` (không cần `libGL`) thay vì `opencv-python`.
- File `app.py` **không** tự import `cv2` — toàn bộ việc vẽ bounding box dùng `PIL.ImageDraw`.
- File `packages.txt` (chứa `libgl1`) được thêm sẵn như một **lớp bảo hiểm bổ sung**, phòng trường hợp một bản cập nhật của `ultralytics` trong tương lai âm thầm kéo theo `opencv-python` (bản đầy đủ, cần GUI) như một dependency ẩn.

Nếu vẫn gặp lỗi này sau khi deploy:
1. Vào trang quản lý app trên Streamlit Cloud → **"Manage app"** → xem log để xác định chính xác thư viện nào đang kéo `opencv-python`.
2. Kiểm tra lại `requirements.txt`: đảm bảo không có dòng `opencv-python` hay `opencv-contrib-python` nào (kể cả do thư viện khác kéo theo gián tiếp).
3. Có thể thử thay `ultralytics` bằng gói `ultralytics-opencv-headless` (bản build sẵn chỉ phụ thuộc `opencv-python-headless`) nếu Ultralytics đã phát hành gói này.
4. Bấm **"Reboot app"** trên Streamlit Cloud để build lại từ đầu (cache cũ đôi khi giữ lại bản `opencv-python` cũ).

### App bị "Out of memory" / crash khi load model

Gói Free của Streamlit Community Cloud giới hạn RAM (khoảng 1GB). Nếu sau này chuyển sang các model nặng hơn (Faster R-CNN ~41M params, RT-DETR-L ~32M params) chạy full PyTorch trên CPU, có thể gặp tình trạng hết RAM. Gợi ý:
- Chỉ load 1 model tại một thời điểm (không load sẵn cả 3 model cùng lúc).
- Cân nhắc dùng bản `.onnx` hoặc export TensorRT/OpenVINO nếu cần tối ưu tốc độ & bộ nhớ.
- Cân nhắc nâng cấp lên gói trả phí hoặc dùng Hugging Face Spaces (16GB RAM) nếu model quá nặng cho Streamlit Free tier.

### App chạy chậm ở lần đầu tiên mở

Đây là hiện tượng bình thường ("cold start"): Streamlit Cloud "ngủ đông" app sau một thời gian không có người dùng, và lần truy cập đầu tiên phải khởi động lại container + tải model. Các lượt truy cập sau sẽ nhanh hơn nhờ cache (`st.cache_resource`).

---

## 📚 Tham khảo

- [Ultralytics YOLOv8 Docs](https://docs.ultralytics.com/)
- [Streamlit Community Cloud Docs](https://docs.streamlit.io/deploy/streamlit-community-cloud)
- [Hugging Face Hub - `hf_hub_download`](https://huggingface.co/docs/huggingface_hub/guides/download)
- Tài liệu thiết kế chi tiết đồ án: xem thư mục [`docs/`](./docs)
