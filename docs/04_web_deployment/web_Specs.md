# ĐẶC TẢ KỸ THUẬT WEB APP (WEB SPECIFICATIONS)

Tài liệu này đóng vai trò là bản Đặc tả Kỹ thuật (Specs) chính xác 100% của hệ thống Web App nhận diện biển báo giao thông đã được lập trình thực tế trong thư mục `Traffic-Sign-Detection-ZaloAI/web_app/`.

---

## 1. Thông số Chung
- **Mục đích:** Giao diện tương tác cho người dùng cuối tải ảnh lên và nhận về tọa độ biển báo.
- **Mô hình Kiến trúc:** Client - Server (Giao tiếp qua HTTP REST API).
- **Ngôn ngữ & Công nghệ:** 
  - Backend: Python 3.10, FastAPI, OpenCV, SAHI.
  - Frontend: HTML5, CSS3 (Vanilla), JavaScript (Vanilla).
- **Môi trường Deploy mục tiêu:** Docker Container trên Hugging Face Spaces (CPU Basic, 16GB RAM).

---

## 2. Đặc tả Lõi Xử lý Backend (`utils.py` & `app.py`)

Backend được thiết kế theo nguyên tắc SRP (Single Responsibility Principle) với khối bắt lỗi an toàn (Clean Code):

### A. Quản lý Model AI (`utils.py`)
- Nạp mô hình một lần duy nhất vào bộ nhớ RAM ngay khi Server khởi động thông qua `AutoDetectionModel` của SAHI.
- File trọng số mặc định được nạp từ: `weights/best.pt`.

### B. Pipeline Tiền xử lý & Dự đoán (`predict_traffic_signs`)
Mỗi khi nhận được file ảnh, Backend thực hiện quy trình sau:
1. **Tiền xử lý:** Chuyển đổi định dạng ảnh từ file upload sang chuẩn `RGB` của OpenCV.
2. **Cắt mảng (Slicing):** Dùng SAHI băm ảnh thành các mảnh lưới `512x512` với độ đè lấp (overlap) 20%. Giúp bảo toàn thông tin của các biển báo siêu nhỏ trên ảnh 4K.
3. **NMS (Non-Maximum Suppression):** Trộn các hộp dự đoán lại với nhau. Giữ lại hộp có điểm tin cậy cao nhất và xóa các hộp rác đè lên nó.
4. **Hậu xử lý (Pruning):** Sắp xếp điểm số từ cao xuống thấp và cắt ngọn lấy đúng **50 hộp cao điểm nhất** (`max_det=50`). Ngăn chặn tình trạng FrontEnd bị lag do phải vẽ hàng trăm khung rác.

### C. Giao tiếp API (`app.py`)
- Cổng kết nối mặc định: `7860`.
- **Route `GET /`**: Trả về giao diện người dùng `index.html`.
- **Route `POST /predict`**: Nhận file, lưu tạm, gọi hàm xử lý AI và trả về định dạng chuẩn JSON sau:
```json
{
  "success": true,
  "predictions": [
    {
      "bbox": [100.5, 200.0, 150.2, 250.8],
      "score": 0.95,
      "label": "Cam Di Nguoc Chieu"
    }
  ]
}
```

---

## 3. Đặc tả Kiến trúc Frontend (`static/` & `templates/`)

Giao diện (UI) được tự code 100% không qua Framework, nhấn mạnh vào trải nghiệm Wow-factor (Aesthetics).

### A. Giao diện Kính mờ (Glassmorphism)
- Chủ đạo màu nền: Đen nhám (Dark Mode ` #0d1117`).
- Khu vực Upload Ảnh và Trả kết quả được đặt trong các thẻ `<section>` phủ hiệu ứng mờ (backdrop-filter: blur(16px)) kết hợp viền ánh sáng Neon.

### B. Hiệu ứng Thị giác (Micro-Animations)
- **Scanning Radar:** Khi đang chờ Backend xử lý (mất 3-5s), Frontend kích hoạt hiệu ứng một tia sáng Neon quét dọc từ trên xuống dưới bức ảnh.
- **Fade-in Canvas:** Kết quả sau khi có sẽ được hiển thị mượt mà.

### C. Logic Javascript (`app.js`)
- Bắt sự kiện Drag & Drop hoặc Click để chọn ảnh.
- Gói ảnh vào đối tượng `FormData` và bắn `fetch()` dạng POST lên API `/predict`.
- Sau khi nhận JSON, hệ thống sẽ thực hiện lọc rác (chỉ giữ lại dự đoán >= 50%) và tự động phiên dịch (Mapping) tên biển báo từ tiếng Anh/số sang Tiếng Việt chuẩn dấu.
- Dùng thư viện 2D Canvas vẽ viền Bounding Box màu đỏ mỏng (không đổ bóng nền).
- Để chống đè chữ, thay vì ghi tên dài dòng lên ảnh, code sẽ đánh số thứ tự `[1]`, `[2]` (sử dụng API `measureText.actualBoundingBoxAscent` để vẽ một lớp nền trắng vừa khít ôm sát phía sau chữ số). Tên chi tiết của biển báo sẽ được liệt kê thành các huy hiệu (Badge) gọn gàng ở bên dưới ảnh.

---

## 4. Hướng dẫn Chạy thử (Local Testing)

Đây là các bước để bạn tự tay chạy thử và kiểm chứng giao diện Web App ngay trên máy tính của mình.

**Bước 1: Mở Terminal tại đúng thư mục Code**
Trong VS Code hoặc PowerShell, hãy chuyển đường dẫn vào thẳng thư mục web_app:
```bash
cd "d:\Học thống kê\Final Project\Traffic-Sign-Detection-ZaloAI\web_app"
```

**Bước 2: Kích hoạt môi trường ảo (Venv) & Cài đặt thư viện**
*(Lưu ý: Môi trường ảo giúp cài đặt hàng GB thư viện AI không làm nặng ổ C).*
Mở môi trường ảo (Nếu làm đúng sẽ hiện chữ `(venv)` đầu dòng lệnh):
```bash
.\venv\Scripts\activate
```
Nếu chưa cài thư viện, hãy chạy lệnh:
```bash
pip install -r requirements.txt
```

**Bước 3: Bật Server AI**
Khởi động FastAPI server với tính năng reload tự động:
```bash
uvicorn app:app --reload --host 0.0.0.0 --port 7860
```
*(Nếu dòng lệnh hiện ra dòng chữ `Đã nạp thành công mô hình AI!` và `Application startup complete` là Server đã sống).*

**Bước 4: Trải nghiệm UI**
1. Mở trình duyệt Web (Chrome/Edge/Safari).
2. Gõ đường dẫn: [http://localhost:7860](http://localhost:7860)
3. Kéo thả 1 bức ảnh vào ô "Kéo thả ảnh vào đây".
4. Ngồi ngắm hiệu ứng quét Radar và xem Bounding Box hiện ra!

*(Mẹo: Bạn có thể vừa chạy web vừa mở file `style.css` lên sửa màu sắc, ấn Save là trang web tự động đổi màu ngay lập tức do đã có cờ `--reload`).*
