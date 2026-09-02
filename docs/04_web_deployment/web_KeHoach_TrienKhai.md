# BẢN THIẾT KẾ KỸ THUẬT & KẾ HOẠCH TRIỂN KHAI WEB APP

Dựa trên các phân tích logic và yêu cầu của Đồ án, hệ thống Web App nhận diện biển báo giao thông sẽ được xây dựng theo **Lựa chọn 3: FastAPI (Backend) + HTML/CSS/JS Thuần (Frontend)** chạy trên nền tảng Docker của Hugging Face Spaces. 

Bản thiết kế này quy định chi tiết cấu trúc, giao diện và các thuật toán sẽ được code trong giai đoạn tới.

---

## 1. Kiến trúc Hệ thống (System Architecture)

Ứng dụng được thiết kế theo mô hình **Client-Server** chuẩn mực, tách biệt hoàn toàn giao diện và logic suy luận (Inference):

*   **Backend (Server):** Viết bằng **FastAPI** (Python). Đảm nhận việc nhận ảnh từ Frontend, gọi model AI (YOLOv8s-P2) và thuật toán SAHI để dự đoán, sau đó trả về tọa độ (Bounding Box) dưới dạng chuẩn `JSON`.
*   **Frontend (Client):** Viết bằng **Vanilla HTML, CSS, JS** (Không dùng framework cồng kềnh). Nhận JSON từ Backend và dùng Javascript (Canvas API) để vẽ khung kết quả lên ảnh.
*   **Deployment:** Toàn bộ hệ thống sẽ được đóng gói bằng **Dockerfile** và đẩy lên Hugging Face Spaces. Không sử dụng Database vì chỉ cần tính năng dự đoán đơn lẻ (Single Image Prediction) theo đúng yêu cầu đề bài.

---

## 2. Kỹ thuật Lõi Dự đoán (Core Inference Engine)

Đúng như Kế hoạch Training đã chốt, Backend sẽ gánh 3 kỹ thuật xử lý ảnh "hạng nặng" để đẩy độ chính xác mAP lên tối đa:

1.  **Thuật toán cắt ảnh SAHI:** Băm ảnh 4K thành các mảnh lưới `512x512` để zoom cận cảnh vào các biển báo siêu nhỏ.
2.  **Test-Time Augmentation (TTA):** Bật cờ augment khi gọi hàm predict để vá các góc nhìn mù.
3.  **Soft-NMS & Hạ max_det=50:** Trộn kết quả các mảnh cắt từ SAHI bằng Soft-NMS để không xóa nhầm các biển báo đè lên nhau. Đồng thời giới hạn tối đa 50 biển báo/ảnh (max_det=50) để đảm bảo trình duyệt Web không bị treo khi vẽ khung rác.

*(Lưu ý: API dự đoán sẽ mất khoảng 3-5 giây xử lý, nên Frontend bắt buộc phải có hiệu ứng Loading đẹp mắt để giữ chân người dùng).*

---

## 3. Thẩm mỹ Giao diện (UI/UX - The WOW Factor)

Để đè bẹp các hệ thống dùng template có sẵn (Gradio/Streamlit) hoặc giao diện đơn giản như các báo cáo mẫu, UI của chúng ta sẽ được code tay 100% với các tính năng:

*   **Chủ đề (Theme):** **Dark Mode** hiện đại, kết hợp viền ánh sáng Neon (Cyberpunk/Tech style).
*   **Hiệu ứng Kính mờ (Glassmorphism):** Các vùng chứa ảnh và thông tin sẽ dùng nền mờ (backdrop-filter: blur) để tạo độ sâu 3D sang trọng.
*   **Hiệu ứng Hoạt hình (Micro-Animations):** 
    *   Khi đang chờ API xử lý (3-5 giây), sẽ có một thanh sáng chạy dọc bức ảnh (Radar Scan Animation) tạo cảm giác AI đang quét từng pixel.
    *   Khung Bounding Box khi vẽ ra sẽ có hiệu ứng viền sáng fade-in mượt mà.

---

## 4. Cấu trúc Thư mục Code (Directory Structure)

Toàn bộ code Web App sẽ được đặt trong thư mục `Traffic-Sign-Detection-ZaloAI/web_app/`:

```text
web_app/
│
├── main.py                 # File chạy Server FastAPI và API /predict
├── requirements.txt        # Danh sách thư viện (fastapi, uvicorn, sahi, ultralytics...)
├── Dockerfile              # Cấu hình môi trường cho Hugging Face Spaces
│
├── weights/
│   └── best.pt             # File trọng số mô hình tốt nhất (best.pt)
│
├── templates/
│   └── index.html          # Cấu trúc giao diện HTML
│
└── static/
    ├── css/
    │   └── style.css       # File code giao diện Glassmorphism & Animations
    └── js/
        └── app.js          # File code Logic gọi API và vẽ Canvas Bounding Box
```

---


---

## 6. Lộ trình Code (Roadmap)

Chúng ta sẽ thi công theo trình tự:
*   **Bước 1:** Viết file `main.py` (FastAPI) và test API `/predict` bằng file weights tạm thời của bạn.
*   **Bước 2:** Viết khung HTML và tô vẽ `style.css` siêu đẹp.
*   **Bước 3:** Viết `app.js` ráp nối dữ liệu JSON từ Backend lên ảnh Frontend.
*   **Bước 4:** Viết `Dockerfile` và hướng dẫn bạn Push lên Hugging Face.
