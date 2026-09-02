# GIẢI THÍCH LOGIC & CHIẾN LƯỢC DEPLOY WEB APP (HUGGING FACE)

Tài liệu này vạch ra chiến lược xây dựng ứng dụng Web dựa trên yêu cầu của đồ án gốc và các kết quả phân tích dữ liệu (EDA).

---

## 1. Phân tích Yêu cầu Đề bài & Ràng buộc
- **Yêu cầu 2 của Đề bài (2 điểm):** *"...select the best-performing model and develop a web-based application that allows users to upload an image and returns the object detection results for that image."*
- **Ràng buộc:** 
  - Chỉ yêu cầu upload **1 ảnh tĩnh**, KHÔNG yêu cầu xử lý luồng Video/Camera Real-time (FPS cao).
  - Nền tảng Deploy: **Hugging Face Spaces**.

### Chốt hạ Mô hình (Model Selection)
Vì không bị ép tốc độ Real-time, ta sẽ hy sinh tốc độ (Inference Time) để đổi lấy **Độ chính xác tuyệt đối (mAP)**. 
- **Mô hình được chọn:** `YOLOv8s-P2` hoặc `RT-DETR-L`.
- **Các Kỹ thuật Inference đính kèm (Bắt buộc theo Kế hoạch):**
  1. **SAHI (Slicing Aided Hyper Inference):** Băm bức ảnh 4K upload thành các mảnh nhỏ `512x512`, đưa qua model dự đoán rồi ghép lại. Kỹ thuật này sống còn để soi các biển báo siêu li ti.
  2. **TTA (Test-Time Augmentation):** Bật tính năng lật/phóng to ảnh tự động lúc inference để dự đoán đa góc độ, sau đó lấy kết quả trung bình để vá các góc mù của model.
  3. **Tối ưu Soft-NMS:** Áp dụng `Soft-NMS` thay vì NMS cứng, kết hợp hạ `max_det=50` để giữ lại các cụm biển báo cắm chùm, đứng sát mép nhau mà không bị thuật toán xóa nhầm.
  
  *(Việc kích hoạt cả 3 cỗ máy này sẽ đẩy thời gian chờ lên khoảng 3-5 giây/ảnh. Tuy nhiên, điều này tuân thủ đúng định hướng Đồ án: Đánh đổi thời gian lấy Độ chính xác mAP cao nhất).*

---

## 2. Các Lựa chọn Công nghệ (Tech Stack Options)
Hugging Face Spaces hỗ trợ nhiều môi trường chạy khác nhau. Dưới đây là 3 hướng đi, mình đã phân tích kỹ ưu/nhược điểm để bạn lựa chọn:

### Lựa chọn 1: Dùng Gradio (Mặc định của Hugging Face)
- **Cơ chế:** Viết code Python thuần, thư viện Gradio tự động sinh ra giao diện Web.
- **Ưu điểm:** Nhanh, tốn khoảng 50 dòng code là xong cả Frontend lẫn Backend.
- **Nhược điểm (Chí mạng):** Giao diện bị "khóa chết" theo template của Gradio. Trông rất nhàm chán, giống các tool nghiên cứu thô sơ, **không thể làm giao diện WOW, không có hiệu ứng Animation xịn sò hay Glassmorphism**. Khó gây ấn tượng mạnh với Hội đồng.

### Lựa chọn 2: Dùng Streamlit
- **Cơ chế:** Tương tự Gradio nhưng UI trông hiện đại hơn một chút, hỗ trợ Layout chia cột.
- **Ưu điểm:** Dễ code bằng Python.
- **Nhược điểm:** Vẫn là UI đúc sẵn, không thể tùy biến CSS quá sâu. Tốc độ load đôi khi bị chậm do cơ chế render lại toàn trang của Streamlit.

### Lựa chọn 3 (Khuyên dùng): FastAPI (Backend) + HTML/CSS/JS Thuần (Frontend) qua Docker Space
- **Cơ chế:** 
  - Backend: Viết 1 API bằng **FastAPI** (Python) để nhận ảnh, xử lý SAHI + YOLOv8 và trả về tọa độ.
  - Frontend: Viết bằng **HTML, CSS Vanilla và JS thuần**.
  - Đóng gói toàn bộ vào 1 file `Dockerfile` và đẩy lên Hugging Face Spaces.
- **Ưu điểm:**
  - **Tự do tuyệt đối về Giao diện (UI/UX):** Ta có thể áp dụng thiết kế Dark Mode sang trọng, hiệu ứng kính mờ (Glassmorphism), các vi-hiệu ứng (micro-animations) khi hover chuột, và hiệu ứng mượt mà khi khung Bounding Box hiện lên ảnh. Chắc chắn sẽ làm **Hội đồng WOW ngay từ cái nhìn đầu tiên**.
  - **Kiến trúc chuẩn Microservices:** Tách biệt rõ Frontend và Backend, cực kỳ ghi điểm về mặt tư duy kỹ sư phần mềm (Software Engineering).
- **Nhược điểm:** Code dài hơn, đòi hỏi kiến thức về API và Docker (Nhưng mình sẽ lo toàn bộ phần code này).

---

## 3. Kế hoạch Thực thi (Implementation Plan)
Nếu bạn chọn **Lựa chọn 3** (Giao diện siêu đẹp với HTML/CSS), đây sẽ là các bước chúng ta sẽ làm:

1. **Phase 1: Backend API (FastAPI)**
   - Khởi tạo app FastAPI, tạo route `POST /predict`.
   - Tích hợp hàm load trọng số (weights) của YOLOv8s-P2 hoặc RT-DETR.
   - Nhúng thuật toán `sahi.predict` kết hợp **TTA** (truyền tham số cấu hình augment lúc load model) và **Soft-NMS** để xử lý ảnh độ phân giải cao và trả kết quả dưới dạng JSON (Tọa độ hộp, Tên nhãn, Độ tin cậy).

2. **Phase 2: Frontend Design (HTML/Vanilla CSS/JS)**
   - Cấu trúc `index.html`: Gồm khu vực Hero Section, Dropzone để kéo thả ảnh, và khu vực hiển thị kết quả.
   - Cấu trúc `style.css`: Phối màu Dark Theme (Nền tối, màu nhấn Neon), hiệu ứng Glassmorphism cho Card chứa ảnh, hiệu ứng loading xoay mượt mà.
   - Cấu trúc `app.js`: Xử lý thao tác kéo thả, gọi Fetch API tới Backend, và quan trọng nhất là **vẽ Canvas 2D** để hiển thị Bounding Box lên trên bức ảnh trả về một cách sinh động.

3. **Phase 3: Docker & Hugging Face Deployment**
   - Viết `Dockerfile` cài đặt Python, OpenCV, Ultralytics.
   - Chạy lệnh Serve cả Frontend tĩnh và Backend API qua cổng 7860 (Cổng mặc định của Hugging Face).
   - Đẩy code và Model Weights lên Hugging Face Hub.

---

## 4. Phụ lục: Giải đáp các quyết định Kỹ thuật

### Tại sao lại dùng Dockerfile và đẩy lên Hugging Face Spaces?
- **Hugging Face Spaces** là nền tảng máy chủ (Cloud Server) miễn phí mạnh mẽ nhất hiện nay dành riêng cho Machine Learning, cung cấp tới 16GB RAM.
- **Dockerfile** là một bản thiết kế (script) hướng dẫn Hugging Face cách "xây một cái máy tính ảo" từ con số không: cài hệ điều hành gì, cài Python phiên bản mấy, tải thư viện OpenCV như thế nào. Không có Dockerfile, máy chủ ảo sẽ không biết cách chạy mã nguồn của chúng ta.

### Tại sao Web App lại dùng FastAPI và Python?
- Lõi AI (YOLO, RT-DETR, PyTorch) đều được lập trình bằng Python. Do đó, Backend bắt buộc phải dùng Python để nạp (load) mô hình AI và xử lý ma trận ảnh gốc.
- **FastAPI** là framework Python hiện đại, tốc độ cực nhanh và xử lý đa luồng (Asynchronous) cực tốt.
- *Tham chiếu:* Trong **báo cáo đồ án mẫu (Trầm AI)**, nhóm sinh viên đó cũng đã sử dụng **chính xác kiến trúc FastAPI (Python) kết hợp với Docker trên Hugging Face**. Việc ta đi theo kiến trúc này chứng tỏ Đồ án của ta đạt quy chuẩn kỹ sư phần mềm (Software Engineering) rất chuyên nghiệp.

### Lựa chọn Tối ưu NMS: Hạ max_det, Nới lỏng IoU hay Soft-NMS?
Trong giai đoạn dự đoán (Inference), mô hình thường vẽ ra hàng chục Bounding Box đè lên cùng một biển báo. Để lọc các hộp thừa, ta có 3 kỹ thuật. **Quyết định CHỐT HẠ của ta là kết hợp Soft-NMS và Hạ max_det=50**:
1. **Nới lỏng IoU (Không dùng):** Phương pháp này dễ làm. Nhưng nếu nới quá lỏng, các biển báo cắm thành chùm (đứng sát nhau ở ngã tư) sẽ bị thuật toán NMS cứng (Hard NMS) tưởng nhầm là 1 cụm rác và xóa mất hoàn toàn các biển báo đằng sau.
2. **Soft-NMS (Chọn):** Thay vì xóa thẳng tay (Hard Delete) các hộp bị đè lên nhau, Soft-NMS hoạt động cực kì tinh tế bằng cách chỉ *hạ điểm tin cậy (Confidence Score)* của hộp bị đè xuống. Nhờ vậy, nếu hộp đó thực sự chứa một biển báo khác (trong chùm biển báo), nó vẫn sống sót qua bộ lọc. Điều này giúp đẩy độ chính xác Recall lên tối đa.
3. **Hạ max_det=50 (Chọn):** Mặc định YOLO cho phép xuất ra tới 300 đối tượng trên 1 ảnh. Việc bắt Frontend (Canvas JS) render hàng trăm khung bounding box sẽ làm treo (lag) trình duyệt Web. Hạ giới hạn xuống 50 (vì ở VN hiếm có ngã tư nào cắm quá 50 cái biển) giúp Web App chạy cực kỳ mượt mà.
