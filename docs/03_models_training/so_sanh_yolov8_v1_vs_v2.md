# Phân tích và So sánh: Hệ thống YOLOv8 v1 (Tích hợp) vs v2 (Phân chia Module)

Tài liệu này trình bày phân tích chi tiết khi đối chiếu hai phiên bản mã nguồn huấn luyện YOLOv8 với tài liệu đặc tả kỹ thuật `models_specs.md`.
Cụ thể, sự so sánh diễn ra giữa phiên bản **v1 nguyên khối** (`train_yolov8.ipynb`) và **hệ thống v2 phân module** (tách biệt thành `data_conversion.ipynb` và `train_yolov8_ver2.ipynb`).

---

### 1. Các điểm GIỐNG NHAU cốt lõi
Dù kiến trúc file tổ chức khác nhau, cả hai phiên bản đều chia sẻ những nền tảng kỹ thuật chung rất quan trọng cho đồ án:
*   **Sử dụng Core Framework:** Cả hai đều dùng thư viện `ultralytics` để khởi tạo và huấn luyện họ mô hình YOLOv8.
*   **Tư tưởng nâng cấp mạng P2 (Small Object Detection):** Cả hai đều không dùng kiến trúc YOLOv8 mặc định mà tiến hành tự định nghĩa file kiến trúc YAML (thêm nhánh P2 - Stride 4) nhằm tối ưu hóa khả năng bắt các biển báo giao thông có kích thước cực nhỏ.
*   **Thuật toán tối ưu (Optimizer):** Cả hai đều sử dụng thuật toán tối ưu `AdamW` (được xác nhận qua log huấn luyện) - một sự lựa chọn lý tưởng cho YOLOv8.
*   **Dữ liệu chung:** Cùng hướng tới việc xử lý và huấn luyện trên bộ dataset khổng lồ Zalo AI Traffic Sign 2020.

---

### 2. Sự khác biệt về Xử lý Dữ liệu (Nguy cơ rò rỉ dữ liệu của v2)
*   **Bản v1 (Chia Data chuẩn mực):** Code v1 có cơ chế tự động trộn ngẫu nhiên dữ liệu và cắt tách bạch **80% cho tập Train** và **20% cho tập Validation**. Sau đó nó tự động rải ảnh vào đúng 4 thư mục độc lập (`images/train`, `images/val`, `labels/train`, `labels/val`).
*   **Hệ thống v2 (`data_conversion.ipynb`):** Tách riêng file xử lý data là một ý tưởng rất hay (chuẩn Software Engineering), nhưng phần lõi bên trong **KHÔNG HỀ chia cắt tập dữ liệu**. Toàn bộ 4500 ảnh được ném chung vào một thư mục `yolo_format/images`.
    *   **Bằng chứng:** Trong dòng log huấn luyện của file `train_yolov8_ver2.ipynb` ghi rất rõ: `"Using 4500 train, 4500 val images for fraction=1.0 at imgsz=640"`. 
    *   **Hậu quả:** Điều này gây ra thảm họa **Data Leakage (Rò rỉ dữ liệu) 100%**. Mô hình lấy chính đề thi (tập Val) ra để làm bài tập về nhà (tập Train), dẫn đến hiện tượng học vẹt nặng nề. Điểm số (Loss) trên máy sẽ rất đẹp nhưng khi test trên dữ liệu mới hoàn toàn, AI sẽ sụp đổ.

### 3. Đường dẫn dữ liệu (Paths) - Nguy cơ Crash trên Kaggle
*   **Bản v1:** Sử dụng lệnh dò tìm tự động `glob.glob('/kaggle/input/**/train_traffic_sign_dataset.json')`. Kỹ thuật này giúp code tự động tìm trúng file dữ liệu trên Kaggle mà không bị phụ thuộc vào sự thay đổi đường dẫn mount của nền tảng máy ảo.
*   **Hệ thống v2:** Cả file chuyển đổi data và file train đều đang sử dụng đường dẫn tuyệt đối (Hardcode) trỏ thẳng vào ổ cứng cá nhân: `LOCAL_JSON_PATH = r"G:\HocKi9\..."`. 
    *   **Hậu quả:** Nếu đưa hệ thống v2 này lên nền tảng Kaggle và chạy, tiến trình sẽ văng lỗi `FileNotFoundError` ngay lập tức.

### 4. Cấu trúc mạng P2 Layer - Khác biệt thông số
*   **Bản v1:** Định nghĩa chuẩn `nc: 7` (7 loại biển báo Zalo AI) ngay từ đầu tại khối kiến trúc YAML.
*   **Bản v2:** Tại khối mã sinh file YAML cấu hình mạng, code đang đặt cứng `nc: 1`. Mặc dù thư viện `ultralytics` rất thông minh tự động ghi đè lại bằng 7 class (nhờ đọc từ file `data.yaml`), nhưng việc khai báo `nc: 1` ngay trong lõi cấu trúc mạng là không chuẩn xác so với Specs. Thêm vào đó, nhánh Head của v2 chèn thêm các lớp tích chập `Conv 1x1` trước các hàm `Upsample`, làm mạng lệch đi đôi chút so với kiến trúc YOLOv8 nguyên bản.

### 5. Cấu hình Huấn luyện & Augmentation - Đi ngược lại Spec
Dựa vào log xuất ra từ quá trình chạy thực tế của file v2, hệ thống đã bắt quả tang các thông số sai lệch nghiêm trọng:
*   **`imgsz=640` và `epochs=1`:** Tài liệu `models_specs.md` bắt buộc dùng ảnh độ phân giải cao `imgsz=1280` và huấn luyện `50` Epochs để mạng P2 bảo toàn được các vật thể siêu nhỏ. Việc v2 chạy `640` và `1` Epoch chứng minh đây chỉ là bản nháp chạy thử (dry run) chưa hoàn chỉnh.
*   **`augment=False`:** Đây là sai sót chí mạng nhất. Khi tham số này tắt, toàn bộ công sức thiết kế các chiến thuật Data Augmentation trong Specs như **Mosaic (Trộn 4 ảnh), Random Shift (Chống Center Bias), Xoay ảnh 10 độ** đều bị hủy bỏ. Mô hình mất đi sức đề kháng và tính bất biến không gian. (Ở chiều ngược lại, bản v1 truyền đầy đủ `mosaic=1.0`, `translate=0.2`, `degrees=10.0`, `fl_gamma=2.0`).

### 6. Khâu đóng gói (Zip kết quả)
*   **Bản v1:** Có một cell code chuyên dụng cuối file để tự động nén toàn bộ thư mục kết quả thành file `yolo_results.zip`. Tính năng này là "cứu cánh" giúp sinh viên dễ dàng tải trọng số khổng lồ (250MB) về máy tính từ Kaggle chỉ với 1 cú click.
*   **Hệ thống v2:** Thiếu vắng ô code nén ZIP này, làm quá trình trích xuất kết quả sau huấn luyện vô cùng vất vả.

---

### 💡 Tổng kết & Khuyến nghị
*   **Hệ thống v1 (`train_yolov8.ipynb`)** là phiên bản hoàn thiện, an toàn, chuẩn mực và **bám sát 100% theo `models_specs.md`**. Nó được thiết kế hoàn hảo để đưa lên Kaggle và huấn luyện đến đích mà không lo lỗi đường dẫn hay rò rỉ dữ liệu.
*   **Hệ thống v2 (Modular)** có tư duy tổ chức file rất chuyên nghiệp (tách riêng xử lý Data, thêm MLOps bằng `wandb`). Tuy nhiên, bản thân các mã nguồn bên trong lại là mã "nháp", chứa lỗi Rò rỉ dữ liệu 100% và tắt toàn bộ Augmentation quan trọng.

**Khuyến nghị:** Ưu tiên sử dụng **v1** làm lõi chính thức nộp đồ án. Nếu bạn vẫn muốn sử dụng **v2** vì thích cấu trúc chia file đẹp mắt, bạn **bắt buộc** phải viết lại hàm phân chia tập Train/Val (80-20) trong file `data_conversion.ipynb` và bật lại toàn bộ thiết lập `augment=True`, `imgsz=1280` trong file huấn luyện.
