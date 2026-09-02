# KẾ HOẠCH TỔNG THỂ DỰ ÁN (PROJECT MASTER PLAN)

## I. Lộ trình thực hiện (Timeline)
<!-- Todo: Điền lộ trình -->

## II. Pipeline 4 Bước
<!-- Todo: Điền pipeline (EDA -> Preprocessing -> Training -> Deployment) -->

## III. Cấu trúc thư mục chuẩn
<!-- Todo: Điền cấu trúc thư mục -->

## IV. Đặc Tả Dữ Liệu Đầu Vào (Data Overview)
*Phần này đúc kết toàn bộ bản chất dữ liệu (được thu thập thực tế từ Zalo AI Challenge) - Trái tim của toàn bộ các quyết định về Hyperparameters phía sau.*

1. **Định dạng dữ liệu (Data Format):** 
   - Tổ chức theo chuẩn **COCO JSON format** chuyên nghiệp (gồm 4 khối chính: `info`, `images`, `annotations`, `categories`).
   - Phù hợp với các kiến trúc Object Detection hiện đại và dễ dàng chuyển đổi sang định dạng YOLO.

2. **Kích thước ảnh (Resolution):**
   - Kích thước phổ quát: **`1622 x 626`** pixels. 
   - Đây là tỉ lệ ảnh siêu ngang (Panorama / Dashcam). Không phải là ảnh 4K tiêu chuẩn nhưng có chiều ngang vô cùng lớn. Việc giữ nguyên tỉ lệ gốc này (không bóp méo thành hình vuông) là tối quan trọng để không làm biến dạng hình tròn của biển báo.

3. **Bài toán Vật thể Siêu Nhỏ (Small Object Detection):**
   - Kích thước diện tích biển báo trung bình (Median Area): Chỉ khoảng **`266 px²`** (tương đương với một hình vuông nhỏ xíu `16 x 16` pixel trên tấm ảnh ngang 1622 pixel).
   - Kích thước nhỏ nhất (Min Area): Có những biển báo nằm tít mù xa, chỉ chiếm vỏn vẹn **`4 px²`** (`2 x 2` pixel).
   - *=> Chẩn đoán:* Đây là thử thách chí mạng của đồ án. Các mô hình YOLO nếu resize thẳng về `640x640` sẽ bóp nát và xóa sổ các biển báo 16x16 này. Bắt buộc phải triển khai thuật toán **SAHI** ở khâu dự đoán và **P2 Layer** ở khâu Training để trị dứt điểm.

4. **Hệ sinh thái Nhãn (Classes):**
   Gồm 7 phân lớp biển báo giao thông đặc trưng của Việt Nam:
   - `0`: Cấm ngược chiều
   - `1`: Cấm dừng và đỗ
   - `2`: Cấm rẽ
   - `3`: Giới hạn tốc độ
   - `4`: Cấm còn lại
   - `5`: Nguy hiểm
   - `6`: Hiệu lệnh
   *(Lưu ý: Đối với mô hình Faster R-CNN, PyTorch yêu cầu phải đếm thêm nhãn `Background` (Nền), nên `num_classes` phải thiết lập là 8).*
