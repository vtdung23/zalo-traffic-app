# Giải thích chi tiết mã nguồn: train_rtdetr.ipynb

Tài liệu này giải thích một cách cặn kẽ và chuyên sâu từng đoạn code trong file `train_rtdetr.ipynb`. Mục tiêu là đối chiếu và làm rõ cách mã nguồn hiện thực hóa các thông số kỹ thuật tiên tiến nhất được định nghĩa tại `models_specs.md` (Đặc biệt là phần **3. RT-DETR (Real-Time DEtection TRansformer)**) và `train_GiaiThich_Hyperparams.md`.

Khác với YOLOv8 hay Faster R-CNN, RT-DETR là một kiến trúc dựa trên **Transformer**, mang trong mình sức mạnh của cơ chế **Self-Attention** nhưng lại đòi hỏi cực kỳ khắt khe về bộ nhớ VRAM. Notebook này được sinh ra để chạy trên **Google Colab** nhằm tận dụng tối đa tài nguyên phần cứng, đồng thời tích hợp các thủ thuật "chống tràn bộ nhớ" đỉnh cao.

---

## Cell 1: Tải dữ liệu siêu tốc bằng Kaggle API

```python
# Tải bộ dữ liệu Zalo AI từ Kaggle về Google Colab bằng API
!pip install -q kaggle
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
!kaggle datasets download -d phhasian0710/za-traffic-2020
!unzip -q -n za-traffic-2020.zip -d /content/dataset
```
- **Ý nghĩa Kỹ thuật**: Đoạn code này thiết lập một "đường ống" truyền dữ liệu trực tiếp từ máy chủ Kaggle sang ổ cứng SSD nội bộ (`/content/`) của máy ảo Google Colab. 
- **Giải quyết nút thắt cổ chai (Bottleneck)**: Nếu tải dữ liệu thủ công lên Google Drive rồi mount sang Colab, tốc độ đọc file ảnh trong lúc train sẽ cực kỳ chậm, làm nghẽn GPU (GPU đói dữ liệu). Việc dùng API tải thẳng file ZIP và giải nén ngay tại local ảo giúp tốc độ đọc (I/O) đạt mức tối đa.

---

## Cell 2 & 3: Tiền xử lý Dữ liệu và Chia tập Train/Val (80/20)

```python
import glob
json_paths = glob.glob('/content/dataset/**/train_traffic_sign_dataset.json', recursive=True)
...
train_dataset, val_dataset = random_split(...) # Cắt 80% Train, 20% Val
```
- **Ý nghĩa (Tránh Data Leakage)**: Tương tự như quy trình chuẩn của nhóm, code sử dụng `random.seed(42)` để xáo trộn toàn bộ danh sách 4500 tấm ảnh, sau đó cắt vạch ranh giới rõ ràng: 80% đưa vào `train` và 20% đưa vào `val`. Việc này **ngăn chặn 100% hiện tượng Rò rỉ dữ liệu (Data Leakage)** - một lỗi sơ đẳng nhưng chí mạng nếu để mô hình học thi bằng chính đề kiểm tra.

```python
def convert_coco_to_yolo(bbox, img_width, img_height):
    x_center = (x_min + w / 2) / img_width
...
```
- **Ý nghĩa Kỹ thuật**: Chuyển đổi hệ tọa độ COCO `[x_min, y_min, w, h]` (pixel tuyệt đối) sang chuẩn YOLO format `[x_center, y_center, w_norm, h_norm]` (chuẩn hóa tỷ lệ từ 0 đến 1). Mặc dù là mô hình Transformer, cấu trúc thư viện `ultralytics` vẫn đòi hỏi dữ liệu đầu vào phải tuân thủ nghiêm ngặt định dạng text này để tối ưu hóa việc nạp batch. Nhãn (label) cũng được lùi 1 đơn vị (`class_id - 1`) để khớp với index mảng bắt đầu từ 0.

---

## Cell 5: Kết nối Google Drive làm "Két sắt"

```python
from google.colab import drive
drive.mount('/content/drive')
save_dir = '/content/drive/MyDrive/DoAn_NhanDienBienBao'
```
- **Ý nghĩa DevOps**: Đây là chiến thuật **Zero-Data Loss** (Không mất mát dữ liệu). Google Colab là một nền tảng điện toán đám mây cấp phát tạm thời. Nếu người dùng tắt trình duyệt hoặc máy chủ hết phiên (Session Timeout), toàn bộ dữ liệu ở `/content/` sẽ bốc hơi sạch sẽ. Đoạn code này mở một kênh lưu trữ vĩnh viễn sang Google Drive, đảm bảo trọng số (weights) nặng hàng trăm MB được cất giữ an toàn tuyệt đối ngay cả khi máy ảo bị sập.

---

## Cell 6: Khởi tạo Kiến trúc Transformer (RT-DETR-L) và Kỹ thuật Gradient Accumulation

Đây là ô code đắt giá nhất và chứa nhiều hàm lượng chất xám nhất của toàn bộ quá trình huấn luyện RT-DETR.

```python
from ultralytics import RTDETR
model = RTDETR('rtdetr-l.pt')
```
- **Ý nghĩa**: Hiện thực hóa mục **3. RT-DETR** trong `models_specs.md`. Khởi tạo mạng Transformer với kích thước **Large (L)**. Thay vì dùng nhánh backbone CNN truyền thống, RT-DETR dùng cơ chế **Multi-Head Self-Attention (Q-K-V)** để quét toàn cục bức ảnh, giúp nó hiểu được bối cảnh (Context) hoàn hảo hơn hẳn YOLO. Khối lượng tham số khổng lồ (pretrained) giúp mô hình "thông minh" sẵn.

```python
results = model.train(
    data='/content/dataset.yaml',
    epochs=50,
    imgsz=1280,
```
- **Ý nghĩa (`imgsz=1280`)**: Trả lời trực tiếp yêu cầu từ EDA. Biển báo giao thông Zalo AI cực kỳ nhỏ. Nếu thu nhỏ ảnh về 640, biển báo sẽ chỉ còn là một đốm nhiễu vài pixel, Attention Mechanism sẽ không thể nhận diện được hình dạng bên trong biển. Giữ nguyên 1280 là **điều kiện tiên quyết** để bảo toàn vật thể nhỏ.

```python
    batch=2,            
    accumulate=4,       
```
- **Ý nghĩa Kỹ thuật Chống Tràn RAM (Gradient Accumulation)**: Đây là một thủ thuật "ma thuật" trong Deep Learning.
  - Tại sao lại dùng `batch=2`? Vì ma trận tính toán Attention của kiến trúc Transformer tăng lên theo cấp số nhân $O(N^2)$ đối với kích thước ảnh. Với ảnh 1280px, một card T4 (16GB VRAM) của Colab sẽ ngay lập tức văng lỗi **CUDA Out Of Memory (OOM)** nếu nhồi batch size = 8. Mức tối đa nó chịu đựng được chỉ là 2 tấm ảnh cùng lúc.
  - Tại sao lại thêm `accumulate=4`? Nếu train với `batch=2`, đạo hàm (Gradient) sẽ cực kỳ nhiễu (Noisy), đồ thị Loss sẽ giật cục và mạng không thể hội tụ. Lệnh `accumulate=4` ra lệnh cho PyTorch: *"Chạy 2 ảnh, tính đạo hàm nhưng ĐỪNG CẬP NHẬT TRỌNG SỐ (optimizer.step) vội. Hãy cộng dồn đạo hàm lại. Lặp lại 4 lần như vậy rồi mới cập nhật 1 lần"*. 
  - **Kết quả**: Batch ảo (Virtual Batch Size) = `2 x 4 = 8`. Hệ thống lách luật thành công: Vừa không bị tràn RAM, vừa giữ nguyên được kích thước Batch chuẩn là 8 giúp đạo hàm mượt mà, ổn định.

```python
    optimizer='AdamW',
    cos_lr=True,
    project=save_dir, 
    name='rtdetr_highres',
```
- **Ý nghĩa**: 
  - `AdamW`: Khác với SGD dùng cho Faster R-CNN, mô hình Transformer cực kỳ nhạy cảm với việc tinh chỉnh Learning Rate và hiện tượng bùng nổ trọng số. Thuật toán `AdamW` (Adam với Weight Decay) là tiêu chuẩn vàng của ngành công nghiệp dành cho Transformer.
  - `cos_lr=True`: Hạ nhiệt độ Learning Rate theo đường cong hình sin (Cosine Annealing), tránh việc mô hình đi lệch khỏi điểm cực tiểu toàn cục ở những epoch cuối.
  - Kết quả cuối cùng được xuất thẳng sang Google Drive thông qua biến `save_dir`.

---

## Cell 7: Thuật toán SAHI (Hỗ trợ Web App)

```python
from sahi import AutoDetectionModel
...
    detection_model = AutoDetectionModel.from_pretrained(
        model_type='yolov8',  # SAHI gọi RT-DETR qua type yolov8
        model_path='/content/drive/MyDrive/DoAn_NhanDienBienBao/rtdetr_highres/weights/best.pt',
...
```
- **Ý nghĩa MLOps**: Chuẩn bị sẵn sàng cho quá trình Deploy (Triển khai lên Web). Trong ứng dụng thực tế (Camera giao thông), ảnh có thể là 4K. Kỹ thuật **SAHI (Slicing Aided Hyper Inference)** giúp băm tấm ảnh 4K ra làm nhiều mảnh nhỏ (ví dụ 4 mảnh 1080p), cho RT-DETR dự đoán trên từng mảnh, rồi tự động nối (Merge) các Bounding Box lại với nhau. Điều thú vị là thư viện SAHI đóng gói API chung cho RT-DETR qua khai báo `model_type='yolov8'`.

---

## Tổng kết: Bảng đối chiếu Code ↔ Spec

| Thông số kỹ thuật | Code trong notebook | `models_specs.md` | Trạng thái |
|---|---|---|---|
| Phân loại kiến trúc | `RTDETR('rtdetr-l.pt')` | Transformer (Self-Attention) Large | ✅ Khớp |
| Chia Dữ liệu | `random.shuffle` (80% Train / 20% Val) | 80% Train / 20% Val | ✅ Khớp |
| Kích thước ảnh (Resolution)| `imgsz=1280` | Giữ độ phân giải cao cho vật siêu nhỏ| ✅ Khớp |
| Tối ưu hóa bộ nhớ (VRAM) | `batch=2`, `accumulate=4` | Kỹ thuật Gradient Accumulation | ✅ Xuất sắc |
| Thuật toán Tối ưu | `optimizer='AdamW'` | AdamW (Chuẩn công nghiệp cho Transformer)| ✅ Khớp |
| Hạ nhiệt độ LR | `cos_lr=True` | Cosine Annealing | ✅ Khớp |
| Triển khai (Deploy) | Gọi thư viện `sahi` | Slicing Aided Hyper Inference | ✅ Khớp |
