# Giải thích chi tiết mã nguồn: train_yolov8.ipynb

Tài liệu này giải thích chi tiết từng dòng code trong file `train_yolov8.ipynb`, tập trung vào việc đối chiếu và làm rõ cách các dòng code này hiện thực hóa các thông số kỹ thuật đã được định nghĩa tại `models_specs.md` (Đặc biệt là phần **1. YOLOv8s-P2 (Custom Architecture)**).

## Cell 1: Khai báo thư viện và Chuẩn bị dữ liệu

```python
import os
import json
import glob
import shutil
import random
from tqdm import tqdm
```
- **Ý nghĩa**: Import các thư viện cần thiết. `os`, `glob`, `shutil` dùng để xử lý file/thư mục. `json` để đọc file dữ liệu gán nhãn gốc của Zalo AI. `random` để chia tập train/val ngẫu nhiên. `tqdm` để hiển thị thanh tiến trình.

```python
# Thiết lập các đường dẫn thư mục
# Tìm file json tự động trong /kaggle/input/ để tránh lỗi sai đường dẫn
json_paths = glob.glob('/kaggle/input/**/train_traffic_sign_dataset.json', recursive=True)
if not json_paths:
    raise FileNotFoundError("Không tìm thấy file JSON. Vui lòng kiểm tra lại dataset đã add vào Kaggle chưa!")

json_path = json_paths[0]
image_dir = os.path.dirname(json_path).replace('traffic_train', 'traffic_train/images')

# Nếu cách nối chuỗi trên không tìm thấy thư mục ảnh, dùng lệnh tìm kiếm thư mục cho an toàn
if not os.path.exists(image_dir):
    img_dirs = glob.glob('/kaggle/input/**/traffic_train/images', recursive=True)
    if img_dirs:
        image_dir = img_dirs[0]

dataset_dir = '/kaggle/working/dataset'
```
- **Ý nghĩa**: Code tìm kiếm tự động đường dẫn đến file JSON chứa nhãn dán của Zalo AI và thư mục chứa ảnh trên môi trường Kaggle. Đích đến của dữ liệu chuẩn hóa sẽ là thư mục `/kaggle/working/dataset`.

```python
# Tạo cấu trúc thư mục YOLO chuẩn (chia 2 tập train và val)
for split in ['train', 'val']:
    os.makedirs(f'{dataset_dir}/{split}/images', exist_ok=True)
    os.makedirs(f'{dataset_dir}/{split}/labels', exist_ok=True)
```
- **Ý nghĩa**: YOLO yêu cầu cấu trúc thư mục nghiêm ngặt gồm `images` và `labels` nằm riêng biệt cho từng tập `train` và `val`. Vòng lặp này tạo ra khung thư mục đó.

```python
# Bắt đầu đọc file JSON và nạp dữ liệu
print("Đang đọc dữ liệu JSON...")
with open(json_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

images_info = {img['id']: img for img in data['images']}

# Gom các bounding box lại theo từng bức ảnh để tiện cho việc xử lý về sau
img_to_anns = {}
for ann in data['annotations']:
    img_id = ann['image_id']
    if img_id not in img_to_anns:
        img_to_anns[img_id] = []
    img_to_anns[img_id].append(ann)
```
- **Ý nghĩa**: Nạp file JSON theo định dạng COCO gốc của Zalo. Tổ chức lại dữ liệu bằng cách gom nhóm tất cả các nhãn (bounding box) tương ứng với từng ID ảnh (vào dictionary `img_to_anns`).

```python
# Phân chia tập dữ liệu ngẫu nhiên thành Train (80%) và Val (20%)
image_ids = list(images_info.keys())
random.seed(42)
random.shuffle(image_ids)
split_idx = int(len(image_ids) * 0.8)
train_ids = image_ids[:split_idx]
val_ids = image_ids[split_idx:]

print(f"Tổng số ảnh: {len(image_ids)}. Train: {len(train_ids)}, Val: {len(val_ids)}")
```
- **Ý nghĩa**: Lấy danh sách toàn bộ ID ảnh, trộn ngẫu nhiên (với seed = 42 để đảm bảo kết quả có thể tái lập) và cắt 80% đưa vào tập huấn luyện (Train), 20% đưa vào tập xác thực (Validation).

---

## Cell 2: Chuyển đổi định dạng từ COCO sang YOLO

```python
# Hàm chuyển đổi tọa độ từ COCO sang chuẩn YOLO
def convert_coco_to_yolo(bbox, img_width, img_height):
    x_min, y_min, w, h = bbox
    x_center = (x_min + w / 2) / img_width
    y_center = (y_min + h / 2) / img_height
    w_norm = w / img_width
    h_norm = h / img_height
    return x_center, y_center, w_norm, h_norm
```
- **Ý nghĩa**: Định dạng COCO lưu bounding box dưới dạng `[x_min, y_min, width, height]` tính bằng pixel thực tế. YOLO yêu cầu định dạng `[x_center, y_center, width, height]` và phải được chuẩn hóa (chia cho chiều rộng/cao của ảnh) về khoảng `[0, 1]`. Hàm này thực hiện phép toán quy đổi đó.

```python
# Thực hiện vòng lặp cắt chuyển đổi và copy ảnh vào đúng thư mục
print("Đang xử lý và tạo file .txt cho YOLO...")

def process_split(ids, split_name):
    for img_id in tqdm(ids, desc=f"Processing {split_name}"):
        img_info = images_info[img_id]
        img_filename = img_info['file_name']
        img_width = img_info['width']
        img_height = img_info['height']
        
        src_img_path = os.path.join(image_dir, img_filename)
        dst_img_path = os.path.join(dataset_dir, split_name, 'images', img_filename)
        
        # Nếu ảnh thực sự tồn tại thì copy sang thư mục YOLO và tạo file nhãn
        if os.path.exists(src_img_path):
            shutil.copy(src_img_path, dst_img_path)
            
            txt_filename = img_filename.rsplit('.', 1)[0] + '.txt'
            txt_path = os.path.join(dataset_dir, split_name, 'labels', txt_filename)
            
            with open(txt_path, 'w') as f_txt:
                if img_id in img_to_anns:
                    for ann in img_to_anns[img_id]:
                        # YOLO class index bắt đầu từ 0, nên id gốc (1->7) phải trừ đi 1
                        class_id = int(ann['category_id']) - 1
                        x_c, y_c, w_n, h_n = convert_coco_to_yolo(ann['bbox'], img_width, img_height)
                        f_txt.write(f"{class_id} {x_c:.6f} {y_c:.6f} {w_n:.6f} {h_n:.6f}\n")

process_split(train_ids, 'train')
process_split(val_ids, 'val')

print("Hoàn tất quá trình chuẩn bị dữ liệu!")
```
- **Ý nghĩa**: Chạy qua từng ảnh trong tập Train/Val, copy file ảnh vật lý sang thư mục của YOLO. Quan trọng nhất là ánh xạ từng class ID (giảm đi 1 do YOLO đếm từ 0) và ghi tọa độ box mới (đã quy đổi) ra file `.txt` tương ứng với mỗi ảnh.

---

## Cell 3: Tạo cấu hình Dataset YOLO (dataset.yaml)

```python
# Tạo file cấu hình dataset.yaml cho mô hình YOLO đọc
yaml_content = f"""
path: {dataset_dir}
train: train/images
val: val/images

# Danh sách 7 loại biển báo (class_id tương ứng từ 0 tới 6)
names:
  0: No entry
  1: No parking / waiting
  2: No turning
  3: Max Speed
  4: Other prohibition signs
  5: Warning signs
  6: Mandatory signs
"""

with open('/kaggle/working/dataset.yaml', 'w', encoding='utf-8') as f:
    f.write(yaml_content.strip())
    
print("Đã tạo xong file cấu hình dataset.yaml")
```
- **Ý nghĩa**: Viết file cấu hình `dataset.yaml` để chỉ điểm cho YOLO biết dữ liệu nằm ở đâu (thư mục `path`, `train`, `val`) và có bao nhiêu class (7 loại biển báo).

---

## Cell 4: Tạo cấu hình Kiến trúc mạng YOLOv8s-P2

```python
# [TỪ EDA E3] TỰ ĐỘNG TẠO KIẾN TRÚC YOLOv8-P2 CHO VẬT THỂ SIÊU NHỎ
import yaml
from ultralytics import YOLO

p2_yaml_content = """
# Ultralytics YOLO 🚀, AGPL-3.0 license
# YOLOv8-p2 architecture
nc: 7  # number of classes
...
head:
...
  - [[18, 21, 24, 27], 1, Detect, [nc]]  # Detect(P2, P3, P4, P5)
"""
```
- **Ý nghĩa**: Dòng này hiện thực hóa **Mục 1.1 Thông số Mạng** trong file `models_specs.md`. Ở mạng YOLOv8 mặc định chỉ có 3 đầu ra nhánh (P3, P4, P5). Việc code lại file YAML ở đây nhằm mở rộng nhánh **P2 Layer (Stride 4)**. Nhánh P2 được nối từ lớp nông của Backbone (Backbone P2), ghép với Head qua C2f layer để phát hiện vật thể cực kỳ li ti (xsmall). Kiến trúc Custom Head sẽ có đoạn `[[18, 21, 24, 27], 1, Detect, [nc]]` xuất ra dự đoán từ 4 nhánh P2, P3, P4, P5. Nhờ đó, ma trận đặc trưng khổng lồ `320x320` ở nhánh P2 (ứng với đầu vào `1280x1280`) được bảo toàn trọn vẹn, không bị mất mát thông tin.

---

## Cell 5: Huấn luyện Mô hình với Hyperparameters đúc kết

```python
from ultralytics import YOLO

# Khởi tạo mô hình kiến trúc P2
model = YOLO('/kaggle/working/yolov8s-p2.yaml')
model.load('yolov8s.pt') # Load pretrained để học nhanh
```
- **Ý nghĩa**: Khởi tạo mạng YOLOv8 chưa có trọng số theo kiến trúc P2 tùy chỉnh ở trên. Lệnh `.load('yolov8s.pt')` mượn lại các trọng số chập (convolution weights) từ mô hình YOLOv8s đã train trên tập dữ liệu chung nhằm sử dụng phương pháp Transfer Learning giúp mô hình hội tụ nhanh hơn.

```python
# Thiết lập Hyperparameters khổng lồ đúc kết từ EDA
results = model.train(
    data='/kaggle/working/dataset.yaml',
    epochs=50,
    imgsz=1280,         # [E3] High-res bảo toàn pixel vật thể nhỏ
```
- **Ý nghĩa**: `epochs=50` (số vòng lặp, theo Mục 1.2), `imgsz=1280` (Độ phân giải đầu vào, theo Mục 1.2). Độ phân giải khổng lồ 1280px đảm bảo rằng khi ảnh bị nén qua nhiều lớp chập, các biển báo li ti vẫn còn ít nhất vài pixel để mô hình học.

```python
    batch=8,
    max_det=50,         # [E1, E2] Tối ưu luồng NMS, tăng tốc luồng xử lý
    iou=0.6,            # [E2] Giữ biển báo đứng cạnh nhau
```
- **Ý nghĩa**: `batch=8` (Kích thước lô, theo Mục 1.2). `max_det=50` (Tham số dọn rác, theo Mục 1.2) - hạ thấp số vật thể tối đa xuống 50 để web không bị quá tải khi xử lý JSON. `iou=0.6` nới lỏng mức gộp hộp (NMS) để không vô tình xóa mất các biển báo đứng kề sát nhau trên thực tế.

```python
    optimizer='AdamW',  # [M4.4] Lịch trình học thuật
    cos_lr=True,        # [M4.4] Hạ nhiệt độ Learning Rate bằng Cosine Annealing
```
- **Ý nghĩa**: Hiện thực hóa Mục 1.2. `AdamW` là thuật toán tối ưu chuẩn xác, chống kẹt vào minima cục bộ. `cos_lr=True` kích hoạt lịch trình Cosine Annealing, độ học (Learning Rate) sẽ hạ êm dần như hình sin giúp trọng số hạ cánh xuống đáy vực hàm Loss tối ưu mượt mà.

```python
    # --- Kỹ thuật Hàm Loss ---
    fl_gamma=2.0,       # [E1] Kích hoạt Focal Loss trị Imbalanced Data
    cls=2.0,            # [M4.1] Tăng cls_gain, ép soi kỹ hình vẽ bên trong biển báo
    box=1.0,
```
- **Ý nghĩa**: Hiện thực hóa Mục **1.4 Cơ chế Hàm độ lỗi**.
  - `fl_gamma=2.0`: Kích hoạt **Focal Loss** - đây là "vũ khí tối thượng" giải quyết hiện tượng chênh lệch dữ liệu (Imbalanced Data).
  - `cls=2.0`: Nhân đôi sức nặng của **Classification Loss**. Ép mạng nơ-ron phải săm soi kỹ họa tiết bên trong lòng biển báo (vì các biển cấm viền đỏ cực giống nhau).

```python
    # --- Augmentation & Khắc phục Center Bias ---
    mosaic=1.0,         # Trộn 4 ảnh
    degrees=10.0,       # Xoay nhẹ
    translate=0.2,      # [E4] Random Shift văng biển báo ra mép ảnh phá Center Bias
```
- **Ý nghĩa**: Hiện thực hóa Mục **1.3 Cơ chế Tiền xử lý & Augmentation**.
  - `mosaic=1.0`: Trộn 4 ảnh làm 1 để ép mạng học cách nhìn xa hơn.
  - `degrees=10.0`: Xoay ngẫu nhiên 10 độ, mô phỏng góc camera nghiêng.
  - `translate=0.2`: **Random Shift** Dịch chuyển điểm ảnh ngang dọc 20%. Đây là thuật toán cốt lõi chống **Center Bias** (lỗi AI chỉ biết nhìn chằm chằm giữa khung hình do hầu hết ảnh trong tập Zalo đều có vật thể ở trung tâm).

```python
    project='/kaggle/working/zalo_traffic',
    name='yolov8s_p2_highres',
    device=0,  # Kích hoạt GPU
)
```
- **Ý nghĩa**: Lưu toàn bộ trọng số (weights) và biểu đồ quá trình học ra thư mục `/kaggle/working/zalo_traffic/yolov8s_p2_highres`. Khởi động quá trình học sử dụng GPU (`device=0`).

---

## Cell 6 & 7: Nén kết quả & Code SAHI hỗ trợ Web App

```python
# TỰ ĐỘNG NÉN TOÀN BỘ KẾT QUẢ ĐỂ TẢI VỀ
!zip -r -q /kaggle/working/yolo_results.zip /kaggle/working/zalo_traffic
```
- **Ý nghĩa**: Đóng gói lại toàn bộ thư mục trọng số sau khi train xong thành 1 file ZIP duy nhất để dễ dàng tải về máy cá nhân từ Kaggle.

```python
# [TỪ E3, M4.2] THUẬT TOÁN DÀNH CHO WEB APP
!pip install -q sahi
from sahi import AutoDetectionModel
...
    detection_model = AutoDetectionModel.from_pretrained(
        model_type='yolov8',
        model_path='/kaggle/working/zalo_traffic/yolov8s_p2_highres/weights/best.pt',
...
```
- **Ý nghĩa**: Khởi tạo thử nghiệm thuật toán **SAHI** (Slicing Aided Hyper Inference). Bằng cách băm nhỏ bức ảnh lúc dự đoán thực tế ra thành từng ô grid và dự đoán trên từng ô nhỏ, sau đó khâu kết quả lại, hệ thống bù đắp rất tốt nhược điểm không thấy được các biển báo siêu nhỏ của ảnh đầu vào độ phân giải lớn khi lên Web App. Đoạn code này minh chứng cho phần kiến trúc kết hợp với kỹ thuật xử lý ảnh sau huấn luyện, sẵn sàng mang vào ứng dụng thực tế.
