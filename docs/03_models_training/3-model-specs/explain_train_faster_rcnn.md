# Giải thích chi tiết mã nguồn: train_faster_rcnn.ipynb

Tài liệu này giải thích chi tiết từng dòng code trong file `train_faster_rcnn.ipynb`, tập trung vào việc đối chiếu và làm rõ cách các dòng code này hiện thực hóa các thông số kỹ thuật đã được định nghĩa tại `models_specs.md` (Đặc biệt là phần **2. Faster R-CNN (Baseline Two-stage)**) và `train_GiaiThich_Hyperparams.md` (phần **3. Mô hình Faster R-CNN**).

## Cell 1: Markdown - Tiêu đề Notebook

```markdown
# Huấn luyện Faster R-CNN cho nhận diện biển báo (Zalo AI 2020)
Notebook này được thiết kế để chạy trên **Kaggle** (GPU P100/T4). Nó sử dụng PyTorch thuần để xây dựng DataLoader và vòng lặp huấn luyện.
```
- **Ý nghĩa**: Ghi chú cho người đọc biết notebook được viết cho môi trường Kaggle sử dụng GPU P100/T4. Khác với YOLO sử dụng framework `ultralytics` chạy trên Kaggle, Faster R-CNN được code bằng **PyTorch thuần túy** nên cần tự tay xây dựng DataLoader, vòng lặp huấn luyện (Training Loop), và logic lưu checkpoint.

---

## Cell 2: Tự động dò tìm Dữ liệu trên Kaggle

```python
import os
import glob

# Tự động tìm đường dẫn dataset trên Kaggle
print("Đang tìm kiếm dataset trên Kaggle...")
json_paths = glob.glob('/kaggle/input/**/train_traffic_sign_dataset.json', recursive=True)
...
```
- **Ý nghĩa**: Thay vì phải gõ lệnh tải dữ liệu thủ công, đoạn code này sử dụng thư viện `glob` để tự động quét toàn bộ phân vùng `/kaggle/input/` (nơi Kaggle mount dữ liệu chỉ đọc) để tìm ra đúng file JSON và thư mục ảnh. Điều này giúp code chạy mượt mà trên mọi Notebook Kaggle mà không cần khai báo đường dẫn cứng (Hardcode).

---

## Cell 3: Import thư viện và Kiểm tra GPU

```python
# Tải thư viện cần thiết
import os
import json
import torch
import torch.utils.data
from PIL import Image
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as F
```
- **Ý nghĩa**: Import toàn bộ thư viện cốt lõi.
  - `torch`, `torch.utils.data`: Framework PyTorch và module DataLoader.
  - `PIL.Image`: Thư viện đọc ảnh chuẩn công nghiệp (Pillow).
  - `torchvision`: Chứa các mô hình detection đã train sẵn (pretrained).
  - `FastRCNNPredictor`: Lớp Head phân loại tùy chỉnh của Faster R-CNN. Đây chính là công cụ để thay đổi số class đầu ra (từ 91 class COCO gốc sang 8 class của ta).
  - `torchvision.transforms.functional as F`: Hàm biến đổi ảnh cấp thấp (convert tensor, normalize...).

```python
# Kiểm tra xem GPU có sẵn sàng không
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f"Đang sử dụng thiết bị: {device}")
```
- **Ý nghĩa**: Tự động phát hiện GPU. Nếu có CUDA (GPU NVIDIA), model sẽ chạy trên GPU cho nhanh. Nếu không, rơi về CPU (sẽ rất chậm). Theo `models_specs.md` Mục 2.4, yêu cầu GPU tối thiểu 12GB VRAM.

```python
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.cluster import KMeans
import numpy as np
```
- **Ý nghĩa**: Import thêm 3 thư viện đặc biệt:
  - `albumentations`: Thư viện Augmentation ảnh chuyên dụng cho Object Detection. Hỗ trợ biến đổi ảnh đồng thời giữ an toàn tọa độ bounding box (bbox-safe). Đây chính là công cụ hiện thực hóa kỹ thuật **BBox-Safe Crop** được mô tả trong `models_specs.md` Mục 2.5 Bước 0.
  - `ToTensorV2`: Chuyển ảnh NumPy sang Tensor PyTorch (kết nối Albumentations với PyTorch).
  - `KMeans` (scikit-learn): Thuật toán phân cụm K-Means dùng để tìm kích thước Anchor Box tối ưu từ dữ liệu thực tế. Đây là hiện thực hóa kỹ thuật **K-Means Anchor 1:1** mô tả tại `models_specs.md` Mục 2.1 và `train_GiaiThich_Hyperparams.md` Mục 3.

---

## Cell 4: Chuỗi Augmentation (Tiền xử lý ảnh)

```python
# Khoi tao chuoi Augmentation
def get_transform():
    return A.Compose([
```
- **Ý nghĩa**: Khai báo hàm trả về một pipeline augmentation. `A.Compose` xếp chồng các phép biến đổi ảnh theo thứ tự tuần tự (Sequential). Mỗi lần DataLoader nạp 1 ảnh, pipeline này sẽ chạy lần lượt từng phép biến đổi.

```python
        # Cắt khuôn 512x512 (Power of 2) phù hợp ảnh gốc 626px
        A.RandomSizedBBoxSafeCrop(width=512, height=512, erosion_rate=0.0, p=0.3),
```
- **Ý nghĩa**: Đây là hiện thực hóa trực tiếp phần **BBox-Safe Crop** mô tả tại `models_specs.md` Mục 2.5 Bước 0 và `train_GiaiThich_Hyperparams.md` Mục 3.
  - `width=512, height=512`: Khuôn cắt cố định 512x512 pixel. Vì ảnh gốc Zalo AI chỉ cao 626px, 512 là giá trị Power of 2 an toàn nhất (1024 sẽ vượt quá chiều cao ảnh gốc). Giá trị Power of 2 giúp tối ưu tính toán trên GPU.
  - `erosion_rate=0.0`: Không cho phép xói mòn (ăn mòn) biên bounding box khi cắt. Tọa độ box được giữ nguyên vẹn.
  - `p=0.3`: Xác suất áp dụng nhát cắt là **30%**. Nghĩa là mỗi lần nạp ảnh, có 30% khả năng ảnh bị cắt 512x512, và 70% giữ nguyên ảnh gốc. Trải qua 10 Epochs, mỗi ảnh sẽ sinh ra khoảng 3 phiên bản bị cắt ở tọa độ khác nhau và 7 lần giữ nguyên. Điều này giúp AI vừa học chi tiết cục bộ (khi cắt), vừa học bối cảnh toàn cục (khi giữ nguyên).
  - **Cơ chế BBox-Safe**: Nếu nhát cắt chém mất >50% diện tích biển báo (do `min_visibility=0.5` ở dưới), hệ thống tự động hủy nhát cắt đó và tung xúc xắc cắt lại chỗ khác. Đảm bảo AI không bao giờ học nhầm "biển báo cụt".

```python
        A.Normalize(),
```
- **Ý nghĩa**: Chuẩn hóa giá trị pixel ảnh từ khoảng `[0, 255]` về khoảng `[0, 1]` (hoặc theo mean/std của ImageNet nếu dùng pretrained). Đây là bước bắt buộc trước khi đưa ảnh vào mạng nơ-ron. Nếu không chuẩn hóa, giá trị pixel 255 sẽ khiến Gradient bùng nổ ngay ở Epoch đầu tiên.

```python
        ToTensorV2(),
```
- **Ý nghĩa**: Chuyển ảnh NumPy (H, W, C) sang Tensor PyTorch (C, H, W). Đây là bước cuối cùng trong pipeline, kết nối đầu ra của Albumentations với đầu vào của PyTorch DataLoader.

```python
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels'], min_visibility=0.5))
```
- **Ý nghĩa**: Cấu hình bảo vệ bounding box trong quá trình augmentation.
  - `format='pascal_voc'`: Tọa độ box ở định dạng `[x_min, y_min, x_max, y_max]` (Pascal VOC). Albumentations sẽ tự động biến đổi tọa độ box theo đúng phép biến đổi ảnh.
  - `label_fields=['labels']`: Chỉ định rằng mảng `labels` đi kèm cùng `bboxes` chứa nhãn class của từng box.
  - `min_visibility=0.5`: **Đây là tham số then chốt** - hiện thực hóa kỹ thuật BBox-Safe mô tả tại `train_GiaiThich_Hyperparams.md`. Nếu sau khi cắt, diện tích còn lại của bounding box < 50% diện tích gốc, box đó bị loại bỏ khỏi batch (không đưa vào train). Ngăn chặn AI học nhầm "cái cuống sắt" thành biển báo.

---

## Cell 5: Dataset, DataLoader và K-Means Anchor

```python
class ZaloTrafficDataset(Dataset):
    ...
```
- **Ý nghĩa**: Khai báo class Dataset tùy chỉnh để đọc file JSON gốc của Zalo AI. 
  - Class này dùng Dictionary `image_annotations` gom nhóm trước tọa độ để truy xuất siêu tốc $O(1)$. 
  - Nó tự động ép hệ tọa độ từ dạng `[x, y, w, h]` (COCO) sang dạng `[xmin, ymin, xmax, ymax]` (Pascal VOC) để tương thích bắt buộc với đầu vào của Faster R-CNN.
  - Tích hợp chuẩn xác module Albumentations (Biến số `transform`) để cắt BBox-Safe Crop.

```python
from torch.utils.data import random_split
def collate_fn(batch):
    return tuple(zip(*batch))
```
- **Ý nghĩa**: Hàm gộp batch tùy chỉnh cho DataLoader. Trong Object Detection, mỗi ảnh có số lượng bounding box khác nhau (ảnh A có 3 biển báo, ảnh B có 7 biển báo). Do đó, ta không thể dùng `collate_fn` mặc định của PyTorch.

```python
full_dataset = ZaloTrafficDataset(root_dir, json_file, transform=get_transform())

# Chia dữ liệu 90% Train, 10% Validation
train_size = int(0.9 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
```
- **Ý nghĩa**: Sử dụng `random_split` để chia toàn bộ dữ liệu thành **90% cho Huấn luyện (Train)** và **10% cho Xác thực (Validation)**. Tập Validation đóng vai trò như "camera giám sát" giúp phát hiện sớm hiện tượng học vẹt (Overfitting).

```python
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=4, shuffle=False, collate_fn=collate_fn)
```
- **Ý nghĩa**: Khởi tạo 2 DataLoader riêng biệt.
  - `batch_size=4`: Hiện thực hóa `models_specs.md` Mục 2.2 - **Batch Size: 4**. Con số này bị khống chế bởi kiến trúc Two-stage cực kỳ ngốn VRAM. Faster R-CNN phải nuôi hàng ngàn Region Proposals ảo trong bộ nhớ, nên nhồi nhiều hơn 4 ảnh sẽ gây lỗi OOM (Out of Memory) như mô tả tại `train_GiaiThich_Hyperparams.md`.

```python
# [TỪ EDA E6] TỰ ĐỘNG CHẠY K-MEANS ĐỂ TÌM ANCHOR BOX
print("Đang phân tích K-Means 5 cụm cho Anchor Box từ tập dữ liệu...")
all_boxes = []
for ann in dataset.coco_data['annotations']:
    w, h = ann['bbox'][2], ann['bbox'][3]
    all_boxes.append([w, h])
```
- **Ý nghĩa**: Bắt đầu hiện thực hóa kỹ thuật **K-Means Anchor** mô tả tại `models_specs.md` Mục 2.1 và `train_GiaiThich_Hyperparams.md` Mục 3. Vòng lặp này quét qua toàn bộ annotations trong dataset, trích xuất chiều rộng (`w`) và chiều cao (`h`) của mỗi bounding box gốc (định dạng COCO: `[x, y, w, h]`). Toàn bộ kích thước box được gom vào danh sách `all_boxes` để chuẩn bị cho thuật toán K-Means.

```python
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
kmeans.fit(all_boxes)
centers = np.sort(kmeans.cluster_centers_, axis=0)
anchor_sizes_kmeans = tuple(int(center[0]) for center in centers)
print(f"5 Kích thước Anchor thu được: {anchor_sizes_kmeans}")
```
- **Ý nghĩa**: Chạy thuật toán **K-Means Clustering** (Machine Learning) trên toàn bộ kích thước biển báo trong dataset Zalo AI.
  - `n_clusters=5`: Gom thành **5 cụm** tương ứng với 5 tầng của FPN (P2, P3, P4, P5, P6) theo mô tả tại `models_specs.md` Mục 2.1.
  - `random_state=42`: Cố định seed để kết quả tái lập được.
  - `n_init=10`: Chạy K-Means 10 lần với tâm cụm khởi tạo khác nhau, lấy kết quả tốt nhất.
  - `np.sort(...)`: Sắp xếp 5 tâm cụm từ nhỏ đến lớn (10px → 133px).
  - Kết quả kỳ vọng theo `models_specs.md`: `(10, 24, 44, 77, 133)`.

```python
# FPN cần 5 tuple riêng biệt cho 5 level
ANCHOR_SIZES = tuple((size,) for size in anchor_sizes_kmeans)
ASPECT_RATIOS = ((1.0,),) * len(ANCHOR_SIZES) # Tỷ lệ 1:1 cho tất cả 5 level
```
- **Ý nghĩa**: Định dạng kết quả K-Means thành cấu trúc mà `AnchorGenerator` của PyTorch yêu cầu.
  - `ANCHOR_SIZES = ((10,), (24,), (44,), (77,), (133,))`: Mỗi tầng FPN chỉ nhận đúng 1 kích thước anchor. Hiện thực hóa chính xác thông số `((10,), (24,), (44,), (77,), (133,))` mô tả tại `models_specs.md` Mục 2.1.
  - `ASPECT_RATIOS = ((1.0,), (1.0,), (1.0,), (1.0,), (1.0,))`: Ép cứng tỷ lệ khung thành **1:1 (Hình vuông tuyệt đối)** cho tất cả 5 tầng. Lý do: Biển báo giao thông Việt Nam (Tròn, Tam giác, Bát giác) có tính đối xứng cao, khung vuông bao phủ hoàn hảo mà không bị rộng/chật. Đây là quyết định thiết kế quan trọng được giải thích kỹ tại `train_GiaiThich_Hyperparams.md` Mục 3 (K-Means Anchor).

---

## Cell 6: Hàm tạo mô hình Faster R-CNN

```python
from torchvision.models.detection.anchor_utils import AnchorGenerator
```
- **Ý nghĩa**: Import class `AnchorGenerator` - công cụ sinh các Anchor Box ảo tại mỗi điểm trên Feature Map. Class này cho phép ta tùy chỉnh kích thước và tỷ lệ anchor thay vì dùng mặc định.

```python
# Hàm tạo mô hình Faster R-CNN với K-Means Anchors
def get_model(num_classes, anchor_sizes, aspect_ratios):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
```
- **Ý nghĩa**: Tải mô hình **Faster R-CNN với Backbone ResNet-50 + FPN** đã được huấn luyện trước trên tập COCO (91 class). Hiện thực hóa `models_specs.md` Mục 2.1: Backbone là ResNet-50, Neck là FPN. Tham số `pretrained=True` áp dụng kỹ thuật **Transfer Learning**: Mượn lại các trọng số tích chập (convolution weights) đã học được từ tập COCO khổng lồ, giúp mô hình hội tụ nhanh hơn rất nhiều so với train từ đầu.

```python
    # Ép K-Means Anchor Generator vào RPN
    anchor_generator = AnchorGenerator(sizes=anchor_sizes, aspect_ratios=aspect_ratios)
    model.rpn.anchor_generator = anchor_generator
```
- **Ý nghĩa**: Đây là thao tác "phẫu thuật" cốt lõi - thay thế bộ Anchor Generator mặc định của RPN bằng bộ K-Means Anchor tùy chỉnh. 
  - `AnchorGenerator(sizes=..., aspect_ratios=...)`: Tạo bộ sinh anchor mới với 5 kích thước từ K-Means và tỷ lệ 1:1.
  - `model.rpn.anchor_generator = ...`: Gán trực tiếp vào mạng RPN. Từ giây phút này, khi RPN trượt cửa sổ 3x3 trên Feature Map, tại mỗi điểm nó sẽ phóng ra 5 khung vuông tối ưu (10x10, 24x24, 44x44, 77x77, 133x133) thay vì 15 khung chữ nhật mặc định. Hiện thực hóa `models_specs.md` Mục 2.1 và 2.5 Bước 2 (RPN).

```python
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
```
- **Ý nghĩa**: Thao tác "phẫu thuật" thứ hai - thay thế đầu phân loại (Classification Head) của mạng.
  - `in_features`: Lấy số chiều đầu vào của lớp phân loại hiện tại (thường là 1024).
  - `FastRCNNPredictor(in_features, num_classes)`: Tạo lớp phân loại mới với `num_classes=8` (7 loại biển báo + 1 background). Mô hình gốc COCO có 91 class, ta thay bằng 8 class cho bài toán Zalo AI.
  - **Lưu ý quan trọng**: Faster R-CNN cần cộng thêm 1 class **background** (nền), khác với YOLO chỉ cần đúng 7 class. Đây là đặc thù của kiến trúc Two-stage: Mạng RoI Head dùng hàm **Softmax** (phân phối xác suất trên tất cả class kể cả nền), trong khi YOLO dùng **Sigmoid** (xác suất độc lập từng class, không cần class nền).

```python
    return model
```
- **Ý nghĩa**: Trả về mô hình đã được tùy chỉnh hoàn chỉnh: Backbone ResNet-50 pretrained + FPN + RPN với K-Means Anchor 1:1 + Head phân loại 8 class.

---

## Cell 7: Vòng lặp Huấn luyện chính (Training Loop)

```python
from tqdm import tqdm
import os

# Cấu hình lưu trữ trên Kaggle
save_dir = '/kaggle/working/faster_rcnn_highres'
os.makedirs(save_dir, exist_ok=True)
```
- **Ý nghĩa**: Khai báo thư viện thanh tiến trình (`tqdm`). Đặt thư mục lưu trữ tại `/kaggle/working/`. Kaggle sẽ giữ lại vĩnh viễn mọi file được lưu ở đây sau khi chạy xong (Save & Run All).

```python
# Chuẩn bị huấn luyện
num_classes = 8  # [QUAN TRỌNG] 7 class biển báo + 1 class nền (background)
model = get_model(num_classes, ANCHOR_SIZES, ASPECT_RATIOS)
model.to(device)
```
- **Ý nghĩa**: 
  - `num_classes = 8`: 7 loại biển báo Zalo AI + 1 class background. Con số này nhất quán với `models_specs.md` Mục 2.1.
  - `model.to(device)`: Đẩy toàn bộ tham số lên GPU.

```python
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
# Thêm bộ hạ tốc StepLR: Đến Epoch thứ 10 thì chia LR cho 10
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
```
- **Ý nghĩa**: 
  - **SGD Momentum** (`lr=0.005`): Cỗ xe lu san phẳng hố Loss, chống lật xe (ổn định hơn AdamW cho mạng Two-stage).
  - **StepLR**: Tự động hạ tốc độ học xuống 10 lần (`gamma=0.1`) khi đến Epoch thứ 10. Giúp mô hình "hạ cánh êm ái" và không bị dao động lố. Khớp hoàn toàn với spec.

```python
num_epochs = 15 
best_val_loss = float('inf')
```
- **Ý nghĩa**: Số vòng lặp là **15**. Khởi tạo `best_val_loss` ở mức cao nhất vô cực để theo dõi kỷ lục Loss mới.

```python
for epoch in range(num_epochs):
    # ==========================
    # PHA 1: TRAINING
    # ==========================
    model.train()  
    train_loss = 0
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
```
- **Ý nghĩa**: Bắt đầu vòng lặp huấn luyện. `model.train()` chuyển mô hình sang chế độ Training. `tqdm` vẽ thanh tiến trình trực quan.

```python
    for images, targets in progress_bar:
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        train_loss += losses.item()
        progress_bar.set_postfix(loss=losses.item())
```
- **Ý nghĩa**: **Tính Loss + Cập nhật trọng số**
  - Đẩy ảnh vào model. PyTorch tự động tính 4 hàm Loss (Objectness, RPN Box, Classification, ROI Box).
  - `losses.backward()`: Chạy Backpropagation tính Gradient.
  - `optimizer.step()`: Cập nhật trọng số.

```python
    avg_train_loss = train_loss / len(train_loader)
```
- **Ý nghĩa**: Tính điểm số Train Loss trung bình của cả epoch.

```python
    # ==========================
    # PHA 2: VALIDATION (Mẹo no_grad với FrozenBatchNorm)
    # ==========================
    val_loss = 0
    with torch.no_grad():
        for images, targets in val_loader:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            val_loss += losses.item()
            
    avg_val_loss = val_loss / len(val_loader)
```
- **Ý nghĩa**: **Mẹo công nghệ cao**. Giữ model ở trạng thái `train()` để ép nó trả về điểm số Loss thay vì tọa độ Bounding Box. Tuy nhiên, ta bọc vòng lặp validation bằng `torch.no_grad()` để chặn dòng Gradient, cấm mô hình học lỏm dữ liệu tập Validation. Vì Faster R-CNN mặc định dùng `FrozenBatchNorm`, thủ thuật này an toàn 100% không bị rò rỉ dữ liệu.

```python
    # ==========================
    # PHA 3: LƯU BEST MODEL & STEP LR
    # ==========================
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        best_path = os.path.join(save_dir, 'faster_rcnn_best.pth')
        torch.save(model.state_dict(), best_path)
        
    last_path = os.path.join(save_dir, 'faster_rcnn_last.pth')
    torch.save(model.state_dict(), last_path)
    
    lr_scheduler.step()
```
- **Ý nghĩa**: **Lưu Best Model thực thụ**. Sau mỗi vòng, nếu Validation Loss giảm xuống mức kỷ lục mới, ta sẽ lưu lại đè lên file `best.pth`. Điều này đảm bảo ta giữ được phiên bản mô hình ở trạng thái hoàn hảo nhất, tránh lấy nhầm mô hình bị Overfitting ở vòng cuối cùng. `lr_scheduler.step()` đếm 1 bước epoch để chuẩn bị hạ tốc.

---

## Cell 8: Tự động Tải về File ZIP

```python
from IPython.display import HTML
!zip -r -q /kaggle/working/faster_rcnn_results.zip /kaggle/working/faster_rcnn_highres
HTML(html_code)
```
- **Ý nghĩa**: Tự động nén thư mục trọng số thành tệp tin `.zip`. Sau đó dùng mã JavaScript chèn thẻ `<a download>` ẩn để ép trình duyệt tự động bật hộp thoại tải file về máy tính cho sinh viên.

---

## Tổng kết: Bảng đối chiếu Code ↔ Spec

| Thông số kỹ thuật | Code trong notebook | `models_specs.md` | Trạng thái |
|---|---|---|---|
| Backbone | `fasterrcnn_resnet50_fpn(pretrained=True)` | ResNet-50 + FPN | ✅ Khớp |
| Chia Dữ liệu | `random_split` (90% Train / 10% Val) | 90% Train / 10% Val | ✅ Khớp |
| LR Scheduler | `StepLR` (step_size=10) | `StepLR` hạ 10 lần ở Epoch 10 | ✅ Khớp |
| Best Model | Lưu theo min `val_loss` (no_grad trick) | Lưu Best Model bằng Val Loss | ✅ Khớp |
| Anchor Generator | K-Means 5 cụm, aspect_ratio = 1.0 | 5 cụm K-Means 1:1 cho FPN | ✅ Khớp |
| num_classes | 8 (7 + 1 background) | 7 nhóm biển báo | ✅ Khớp (R-CNN cần +1 bg) |
| Optimizer | SGD(lr=0.005, momentum=0.9, wd=0.0005) | SGD(lr=0.005, momentum=0.9, wd=0.0005) | ✅ Khớp |
| Batch Size | 4 | 4 | ✅ Khớp |
| Epochs | 15 | 15 | ✅ Khớp |
| Augmentation | RandomSizedBBoxSafeCrop 512x512, p=0.3 | BBox-Safe Crop 512x512 | ✅ Khớp |
| min_visibility | 0.5 | 0.5 | ✅ Khớp |
| RPN Loss | Objectness (BCE) + Box (Smooth L1) | BCE + Smooth L1 | ✅ Khớp |
| RoI Loss | Cross-Entropy + Smooth L1 | Cross-Entropy + Smooth L1 | ✅ Khớp |
