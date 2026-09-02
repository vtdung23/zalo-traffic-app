# KẾ HOẠCH THỰC HIỆN ĐỀ 1: ỨNG DỤNG NHẬN DIỆN BIỂN BÁO GIAO THÔNG (OBJECT DETECTION)

Tài liệu này là bản kế hoạch chi tiết, chuẩn hóa từ A-Z để thực hiện Đề 1, bám sát các yêu cầu từ file PDF gốc của môn học.

**Quyết định cốt lõi:**
1. **Dataset chốt:** Zalo AI Traffic Sign 2020 (Biển báo giao thông đường phố Việt Nam).
2. **Môi trường:** Kaggle Notebooks (để Train AI) và Hugging Face Spaces (để chạy Web App).
3. **Mô hình (Models):** YOLOv8/v11, Faster R-CNN, và DETR.

---

## I. Phân tích Yêu cầu & Đặc thù của Dataset

### 1. Cấu trúc dữ liệu thô (Raw Data) trên Kaggle
Tôi đã nắm cực kỳ rõ cấu trúc của bộ dữ liệu này. Nó được tổ chức theo chuẩn COCO Format chứ không phải chuẩn YOLO:

```text
/kaggle/input/za-traffic-2020/za_traffic_2020/
├── traffic_train/
│   ├── images/                           # Chứa ~4500 tấm ảnh huấn luyện (.png)
│   └── train_traffic_sign_dataset.json   # TẤT CẢ nhãn bị gộp chung vào 1 file này!
└── traffic_public_test/
    └── images/                           # Chứa ảnh test (không có nhãn để tự chấm)
```

**Bên trong file `train_traffic_sign_dataset.json` có gì?**
Nó chứa 3 danh sách (List) khổng lồ:
1. `categories`: Danh sách Tên và ID của 7 loại biển báo giao thông.
2. `images`: Danh sách chiều cao (height), chiều rộng (width) và tên file của 4500 tấm ảnh.
3. `annotations`: Chứa hàng chục ngàn tọa độ biển báo dưới dạng `[x_min, y_min, width, height]` (tọa độ góc trên bên trái và chiều ngang/dọc của hộp). Tọa độ này tính bằng pixel tuyệt đối.

### 2. Thách thức cốt lõi của Dataset
- **Đạt yêu cầu của đề:** Dataset chứa ảnh đường phố Việt Nam, số lượng classes > 5.
- **Thách thức cực đại (Vật thể siêu nhỏ):** Các biển báo trong ảnh rất nhỏ (ví dụ chỉ chiếm 19x18 pixel trong một tấm ảnh độ phân giải 1622x626). Đây là bài toán *Small Object Detection*.
  - **Tác động:** Faster R-CNN (vốn giỏi quét chi tiết) sẽ hoạt động khá ổn. Tuy nhiên, YOLO và DETR nếu nhận ảnh nguyên bản sẽ rớt hiệu suất thê thảm do vật thể bị mất tích trong quá trình trích xuất đặc trưng (Feature Extraction).
  - **Giải pháp:** Bắt buộc phải áp dụng kỹ thuật **Image Tiling** (Cắt 1 ảnh to thành 4-6 ảnh nhỏ) hoặc dùng thư viện **SAHI** (Slicing Aided Hyper Inference).

---

## II. Lộ trình thực hiện chi tiết (Step-by-Step)

Dự án chia làm 2 giai đoạn: **Huấn luyện trên Đám mây (Kaggle)** và **Triển khai Web App (Hugging Face Spaces)**.

### Giai đoạn 1: PHÂN TÍCH LOCAL & HUẤN LUYỆN TRÊN ĐÁM MÂY (KAGGLE/COLAB)

#### Bước 1: Phân tích Dữ liệu (EDA) trực tiếp trên máy tính cá nhân
- **Môi trường:** Chạy file `eda_analysis.ipynb` trực tiếp trên máy tính của bạn (Local).
- **Thực thi:** Trích xuất thông tin từ file JSON (Class Balance, Object Density, Aspect Ratio, Spatial Heatmap...) và thuật toán K-Means để xuất ra biểu đồ. Các ảnh biểu đồ tự động lưu vào `reports/charts/` để chép vào file báo cáo.

#### Bước 2: Tiền xử lý dữ liệu (Chạy trên Kaggle Notebook)
- **Tạo Dataset chuẩn format YOLO:** Viết script Python chuyển từ tọa độ JSON gốc sang format YOLO chuẩn hóa `[class_id, x_center, y_center, w, h]`.
- **Chia tập Train/Val:** Tách ngẫu nhiên tập Train thành 80% (train) và 20% (val).
- **Giải quyết vùng mù & mất cân bằng:** Gài hàm **BBox-Safe Augmentation** (chống cắt nhầm lề phải ảnh) và cấu hình lại trọng số để trị dứt điểm mất cân bằng class.

#### Bước 3: Huấn luyện 3 mô hình (Đỉnh cao Tối ưu)
- **Model 1: YOLOv8 hoặc v11 (Đại diện One-stage)**
  - Tải folder YOLO `.txt`.
  - **Tối ưu:** Huấn luyện ở độ phân giải siêu cao `imgsz=1280` kết hợp mở **P2 Layer** và **Focal Loss** để mô hình "mở to mắt" nhìn vật thể nhỏ.
- **Model 2: Faster R-CNN (Đại diện Two-stage)**
  - Nạp thẳng file COCO JSON gốc bằng Custom DataLoader (PyTorch).
  - **Tối ưu:** Truyền 5 kích thước chuẩn từ thuật toán **K-Means EDA** vào thẳng hàm `AnchorGenerator` thay cho đồ mặc định.
- **Model 3: RT-DETR (Đại diện Transformer)**
  - Nạp dữ liệu YOLO txt (Dùng chung bộ thư viện Ultralytics với YOLO).
  - Đảm bảo Real-time, khắc phục tốc độ rùa bò của DETR truyền thống.

> ⚠️ **LUẬT SINH TỒN TRÊN KAGGLE:** 
> Mỗi tuần Kaggle cho bạn 30 tiếng dùng GPU miễn phí. Hãy lưu file trọng số (model weights) `best.pt` liên tục vào thư mục `/kaggle/working/` và tải về máy tính để phòng hờ trường hợp hết giờ chạy.

#### Bước 4: Đánh giá & Suy luận (Inference)
- Quét qua tập Test hoặc ảnh thực tế.
- **Vũ khí tối thượng:** Bọc các mô hình bên trong thư viện **SAHI (Slicing Aided Hyper Inference)**. SAHI sẽ tự động cắt khung ảnh trượt đè lấp (Overlapping Sliding Window) để zoom cận cảnh biển báo cho mô hình nhận diện, sau đó dùng NMS gộp lại, bóp nát tỷ lệ trượt của bài toán Small Object!
- Lập bảng so sánh và phân tích sự đánh đổi giữa Tốc độ và Độ chi tiết.

---

### Giai đoạn 2: LÀM VIỆC TRÊN HUGGING FACE SPACES

Sau khi có kết quả từ Giai đoạn 1, ta chọn ra mô hình có kết quả tổng hợp tốt nhất (Ví dụ YOLOv8) để làm sản phẩm cuối.

#### Bước 5: Triển khai Web App ứng dụng
- **Tải model:** Tải duy nhất file trọng số `best.pt` của mô hình chiến thắng từ Kaggle về máy tính (dung lượng chỉ khoảng vài chục MB).
- **Lập trình giao diện:** Sử dụng thư viện `Streamlit` hoặc `Gradio` viết mã nguồn tạo trang Web và đẩy code lên **Hugging Face Spaces**.
- **Tính năng Web:** 
  1. Người dùng truy cập đường link Public, bấm nút Upload 1 tấm ảnh đường phố.
  2. Load file `best.pt` để dự đoán (Chạy trực tiếp trên server miễn phí của Hugging Face).
  3. Trả về kết quả là tấm ảnh đã được vẽ các Bounding Box xung quanh biển báo và tên biển báo.

---

## III. Các Script / File Code cụ thể cần phải viết

Để hoàn thành Đề 1, danh sách các công việc thực hành (Coding) mà chúng ta cần viết theo thứ tự là:

- [ ] 1. Khởi tạo Kaggle Notebook đính kèm sẵn dataset Zalo AI.
- [ ] 2. Script Data EDA (Phân tích dữ liệu JSON và vẽ biểu đồ).
- [ ] 3. Script Chuyển đổi định dạng: COCO JSON sang YOLO TXT.
- [ ] 4. Script Tiền xử lý: Cắt nhỏ ảnh (Image Tiling) để xử lý biển báo siêu nhỏ.
- [ ] 5. Notebook (File .ipynb) để Train và Đánh giá YOLO.
- [ ] 6. Notebook (File .ipynb) để Train và Đánh giá Faster R-CNN.
- [ ] 7. Notebook (File .ipynb) để Train và Đánh giá DETR.
- [ ] 8. Source code `app.py` (Streamlit) để deploy Web App lên Hugging Face Spaces.

---

## IV. Cấu trúc thư mục chuẩn (Project Structure)

Để quản lý code một cách chuyên nghiệp (tránh tình trạng code vứt lung tung không biết file nào chạy ở đâu), toàn bộ dự án trên máy tính/GitHub của bạn cần được tổ chức theo cấu trúc sau:

```text
Traffic-Sign-Detection-ZaloAI/
│
├── data_preparation/           # Chứa các file kịch bản (Script) xử lý dữ liệu
│   ├── convert_coco_to_yolo.py # Chuyển đổi nhãn COCO gốc sang định dạng YOLO TXT
│   ├── image_tiling.py         # Code cắt nhỏ ảnh (Tiling) cho vật thể nhỏ
│   └── eda_analysis.ipynb      # Phân tích biểu đồ dữ liệu
│
├── notebooks/                  # Các file này SẼ ĐƯỢC UPLOAD LÊN KAGGLE ĐỂ TRAIN
│   ├── train_yolov8.ipynb      # Notebook chứa code tải data, train và test YOLO
│   ├── train_faster_rcnn.ipynb # Notebook chứa code train Faster R-CNN
│   └── train_detr.ipynb        # Notebook chứa code train DETR
│
├── web_app/                    # Các file này CHỈ ĐỂ ĐẨY LÊN HUGGING FACE SPACES
│   ├── app.py                  # Source code chính của giao diện Web
│   ├── utils.py                # Chứa các hàm hỗ trợ vẽ ảnh, xử lý dự đoán
│   ├── requirements.txt        # Danh sách thư viện cần thiết để Hugging Face cài đặt
│   └── weights/                
│       └── best_yolov8.pt      # File trọng số bạn tải từ Kaggle về đặt vào đây
│
├── reports/                    # Thư mục lưu kết quả để viết Báo cáo
│   ├── comparison_table.csv    # Bảng csv so sánh mAP, Speed 3 mô hình
│   └── charts/                 # Lưu các hình ảnh chụp biểu đồ Loss, Confusion Matrix
│
└── ke_hoach_de_1.md            # File kế hoạch bạn đang xem
```


---


---

## VII. Đặc tả (Specification) chi tiết các Mô hình

### 1. Specification Mô hình 1 - YOLOv8
- **File thực hiện:** `notebooks/train_yolov8.ipynb`
- **Phong cách Code:** Tuân thủ chặt chẽ `AGENTS.md` (Viết code mộc mạc kiểu sinh viên, comment tiếng Việt cho các khối logic).
- **Quy trình chi tiết trong file Notebook (Chạy trên Kaggle):**
  1. **Khối 1 (Tiền xử lý):** Sử dụng hàm `json.load()` đọc file `train_traffic_sign_dataset.json`. Tạo thư mục format YOLO. (Tách ngẫu nhiên 80% train, 20% val).
  2. **Khối 2 (Cấu hình Nâng cao - M2.3 & M1.1):** 
     - Sửa file cấu hình `yolov8.yaml` để **kích hoạt P2 Layer** (tăng khả năng bắt vật thể siêu nhỏ). 
     - Code tự động tạo file `dataset.yaml` chứa đường dẫn data.
  3. **Khối 3 (Huấn luyện):** Cài đặt thư viện `ultralytics`. Dùng lệnh `model.train()` với cấu hình: `epochs=50`, `imgsz=1280` (trị vật thể nhỏ), `batch=8`. (Lưu ý gài các tham số phạt class thiểu số để kích hoạt **Focal Loss** xử lý mất cân bằng dữ liệu).
  4. **Khối 4 (Suy luận - M2.1):** Cài đặt thư viện **SAHI** để tiến hành cắt ảnh trượt đè lấp (Overlapping Sliding Window) lúc dự đoán (Inference), tối đa hóa mAP.
- **Lưu ý triển khai:** Khối 1 và Khối 2 xử lý việc tạo folder và json ngay trên Kaggle RAM, tránh việc phải upload hàng vạn file txt từ máy cá nhân lên gây lỗi mạng.

### 2. Specification Mô hình 2 - Faster R-CNN
- **File thực hiện:** `notebooks/train_faster_rcnn.ipynb`
- **Môi trường chạy:** Google Colab (GPU T4 miễn phí) hoặc Kaggle.
- **Phong cách Code:** Code bằng PyTorch thuần (`torchvision`), comment tiếng Việt chi tiết cách xây dựng hàm DataLoader và vòng lặp.
- **Quy trình chi tiết trong file Notebook:**
  1. **Khối 1 (Tải dữ liệu):** Sử dụng Kaggle API để tải trực tiếp dataset Zalo AI về môi trường Cloud.
  2. **Khối 2 (Tạo lớp Dataset):** Code class `ZaloTrafficDataset`. Hàm `__getitem__` sẽ đọc thẳng tọa độ từ file COCO JSON gốc, xử lý bounding box thành cấu trúc tensor dictionary `{boxes, labels}`.
  3. **Khối 3 (Khởi tạo Mô hình & M3.1 Anchor K-Means):** Import `fasterrcnn_resnet50_fpn`. **QUAN TRỌNG:** Phải gọi hàm `AnchorGenerator` của Torchvision, nạp 5 kích thước hộp vuông (đã tính bằng K-Means ở bước EDA) để thay thế bộ Anchor hình chữ nhật mặc định của COCO. Sau đó mới thay lớp phân loại mặc định thành 8 classes (7 biển báo + 1 nền).
  4. **Khối 4 (Huấn luyện):** Thiết lập `SGD Optimizer`. Viết vòng lặp `for epoch in range(epochs):` thủ công, nạp batch vào GPU, tính toán Loss, gọi `backward()` và cập nhật trọng số. Lưu trọng số `faster_rcnn_best.pth` nếu Loss giảm.

### 3. Specification Mô hình 3 - DETR (RT-DETR)
- **File thực hiện:** `notebooks/train_rtdetr.ipynb`
- **Môi trường chạy:** Kaggle Notebook (có thể chạy sau khi YOLO train xong).
- **Kiến trúc:** Nhóm quyết định sử dụng **RT-DETR** (Real-Time DETR) thay cho DETR truyền thống. Sự thay đổi này mang tính chiến lược vì RT-DETR khắc phục điểm yếu chí mạng của Transformer là "tốc độ chậm", giúp mô hình có thể đạt Real-time như YOLO nhưng mang trong mình sự chính xác của Transformer.
- **Quy trình chi tiết trong file Notebook:**
  1. **Khối 1 (Chuẩn bị):** Tái sử dụng lại toàn bộ cấu trúc folder YOLO (images, labels) đã được tạo ra từ file `train_yolov8.ipynb`. (Bởi vì RT-DETR của Ultralytics hỗ trợ đọc chung format với YOLO).
  2. **Khối 2 (Huấn luyện):** Load mô hình `rtdetr-l.pt` (bản Large). Gọi hàm `model.train()` với `imgsz=1280` và `epochs=50`.
- **Lưu ý:** Việc sử dụng chung thư viện `ultralytics` cho 2 mô hình (YOLO và RT-DETR) là một điểm sáng, giúp đơn giản hóa pipeline tiền xử lý, tránh viết code rườm rà dễ sinh lỗi.
