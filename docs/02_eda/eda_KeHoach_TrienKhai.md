# KẾ HOẠCH TRIỂN KHAI CODE EDA & TỐI ƯU MODEL

Tài liệu này là "Bản thiết kế thi công" (Blueprint) chứa toàn bộ các hạng mục cần lập trình trong file `eda_analysis.ipynb` và các thao tác cấu hình nâng cao khi huấn luyện mô hình. Kế hoạch này được tổng hợp và nâng cấp từ hệ thống tự động của Dataset Ninja, đảm bảo tính học thuật cao nhất.

---

## PHẦN I: KẾ HOẠCH LẬP TRÌNH BIỂU ĐỒ EDA (Code trong Jupyter Notebook)

Khi viết code Python phân tích file `train_traffic_sign_dataset.json`, chúng ta bắt buộc phải lập trình 7 hạng mục (gồm 1 hạng mục trực quan hóa và 6 biểu đồ chuyên sâu) sau đây:

### [E0] Trực quan hóa Dữ liệu mẫu (Visual Inspection)
- **Mục tiêu code**: Code vòng lặp để vẽ bounding box lên ảnh gốc và cắt riêng vật thể (Crop).
- **Nội dung cần tính**: Lấy ngẫu nhiên 5-10 bức ảnh, dùng tọa độ bbox trong JSON để vẽ khung chữ nhật lên ảnh. Sau đó, cắt riêng (crop) các biển báo này và hiển thị thành 1 thư viện ảnh thu nhỏ (Gallery).
- **Giá trị**: Giúp kiểm tra bằng mắt thường (sanity check) xem dữ liệu JSON có bị lệch tọa độ hay không, và giúp hội đồng thấy được mức độ mờ/nhiễu của biển báo thực tế. (Tương đương tính năng "Images" và "Objects" của Dataset Ninja).

### [E1] Tổng quan Phân bố nhãn (Class Balance Dashboard)
- **Mục tiêu code**: Lập bảng thống kê đa chiều và vẽ biểu đồ Bar chart.
- **Nội dung cần tính**: Đếm số lượng ảnh, số lượng vật thể (object), tính số lượng vật thể trung bình/ảnh, và tính diện tích trung bình cho từng loại biển báo (class).
- **Giá trị**: Cho cái nhìn toàn cảnh về độ mất cân bằng dữ liệu và mật độ biển báo.
- 💡 **Kỹ thuật Model áp dụng sau đó**: Bắt buộc kích hoạt **[M1.1] Focal Loss**, **[M1.2] Mosaic Augmentation**, và hạ **[M1.3] max_det**.

### [E2] Phân bố Mật độ vật thể (Object Distribution Heatmap)
- **Mục tiêu code**: Vẽ lưới Heatmap (Hàng là Class, Cột là Số lượng object 1, 2, 3...).
- **Nội dung cần tính**: Thống kê số lượng bức ảnh chứa đúng 1, 2, 3... biển báo của một class cụ thể.
- **Giá trị**: Cho thấy mức độ "đông đúc" của từng loại biển báo.
- 💡 **Kỹ thuật Model áp dụng sau đó**: Bắt buộc cấu hình tăng **[M3.3] IoU Threshold** của thuật toán NMS để không xóa nhầm biển báo xếp chồng.

### [E3] Kích thước chi tiết & Tree Map (Class Sizes & Tree Map)
- **Mục tiêu code**: Lập bảng Min/Max/Avg và vẽ biểu đồ Tree Map.
- **Nội dung cần tính**: Tính toán Chiều Rộng (Width), Chiều Cao (Height) và Diện tích (Area) của toàn bộ các Bounding Box so với khung hình gốc. Tại đây bổ sung hạng mục mới là **COCO Size Metrics**, trong đó bbox được phân loại theo diện tích pixel thật thành: **Small** (< 32x32 px), **Medium** (32x32 đến 96x96 px), **Large** (> 96x96 px).
- **Giá trị**: Chứng minh định lượng dataset Zalo AI thuộc bài toán "Small Object Detection". Khi phần lớn bbox rơi vào nhóm Small, nghĩa là mô hình cần giữ thông tin chi tiết ở tầng feature map thấp để không mất đối tượng li ti.
- 💡 **Kỹ thuật Model áp dụng sau đó**: Bắt buộc dùng **[M2.1] SAHI**, Train độ phân giải cao **[M2.2] High-Res**, và mở **[M2.3] P2 Layer** cho YOLO. Đây là lý do vì sao chuẩn COCO Size Metrics rất quan trọng với kiến trúc FPN: tầng P2/P3 giúp giữ đặc trưng cho vật thể nhỏ, còn tầng cao hơn thích hợp cho đối tượng lớn hơn.

### [E4] Bản đồ nhiệt Không gian (Spatial Heatmap)
- **Mục tiêu code**: Vẽ Bản đồ nhiệt 2D trên khung ảnh.
- **Nội dung cần tính**: Trích xuất tọa độ $(x\_center, y\_center)$ của tất cả biển báo và chồng chúng lên nhau trên một ma trận ảnh trống để tìm "điểm nóng".
- **Giá trị**: Phát hiện vùng mù và vùng mật độ cao (ví dụ lề phải).
- 💡 **Kỹ thuật Model áp dụng sau đó**: Bắt buộc dùng thư viện Albumentations để gài **[M3.2] BBox-Safe Augmentation (min_visibility=0.5)**.

### [E5] Ma trận Đồng xuất hiện (Co-occurrence Matrix)
- **Mục tiêu code**: Vẽ Heatmap Matrix vuông (N x N class).
- **Nội dung cần tính**: Tính xác suất / số lần 2 loại biển báo bất kỳ cùng xuất hiện trong một bức ảnh.
- **Giá trị**: Phân tích nhận thức ngữ cảnh (Contextual Awareness). Dùng làm luận điểm trong báo cáo để giải thích vì sao model hay nhầm lẫn các biển báo hay đi kèm nhau.

### [E6] Phổ phân bố Tỷ lệ khung hình (Aspect Ratio Distribution)
- **Mục tiêu code**: Vẽ biểu đồ Scatter Plot (Width vs Height) hoặc Histogram.
- **Nội dung cần tính**: Tính tỷ lệ $Width/Height$ của mọi biển báo (Kỳ vọng đa số tập trung ở mức 1:1 do biển tròn/vuông).
- **Giá trị**: Chứng minh tính đặc thù hình dáng của bộ dữ liệu Việt Nam.
- 💡 **Kỹ thuật Model áp dụng sau đó**: Dữ liệu đầu vào bắt buộc để chạy **[M3.1] Anchor Box K-Means Clustering** cho Faster R-CNN.

### [E7] Phân bố độ sáng (Brightness & Illumination)
- **Mục tiêu code**: Dùng toàn bộ tập train, chuyển từng ảnh về ảnh xám (Grayscale), tính cường độ pixel trung bình của từng ảnh rồi vẽ biểu đồ Histogram.
- **Nội dung cần tính**: Tính giá trị brightness trung bình cho toàn bộ tập train, đồng thời quan sát phân phối lệch về tối, trung bình hay sáng quá mức.
- **Giá trị**: Dự đoán điều kiện ánh sáng của tập dữ liệu một cách toàn diện nhất. Vì dataset không lớn, việc lấy toàn bộ ảnh giúp đánh giá chính xác hơn so với sampling ngẫu nhiên và tránh bỏ sót vùng sáng/tối bất thường.
- 💡 **Kỹ thuật Model áp dụng sau đó**: Nếu Histogram bị lệch rõ về tối/sáng, bắt buộc dùng **Color Jitter Augmentation** để tăng độ đa dạng ánh sáng, giúp mô hình robust hơn trước thay đổi điều kiện chiếu sáng. Đây là kỹ thuật quan trọng để tránh model quá phụ thuộc vào không gian ánh sáng của tập train.

### [E8] Chuyển đổi định dạng dữ liệu (COCO -> YOLO)
- **Mục tiêu code**: Tạo bước chuẩn hóa dữ liệu trước khi train. Dữ liệu gốc được lưu theo chuẩn COCO (`images`, `annotations`, `categories`), nhưng các mô hình YOLO/RT-DETR yêu cầu định dạng biểu diễn lại dưới dạng file `.txt` cho từng ảnh.
- **Nội dung cần tính**: Chuyển bộ tọa độ bounding box từ dạng COCO `[x, y, width, height]` (tính theo pixel gốc) sang dạng YOLO `[class_id, x_center, y_center, width, height]` với các giá trị được chuẩn hóa theo kích thước ảnh (`0..1`). Kết quả lưu vào thư mục mới `data/yolo_format/images/` và `data/yolo_format/labels/` để tránh rác trong thư mục gốc.
- **Giá trị**: Giúp chuẩn hóa input cho mọi mô hình. Faster R-CNN thường đọc trực tiếp từ COCO JSON, trong khi YOLO và RT-DETR cần file `.txt` tương ứng với từng ảnh. Nếu bỏ qua bước này, mô hình sẽ không hiểu tọa độ nhãn đúng cách hoặc sẽ nhận số liệu lệch không thể train được.
- 💡 **Khác biệt input giữa các model**:
  - **Faster R-CNN / torchvision**: đọc trực tiếp cấu trúc COCO, thường nhận `bbox` ở dạng pixel và `labels` là category id.
  - **YOLOv8 / RT-DETR**: yêu cầu file `.txt` riêng cho mỗi ảnh, trong đó mỗi dòng là một đối tượng và tọa độ phải chuẩn hóa theo tỷ lệ ảnh.
- **Lưu ý thực tế**: Đây là bước bắt buộc trong pipeline, vì mô hình không thể auto-translate giữa hai định dạng annotation khác nhau. Từ khóa để hiểu là: COCO là định dạng “gốc/đầy đủ”, YOLO là định dạng “train-ready”.

---

## PHẦN II: KẾ HOẠCH TỐI ƯU MÔ HÌNH (Áp dụng khi Train/Test)

Dựa trên kết quả từ Phần I, đây là các "Tuyệt chiêu" bắt buộc phải code vào mô hình để đẩy mAP lên mức tối đa. Bạn TUYỆT ĐỐI không được bỏ sót các kỹ thuật này:

### Nhóm M1: Trị bệnh "Mất cân bằng dữ liệu" (Từ kết quả E1)
- **[M1.1] Kích hoạt Focal Loss**: Dùng hàm loss có trọng số để phạt nhẹ class dễ (nhiều data) và phạt nặng class khó (hiếm data).
- **[M1.2] Mosaic / MixUp Augmentation**: Bật kỹ thuật trộn nhiều ảnh lại với nhau trong code Augmentation để sinh thêm dữ liệu và bối cảnh lộn xộn.
- **[M1.3] Ép tham số `max_det`**: Dựa vào số liệu "Avg count" của E1, hạ thông số `max_det` (số hộp tối đa) của YOLO xuống mức 30-50 để thuật toán NMS chạy cực nhanh, tăng trực tiếp FPS.

### Nhóm M2: Trị bệnh "Vật thể siêu nhỏ" (Từ kết quả E3)
- **[M2.1] Sử dụng SAHI (Inference)**: Cài đặt thư viện `sahi`. Lúc chạy Test, thay vì đưa ảnh gốc vào model, dùng SAHI cắt ảnh thành các ô nhỏ 512x512 quét đè lấp, giúp model soi rõ vật nhỏ. (Áp dụng cho cả 3 model: YOLO, Faster R-CNN, RT-DETR).
- **[M2.2] High-Resolution Training**: Train với kích thước ảnh lớn `imgsz=1280` thay vì 640 để bảo toàn pixel vật thể (Áp dụng YOLO, RT-DETR).
- **[M2.3] Cấu hình P2 Layer (YOLOv8)**: Sửa file `yolov8.yaml` để mở thêm nhánh dự đoán P2 (bị downsample ít, có mắt lưới siêu dày 160x160). Kỹ thuật này sinh ra trọng số cực kỳ nhạy với biển báo li ti. (Chỉ áp dụng YOLO).

### Nhóm M3: Trị bệnh "Khung hình & Vị trí" (Từ kết quả E2, E4, E6)
- **[M3.1] Anchor Box K-Means Clustering (Từ E6)**: Chạy thuật toán K-Means gom cụm (Width, Height) của toàn bộ biển báo để tìm ra $K$ kích thước phổ biến nhất (ví dụ toàn hình vuông 1:1). Nạp các kích thước này vào hàm `AnchorGenerator` của Faster R-CNN thay cho anchor COCO mặc định. Giúp model không tốn Epoch để học cách co giãn hộp.
- **[M3.2] BBox-Safe Augmentation (Từ E4)**: Chuyển sang dùng thư viện `Albumentations`. Khi gọi hàm `RandomCrop`, bắt buộc gài tham số `BboxParams(min_visibility=0.5)`. Nhờ vậy, nếu máy tính cắt nhầm ảnh làm mất > 50% diện tích biển báo, nó sẽ tự động vứt ảnh đó và cắt lại ở chỗ an toàn hơn. Tuyệt đối không sinh ra dữ liệu rác.
- **[M3.3] Chỉnh IoU Threshold (Từ E2)**: Nếu mật độ biển báo dày đặc (hay đứng sát nhau trên cùng 1 cột điện), phải cấu hình tăng chỉ số `IoU threshold` của thuật toán NMS lên 0.65 - 0.70 để tránh việc NMS xóa nhầm các biển báo thật đứng cạnh nhau.

---
*Lưu ý cho Dev: Dùng file này làm Checklist. Bắt đầu từ việc mở Jupyter Notebook và code hoàn thiện 6 biểu đồ ở Phần I.*
