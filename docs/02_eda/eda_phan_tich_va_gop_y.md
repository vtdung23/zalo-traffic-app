# Phân tích báo cáo "Trầm A.I chính" và Góp ý cho Đồ án Object Detection

## 1. Phân tích đồ án mẫu (Text Classification - Sinh số sao)

Đồ án mẫu thực hiện phân loại văn bản để dự đoán số sao từ bình luận trên Shopee và Tiki. Quy trình họ đã làm rất bài bản, bao gồm:

### Các bước thực hiện:
1. **Thu thập dữ liệu**: Sử dụng Extension cho Shopee và viết script Python (Selenium, Regex) gọi API của Tiki.
2. **Tiền xử lý (Preprocessing)**: 
   - Gom nhóm nhãn (Tích cực > 3 sao, Tiêu cực <= 3 sao).
   - Xóa dữ liệu trùng lặp, xử lý giá trị thiếu (NaN), lọc bỏ rating không hợp lệ.
3. **Khám phá dữ liệu (EDA)**: *Họ có làm EDA rất chi tiết*.
   - Phân bố class (mất cân bằng, lệch về 5 sao).
   - Vẽ Word Cloud (đám mây từ khóa) để tìm từ phổ biến trong đánh giá tích cực/tiêu cực.
   - Phân tích chiều dài bình luận (Comment length) qua từng mức sao và tìm Outliers.
   - **Đặc biệt**: Họ phân tích cả các mẫu bị dán nhãn sai (đánh giá tốt nhưng sao thấp) và dùng mô hình phụ để làm sạch dữ liệu lần 2.
4. **Huấn luyện mô hình**: Triển khai rất nhiều kiến trúc từ dễ đến khó:
   - *Baseline*: Softmax Regression (TF-IDF), SVM.
   - *Deep Learning*: TextCNN (Word2Vec), BiLSTM + Attention.
   - *Transformers*: PhoBERT (thử cả Freeze và Unfreeze), ViBERT, XLM-RoBERTa, mT5-base.
5. **Đánh giá**: Dùng Confusion Matrix, Accuracy, F1-Score, MAE. Vẽ biểu đồ Loss/Accuracy để chỉ ra hiện tượng Overfitting.
6. **Triển khai Ứng dụng**: Web app bằng HTML/Tailwind/JS ở frontend và FastAPI ở backend, deploy lên Hugging Face Spaces.

---

## 2. Góp ý cho đồ án Object Detection hiện tại của bạn

Dựa vào mã nguồn trong thư mục `Traffic-Sign-Detection-ZaloAI`, bạn đã làm được khá nhiều việc (EDA cơ bản, Train YOLOv8, Faster R-CNN, RT-DETR, Web App). 

### Ý tưởng dùng đường Pareto (Pareto Front) để so sánh model
- **Tính đại diện**: Ý tưởng này **cực kỳ xuất sắc và chuyên nghiệp**! Trong Object Detection, sự đánh đổi (trade-off) giữa **Tốc độ (FPS/Inference Time)** và **Độ chính xác (mAP)** là quan trọng nhất. Việc dùng đường viền Pareto để loại các model bị "dominate" là hoàn toàn hợp lý về mặt khoa học.

### Điểm mạnh và điểm yếu của dataset Zalo AI Traffic Sign
- **Điểm mạnh**: Dữ liệu thực tế tại Việt Nam, mang tính ứng dụng cao.
- **Điểm yếu**: Vật thể quá nhỏ (rất nhiều biển báo < 1% khung hình), mất cân bằng dữ liệu, và điều kiện môi trường nhiễu (mưa, nắng, khuất tầm nhìn).

---

## 3. Cấu trúc dữ liệu và Tại sao mình biết rõ?

Mình đã nắm rất rõ cấu trúc của bộ dữ liệu Zalo AI Traffic Sign này vì:
1. **Đọc trực tiếp file code của bạn**: Trong file `eda_analysis.ipynb`, bạn đang parse mảng `data['images']` và `data['annotations']` chứa `category_id` và `bbox`. Đây chính là chuẩn **COCO format** huyền thoại.
2. **Từ trang Dataset Ninja**: Mình cũng vừa chạy lệnh crawl toàn bộ source code của trang web `datasetninja.com/zalo-traffic-sign` để kiểm tra các metadata và biểu đồ mà họ đã tự động generate.

---

## 4. Phân tích chi tiết EDA từ trang Dataset Ninja & Đề xuất cho bạn

Dựa trên nội dung chính xác từ trang **Dataset Ninja**, họ đã cung cấp một bộ công cụ EDA tự động cực kỳ chi tiết. Dưới đây là các phần họ đã làm:

1. **Class balance**: Thống kê số lượng ảnh, số lượng object, trung bình số object/ảnh và diện tích trung bình cho từng class.
2. **Co-occurrence matrix**: Ma trận đồng xuất hiện cho thấy tần suất các cặp biển báo xuất hiện cùng nhau trong một bức ảnh.
3. **Images & Object distribution**: Phân bố số lượng object trên từng bức ảnh (ảnh nào có 1, 2 hay nhiều biển báo).
4. **Class sizes**: Kích thước chi tiết (Area, Height, Width - Min/Max/Avg) của từng loại biển báo, kèm theo Tree map.
5. **Spatial Heatmap**: Bản đồ nhiệt thể hiện sự phân bố vị trí không gian của các biển báo trên khung hình gốc.

### Họ làm vậy đã đủ chưa?
**Trả lời:** Về mặt số lượng biểu đồ và thống kê mô tả (Descriptive Statistics), họ làm **đã quá đủ và rất chi tiết**. 
Tuy nhiên, đối với một đồ án học thuật, nếu bạn chỉ đưa các biểu đồ này vào báo cáo thì **chưa đạt yêu cầu tối đa**. Bởi vì Dataset Ninja là một công cụ tự động, họ không đưa ra các **phân tích định hướng (Prescriptive Analysis)** để giúp tối ưu các kiến trúc model cụ thể mà bạn chọn (YOLOv8, Faster R-CNN, RT-DETR).

### Ta cần làm gì thêm và tại sao?
Chúng ta sẽ **kế thừa** các biểu đồ của Dataset Ninja, nhưng áp dụng format **"Kiềng 3 chân"** (Biểu đồ -> Nhận xét -> Kết luận xử lý model) giống báo cáo mẫu để biến các con số khô khan thành Insight có giá trị. Đồng thời, ta sẽ **bổ sung thêm 1 biểu đồ cốt lõi** mà họ chưa trực tiếp vẽ.

#### A. Kế thừa và Nâng cấp EDA của Dataset Ninja
1. **Phân bố nhãn (Class Balance)**
   - *Dataset Ninja đã làm*: Vẽ bảng thống kê số lượng.
   - *Ta làm thêm*: Bổ sung phần kết luận trong báo cáo: "Do dữ liệu mất cân bằng nặng (warning > 3000 objects, no turning < 600 objects), ta bắt buộc phải cấu hình **Focal Loss** cho YOLOv8 hoặc gán **Class Weights** cho Faster R-CNN để phạt nặng model khi đoán sai class thiểu số, tránh tình trạng bias."
2. **Kích thước Bounding Box (Class Sizes / Area)**
   - *Dataset Ninja đã làm*: Đưa ra tỷ lệ diện tích trung bình (Avg area chỉ ở mức cực nhỏ từ 0.06% - 0.17%).
   - *Ta làm thêm*: Dùng số liệu này để mạnh dạn tuyên bố bài toán thuộc nhóm khó: **Small Object Detection**. Từ đó, đề xuất hướng xử lý bắt buộc: Áp dụng kỹ thuật chia nhỏ ảnh **SAHI (Slicing Aided Hyper Inference)** khi suy luận, hoặc nâng cao độ phân giải đầu vào để các Feature Map cuối cùng không làm mất hoàn toàn pixel của biển báo.
3. **Bản đồ nhiệt (Spatial Heatmap)**
   - *Dataset Ninja đã làm*: Cung cấp heatmap cho thấy vị trí tập trung của biển báo.
   - *Ta làm thêm*: Nhận định rõ Insight từ heatmap: Biển báo giao thông Việt Nam đa số tụ lại ở lề phải hoặc góc trên bên phải khung hình. Kết luận: Khi áp dụng Data Augmentation (như Random Crop, Cutout), phải thiết lập thông số để **tuyệt đối không cắt phạm vào vùng có mật độ cao này**, đảm bảo không tự phá hỏng bộ dữ liệu của mình.

#### B. Phần EDA CHÚNG TA CẦN BỔ SUNG THÊM
1. **Phân bố Tỷ lệ khung hình (Aspect Ratio = Width / Height)**
   - *Tại sao cần thêm*: Dataset Ninja chỉ cho thông số Width, Height riêng lẻ (Min/Max/Avg), nhưng họ không trực tiếp vẽ phổ phân bố tỷ lệ Width/Height. Tuy nhiên, các kiến trúc anchor-based như **Faster R-CNN** lại cực kỳ cần thông số Aspect Ratio này để cấu hình các **Anchor Boxes**.
   - *Giá trị mang lại*: Ta code thêm biểu đồ vẽ phân bố Aspect Ratio để chứng minh: "Đa số biển báo là hình tròn, hình vuông hoặc tam giác đều, nên tỷ lệ Width/Height tập trung dày đặc ở mức 1:1. Dựa vào EDA này, ta tinh chỉnh cấu hình anchor size mặc định của Faster R-CNN (vốn là 1:2 và 2:1) tập trung về 1:1. Thao tác này giúp mạng RPN (Region Proposal Network) dự đoán bounding box khớp nhanh hơn và tăng trực tiếp mAP."

---

## 6. Menu Lựa chọn Kỹ thuật Triển khai

Dưới đây là danh sách các kỹ thuật vẽ biểu đồ EDA và kỹ thuật tối ưu Model tương ứng để bạn cân nhắc đưa vào đồ án:

### 🎯 6.1. Các kỹ thuật vẽ biểu đồ EDA chuyên sâu (Kết hợp Dataset Ninja & Tối ưu)
- **[E1] Tổng quan Phân bố nhãn (Class Balance Dashboard)**: Bảng thống kê toàn diện (giống Dataset Ninja) gồm: Số lượng ảnh, Số lượng object, Số object trung bình/ảnh, và Diện tích trung bình cho từng class. Kết hợp biểu đồ Bar chart.
- **[E2] Phân bố Mật độ vật thể (Object Distribution Heatmap)**: Biểu đồ Heatmap (Class vs. Number of Objects) thể hiện có bao nhiêu ảnh chứa 1, 2, 3... biển báo cho từng class cụ thể (giống Dataset Ninja).
- **[E3] Kích thước chi tiết & Tree Map (Class Sizes & Tree Map)**: Bảng thống kê Min/Max/Avg của Area, Height, Width từng class, kết hợp biểu đồ Tree Map trực quan hóa tỷ trọng kích thước (giống Dataset Ninja).
- **[E4] Bản đồ nhiệt Không gian (Spatial Heatmap)**: Bản đồ nhiệt 2D cho thấy vị trí địa lý của biển báo trên khung hình, vẽ tách biệt cho từng class (giống Dataset Ninja).
- **[E5] Ma trận Đồng xuất hiện (Co-occurrence Matrix)**: Heatmap matrix thể hiện xác suất các cặp biển báo xuất hiện cùng nhau (giống Dataset Ninja).
- **[E6] Tỷ lệ Khung hình (Aspect Ratio Distribution)**: *(Phần bổ sung)* Phổ phân bố tỷ lệ Rộng/Cao của toàn bộ dataset để tối ưu cấu hình mạng.

### ⚙️ 6.2. Các kỹ thuật tối ưu Model dựa trên EDA (Bạn muốn dùng chiêu nào?)

**Nhóm kỹ thuật xử lý "Mất cân bằng dữ liệu" (Từ kết quả E1):**
- **[M1.1] Focal Loss / Class Weights**
- **[M1.2] Mosaic & MixUp Augmentation**

**Nhóm kỹ thuật xử lý "Vật thể siêu nhỏ - Small Objects" (Từ kết quả E2):**
- **[M2.1] SAHI (Slicing Aided Hyper Inference)**
- **[M2.2] High-resolution Training / Zoom-in Augmentation**
- **[M2.3] Tinh chỉnh Feature Map (Ví dụ kích hoạt P2 layer trong YOLOv8)**

**Nhóm kỹ thuật xử lý "Khung hình & Vị trí" (Từ kết quả E3, E4):**
- **[M3.1] Anchor Box K-Means Clustering**
- **[M3.2] Safe Spatial Augmentation (Hạn chế crop vùng lề phải)**

*(Chi tiết giải thích tại sao chọn và công dụng của từng kỹ thuật được trình bày trong file `eda_GiaiThich.md`)*
