## 5. Giải thích chi tiết các khái niệm EDA trong Object Detection

Để bạn dễ dàng đưa vào báo cáo và trình bày trước hội đồng, dưới đây là giải thích chi tiết cho các thuật ngữ và khái niệm xuất hiện trong phần EDA ở trên:

### 5.1. Object (Vật thể) trong ảnh là gì?
- **Khái niệm**: Trong bài toán Object Detection (Nhận diện vật thể), "Object" không phải là toàn bộ bức ảnh, mà là **một thực thể cụ thể** nằm bên trong bức ảnh đó và được đánh dấu bằng một hộp giới hạn (Bounding Box). 
- **Ví dụ cụ thể**: Trong 1 bức ảnh chụp ngã tư đường, có thể có 3 biển báo giao thông. Khi đó:
  - Số lượng Image (ảnh) = 1
  - Số lượng Object (vật thể/biển báo) = 3. Mỗi object sẽ có tọa độ Bounding Box riêng (x, y, width, height) và nhãn (class) riêng (vd: 1 biển cấm rẽ, 2 biển giới hạn tốc độ).

### 5.2. Co-occurrence Matrix (Ma trận đồng xuất hiện) là gì?
- **Khái niệm**: "Đồng xuất hiện" (Co-occurrence) nghĩa là 2 loại biển báo (class) cùng xuất hiện **chung trong một bức ảnh**. Ma trận đồng xuất hiện là một bảng dạng lưới (grid), trong đó mỗi ô giao nhau giữa hàng A và cột B thể hiện số lượng bức ảnh có chứa **cả biển báo loại A và biển báo loại B**.
- **Ý nghĩa thực tiễn**: Giúp phát hiện các quy luật giao thông. Ví dụ: Biển "Cấm đỗ xe" thường hay xuất hiện cùng biển "Cấm quay đầu" tại các ngã tư hẹp. Nếu mô hình học được mối tương quan ngữ nghĩa (Contextual correlation) này, khi nó nhận diện được biển "Cấm đỗ xe", nó sẽ có xu hướng chú ý hơn để tìm biển "Cấm quay đầu" gần đó.

### 5.3. Class Balance (Độ cân bằng nhãn)
- **Khái niệm**: Đo lường xem số lượng object (biển báo) của mỗi class có xấp xỉ bằng nhau hay không.
- **Ý nghĩa thực tiễn**: Nếu class "Cảnh báo nguy hiểm" chỉ có 50 object, trong khi "Cấm đỗ xe" có tới 5000 object, dữ liệu bị **mất cân bằng (Imbalanced)**. Nếu không xử lý, mô hình học máy sẽ bị "lười" và luôn dự đoán là "Cấm đỗ xe" để dễ đạt độ chính xác cao (Accuracy Paradox), dẫn đến việc bỏ sót các biển báo hiếm nhưng quan trọng.

### 5.4. Images & Object Distribution (Phân bố vật thể trên ảnh)
- **Khái niệm**: Biểu đồ thống kê xem phần lớn các bức ảnh trong dataset chứa bao nhiêu vật thể. Có bao nhiêu ảnh chỉ có 1 biển báo? Bao nhiêu ảnh có 2, 3 hay 10 biển báo cùng lúc?
- **Ý nghĩa thực tiễn**: Giúp ta biết mức độ phức tạp của khung hình. Nếu đa số ảnh chỉ có 1 biển báo, mô hình sẽ dễ hội tụ hơn. Nếu ảnh chứa chằng chịt 10-20 biển báo chồng chéo, ta phải dùng các kỹ thuật Non-Maximum Suppression (NMS) gắt gao hơn để tránh việc mô hình nhận diện trùng lặp 1 biển báo nhiều lần.

### 5.5. Class Sizes & Bounding Box Area (Kích thước vật thể)
- **Khái niệm**: Đo lường diện tích của Bounding Box (Width × Height) so với toàn bộ bức ảnh gốc. 
- **Ý nghĩa thực tiễn**: Được dùng để phân loại bài toán thành Small, Medium hay Large Object Detection. Nếu vật thể quá nhỏ (diện tích < 1%), khi đi qua các tầng tích chập (Convolutional Layers) của YOLO, thông tin pixel của biển báo sẽ bị gộp lại, mờ đi và biến mất hoàn toàn ở các Feature Map cuối cùng. Hiểu được điều này giúp ta quyết định không thu nhỏ ảnh gốc (resize) quá đà khi train.

### 5.6. Spatial Heatmap (Bản đồ nhiệt không gian)
- **Khái niệm**: Tưởng tượng bạn xếp chồng toàn bộ 4500 bức ảnh lên nhau, sau đó chấm một điểm đỏ vào vị trí tâm của mọi biển báo giao thông xuất hiện. Nơi nào có nhiều dấu chấm đỏ chồng lên nhau, nơi đó sẽ "nóng" lên (chuyển sang màu đỏ rực), nơi nào ít sẽ có màu xanh lạnh. Đó là bản đồ nhiệt.
- **Ý nghĩa thực tiễn**: Cho ta biết **vị trí địa lý ưu tiên** của vật thể trên khung hình. Trong dataset này, heatmap sẽ đỏ rực ở nửa trên bên phải của ảnh. Từ đó, ta rút ra kinh nghiệm để tinh chỉnh thuật toán Augmentation: Cấm hệ thống tự động cắt (Crop) bỏ phần trên bên phải của bức ảnh khi sinh dữ liệu huấn luyện, vì như thế là tự cắt đi biển báo.

### 5.7. Chuyển đổi định dạng dữ liệu (COCO -> YOLO)
- **Khái niệm**: Dữ liệu gốc của Zalo AI lưu theo chuẩn COCO với cấu trúc JSON gồm `images`, `annotations`, `categories`. Tuy nhiên, các mô hình YOLO và RT-DETR không đọc trực tiếp JSON này theo cách của COCO; thay vào đó, chúng cần file label `.txt` riêng cho từng ảnh, trong đó mỗi dòng biểu diễn một bbox dưới dạng `[class_id, x_center, y_center, width, height]` được chuẩn hóa theo tỷ lệ ảnh.
- **Lý do phải chuyển đổi**: Đây là bước bắt buộc để đưa dữ liệu từ “formatted dataset” sang “train-ready dataset”. Nếu không chuẩn hóa, các model sẽ nhận nhãn lệch và không hiểu trong ảnh có bao nhiêu pixel, tọa độ ở đâu, hộp bao phủ tương ứng với class nào.
- **Sự khác biệt input giữa các model**:
  - **Faster R-CNN / Torchvision**: đọc trực tiếp annotation dạng COCO, không cần file `.txt` cho từng ảnh, nhưng cần `Dataset` class custom vừa nạp JSON vừa xử lý bbox.
  - **YOLOv8 / RT-DETR**: đòi hỏi mỗi ảnh phải có đồng bộ một file label `.txt` và tọa độ chuẩn hóa theo tỷ lệ ảnh để dễ tính toán IoU, loss và NMS.
- **Giá trị thực tiễn**: Bước này làm cho quá trình training thống nhất và dễ quản lý. Ngoài ra, việc lưu ra thư mục mới `data/yolo_format/images` và `data/yolo_format/labels` giúp tập dữ liệu train sạch, không làm lẫn giữa file gốc và file chuyển đổi.

---

## 6. Giải thích chi tiết Menu Lựa chọn Kỹ thuật Triển khai

Phần này giải thích chi tiết ý nghĩa, lý do lựa chọn và tác dụng thực tế của từng kỹ thuật (EDA và Modeling) được liệt kê trong `eda_phan_tich_va_gop_y.md`. Bạn hãy đọc kỹ để chọn ra "combo" phù hợp nhất cho đồ án của mình.

### 🎯 6.1. Giải thích các kỹ thuật vẽ biểu đồ EDA chuyên sâu

*   **[E1] Tổng quan Phân bố nhãn (Class Balance Dashboard)**:
    *   *So với Dataset Ninja*: Làm giống hệt họ bằng cách lập bảng thống kê đa chiều (Images, Objects, Avg count, Avg area) thay vì chỉ vẽ Bar chart đếm số lượng đơn điệu.
    *   *Giúp gì cho model*: Bảng này cung cấp cái nhìn toàn cảnh, tác động trực tiếp đến việc tinh chỉnh tham số huấn luyện:
        - **Cột "Objects" (Dùng Focal Loss)**: Bình thường hàm loss đánh giá mọi lỗi sai như nhau. Nếu biển Cấm đỗ có 10.000 cái, biển Cấm rẽ chỉ có 100 cái, model sẽ có xu hướng đoán mọi thứ là Cấm đỗ để dễ có điểm cao. Nhìn vào cột Objects bị lệch, ta biết phải bật `Focal Loss`. Hàm này hoạt động theo cơ chế tự động giảm nhẹ hình phạt với class dễ (nhiều data) và "phạt cực kỳ nặng" khi model đoán sai class khó (hiếm data), ép model không được bỏ rơi biển Cấm rẽ.
        - **Cột "Avg count" (Giới hạn tham số `max_det`)**: Các mạng như YOLO có thông số `max_det` (số lượng bounding box tối đa xuất ra trên 1 ảnh, mặc định là 300). Nếu Avg count chỉ báo trung bình 1.5 biển báo/ảnh, ta có thể tự tin hạ `max_det` xuống 30-50. Việc này giúp **tăng trực tiếp chỉ số FPS (Frame Per Second - Tốc độ khung hình/giây)** nhờ giảm tải cực lớn cho thuật toán hậu xử lý **NMS (Non-Maximum Suppression)**. 
          *(Giải thích thêm về cơ chế NMS & FPS)*: Khi YOLO phân tích 1 bức ảnh, nó thường vạch ra hàng trăm bounding box chồng chéo lên nhau quanh 1 vật thể. Thuật toán NMS có nhiệm vụ quét qua toàn bộ các box này, so sánh độ trùng lặp (chỉ số IoU) để giữ lại duy nhất 1 box có điểm tự tin (Confidence score) cao nhất và xóa bỏ các box thừa đi. Quá trình quét và so sánh này rất tốn tài nguyên CPU. Bằng cách ép model chỉ được xuất ra tối đa 50 box (`max_det=50`) ngay từ đầu thay vì 300, NMS sẽ có cực ít dữ liệu phải xử lý. Nhờ vậy, tốc độ phản hồi tổng thể của model (FPS) sẽ nhanh hơn đáng kể, rất quan trọng khi chạy thực tế trên camera hành trình ô tô.
*   **[E2] Phân bố Mật độ vật thể (Object Distribution Heatmap)**:
    *   *So với Dataset Ninja*: Nâng cấp từ biểu đồ cột thông thường thành một lưới Heatmap (Hàng là Class, Cột là Số lượng object 1,2,3...).
    *   *Giúp gì cho model*: Cho thấy mức độ "đông đúc" của từng loại biển báo. Nếu biển báo "Cấm đỗ" thường xuất hiện 3-4 cái trong 1 ảnh (tức là mật độ dày đặc), model sẽ bắt buộc phải tinh chỉnh tham số **IoU (Intersection over Union) threshold** trong hàm NMS.
        *(Giải thích thêm về cách chỉnh IoU Threshold)*: Khi có nhiều biển báo đứng sát nhau (ví dụ cắm chung trên 1 cột điện), các bounding box của chúng sẽ bị đè (chồng chéo) lên nhau. Hàm NMS dùng `IoU threshold` (mặc định thường là 0.45) làm mốc: nếu 2 hộp chồng lên nhau lớn hơn 45%, nó sẽ xóa bớt 1 hộp vì tưởng model dự đoán lặp lại cùng 1 vật. Nhưng vì EDA (biểu đồ E2) báo cho ta biết biển báo thực tế đứng sát nhau rất nhiều, ta phải **tăng chỉ số IoU threshold lên cao (ví dụ 0.65 - 0.7)**. Việc này "ra lệnh" cho model: *"Chỉ xóa hộp khi chúng chồng lên nhau quá 70%, còn nếu chỉ đè 50% thì hãy giữ lại cả hai, vì đó rất có thể là 2 biển báo khác nhau nằm cạnh nhau!"*. Nhờ sự can thiệp này từ EDA, model sẽ không bị xóa nhầm các biển báo hợp lệ.
*   **[E3] Kích thước chi tiết & Tree Map (Class Sizes & Tree Map)**:
    *   *So với Dataset Ninja*: Khôi phục lại trọn vẹn bảng thống kê chi tiết (Min/Max/Avg cho Width, Height, Area) và vẽ biểu đồ Tree Map diện tích.
    *   *Giúp gì cho model*: Chứng minh một cách định lượng đây là bài toán "Small Object Detection" (khi Avg area chỉ loanh quanh 0.1%). Đây là căn cứ khoa học tuyệt đối để kích hoạt các tính năng chuyên trị vật thể nhỏ như **P2 Layer (trong YOLO)** hoặc áp dụng kỹ thuật **SAHI (Slicing Aided Hyper Inference)**.
        *(Giải thích & Tư vấn chi tiết về P2 Layer và SAHI)*:
        - **P2 Layer là gì?** Mặc định, YOLO có 3 đầu ra dự đoán (gọi là P3, P4, P5) để tìm vật thể kích thước Vừa, Lớn, Siêu Lớn. Vì ảnh phải đi qua nhiều tầng tích chập, nó bị thu nhỏ (downsample) nhiều lần, khiến các pixel của biển báo siêu nhỏ bị hòa tan và biến mất. Kích hoạt P2 Layer nghĩa là ta mở thêm 1 đầu ra dự đoán ở tầng nông hơn (khi ảnh chưa bị thu nhỏ quá nhiều), giúp YOLO "nhìn" rõ được các biển báo li ti. 
        - **SAHI là gì?** SAHI không phải là sửa model, mà là kỹ thuật xử lý ảnh đầu vào. Thay vì nhét cả bức ảnh 1622x626 khổng lồ vào model để dự đoán, SAHI sẽ cắt bức ảnh đó thành các ô vuông nhỏ hơn (ví dụ cắt thành các mảng 512x512) và di chuyển lướt qua toàn bộ ảnh. Nhờ cắt nhỏ ảnh, biển báo tự nhiên trở nên "to hơn" một cách tương đối so với cái khung hình 512x512 mới, giúp model nhận diện siêu nét. Sau khi đoán xong các ô nhỏ, SAHI tự động ráp tọa độ lại vào vị trí trên ảnh gốc.
        - **Tư vấn (Nên dùng cái nào?):** Khuyên bạn nên **DÙNG CẢ HAI** nhưng ở 2 thời điểm khác nhau. Khi huấn luyện (Training YOLO), hãy cấu hình bật P2 Layer để model sinh ra trọng số nhạy cảm với vật nhỏ. Khi đem model đi dự đoán thực tế (Testing/Inference), hãy nhúng model đó vào luồng chạy của SAHI. Đây là combo hủy diệt giúp mAP tăng vọt.
        - **Hai mô hình kia (Faster R-CNN, RT-DETR) có cần không?** 
          - *Với P2 Layer*: Không cần. P2 là tên gọi đặc thù của kiến trúc YOLO. Mạng Faster R-CNN bản thân nó đã xài kiến trúc FPN (Feature Pyramid Network) - một cơ chế khai thác đặc trưng đa tầng tự nhiên đã rất mạnh với vật nhỏ rồi. Mạng RT-DETR dùng kiến trúc Transformer (xét mọi pixel đồng thời) nên cũng không có khái niệm P2 Layer.
          - *Với SAHI*: **Vẫn rất CẦN**. SAHI là thuật toán độc lập không phụ thuộc vào model (Model-Agnostic). Tức là bạn bọc YOLO, Faster R-CNN hay RT-DETR vào trong SAHI thì nó đều tự động cắt ảnh ra giùm bạn. Áp dụng chung SAHI cho cả 3 model khi test sẽ tạo ra một môi trường so sánh công bằng nhất cho báo cáo đồ án.
*   **[E4] Bản đồ nhiệt Không gian (Spatial Heatmap)**:
    *   *So với Dataset Ninja*: Tái hiện lại bản đồ nhiệt quét vị trí tâm của biển báo trên toàn bộ 4500 ảnh.
    *   *Giúp gì cho model*: Khám phá ra "vùng mù" và "vùng mật độ cao" (biển báo hay nằm ở lề phải khung hình). Từ đó, ta code các lớp Data Augmentation (như Random Crop/Cutout) một cách thông minh: cấm cắt xén ngẫu nhiên vào góc phải của bức ảnh để không tự hủy dữ liệu huấn luyện.
*   **[E5] Ma trận Đồng xuất hiện (Co-occurrence Matrix)**:
    *   *So với Dataset Ninja*: Tính toán số lần 2 class bất kỳ cùng xuất hiện trong một bức ảnh và vẽ Heatmap Matrix.
    *   *Giúp gì cho model*: Cung cấp Contextual Awareness (Nhận thức ngữ cảnh). Nếu "Biển cảnh báo nguy hiểm" và "Biển hạn chế tốc độ" hay đi liền với nhau, ta có thể dùng kiến thức này để sửa lỗi sai của mô hình khi nó dự đoán các biển báo mờ lân cận.
*   **[E6] Tỷ lệ Khung hình (Aspect Ratio Distribution) - *Phần bổ sung độc quyền***:
    *   *Tại sao phải bổ sung*: Dataset Ninja chỉ cung cấp thông số Width/Height đơn lẻ, nhưng thứ mà mô hình AI thực sự quan tâm là tỷ lệ tỷ đối (Aspect Ratio = Width / Height).
    *   *Giúp gì cho model*: Các kiến trúc mạng Anchor-based (như Faster R-CNN, YOLOv5, YOLOv7) khi bắt đầu nhận diện sẽ tung ra hàng ngàn các "hộp ảo" (Anchor Box) với nhiều hình dáng mặc định (ví dụ: hộp dài như con người tỷ lệ 1:3, hộp dẹt như ô tô tỷ lệ 2:1). 

### ⚙️ 6.2. Giải thích chi tiết các kỹ thuật tối ưu Model (Từ cơ bản đến nâng cao)

**Nhóm M1: Xử lý Mất cân bằng dữ liệu (Imbalanced Data)**
*   **[M1.1] Focal Loss**:
    *   *Lý do & Tác dụng*: Các hàm loss thông thường sẽ đánh giá mọi sai sót như nhau. Nếu biển Cấm đỗ có 10.000 cái, biển Cấm rẽ chỉ có 100 cái, model sẽ lười biếng đoán mọi thứ là Cấm đỗ. Focal Loss tự động hạ thấp "tiền phạt" với các class dễ (số lượng nhiều) và "phạt cực nặng" khi model đoán sai các class khó/hiếm, ép model phải học cho bằng được lớp thiểu số.
*   **[M1.2] Mosaic, MixUp & Copy-Paste Augmentation**:
    *   *Lý do & Tác dụng*: Sinh thêm dữ liệu bằng cách trộn nhiều ảnh (Mosaic/MixUp) hoặc cắt/dán trực tiếp vật thể hiếm sang ảnh nền khác (Copy-Paste). Đặc biệt Copy-Paste là cứu cánh tuyệt đối cho việc ép tỷ lệ cân bằng và tăng chỉ số Recall của các biển báo siêu hiếm.

**Nhóm M2: Xử lý Vật thể siêu nhỏ (Small Objects)**
*   **[M2.1] Tăng độ phân giải Train (High-Res 1280) & Mở P2 Layer (YOLOv8)**:
    *   *Lý do & Tác dụng*: Biển báo 20x20 đi qua mạng CNN (bị thu nhỏ bằng Stride 8) sẽ teo lại thành 1x1 pixel, khiến máy tính mù tịt. Train ở độ phân giải 1280 giúp bảo toàn pixel. Cấu hình mở thêm nhánh P2 (Stride 4) giúp YOLO trích xuất đặc trưng ở tầng nông hơn, bắt được các điểm ảnh li ti này.
*   **[M2.2] SAHI (Slicing Aided Hyper Inference)**:
    *   *Lý do & Tác dụng*: Thay vì nhét ảnh 1600x600 khổng lồ vào dự đoán, SAHI cắt ảnh thành nhiều ô 512x512 quét đè lên nhau. Vật thể nhỏ bỗng chốc trở nên "to" một cách tương đối so với khung hình mới, đẩy độ chính xác mAP vọt lên.

**Nhóm M3: Xử lý Khung hình, Vị trí & Context**
*   **[M3.1] BBox-Safe Augmentation & Random Shift**:
    *   *Lý do & Tác dụng*: Nếu cắt ảnh (Crop) ngẫu nhiên sẽ dễ làm mất nửa biển báo. Thuật toán `BBox-Safe Crop` (min_visibility=0.5) đảm bảo hộp giới hạn luôn được giữ. Hơn nữa, dùng `Random Shift` để hất biển báo từ giữa ảnh văng ra mép lề ép model phải từ bỏ định kiến (Center Bias) và quét mắt toàn diện khung hình.
*   **[M3.2] Nhận thức Ngữ cảnh với RT-DETR (Self-Attention)**:
    *   *Lý do & Tác dụng*: CNN truyền thống bị giới hạn vùng nhìn (Receptive Field). Mạng Transformer như RT-DETR tính toán sự tương quan giữa tất cả các điểm ảnh. Nhờ đó nó học được quy luật: "Biển Cấm ngược chiều luôn đi kèm với biển Hiệu lệnh", giúp nó suy luận được biển báo dù 1 trong 2 cái bị lá cây che khuất.
*   **[M3.3] Thuật toán gom cụm Anchor (K-Means Clustering)**:
    *   *Lý do & Tác dụng*: Các mạng Faster R-CNN dùng anchor mặc định của COCO (hình dẹt, chữ nhật đứng). Dùng K-Means tìm ra kích thước thực tế của tập Zalo (chủ yếu là 1:1 hình vuông) và nạp vào mạng, giúp model hội tụ siêu tốc vì không phải tốn hàng chục epoch để nắn lại hình hộp vuông nữa.

**Nhóm M4: Tinh chỉnh Hậu xử lý & Siêu tham số (Advanced Tuning)**
*   **[M4.1] Tinh chỉnh hệ số Loss (cls_gain, box_gain)**:
    *   *Lý do & Tác dụng*: Biển báo giao thông rất giống nhau về hình học (tròn, viền đỏ), chỉ khác hình bên trong. Ta cần cấu hình hệ số `cls_gain` (Phạt phân loại sai) cao hơn nhiều so với `box_gain` (Phạt trượt tọa độ) để ép model săm soi kỹ vào hình vẽ bên trong biển báo.
*   **[M4.2] Tối ưu NMS (max_det, IoU Threshold, Soft-NMS)**:
    *   *Lý do & Tác dụng*: Hạ `max_det = 50` giúp thuật toán NMS chạy nhanh hơn (tăng FPS). Tăng `IoU Threshold = 0.6` hoặc dùng `Soft-NMS` giúp giữ lại các biển báo đứng sát nhau (không bị hàm NMS xóa nhầm do lầm tưởng là 1 vật).
*   **[M4.3] Test-Time Augmentation (TTA)**:
    *   *Lý do & Tác dụng*: Lúc dự đoán, đưa ảnh gốc, ảnh lật ngang, ảnh phóng to vào model cùng lúc rồi gom kết quả trung bình (Ensemble). Cực kỳ mạnh để thi đấu Kaggle, tăng chắc chắn 1.5 - 2% mAP nhưng bù lại chạy dự đoán chậm gấp 3 lần.
*   **[M4.4] Lịch trình Học thuật (AdamW & Cosine Annealing)**:
    *   *Lý do & Tác dụng*: Khởi đầu với Warmup (LR nhỏ) để chống nổ Gradient khi học các vật thể bé, sau đó hạ nhiệt độ LR từ từ theo đường cong Cosine giúp model tránh hố sâu cục bộ và giảm overfitting.

---

### 🎯 6.3. Hướng dẫn Lựa chọn Kỹ thuật (Chiến lược Thực chiến)

Để giúp bạn tối ưu hóa thời gian và tài nguyên, dưới đây là danh sách đánh giá và phân loại toàn bộ các kỹ thuật tinh chỉnh đã đề cập trong suốt quá trình EDA. Hãy chọn "combo" phù hợp với cấu hình máy của bạn để báo cáo trước Hội đồng.

#### 🌟 1. BẮT BUỘC PHẢI DÙNG (Cốt lõi - Không có là Thất bại)
Đây là các kỹ thuật sống còn. Nếu thiếu chúng, mô hình sẽ sụp đổ trước các vật thể siêu nhỏ và sự mất cân bằng dữ liệu trầm trọng.
- **Focal Loss (E1)**: Bắt buộc bật. Không có nó, class thiểu số như "Cấm Rẽ" sẽ bị phớt lờ, độ chính xác Recall sẽ chạm đáy 0.
- **P2 Layer cho YOLO hoặc K-Means Anchor cho Faster R-CNN (E3, E6)**: Kiến trúc bắt buộc để mô hình học được kích thước vật thể li ti và hình khối vuông 1:1 chuẩn xác của biển báo Việt Nam.
- **BBox-Safe Crop & Random Shift (E4)**: Dùng `Albumentations` để cắt ảnh an toàn (min_visibility=0.5) và dịch chuyển ảnh để phá vỡ "định kiến trung tâm" (Center Bias). Cực kỳ quan trọng để máy không tự sinh ra rác dữ liệu làm nhiễu mô hình.
- **SAHI (Khi suy luận) (E3)**: Lên đồ án Object Detection vật thể nhỏ mà thiếu SAHI là một lỗ hổng vô cùng lớn. Nó là "thần dược" cắt nhỏ ảnh giúp mAP vọt lên mà không cần tốn công train lại model.

#### ⚡ 2. NÊN DÙNG (Lấy điểm tuyệt đối từ Hội đồng)
Các kỹ thuật này chứng minh bạn am hiểu sâu sắc bản chất Toán học và cơ chế vận hành bên trong mô hình, không chỉ xài AI như một "hộp đen".
- **Bộ Augmentation Đa dạng (Mosaic, Copy-Paste, CLAHE) (E0, M1.2)**: Sử dụng Copy-Paste để cắt dán trực tiếp biển báo hiếm sang bối cảnh khác là luận điểm cực mạnh để giải quyết Imbalanced Data thay vì chỉ ỷ lại vào hàm Loss.
- **Loss Multipliers Tuning (cls_gain) (M4.1)**: Việc biết cách tăng tham số phạt phân loại (Class Loss) cao hơn phạt tọa độ (Box Loss) chứng tỏ bạn thấu hiểu đặc thù: Biển giao thông rất giống nhau về viền đỏ bên ngoài, chỉ phân biệt nhờ hình vẽ mờ nhạt bên trong.
- **Tối ưu NMS (Hạ max_det = 50, Tăng IoU = 0.6) (E2, M4.2)**: Phân tích được mật độ biển báo để hạ max_det giúp tăng mạnh tốc độ khung hình (FPS) khi chạy thực tế, chứng tỏ tư duy tối ưu tài nguyên hệ thống rất tốt.
- **Lịch trình Học thuật (AdamW & Cosine Annealing) (M4.4)**: Sử dụng thuật toán tối ưu chuẩn nghiên cứu với Warmup chống nổ Gradient và hạ nhiệt theo đường cong Cosine để tránh hố sâu cục bộ.

#### ⚖️ 3. CÂN NHẮC DÙNG (Dành cho việc thi đấu đua Top)
Các kỹ thuật này mang lại kết quả trên giấy rất đẹp nhưng lại ngốn tài nguyên phần cứng khủng khiếp hoặc không phù hợp để Demo thời gian thực.
- **Test-Time Augmentation - TTA (M4.3)**: Cực tốt để đẩy % mAP lên kịch trần làm báo cáo slide. NHƯNG, nếu hội đồng yêu cầu chạy Demo thực tế trên Webcam, bạn PHẢI TẮT TTA đi, vì nó bắt máy tính phải lật/phóng to ảnh liên tục sẽ kéo FPS tụt thê thảm khiến màn hình giật lag.
- **Multi-Scale Training & High-Res 1280 (E3)**: Ép mô hình học ở nhiều độ phân giải khác nhau (từ bé đến lớn) làm mệt máy, tốn VRAM và kéo dài thời gian train gấp nhiều lần. Chỉ dùng nếu có GPU thật khỏe.
- **Kiến trúc Transformer - RT-DETR (E5)**: Rất xuất sắc trong việc hiểu Ngữ cảnh toàn cục (minh chứng cho bài toán "Biển cấm ngược chiều luôn đi chung biển Hiệu lệnh"), nhưng chạy khá nặng nề. Dùng để làm đối trọng so sánh với YOLO.
- **Soft-NMS (M4.2)**: Khắc phục triệt để việc xóa nhầm các biển báo đứng sát cạnh nhau, nhưng làm chậm tốc độ suy luận đôi chút so với NMS cứng mặc định.

> **Tư vấn CHỐT HẠ:** Hãy dồn 80% công lực vào mô hình **YOLOv8** làm át chủ bài. Bật nhánh **P2**, kết hợp bộ ba thuật toán Augmentation (**Copy-Paste + Bbox-Safe + Focal Loss**) và chốt hạ bằng lớp giáp **SAHI** ở khâu Testing. Thiết lập thêm `max_det=50` để tối ưu FPS. Khung sườn này tạo nên một lộ trình "Giải thích vấn đề -> Tìm phương án tối ưu -> Giải quyết triệt để" cực kỳ hoàn hảo cho cuốn Đồ án tốt nghiệp!
