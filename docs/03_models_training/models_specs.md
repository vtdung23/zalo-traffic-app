# THÔNG SỐ KỸ THUẬT MÔ HÌNH (MODEL DATASHEETS)

Tài liệu này đóng vai trò như một **Datasheet tiêu chuẩn công nghiệp**, cung cấp thông số kỹ thuật chi tiết nhất (cấp độ mạng nơ-ron và hàm toán học) cho 3 mô hình được lựa chọn. Những thông số này là vũ khí đắc lực để báo cáo trước Hội đồng khoa học nhằm chứng minh độ hiểu biết sâu sắc về kiến trúc mạng.

---

## 1. YOLOv8s-P2 (Custom Architecture)
**Phân loại:** One-stage Anchor-free Detector
**Mục tiêu thiết kế:** Đạt tốc độ mượt mà nhất cho Web App nhưng vẫn bắt được vật thể cực nhỏ (nhờ P2 Layer).

### 1.1 Thông số Mạng (Architecture)
- **Backbone:** Modified CSPDarknet53. Sử dụng cấu trúc `C2f` (Cross Stage Partial Bottleneck với 2 chập) thay cho `C3` cũ, giúp dòng gradient chảy mượt hơn và giảm thiểu suy hao đặc trưng.
- **Neck:** PANet (Path Aggregation Network). Nối các đặc trưng từ dưới lên trên (Bottom-up) và từ trên xuống (Top-down).
- **Head (Custom):** 
  - Mở rộng nhánh **P2 Layer (Stride 4)**. Ở ảnh `1280x1280`, nhánh P2 xuất ra Feature Map kích thước khổng lồ `320x320` pixel.
  - Hỗ trợ phát hiện ở 4 cấp độ (P2, P3, P4, P5) thay vì 3 cấp độ mặc định.
- **Số lượng tham số (Parameters):** ~11.5 Triệu. (Nhỉnh hơn bản gốc 11.1M do thêm nhánh P2).
- **Khối lượng tính toán (GFLOPs):** ~30 GFLOPs.

### 1.2 Thiết lập Huấn luyện (Training Config)
- **Độ phân giải đầu vào (Resolution):** `1280 x 1280` (High-resolution).
- **Thuật toán Tối ưu (Optimizer):** `AdamW` (Tự động thích ứng Learning Rate và phạt Weight Decay chuẩn xác hơn Adam gốc).
- **Lịch trình LR (Learning Rate Scheduler):** `Cosine Annealing`. Khởi đầu (Warmup) ở mức rất nhỏ, sau đó vọt lên và giảm dần theo đường cong hình sin lượn sóng.
- **Kích thước Lô (Batch Size):** 8.
- **Số vòng lặp (Epochs):** 50.
- **Tham số dọn rác (NMS):** Hạ `max_det = 50` (Giới hạn tối đa 50 vật thể/ảnh để tối ưu luồng xử lý Web) và `iou = 0.6` (Bảo vệ các biển báo cắm sát nhau).

### 1.3 Cơ chế Tiền xử lý & Augmentation (CPU DataLoader)
- **Mosaic (`1.0`):** Kỹ thuật đập vụn 4 bức ảnh và nén vào 1 lưới (Grid). Vô tình thu nhỏ kích thước thật của vật thể, ép mạng P2 Layer phải học cách nhìn xa. Đồng thời hack dung lượng VRAM (Batch 8 mang bối cảnh của 32 ảnh).
- **Random Shift (`translate=0.2`):** Sử dụng ma trận biến đổi Affine để dịch chuyển toàn bộ tọa độ điểm ảnh ngẫu nhiên 20%. Kỹ thuật này sinh ra để triệt tiêu hội chứng **Center Bias** (Khi thống kê cho thấy 67.4% biển báo Zalo tập trung ở chính giữa bức ảnh).
- **Xoay nhẹ (`degrees=10.0`):** Xoay ảnh ngẫu nhiên trong khoảng ±10° để tăng tính đa dạng dữ liệu, mô phỏng góc nghiêng thực tế của camera Dashcam.

### 1.4 Cơ chế Hàm độ lỗi (Loss Functions)
Tổng Loss = $\lambda_1 L_{cls} + \lambda_2 L_{box} + \lambda_3 L_{dfl}$
- **Classification Loss ($L_{cls}$):** Dùng **BCE (Binary Cross-Entropy)** kết hợp với **Focal Loss** (`fl_gamma=2.0`). Trọng số `cls_gain` được ép lên 2.0 để trừng phạt thật nặng các lỗi nhận diện nhầm biển báo có viền đỏ giống nhau.
- **Bounding Box Loss ($L_{box}$):** Sử dụng **CIoU Loss** (Complete IoU). Không chỉ xét diện tích đè lấp, mà còn đo khoảng cách giữa 2 tâm (Center distance) và tỷ lệ khung hình (Aspect Ratio).
- **Distribution Focal Loss ($L_{dfl}$):** Tối ưu hóa xác suất ranh giới mờ của các hộp (Fuzzy boundaries).

### 1.5 Yêu cầu Phần cứng & Hiệu năng
- **Tài nguyên Training:** Yêu cầu GPU tối thiểu 12GB VRAM (như T4, RTX 3060). Thời gian train ~ 3-4 phút / Epoch.
- **Tốc độ Inference:** ~80-100 FPS trên GPU, và ~15-20 FPS nếu chạy thuần CPU (Hoàn toàn vượt chuẩn cho Web App).

### 1.6 Bản chất Kỹ thuật: Quy trình vòng lặp học tập của YOLOv8
Để thấu hiểu sự hiệp đồng của tất cả các kỹ thuật trên, ta cần mổ xẻ quy trình một vòng lặp huấn luyện (Training Iteration) theo đúng trình tự thời gian:
0. **Tiền xử lý (DataLoader trên CPU):** Trước khi nạp vào mạng, CPU sẽ bốc ngẫu nhiên 4 bức ảnh Zalo (dạng Panorama ngang 1622x626) khác nhau. Thay vì cắt xén làm xói mòn dữ liệu, DataLoader dùng kỹ thuật **Letterboxing (Đệm viền xám)** để bóp tỷ lệ ảnh lọt thỏm vào khung vuông 1280x1280. Sau đó, nó áp dụng ma trận Affine để dịch chuyển tọa độ (văng vật thể ra rìa), rồi dán đè 4 ảnh lại bằng lưới Mosaic. Kết quả tạo ra một bức ảnh "giả lập" 1280x1280 hoàn toàn mới mẻ, xóa bỏ Center Bias mà vẫn bảo toàn 100% pixel gốc.
1. **Forward Pass (Lan truyền tiến trên GPU):** Bức ảnh 1280x1280 (sau khi Augment) đi qua mạng Backbone (CSPDarknet). Tại nhánh P2 (tầng nông nhất), mạng trích xuất ra một ma trận đặc trưng khổng lồ 320x320 chứa nguyên vẹn điểm ảnh của các biển báo siêu li ti (đã bị thu nhỏ thêm bởi Mosaic). Thông tin này qua cổ chai PANet rồi đẩy ra Head để đưa ra 2 dự đoán: Tọa độ viền hộp và Xác suất phân loại của từng lớp ($p_t$).
2. **Calculate Loss (Đo lường sai số ĐỒNG THỜI):** 
   - Sai số tọa độ được đo bằng thuật toán **CIoU** ($L_{box}$).
   - Sai số phân loại được đo bằng **Focal Loss** ($L_{cls}$). Hàm toán học $FL(p_t) = -(1 - p_t)^\gamma \log(p_t)$ được kích hoạt để bóp nghẹt sai số của mẫu dễ và giữ lại sai số của mẫu khó.
   - Hàm Loss tổng hợp được tính toán ngay lập tức: **$Loss_{total} = 1.0 \cdot L_{box} + 2.0 \cdot L_{cls}$**.
3. **Backward Pass (Lan truyền ngược):** PyTorch tính đạo hàm (Gradient) từ $Loss_{total}$ truyền ngược về lại mạng Backbone. Nhờ hệ số nhân `cls_gain=2.0` và sự thanh lọc rác của Focal Loss, dòng thác Gradient lúc này mang theo một mệnh lệnh sắt đá duy nhất: *"Chỉ dồn lực sửa lỗi nơ-ron đọc chữ cho các biển báo li ti mờ nhòe, bỏ qua các biển to dễ nhìn"*.
4. **Weight Update (Cập nhật trọng số):** Thuật toán tối ưu **AdamW** tiếp nhận dòng thác Gradient này. Nó dùng thuật toán tự động thích ứng để tinh chỉnh (update) hàng triệu ma trận tham số trong mạng nơ-ron theo đúng hướng dẫn của Gradient. Kết hợp với chiến thuật phanh **Cosine Annealing LR**, AdamW sẽ hạ cánh lướt êm các trọng số này đáp chính xác xuống đáy của hố Loss (Global Minima) mà không bị học vẹt.
---

## 2. Faster R-CNN (Baseline Two-stage)
**Phân loại:** Two-stage Anchor-based Detector
**Mục tiêu thiết kế:** Đóng vai trò là đường cơ sở (Baseline) mang tính học thuật cao nhất, chuẩn mực của Object Detection truyền thống.

### 2.1 Thông số Mạng (Architecture)
- **Backbone:** ResNet-50 (Residual Network với 50 lớp). Sử dụng các khối Skip Connection để chống hiện tượng tiêu biến đạo hàm (Vanishing Gradient) khi mạng quá sâu.
- **Neck:** FPN (Feature Pyramid Network).
- **RPN (Region Proposal Network):**
  - Mạng lưới sinh hộp neo (Anchor Generator). 
  - **Sự khác biệt cốt lõi (K-Means 1:1 Anchor cho FPN):** Chạy thuật toán K-Means Clustering để phân tích toàn bộ kích thước biển báo trong kho dữ liệu Zalo, gom chúng thành 5 cụm tối ưu nhất: 10px, 24px, 44px, 77px, 133px. Vì biển báo giao thông (Tròn, Tam giác) có tính đối xứng cao, ta ép cứng tỷ lệ khung thành 1:1 (Khung vuông tuyệt đối). Dữ liệu này được định dạng thành một Tuple chứa 5 Tuple con `((10,), (24,), (44,), (77,), (133,))` nhằm phân phối chính xác 5 loại khung mồi này cho 5 tầng của tháp FPN (P2, P3, P4, P5, P6).
- **RoI Heads:** Dùng RoIAlign để trích xuất đặc trưng chính xác tới cấp độ số thập phân, khắc phục lỗi lệch pixel của RoIPool cũ.
- **Số lượng tham số (Parameters):** ~41 Triệu.
- **Khối lượng tính toán (GFLOPs):** ~130 GFLOPs.

### 2.2 Thiết lập Huấn luyện (Training Config)
- **Tỷ lệ chia dữ liệu:** 90% Train / 10% Validation. Đo lường Validation Loss (bằng mẹo no_grad kết hợp FrozenBatchNorm) để chọn ra Best Model.
- **Độ phân giải đầu vào:** Tự động scale sao cho cạnh nhỏ nhất là 800px.
- **Thuật toán Tối ưu (Optimizer):** `SGD` (`lr=0.005`, `momentum=0.9`, `weight_decay=0.0005`). Với ResNet, SGD kèm Momentum luôn mang lại sự hội tụ ổn định và sâu hơn so với các thuật toán họ Adam.
- **Lịch trình LR (Learning Rate Scheduler):** `StepLR`. Hạ tốc độ học xuống 10 lần ở Epoch thứ 10 để mô hình hạ cánh êm ái, chống dao động và chống học vẹt.
- **Batch Size:** 4 (Do kiến trúc Two-stage rất tốn VRAM).
- **Số vòng lặp (Epochs):** 15.

### 2.3 Cơ chế Hàm độ lỗi (Loss Functions)
- **RPN Loss:** Gồm 2 hàm: Objectness Loss (BCE) để phân biệt có vật thể hay nền, và RPN Box Loss (Smooth L1).
- **RoI Loss:** Gồm 2 hàm: Classification Loss (**Cross-Entropy**, do giới hạn mã nguồn không dùng được Focal Loss) và Box Regression Loss (Smooth L1).

### 2.4 Yêu cầu Phần cứng & Hiệu năng
- **Tài nguyên Training:** Yêu cầu VRAM >= 12GB. Chạy khá lâu do phải huấn luyện cả 2 mạng (RPN và Fast R-CNN độc lập).
- **Tốc độ Inference:** ~10 FPS trên GPU. (Khá chậm, chỉ phù hợp xử lý ảnh tĩnh).

### 2.5 Bản chất Kỹ thuật: Quy trình vòng lặp học tập của Faster R-CNN
Dây chuyền Two-stage của Faster R-CNN trải qua 5 bước cực kỳ cồng kềnh nhưng độ chính xác lại vươn lên đỉnh cao của học thuật:
0. **Tiền xử lý (BBox-Safe Crop trên CPU):** Khác với YOLO dùng Mosaic, mô hình này dùng `Albumentations` cắt ảnh ngẫu nhiên (On-the-fly Random Crop). Kích thước khuôn cắt chốt cứng là `512x512` (Vì ảnh gốc Zalo AI chỉ cao 626px, 512 là giới hạn Power of 2 an toàn nhất). Cụ thể cắt bao nhiêu lần? Mỗi lần nạp 1 bức ảnh vào GPU (ở mỗi Epoch), CPU chỉ tung xúc xắc cắt đúng **1 lần duy nhất** với xác suất `p=0.3` (tức là 30% khả năng bị cắt 512x512, 70% giữ nguyên ảnh gốc). Trải qua 10 Epochs, 1 bức ảnh gốc sẽ sinh ra khoảng 3 phiên bản bị cắt ở 3 tọa độ khác nhau, và 7 lần giữ nguyên. Điều này giúp AI vừa học được chi tiết cục bộ (khi bị cắt phóng to), vừa học được bối cảnh toàn cục (khi giữ nguyên). Nếu nhát cắt chém mất $> 50\%$ diện tích biển báo (`min_visibility=0.5`), hàm `RandomSizedBBoxSafeCrop` lập tức vứt bỏ và tung xúc xắc cắt lại, đảm bảo AI không bao giờ học nhầm "biển báo cụt".
1. **Trích xuất Đặc trưng (ResNet-50 & FPN):** Ảnh nạp vào bị hàng ngàn ma trận Tích chập (Convolutional Kernel 3x3) trượt qua để tìm viền và hình khối, nén lại thành khối ma trận đặc trưng.
   - **Skip Connection:** Khắc phục lỗi Vanishing Gradient (Khi tín hiệu lùi qua 50 lớp nơ-ron, Gradient bị nhân liên tiếp với số $<1$ và tiêu biến về 0). Nhờ đường vòng $F(x)+x$, đạo hàm bằng 1 giữ cho Gradient truyền thẳng về gốc mạng.
   - **FPN (Feature Pyramid):** Lớp nông rất Nét nhưng Ngu ngốc, lớp sâu rất Thông minh nhưng Mờ nhòe. FPN dùng Lateral Connection cộng dồn ma trận, đúc ra Kim tự tháp đặc trưng vừa hiểu ngữ nghĩa vừa sắc nét tọa độ.
2. **Mạng lưới đề xuất (RPN - Giai đoạn 1):** RPN trượt cửa sổ 3x3 trên ma trận đặc trưng. Tại mỗi điểm $(x, y)$, nó chiếu ngược về tâm ảnh gốc và "phóng ra" 5 Khung mồi ảo (Anchor Boxes). Kích thước chốt cứng từ thuật toán **K-Means 1:1** trên Zalo Dataset: $10\times10, 24\times24, 44\times44, 77\times77, 133\times133$. Thông qua phép nhân vô hướng (Dot Product) tạo ra sự cộng hưởng, RPN đánh giá 2 thứ: Điểm Objectness (BCE Loss) và độ lệch Box Deltas, từ đó "ghim" lại tọa độ những vùng CÓ THỂ chứa vật thể (Lọc bỏ 99% rác nền).
3. **Trạm kiểm duyệt (RoI Align & Head - Giai đoạn 2):** Các vùng đề xuất từ RPN được đẩy vào Mạng Head để trả lời "Đó là biển báo gì?". 
   - **RoI Align (Nội suy điểm ảnh lẻ):** Tọa độ RPN đẩy ra thường bị lẻ (VD: $15.6$). RoIPool cũ ép làm tròn thành $15$. Khi phóng ngược lên ảnh thật (nhân tỷ lệ nén 32 lần), sai số $0.6$ bị khuếch đại thành lệch $19.2$ pixel gốc (đủ chém đứt nửa biển báo). RoI Align cấm làm tròn, dùng Nội suy song tuyến tính (Bilinear Interpolation) tính chính xác ma trận của điểm ảo $15.6$. Khối ma trận trích xuất hoàn hảo này được đưa vào hàm Softmax chốt xác suất $p_t$ cho 7 nhóm biển báo.
4. **Tính Loss Kép:** Vì là 2 mạng nơ-ron chạy nối tiếp, model BẮT BUỘC phải tính 2 loại Loss độc lập cùng lúc cho RPN và RoI Head. 
   - **Phân loại:** Dùng **Cross-Entropy** (phóng to điểm phạt bằng Logarit nếu mô hình đoán sai nhưng lại tự tin mù quáng 99%). 
   - **Tọa độ:** Dùng **Smooth L1**. Xa đích thì chạy như hàm L1 thẳng tắp để chống nổ Gradient (Exploding Gradient), về sát đích thì uốn cong thành hàm bậc 2 (Parabol L2) để hạ cánh êm ái chống rung lắc.
5. **Cập nhật Trọng số (SGD Momentum):** Gradient kép dội về thuật toán **SGD (Momentum=0.9)**. Mạng Two-stage rất dễ vỡ nên kén AdamW. Phương trình động lượng $v_{t+1} = \mu v_t + \nabla L$ hoạt động như một cỗ xe lu. Nó giữ lại 90% vận tốc cũ ($\mu = 0.9$), tạo lực quán tính khổng lồ. Nếu gặp rãnh zig-zag, lực đối nghịch tự triệt tiêu. Nếu sụp vào "ổ gà" (Local Minima), đà quá khứ sẽ hất tung xe lu vọt qua miệng hố, lướt êm ái xuống đáy tối ưu vĩ mô.

---

## 3. RT-DETR-L (State-of-the-Art Transformer)
**Phân loại:** End-to-End Transformer-based Detector
**Mục tiêu thiết kế:** "Cỗ máy hủy diệt" dùng để phô diễn sức mạnh công nghệ mới nhất. Giải quyết bài toán ngữ cảnh toàn cục (Biển cấm ngược chiều thường đi với biển cấm quẹo).

### 3.1 Thông số Mạng (Architecture)
- **Backbone:** HGNetv2 (Hierarchical Graph Network). Cực kỳ mạnh mẽ trong việc trích xuất đặc trưng cấp thấp.
- **Neck / Encoder:** Hybrid Encoder thay thế cho Transformer Encoder tiêu chuẩn. Giảm bớt số lớp Attention để tăng tốc, kết hợp với các khối chập (Conv) cục bộ.
- **Decoder:** Transformer Decoder.
  - Sử dụng cơ chế Attention để mô hình tự động "nhìn" vào các mối liên kết toàn cục của bức ảnh.
  - Bỏ hoàn toàn thuật toán NMS. Mô hình tự động xuất ra số lượng hộp cố định và dùng thuật toán **Bipartite Matching** (Hungarian Algorithm) để khớp 1-1 với vật thể thật.
- **Số lượng tham số (Parameters):** ~32 Triệu (Bản L).
- **Khối lượng tính toán (GFLOPs):** ~114 GFLOPs.

### 3.2 Thiết lập Huấn luyện (Training Config)
- **Độ phân giải đầu vào:** `1280 x 1280`. (Cần độ phân giải cực lớn để Self-Attention vươn vòi bao quát toàn bộ ngữ cảnh bức ảnh).
- **Thủ thuật Chống OOM (Out-Of-Memory):** **Gradient Accumulation**. Khóa `batch_size = 2` nhưng cài `accumulate = 4`. Mô hình cộng dồn đạo hàm qua 4 chu kỳ liên tiếp mới cập nhật trọng số 1 lần, tạo hiệu ứng batch ảo là 8.
- **Thuật toán Tối ưu (Optimizer):** `AdamW` + `Cosine Annealing`.
- **Số vòng lặp (Epochs):** 50.

### 3.3 Cơ chế Hàm độ lỗi (Loss Functions)
- Bỏ hẳn tư duy Anchor Box (Khung mồi).
- Sử dụng **Hungarian Loss**: Là sự kết hợp của Focal Loss (Phân loại) và L1/GIoU Loss (Tọa độ). 

### 3.4 Bản chất Kỹ thuật: Bức tranh toàn cảnh 5 bước của RT-DETR
Khác hoàn toàn với tư duy "Trượt cửa sổ" của CNN (như YOLO, R-CNN), RT-DETR mang tư duy "Nhìn toàn cục" của Transformer:
0. **Tiền xử lý (Nạp ảnh Panorama 1280px):** Trái ngược với Faster R-CNN phải cắt vụn ảnh ra 512x512, RT-DETR nuốt trọn bức ảnh toàn cảnh khổng lồ `1280x1280` để giữ lại 100% bối cảnh không gian (Ví dụ: Mô hình tự hiểu biển báo cấm rẽ thường đứng chung cột với biển cấm ngược chiều).
1. **Trích xuất cục bộ (HGNetv2 Backbone):** Dù là mạng Transformer, lớp đầu tiên của nó BẮT BUỘC phải là mạng Tích chập (CNN) HGNetv2. Lý do: Transformer rất ngu ngốc trong việc nhận diện viền/góc cạnh ở giai đoạn đầu. Mạng CNN sẽ giải quyết phần "chân tay" này và nén ảnh thành ma trận.
2. **Cầu nối đa tầng (CCFM Neck):** Thay vì dùng FPN, RT-DETR dùng mô-đun lai CCFM (Cross-Scale Feature-fusion Module) để trộn lẫn ma trận từ tầng nông và tầng sâu, chuẩn bị "thức ăn" tinh gọn nhất trước khi tống vào lõi Transformer.
3. **Bộ não Transformer (Self-Attention Decoder):** Đây là lõi sức mạnh. Mô hình ném vào không gian đúng 300 "Hạt giống" (Object Queries). Chẳng cần trượt cái cửa sổ nào cả! Mỗi hạt giống phóng tầm mắt bao quát toàn bộ 1,600 mảnh ghép bằng cơ chế toán học **Q-K-V (Query-Key-Value)**. Thao tác này hoàn toàn dựa trên Đại số tuyến tính: Nó dùng phép Nhân vô hướng (Dot Product) để đo **Cosine Similarity** (Độ tương đồng) giữa lệnh truy nã ($Q$) và biển quảng cáo của pixel ($K$). Sau khi trúng mục tiêu, nó dùng phép Cộng ma trận (Residual Connection) để "nuốt" trọn khối lượng dữ liệu thật ($V$) vào bản thân hạt giống. Trải qua 6 vòng lặp Decoder, 300 hạt giống này tự động bù trừ Offset (Độ lệch) và nở thành đúng 300 Khung dự đoán.
4. **Khớp nối 1-1 & Tính Loss (Hungarian Algorithm):** Đây là cuộc cách mạng chấm dứt kỷ nguyên của NMS! R-CNN hay YOLO phọt ra 10,000 khung rồi phải dùng NMS xóa bớt. RT-DETR chỉ xuất đúng 300 khung. Nó dùng thuật toán **Bipartite Matching** (Kuhn-Munkres) trong thời gian đa thức $O(N^3)$ để lập Ma trận chi phí (Dựa trên $\mathcal{L}_{\text{L1}}$, $\mathcal{L}_{\text{GIoU}}$ và $\text{Focal Loss}$). Toán học giải bài toán Tối ưu Tổ hợp sao cho 5 cái biển báo thật được gán cho đúng 5 khung dự đoán với chi phí RẺ NHẤT (1-kèm-1). Sự thanh lịch tuyệt đối nằm ở chỗ: 5 khung khớp nhất được lôi ra tính Loss để bay về đích. 295 khung rớt đài còn lại bị ép gán nhãn "$\varnothing$" (Background) và lập tức chịu sự trừng phạt tàn khốc của **Focal Loss**, sinh ra dòng Gradient âm khổng lồ đè bẹp trọng số của chúng về 0. Toàn bộ rác nền tự động bị triệt tiêu bằng Toán học thuần túy mà không cần NMS!

### 3.4 Yêu cầu Phần cứng & Hiệu năng
- **Tài nguyên Training:** Cực kỳ "ăn" VRAM do tính chất ma trận vuông khổng lồ của Attention. Nếu không dùng Gradient Accumulation, card 16GB sẽ nổ tung ở ảnh 1280.
- **Tốc độ Inference:** ~60-80 FPS trên GPU. Cực kỳ xuất sắc, có thể dùng thay thế YOLO trong hệ thống thực tế.
