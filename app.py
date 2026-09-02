"""
Traffic Sign Detection - Streamlit App
=======================================
Ứng dụng demo nhận diện Biển báo Giao thông Việt Nam (Zalo AI Challenge Dataset)
bằng các mô hình Object Detection: YOLOv8, Faster R-CNN, RT-DETR.

Trạng thái hiện tại: PLACEHOLDER MODE
--------------------------------------
Các mô hình thật (YOLOv8s-P2, Faster R-CNN, RT-DETR-L) huấn luyện trên 7 lớp
biển báo giao thông vẫn đang trong quá trình huấn luyện. Vì vậy, bất kể người
dùng chọn model nào ở Sidebar, luồng suy luận (Inference) bên dưới TẠM THỜI
luôn tải model `yolov8n.pt` mặc định của thư viện `ultralytics` (huấn luyện
sẵn trên bộ dữ liệu COCO - 80 lớp vật thể thông dụng) để làm "mồi" test luồng
xử lý (upload -> inference -> vẽ bounding box) từ đầu đến cuối.

Khi có file trọng số `best.pt` thật, chỉ cần bỏ comment hàm
`load_model_from_huggingface()` bên dưới và trỏ `REPO_ID` tới đúng Hugging
Face Hub Model Repository là toàn bộ pipeline sẽ hoạt động với model thật mà
KHÔNG cần sửa bất kỳ logic vẽ bounding box hay UI nào (vì nhãn lớp được đọc
động từ `model.names` của chính model đang được nạp).
"""

import io

import streamlit as st
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

# ============================================================================
# CẤU HÌNH TRANG
# ============================================================================
st.set_page_config(
    page_title="🚦 Traffic Sign Detection",
    page_icon="🚧",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================================
# CUSTOM CSS
# ============================================================================
st.markdown(
    """
<style>
    /* Tăng cỡ chữ gốc của toàn bộ trang (Streamlit dùng đơn vị rem/em tương
       đối theo root font-size, nên chỉnh 1 chỗ này sẽ scale hầu hết chữ,
       khoảng cách trong toàn app - kể cả các widget mặc định của Streamlit). */
    html, body, [class*="css"] {
        font-size: 32px;
    }

    /* Ép Sidebar rộng ra ~1/4 màn hình thay vì cỡ hẹp mặc định của Streamlit. */
    section[data-testid="stSidebar"] {
        width: 30vw !important;
        min-width: 420px !important;
        max-width: 640px !important;
    }
    section[data-testid="stSidebar"] > div:first-child {
        width: 30vw !important;
        min-width: 420px !important;
        max-width: 640px !important;
    }

    .main-header {
        font-size: 3.5rem;
        font-weight: 800;
        text-align: center;
        background: linear-gradient(90deg, #FF6B35, #F7C548);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        text-align: center;
        color: #666;
        font-size: 1.4rem;
        margin-bottom: 2rem;
    }
    .result-box {
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        background: linear-gradient(135deg, #232526 0%, #414345 100%);
        color: white;
        text-align: center;
    }
    .result-box h2 {
        margin: 0;
        font-size: 2.2rem;
    }
    .model-info-box {
        padding: 1.2rem 1.4rem;
        border-radius: 10px;
        background: #f0f2f6;
        border-left: 5px solid #FF6B35;
        font-size: 1.1rem;
        line-height: 1.7;
        margin-bottom: 0.5rem;
    }
    .model-info-box b {
        font-size: 1.2rem;
    }
    section[data-testid="stSidebar"] h2 {
        font-size: 1.8rem;
    }
    section[data-testid="stSidebar"] h3 {
        font-size: 1.4rem;
    }
    section[data-testid="stSidebar"] p, section[data-testid="stSidebar"] li,
    section[data-testid="stSidebar"] label {
        font-size: 1.1rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 1.2rem;
        font-weight: 600;
    }
</style>
""",
    unsafe_allow_html=True,
)

# ============================================================================
# HẰNG SỐ CẤU HÌNH
# ============================================================================

# 7 lớp biển báo giao thông mục tiêu của đồ án (theo docs/01_project_master).
# Danh sách này chỉ dùng để HIỂN THỊ thông tin ở Sidebar cho người dùng biết
# bài toán thật sẽ phân loại các lớp gì. Nó KHÔNG được dùng trực tiếp để gán
# nhãn cho kết quả suy luận (nhãn suy luận luôn lấy động từ `model.names`),
# vì model placeholder hiện tại (yolov8n.pt gốc) được train trên COCO-80 lớp,
# chưa hề biết tới 7 lớp này.
TRAFFIC_SIGN_CLASSES = [
    "Cấm ngược chiều",
    "Cấm dừng và đỗ",
    "Cấm rẽ",
    "Giới hạn tốc độ",
    "Cấm còn lại",
    "Nguy hiểm",
    "Hiệu lệnh",
]

# Danh sách model hiển thị cho người dùng lựa chọn ở Sidebar, kèm mô tả kỹ
# thuật (tham khảo docs/03_models_training/models_specs.md) để người xem demo
# hiểu rõ định hướng thiết kế của từng kiến trúc, dù hiện tại cả 3 lựa chọn
# đều đang chạy chung 1 model placeholder YOLOv8n.
MODEL_OPTIONS = {
    "YOLOv8 (Tối ưu Tốc độ)": {
        "emoji": "⚡",
        "desc": "One-stage, Anchor-free. Backbone CSPDarknet + PANet neck, "
        "mở rộng nhánh P2 để bắt vật thể siêu nhỏ. Nhanh nhất, phù hợp Web App.",
        "params": "~11.5M",
        "speed": "~15-20 FPS (CPU)",
    },
    "Faster R-CNN (Tối ưu Độ chính xác)": {
        "emoji": "🎯",
        "desc": "Two-stage, Anchor-based. Backbone ResNet-50 + FPN + RPN. "
        "Độ chính xác học thuật cao, đánh đổi tốc độ suy luận chậm hơn.",
        "params": "~41M",
        "speed": "~10 FPS (GPU)",
    },
    "RT-DETR (Cân bằng)": {
        "emoji": "⚖️",
        "desc": "End-to-end Transformer, bỏ hoàn toàn NMS nhờ Bipartite Matching. "
        "Hiểu ngữ cảnh toàn cục, cân bằng giữa tốc độ và độ chính xác.",
        "params": "~32M",
        "speed": "~60-80 FPS (GPU)",
    },
}

PLACEHOLDER_WEIGHTS = "yolov8n.pt"


# ============================================================================
# [TƯƠNG LAI] TẢI MODEL THẬT (best.pt) TỪ HUGGING FACE HUB
# ============================================================================
# Khi đã huấn luyện xong model thật cho từng kiến trúc và upload file trọng
# số `best.pt` lên một Hugging Face Hub Model Repository, hãy:
#   1. Bỏ comment khối code dưới đây.
#   2. Điền đúng `repo_id` cho từng model trong MODEL_OPTIONS (VD: thêm key
#      "hf_repo_id" vào từng dict phía trên).
#   3. Thay lời gọi `load_placeholder_model()` trong `get_active_model()`
#      bằng `load_model_from_huggingface(repo_id)`.
#
# from huggingface_hub import hf_hub_download
#
# def load_model_from_huggingface(repo_id: str, filename: str = "best.pt") -> YOLO:
#     """
#     Tải file trọng số `best.pt` đã huấn luyện thật từ một Hugging Face Hub
#     Model Repository về máy chủ (cache local), sau đó nạp vào đối tượng YOLO.
#
#     Args:
#         repo_id: ID repository trên HF Hub, dạng "username/model-name".
#                  VD: "your-username/traffic-sign-yolov8s-p2"
#         filename: Tên file trọng số trên Hub (mặc định "best.pt").
#
#     Returns:
#         model: Đối tượng ultralytics.YOLO đã sẵn sàng để .predict()/.__call__().
#     """
#     model_path = hf_hub_download(repo_id=repo_id, filename=filename)
#     model = YOLO(model_path)
#     return model
#
# Ví dụ sử dụng:
#   model = load_model_from_huggingface("your-username/traffic-sign-yolov8s-p2")
#   results = model.predict(image, conf=0.25)


# ============================================================================
# LOAD MODEL (CACHE) - PLACEHOLDER
# ============================================================================
@st.cache_resource(show_spinner=False)
def load_placeholder_model(weights: str = PLACEHOLDER_WEIGHTS) -> YOLO:
    """
    Tải model YOLOv8n mặc định của `ultralytics` để làm luồng suy luận mồi.

    Lần chạy đầu tiên, ultralytics sẽ tự động tải file trọng số `yolov8n.pt`
    (~6MB) từ GitHub Releases của Ultralytics về, sau đó cache lại nhờ
    `st.cache_resource` để các lượt chạy sau không phải tải/nạp lại.
    """
    model = YOLO(weights)
    return model


def get_active_model(selected_model_label: str) -> YOLO:
    """
    Trả về model sẽ thực sự được dùng để suy luận.

    HIỆN TẠI: bất kể `selected_model_label` là gì trong 3 lựa chọn Sidebar,
    hàm luôn trả về model placeholder `yolov8n.pt` (xem docstring đầu file).
    """
    return load_placeholder_model()


# ============================================================================
# VẼ BOUNDING BOX BẰNG PILLOW (KHÔNG DÙNG opencv-python)
# ============================================================================
# Lưu ý quan trọng cho việc Deploy lên Streamlit Community Cloud:
# Toàn bộ việc vẽ khung/nhãn ở đây dùng thuần `PIL.ImageDraw`, KHÔNG import
# `cv2` trong code của app. Nhờ vậy app không phụ thuộc vào các thư viện đồ
# họa hệ thống như `libGL.so.1` (vốn không có sẵn trên máy chủ Linux headless
# của Streamlit Cloud) - đây chính là nguyên nhân phổ biến nhất gây lỗi
# `ImportError: libGL.so.1: cannot open shared object file`.
def _get_font(size: int = 16) -> ImageFont.ImageFont:
    """Lấy font để vẽ nhãn, có fallback an toàn cho mọi môi trường."""
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        # Pillow < 10.1 chưa hỗ trợ tham số `size` cho load_default()
        return ImageFont.load_default()


# Bảng màu để phân biệt các box theo class_id (lặp vòng nếu nhiều hơn 10 lớp)
BOX_COLORS = [
    "#FF6B35", "#4ECDC4", "#F7C548", "#5B5FEF", "#E5484D",
    "#2ECC71", "#9B5DE5", "#00B4D8", "#F15BB5", "#118AB2",
]


def draw_detections(image: Image.Image, result, conf_threshold: float = 0.25):
    """
    Vẽ bounding box + nhãn lớp + confidence score lên ảnh gốc bằng PIL.

    Args:
        image: Ảnh gốc (PIL Image, RGB).
        result: Một phần tử trong list trả về bởi `model.predict(...)`
                (đối tượng `ultralytics.engine.results.Results`).
        conf_threshold: Ngưỡng lọc confidence tối thiểu để vẽ box.

    Returns:
        annotated: Ảnh PIL đã vẽ xong bounding box.
        detections: list[dict] chi tiết từng box đã vẽ (để hiển thị bảng bên dưới).
    """
    annotated = image.convert("RGB").copy()
    draw = ImageDraw.Draw(annotated)
    font = _get_font(size=max(14, image.width // 80))

    class_names = result.names  # dict {class_id: class_name} lấy động từ model
    detections = []

    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return annotated, detections

    for box in boxes:
        conf = float(box.conf[0])
        if conf < conf_threshold:
            continue

        cls_id = int(box.cls[0])
        label = class_names.get(cls_id, str(cls_id))
        x1, y1, x2, y2 = [float(v) for v in box.xyxy[0]]

        color = BOX_COLORS[cls_id % len(BOX_COLORS)]
        line_width = max(2, image.width // 400)

        # Khung bounding box
        draw.rectangle([x1, y1, x2, y2], outline=color, width=line_width)

        # Nhãn + confidence, có nền để dễ đọc
        caption = f"{label} {conf * 100:.1f}%"
        text_bbox = draw.textbbox((0, 0), caption, font=font)
        text_w, text_h = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
        pad = 3
        label_y0 = max(0, y1 - text_h - 2 * pad)
        draw.rectangle(
            [x1, label_y0, x1 + text_w + 2 * pad, label_y0 + text_h + 2 * pad],
            fill=color,
        )
        draw.text((x1 + pad, label_y0 + pad), caption, fill="white", font=font)

        detections.append({"label": label, "confidence": conf, "box": (x1, y1, x2, y2)})

    return annotated, detections


# ============================================================================
# SIDEBAR
# ============================================================================
def render_sidebar() -> tuple[str, float]:
    """Render sidebar. Trả về (model được chọn, ngưỡng confidence)."""
    with st.sidebar:
        st.markdown("## 🚦 Traffic Sign Detection")
        st.markdown("---")

        st.markdown("### 🧠 Chọn mô hình")
        selected_model = st.selectbox(
            "Model nhận diện",
            options=list(MODEL_OPTIONS.keys()),
            index=0,
            help="Chọn kiến trúc model muốn dùng để nhận diện biển báo.",
        )

        info = MODEL_OPTIONS[selected_model]
        st.markdown(
            f"""
            <div class="model-info-box">
            <b>{info['emoji']} {selected_model}</b><br/><br/>
            {info['desc']}<br/><br/>
            📦 Số tham số: <b>{info['params']}</b><br/>
            ⚡ Tốc độ: <b>{info['speed']}</b>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.warning(
            "🚧 **Đang ở chế độ Placeholder**\n\n"
            "Model thật cho 7 lớp biển báo chưa huấn luyện xong. "
            "App đang chạy tạm `yolov8n.pt` (COCO-80 lớp) để demo "
            "luồng xử lý end-to-end.",
            icon="🚧",
        )

        st.markdown("---")

        st.markdown("### 🎚️ Ngưỡng tin cậy")
        conf_threshold = st.slider(
            "Chỉ hiển thị box có confidence ≥",
            min_value=0.05,
            max_value=0.95,
            value=0.25,
            step=0.05,
        )

        st.markdown("---")

        st.markdown("### 🏷️ 7 lớp biển báo mục tiêu")
        st.markdown("\n".join(f"- {c}" for c in TRAFFIC_SIGN_CLASSES))

        st.markdown("---")

        st.markdown("### 📖 Về project")
        st.markdown(
            """
            Đồ án nhận diện biển báo giao thông Việt Nam, dữ liệu từ
            **Zalo AI Challenge** (định dạng COCO JSON, ảnh Panorama 1622x626).
            """
        )

        st.markdown("---")

        st.markdown("### 🔗 Links")
        st.markdown(
            """
            - [Ultralytics YOLOv8 Docs](https://docs.ultralytics.com/)
            - [Hugging Face Hub](https://huggingface.co/)
            """
        )

    return selected_model, conf_threshold


# ============================================================================
# TAB: DEMO
# ============================================================================
def render_demo_tab(selected_model: str, conf_threshold: float):
    st.markdown("### 📸 Upload ảnh để nhận diện biển báo")
    st.markdown("Hỗ trợ định dạng: JPG, JPEG, PNG")

    uploaded_file = st.file_uploader(
        "Chọn ảnh...",
        type=["jpg", "jpeg", "png"],
        help="Upload ảnh đường phố có chứa biển báo giao thông.",
    )

    if uploaded_file is None:
        st.info("👆 Vui lòng upload một ảnh để bắt đầu!")
        with st.expander("💡 Gợi ý"):
            st.markdown(
                """
                - Ảnh nên rõ ràng, đủ sáng, biển báo không bị che khuất quá nhiều.
                - Hỗ trợ các định dạng phổ biến: JPG, PNG.
                - Vì đang ở chế độ Placeholder (model gốc COCO-80 lớp), ảnh chứa
                  các vật thể đời thường (người, xe, ...) sẽ minh họa luồng xử lý
                  tốt hơn là ảnh chỉ có biển báo đơn thuần.
                """
            )
        return

    image = Image.open(uploaded_file).convert("RGB")

    with st.spinner("🔄 Đang tải model..."):
        try:
            model = get_active_model(selected_model)
        except Exception as e:
            st.error(f"❌ **Lỗi khi tải model!**\n\nChi tiết lỗi: {str(e)}")
            return

    with st.spinner("🔍 Đang phân tích ảnh..."):
        try:
            results = model.predict(image, conf=conf_threshold, verbose=False)
            result = results[0]
            annotated_image, detections = draw_detections(image, result, conf_threshold)
        except Exception as e:
            st.error(f"❌ **Lỗi khi suy luận!**\n\nChi tiết lỗi: {str(e)}")
            return

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 🖼️ Ảnh gốc")
        st.image(image, use_container_width=True)
    with col2:
        st.markdown("#### 🎯 Kết quả nhận diện")
        st.image(annotated_image, use_container_width=True)

    st.markdown("---")
    st.markdown("### 📋 Chi tiết các đối tượng phát hiện")

    if not detections:
        st.info(
            f"Không phát hiện đối tượng nào với ngưỡng confidence ≥ {conf_threshold:.0%}. "
            "Thử hạ ngưỡng confidence ở Sidebar hoặc upload ảnh khác."
        )
    else:
        st.markdown(
            f"""
            <div class="result-box">
                <h2>✅ Phát hiện {len(detections)} đối tượng</h2>
            </div>
            """,
            unsafe_allow_html=True,
        )
        for i, det in enumerate(sorted(detections, key=lambda d: -d["confidence"]), start=1):
            c1, c2 = st.columns([3, 1])
            with c1:
                st.markdown(f"**[{i}] {det['label']}**")
                st.progress(det["confidence"])
            with c2:
                st.markdown(f"**{det['confidence'] * 100:.2f}%**")


# ============================================================================
# TAB: THÔNG TIN MODEL
# ============================================================================
def render_info_tab():
    st.markdown("### 🔬 So sánh 3 kiến trúc Model")
    st.markdown(
        "Bảng dưới đây tóm tắt định hướng thiết kế của 3 model sẽ được huấn luyện "
        "cho bài toán này (xem chi tiết tại `docs/03_models_training/models_specs.md`)."
    )

    cols = st.columns(3)
    for col, (name, info) in zip(cols, MODEL_OPTIONS.items()):
        with col:
            st.markdown(f"#### {info['emoji']} {name}")
            st.markdown(info["desc"])
            st.metric("Số tham số", info["params"])
            st.metric("Tốc độ suy luận", info["speed"])

    st.markdown("---")
    st.markdown("### 🗂️ Dataset")
    st.markdown(
        """
        - **Nguồn:** Zalo AI Challenge - Traffic Sign Detection.
        - **Định dạng:** COCO JSON (`info`, `images`, `annotations`, `categories`).
        - **Độ phân giải:** ~1622 x 626 px (ảnh Panorama/Dashcam).
        - **Thử thách chính:** Vật thể siêu nhỏ (median area chỉ ~266 px²).
        """
    )


# ============================================================================
# MAIN APP
# ============================================================================
def main():
    selected_model, conf_threshold = render_sidebar()

    st.markdown(
        '<h1 class="main-header">🚦 Traffic Sign Detection 🚧</h1>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="sub-header">Hệ thống nhận diện biển báo giao thông Việt Nam '
        "bằng YOLOv8 / Faster R-CNN / RT-DETR</p>",
        unsafe_allow_html=True,
    )

    tab1, tab2 = st.tabs(["🎮 Demo", "🔬 Về Model"])
    with tab1:
        render_demo_tab(selected_model, conf_threshold)
    with tab2:
        render_info_tab()

    st.markdown("---")
    st.markdown(
        "<p style='text-align: center; color: #888;'>Made with ❤️ using Streamlit "
        "| Zalo AI Challenge - Traffic Sign Detection</p>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
