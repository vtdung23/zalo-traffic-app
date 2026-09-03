"""
Traffic Sign Detection - Streamlit App
=======================================
Ứng dụng nhận diện Biển báo Giao thông Việt Nam (bộ dữ liệu Zalo AI Challenge 2020).

Mô hình đang triển khai: YOLOv8s-P2
-----------------------------------
Sau khi huấn luyện và đánh giá cả 3 kiến trúc (YOLOv8s-P2, Faster R-CNN R50-FPN,
RT-DETR-L) trên tập Hold-out Test 20%, YOLOv8s-P2 được chọn để đưa lên Web App.

Lý do chọn nằm ở ràng buộc triển khai chứ không thuần độ chính xác: RT-DETR-L có
mAP@50-95 nhỉnh hơn (47.15% so với 42.60%) nhưng file trọng số nặng 251.5 MB,
trong khi YOLOv8s-P2 chỉ 20.8 MB, lại nhanh hơn và có mAP@50 cao hơn. Streamlit
Community Cloud chạy thuần CPU với khoảng 1 GB RAM, nên bản nhẹ là lựa chọn khả thi.
Xem phần "Vì sao chọn mô hình này" ở tab Về Model để biết đầy đủ lập luận.

App chỉ phục vụ đúng một mô hình. Người dùng upload ảnh, app trả về ảnh đã vẽ
bounding box - không có chức năng chọn giữa nhiều mô hình.
"""

import json
import os
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

# ============================================================================
# CẤU HÌNH TRANG
# ============================================================================
st.set_page_config(
    page_title="🚦 Nhận diện Biển báo Giao thông",
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

TEN_MODEL = "YOLOv8s-P2"

# Repo trên Hugging Face Hub chứa file trọng số. Có thể ghi đè bằng Secrets của
# Streamlit hoặc biến môi trường HF_REPO_ID để đổi repo mà không phải sửa code.
HF_REPO_ID_MAC_DINH = "vtdung23/traffic-sign-yolov8s-p2"
HF_TEN_FILE = "best.pt"

# Tham số suy luận phải khớp đúng với lúc huấn luyện và lúc đánh giá, nếu không
# thì kết quả trên Web App sẽ lệch so với con số mAP đã công bố trong báo cáo.
IMGSZ = 1280   # Spec mục 1.2: huấn luyện ở độ phân giải cao để bắt vật thể nhỏ
IOU_NMS = 0.6  # [E2] Giữ được các biển báo đứng sát cạnh nhau
MAX_DET = 50   # [E1, E2] Chặn trần số box mỗi ảnh cho nhẹ khâu NMS

# 7 lớp biển báo của bài toán. Thứ tự trong danh sách này khớp đúng với thứ tự
# class_id 0-6 mà `split_dataset.py` sinh ra, nên có thể tra tên tiếng Việt
# trực tiếp theo chỉ số. Model trả về tên tiếng Anh, ta dịch lại khi hiển thị.
TEN_LOP_TIENG_VIET = [
    "Cấm ngược chiều",
    "Cấm dừng và đỗ",
    "Cấm rẽ",
    "Giới hạn tốc độ",
    "Cấm còn lại",
    "Nguy hiểm",
    "Hiệu lệnh",
]

# Tìm thư mục chứa kết quả đánh giá. Phải dò nhiều vị trí vì file app.py này có
# thể được copy sang repo khác, nơi app.py nằm ngay thư mục gốc chứ không nằm
# trong app/. Có thể ép cứng đường dẫn bằng biến môi trường EVAL_RESULTS_DIR.
THU_MUC_APP = Path(__file__).resolve().parent


def tim_thu_muc_ket_qua() -> Path:
    """Dò các vị trí có thể chứa eval-results, trả về vị trí đầu tiên tìm thấy."""
    duong_dan_ep = os.environ.get("EVAL_RESULTS_DIR")
    if duong_dan_ep:
        return Path(duong_dan_ep)

    cac_vi_tri_thu = [
        THU_MUC_APP / "eval-results",         # app.py nằm cùng cấp với eval-results
        THU_MUC_APP.parent / "eval-results",  # app.py nằm trong app/, eval-results ở gốc
    ]
    for vi_tri in cac_vi_tri_thu:
        if vi_tri.is_dir():
            return vi_tri

    # Không tìm thấy thì vẫn trả về vị trí mặc định, app sẽ hiện cảnh báo thiếu file
    return cac_vi_tri_thu[-1]


THU_MUC_KET_QUA = tim_thu_muc_ket_qua()

DUONG_DAN_BANG_SO_SANH = THU_MUC_KET_QUA / "final_comparison_table.csv"
DUONG_DAN_LICH_SU = THU_MUC_KET_QUA / "yolov8_training_history.json"
DUONG_DAN_ANH_PARETO = THU_MUC_KET_QUA / "pareto_accuracy_vs_speed.png"
DUONG_DAN_MA_TRAN = THU_MUC_KET_QUA / "confusion_matrix_YOLOv8s-P2.png"

# Bảng màu để phân biệt các box theo class_id
MAU_BOX = [
    "#FF6B35", "#4ECDC4", "#F7C548", "#5B5FEF", "#E5484D",
    "#2ECC71", "#9B5DE5",
]


# ============================================================================
# NẠP MODEL TỪ HUGGING FACE HUB
# ============================================================================
def lay_repo_id() -> str:
    """Lấy repo id của Hugging Face, ưu tiên Secrets rồi tới biến môi trường."""
    try:
        if "HF_REPO_ID" in st.secrets:
            return st.secrets["HF_REPO_ID"]
    except Exception:
        # Chạy local không có file secrets thì st.secrets ném lỗi, bỏ qua là được
        pass
    return os.environ.get("HF_REPO_ID", HF_REPO_ID_MAC_DINH)


@st.cache_resource(show_spinner=False)
def tai_model(repo_id: str) -> YOLO:
    """Tải file best.pt từ Hugging Face Hub rồi nạp vào đối tượng YOLO.

    Dùng `st.cache_resource` nên file chỉ tải và nạp đúng một lần cho mỗi phiên
    server, các lượt upload ảnh sau đó dùng lại model đã nằm sẵn trong bộ nhớ.
    """
    from huggingface_hub import hf_hub_download

    duong_dan_weight = hf_hub_download(repo_id=repo_id, filename=HF_TEN_FILE)
    return YOLO(duong_dan_weight)


# ============================================================================
# ĐỌC DỮ LIỆU KẾT QUẢ ĐÁNH GIÁ
# ============================================================================
@st.cache_data(show_spinner=False)
def doc_bang_so_sanh():
    """Đọc bảng so sánh 3 mô hình. Trả về None nếu chưa có file."""
    if not DUONG_DAN_BANG_SO_SANH.exists():
        return None
    return pd.read_csv(DUONG_DAN_BANG_SO_SANH)


@st.cache_data(show_spinner=False)
def doc_lich_su_huan_luyen():
    """Đọc file JSON nhật ký huấn luyện. Trả về None nếu chưa có file."""
    if not DUONG_DAN_LICH_SU.exists():
        return None
    with open(DUONG_DAN_LICH_SU, "r", encoding="utf-8") as f:
        return json.load(f)


def lay_chi_so_model_dang_dung(bang):
    """Bóc dòng kết quả của riêng mô hình đang triển khai ra khỏi bảng."""
    if bang is None:
        return None
    dong_khop = bang[bang["Model"] == TEN_MODEL]
    if len(dong_khop) == 0:
        return None
    return dong_khop.iloc[0]


# ============================================================================
# VẼ BOUNDING BOX BẰNG PILLOW (KHÔNG DÙNG opencv-python)
# ============================================================================
# Lưu ý quan trọng cho việc Deploy lên Streamlit Community Cloud:
# Toàn bộ việc vẽ khung/nhãn ở đây dùng thuần `PIL.ImageDraw`, KHÔNG import
# `cv2` trong code của app. Nhờ vậy app không phụ thuộc vào các thư viện đồ
# họa hệ thống như `libGL.so.1` (vốn không có sẵn trên máy chủ Linux headless
# của Streamlit Cloud) - đây chính là nguyên nhân phổ biến nhất gây lỗi
# `ImportError: libGL.so.1: cannot open shared object file`.
def _lay_font(co_chu: int = 16):
    """Lấy font để vẽ nhãn, có phương án dự phòng cho mọi môi trường."""
    try:
        return ImageFont.load_default(size=co_chu)
    except TypeError:
        # Pillow < 10.1 chưa hỗ trợ tham số `size` cho load_default()
        return ImageFont.load_default()


def doi_ten_lop(class_id: int, ten_goc: str) -> str:
    """Đổi tên lớp tiếng Anh của model sang tiếng Việt cho người dùng dễ đọc."""
    if 0 <= class_id < len(TEN_LOP_TIENG_VIET):
        return TEN_LOP_TIENG_VIET[class_id]
    return ten_goc


def ve_ket_qua(anh: Image.Image, ket_qua, nguong_conf: float = 0.25):
    """Vẽ bounding box, tên lớp và độ tin cậy lên ảnh gốc.

    Trả về (ảnh đã vẽ, danh sách chi tiết từng box) để phần dưới còn hiển thị bảng.
    """
    anh_ve = anh.convert("RGB").copy()
    but_ve = ImageDraw.Draw(anh_ve)
    font = _lay_font(co_chu=max(14, anh.width // 80))

    ten_cac_lop = ket_qua.names  # dict {class_id: tên} lấy động từ chính model
    danh_sach_box = []

    boxes = ket_qua.boxes
    if boxes is None or len(boxes) == 0:
        return anh_ve, danh_sach_box

    for box in boxes:
        do_tin_cay = float(box.conf[0])
        if do_tin_cay < nguong_conf:
            continue

        class_id = int(box.cls[0])
        nhan = doi_ten_lop(class_id, ten_cac_lop.get(class_id, str(class_id)))
        x1, y1, x2, y2 = [float(v) for v in box.xyxy[0]]

        mau = MAU_BOX[class_id % len(MAU_BOX)]
        do_day = max(2, anh.width // 400)

        but_ve.rectangle([x1, y1, x2, y2], outline=mau, width=do_day)

        # Nhãn kèm độ tin cậy, có nền màu cho dễ đọc trên mọi ảnh nền
        chu_thich = f"{nhan} {do_tin_cay * 100:.1f}%"
        khung_chu = but_ve.textbbox((0, 0), chu_thich, font=font)
        rong_chu = khung_chu[2] - khung_chu[0]
        cao_chu = khung_chu[3] - khung_chu[1]
        dem = 3
        y_nhan = max(0, y1 - cao_chu - 2 * dem)
        but_ve.rectangle(
            [x1, y_nhan, x1 + rong_chu + 2 * dem, y_nhan + cao_chu + 2 * dem],
            fill=mau,
        )
        but_ve.text((x1 + dem, y_nhan + dem), chu_thich, fill="white", font=font)

        danh_sach_box.append({
            "nhan": nhan,
            "do_tin_cay": do_tin_cay,
            "toa_do": (x1, y1, x2, y2),
        })

    return anh_ve, danh_sach_box


# ============================================================================
# SIDEBAR
# ============================================================================
def render_sidebar() -> float:
    """Render sidebar. Trả về ngưỡng confidence do người dùng chọn."""
    with st.sidebar:
        st.markdown("## 🚦 Nhận diện Biển báo")
        st.markdown("---")

        st.markdown("### 🧠 Mô hình đang dùng")

        chi_so = lay_chi_so_model_dang_dung(doc_bang_so_sanh())
        if chi_so is not None:
            dong_chi_so = (
                f"🎯 mAP@50: <b>{chi_so['mAP@50 (%)']:.2f}%</b><br/>"
                f"📏 mAP@50-95: <b>{chi_so['mAP@50-95 (%)']:.2f}%</b><br/>"
                f"⚡ Tốc độ: <b>{chi_so['FPS']:.1f} FPS</b> (đo trên GPU)"
            )
        else:
            dong_chi_so = "📊 Chưa có số liệu đánh giá"

        st.markdown(
            f"""
            <div class="model-info-box">
            <b>⚡ {TEN_MODEL}</b><br/><br/>
            One-stage, Anchor-free. Backbone CSPDarknet + PANet neck, mở rộng
            nhánh P2 để bắt vật thể siêu nhỏ. Huấn luyện ở độ phân giải
            {IMGSZ}px.<br/><br/>
            {dong_chi_so}
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("---")

        st.markdown("### 🎚️ Ngưỡng tin cậy")
        nguong_conf = st.slider(
            "Chỉ hiển thị box có độ tin cậy ≥",
            min_value=0.05,
            max_value=0.95,
            value=0.25,
            step=0.05,
            help="Hạ ngưỡng sẽ bắt được nhiều biển hơn nhưng cũng dễ báo nhầm hơn.",
        )

        st.markdown("---")

        st.markdown("### 🏷️ 7 lớp biển báo nhận diện được")
        st.markdown("\n".join(f"- {ten}" for ten in TEN_LOP_TIENG_VIET))

        st.markdown("---")

        st.markdown("### 📖 Về đồ án")
        st.markdown(
            """
            Đồ án nhận diện biển báo giao thông Việt Nam, dữ liệu từ
            **Zalo AI Challenge 2020** (định dạng COCO JSON, ảnh Panorama
            1622x626). Ba kiến trúc được huấn luyện và so sánh trên cùng một
            tập Hold-out Test 20% được tách riêng từ đầu.
            """
        )

        st.markdown("---")

        st.markdown("### 🔗 Liên kết")
        st.markdown(
            """
            - [Ultralytics YOLOv8 Docs](https://docs.ultralytics.com/)
            - [Zalo AI Challenge](https://challenge.zalo.ai/)
            """
        )

    return nguong_conf


# ============================================================================
# TAB: DEMO
# ============================================================================
def render_tab_demo(nguong_conf: float):
    st.markdown("### 📸 Upload ảnh để nhận diện biển báo")
    st.markdown("Hỗ trợ định dạng: JPG, JPEG, PNG")

    file_tai_len = st.file_uploader(
        "Chọn ảnh...",
        type=["jpg", "jpeg", "png"],
        help="Upload ảnh đường phố có chứa biển báo giao thông Việt Nam.",
    )

    if file_tai_len is None:
        st.info("👆 Vui lòng upload một ảnh để bắt đầu!")
        with st.expander("💡 Gợi ý để có kết quả tốt nhất"):
            st.markdown(
                f"""
                - Mô hình được huấn luyện trên ảnh dashcam/panorama của bộ Zalo AI,
                  nên ảnh chụp từ góc nhìn trên đường sẽ cho kết quả sát nhất.
                - Chỉ nhận diện được **7 lớp biển báo** liệt kê ở sidebar, không
                  nhận diện người, xe hay vật thể khác.
                - Biển báo trong bộ dữ liệu rất nhỏ (diện tích trung vị chỉ khoảng
                  266 px²), nên ảnh càng nét càng dễ bắt.
                - Ảnh được suy luận ở độ phân giải {IMGSZ}px. Trên máy chủ chỉ có
                  CPU, mỗi ảnh mất khoảng vài giây - đây là chuyện bình thường.
                """
            )
        return

    anh = Image.open(file_tai_len).convert("RGB")

    with st.spinner("🔄 Đang tải mô hình từ Hugging Face Hub..."):
        try:
            model = tai_model(lay_repo_id())
        except Exception as loi:
            st.error(
                f"❌ **Không tải được mô hình!**\n\n"
                f"Repo đang trỏ tới: `{lay_repo_id()}`\n\n"
                f"Chi tiết lỗi: {loi}"
            )
            st.info(
                "Kiểm tra lại repo id trên Hugging Face Hub, và chắc chắn repo đó "
                f"có file `{HF_TEN_FILE}`. Nếu repo ở chế độ riêng tư thì phải khai "
                "báo thêm token truy cập."
            )
            return

    with st.spinner(f"🔍 Đang phân tích ảnh ở độ phân giải {IMGSZ}px..."):
        try:
            ket_qua = model.predict(
                anh,
                imgsz=IMGSZ,
                conf=nguong_conf,
                iou=IOU_NMS,
                max_det=MAX_DET,
                verbose=False,
            )[0]
            anh_ve, danh_sach_box = ve_ket_qua(anh, ket_qua, nguong_conf)
        except Exception as loi:
            st.error(f"❌ **Lỗi khi suy luận!**\n\nChi tiết lỗi: {loi}")
            return

    cot_trai, cot_phai = st.columns(2)
    with cot_trai:
        st.markdown("#### 🖼️ Ảnh gốc")
        st.image(anh, use_container_width=True)
    with cot_phai:
        st.markdown("#### 🎯 Kết quả nhận diện")
        st.image(anh_ve, use_container_width=True)

    st.markdown("---")
    st.markdown("### 📋 Chi tiết các biển báo phát hiện được")

    if not danh_sach_box:
        st.info(
            f"Không phát hiện biển báo nào với ngưỡng tin cậy ≥ {nguong_conf:.0%}. "
            "Thử hạ ngưỡng ở sidebar hoặc upload ảnh khác."
        )
        return

    st.markdown(
        f"""
        <div class="result-box">
            <h2>✅ Phát hiện {len(danh_sach_box)} biển báo</h2>
        </div>
        """,
        unsafe_allow_html=True,
    )

    danh_sach_sap_xep = sorted(danh_sach_box, key=lambda b: -b["do_tin_cay"])
    for thu_tu, box in enumerate(danh_sach_sap_xep, start=1):
        c1, c2 = st.columns([3, 1])
        with c1:
            st.markdown(f"**[{thu_tu}] {box['nhan']}**")
            st.progress(box["do_tin_cay"])
        with c2:
            st.markdown(f"**{box['do_tin_cay'] * 100:.2f}%**")


# ============================================================================
# TAB: VỀ MODEL
# ============================================================================
def _ve_bieu_do_epoch(lich_su: dict):
    """Vẽ 2 biểu đồ đường từ nhật ký huấn luyện: Loss và mAP theo từng epoch."""
    cac_moc = lich_su.get("history", [])
    if not cac_moc:
        st.warning("File nhật ký không có dữ liệu epoch nào.")
        return

    bang_epoch = pd.DataFrame(cac_moc).set_index("epoch_id")

    # Tìm epoch đạt mAP@50-95 cao nhất - đây chính là epoch mà best.pt lưu lại
    moc_tot_nhat = max(cac_moc, key=lambda m: m["mAP_50_95"])

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Số epoch đã chạy", f"{lich_su.get('epochs_thuc_te', len(cac_moc))}")
    c2.metric("Epoch dự kiến", f"{lich_su.get('epochs_du_kien', '?')}")
    c3.metric("Early Stopping patience", f"{lich_su.get('early_stopping_patience', '?')}")
    c4.metric("Epoch tốt nhất", f"{moc_tot_nhat['epoch_id']}")

    so_epoch = lich_su.get("epochs_thuc_te", len(cac_moc))
    so_epoch_du_kien = lich_su.get("epochs_du_kien")
    so_epoch_sau_dinh = so_epoch - moc_tot_nhat["epoch_id"]
    patience = lich_su.get("early_stopping_patience")

    # Phân biệt rõ 2 lý do dừng: hội tụ thật sự hay bị cắt ngang. Đây là chi tiết
    # bắt buộc phải trung thực trong báo cáo, nhìn biểu đồ là hội đồng thấy ngay.
    if so_epoch_du_kien and so_epoch >= so_epoch_du_kien:
        st.success(f"Mô hình chạy trọn {so_epoch} epoch theo đúng kế hoạch.")
    elif patience and so_epoch_sau_dinh >= patience:
        st.success(
            f"**Early Stopping đã kích hoạt.** Sau epoch {moc_tot_nhat['epoch_id']}, "
            f"mô hình chạy thêm {so_epoch_sau_dinh} epoch mà không cải thiện nên "
            "dừng lại. Đây là dấu hiệu mô hình đã hội tụ."
        )
    else:
        st.warning(
            f"**Quá trình huấn luyện bị cắt ngang.** Đỉnh rơi vào epoch "
            f"{moc_tot_nhat['epoch_id']}, mới chạy thêm {so_epoch_sau_dinh} epoch "
            f"thì dừng, trong khi ngưỡng Early Stopping là {patience}. "
            "Nghĩa là mô hình chưa hội tụ hẳn khi phiên huấn luyện kết thúc."
        )

    st.markdown("#### 📉 Đường cong Loss")
    st.markdown(
        "Dùng để phát hiện overfitting: nếu `val_loss` quay đầu đi lên trong khi "
        "`train_loss` vẫn giảm thì mô hình đã bắt đầu học vẹt."
    )
    st.line_chart(bang_epoch[["train_loss", "val_loss"]], height=320)

    st.markdown("#### 📈 Đường cong mAP trên tập Validation")
    st.markdown(
        "Dùng để trả lời câu hỏi mô hình đã hội tụ chưa: đường đi ngang ở cuối "
        "nghĩa là đã tới hạn, còn vẫn đang dốc lên nghĩa là huấn luyện thêm còn cải thiện."
    )
    st.line_chart(bang_epoch[["mAP_50", "mAP_50_95"]], height=320)

    with st.expander("📄 Xem bảng số liệu chi tiết từng epoch"):
        st.dataframe(bang_epoch, use_container_width=True)


def render_tab_model():
    bang = doc_bang_so_sanh()
    chi_so = lay_chi_so_model_dang_dung(bang)

    # ---------- Phần 1: Chỉ số của mô hình đang triển khai ----------
    st.markdown(f"### 🎯 Kết quả của {TEN_MODEL} trên tập Hold-out Test")

    if chi_so is None:
        st.warning(
            f"Chưa tìm thấy file `{DUONG_DAN_BANG_SO_SANH.name}` trong thư mục "
            f"`eval-results/`. Chạy notebook `evaluate_3_models.ipynb` rồi chép "
            "file kết quả vào đó."
        )
    else:
        st.markdown(
            "Toàn bộ con số dưới đây đo trên **tập Hold-out Test 20%** — phần dữ "
            "liệu được tách riêng và giấu đi từ đầu, mô hình chưa từng nhìn thấy "
            "trong suốt quá trình huấn luyện."
        )
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("mAP@50", f"{chi_so['mAP@50 (%)']:.2f}%")
        c2.metric("mAP@50-95", f"{chi_so['mAP@50-95 (%)']:.2f}%")
        c3.metric("mAP vật thể nhỏ", f"{chi_so['mAP_small (%)']:.2f}%")
        c4.metric("Tốc độ (GPU)", f"{chi_so['FPS']:.1f} FPS")
        st.caption(
            f"Suy luận {chi_so['Inference (ms/anh)']:.2f} ms/ảnh ở độ phân giải "
            f"{chi_so['imgsz']}px. Lưu ý con số FPS đo trên GPU; Web App chạy "
            "thuần CPU nên chậm hơn đáng kể."
        )

    st.markdown("---")

    # ---------- Phần 2: Quá trình huấn luyện theo từng epoch ----------
    st.markdown("### 📊 Quá trình huấn luyện theo từng epoch")

    lich_su = doc_lich_su_huan_luyen()
    if lich_su is None:
        st.warning(
            f"Chưa có file `{DUONG_DAN_LICH_SU.name}` trong thư mục `eval-results/`. "
            "File này do notebook huấn luyện sinh ra, nằm trong gói "
            "`yolov8_v3_results.zip` tải về từ Kaggle."
        )
    else:
        _ve_bieu_do_epoch(lich_su)

    st.markdown("---")

    # ---------- Phần 3: Ma trận nhầm lẫn ----------
    st.markdown("### 🔢 Ma trận nhầm lẫn")
    if DUONG_DAN_MA_TRAN.exists():
        st.markdown(
            "Cho biết mô hình hay nhầm lớp nào với lớp nào. Các biển báo cùng có "
            "viền đỏ tròn thường là nhóm dễ lẫn nhất."
        )
        st.image(str(DUONG_DAN_MA_TRAN), use_container_width=True)
    else:
        st.info(f"Chưa có file `{DUONG_DAN_MA_TRAN.name}` trong `eval-results/`.")

    st.markdown("---")

    # ---------- Phần 4: Dataset ----------
    st.markdown("### 🗂️ Bộ dữ liệu")
    st.markdown(
        """
        - **Nguồn:** Zalo AI Challenge 2020 - Traffic Sign Detection.
        - **Định dạng gốc:** COCO JSON (`info`, `images`, `annotations`, `categories`).
        - **Độ phân giải ảnh:** khoảng 1622 x 626 px (ảnh Panorama từ dashcam).
        - **Số lớp:** 7 loại biển báo giao thông Việt Nam.
        - **Cách chia:** 70% Train / 10% Validation / **20% Hold-out Test**, chia
          bằng `split_dataset.py` với `random_seed=42` cố định. Tập Test được cắt
          ra **đầu tiên** và lưu ở thư mục riêng, không hề khai báo trong `data.yaml`
          dùng để huấn luyện, nên tuyệt đối không có rò rỉ dữ liệu.
        - **Thử thách chính:** vật thể siêu nhỏ (diện tích trung vị chỉ khoảng
          266 px²) và mất cân bằng lớp ở mức 1:5.5.
        """
    )

    st.markdown("---")

    # ---------- Phần 5: So sánh 3 mô hình và lý do chọn ----------
    st.markdown("### 🔬 So sánh 3 kiến trúc và lý do chọn mô hình này")

    if bang is None:
        st.info("Chưa có bảng so sánh để hiển thị.")
        return

    st.dataframe(bang, use_container_width=True, hide_index=True)

    if DUONG_DAN_ANH_PARETO.exists():
        st.markdown("#### ⚖️ Đồ thị Pareto: Độ chính xác đổi lấy Tốc độ")
        st.markdown(
            "Mô hình nằm trên đường biên Pareto là những mô hình không bị mô hình "
            "nào khác vừa nhanh hơn vừa chính xác hơn — tức là các ứng viên đáng cân nhắc."
        )
        st.image(str(DUONG_DAN_ANH_PARETO), use_container_width=True)

    st.markdown(
        f"""
        #### 💡 Vì sao chọn {TEN_MODEL} thay vì RT-DETR-L?

        Xét thuần độ chính xác thì RT-DETR-L nhỉnh hơn ở mAP@50-95 (47.15% so với
        42.60%). Nhưng quyết định triển khai không chỉ dựa vào một con số:

        - **Dung lượng:** RT-DETR-L nặng 251.5 MB, YOLOv8s-P2 chỉ 20.8 MB — chênh
          nhau 12 lần. Streamlit Community Cloud chỉ cấp khoảng 1 GB RAM, nạp mô
          hình quá nặng rất dễ bị ngắt tiến trình.
        - **Tốc độ:** YOLOv8s-P2 đạt 18.2 FPS so với 13.0 FPS của RT-DETR-L. Khoảng
          cách này còn giãn rộng hơn khi chuyển sang chạy thuần CPU.
        - **mAP@50 lại cao hơn:** 73.11% so với 71.98%. Với bài toán demo trên web,
          ngưỡng IoU 0.5 phản ánh trải nghiệm người dùng sát hơn là mAP@50-95.

        Nói cách khác, đây là đánh đổi 4.55 điểm mAP@50-95 để lấy mô hình nhẹ hơn
        12 lần và nhanh hơn 40%. Với ràng buộc CPU-only của nền tảng triển khai,
        đây là lựa chọn khả thi duy nhất.
        """
    )


# ============================================================================
# MAIN APP
# ============================================================================
def main():
    nguong_conf = render_sidebar()

    st.markdown(
        '<h1 class="main-header">🚦 Nhận diện Biển báo Giao thông 🚧</h1>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="sub-header">Hệ thống nhận diện biển báo giao thông Việt Nam '
        f"bằng {TEN_MODEL} — dữ liệu Zalo AI Challenge 2020</p>",
        unsafe_allow_html=True,
    )

    tab_demo, tab_model = st.tabs(["🎮 Demo", "🔬 Về Model"])
    with tab_demo:
        render_tab_demo(nguong_conf)
    with tab_model:
        render_tab_model()

    st.markdown("---")
    st.markdown(
        "<p style='text-align: center; color: #888;'>Đồ án Nhận diện Biển báo "
        "Giao thông | Dữ liệu: Zalo AI Challenge 2020</p>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
