import streamlit as st
import os

st.set_page_config(
    page_title="Multimodal Disaster Response DSS",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Premium CSS Injection
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .main-title {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(135deg, #FF4B4B 0%, #FF8F00 50%, #FFD600 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }

    .subtitle {
        font-size: 1.25rem;
        color: #B0BEC5;
        font-weight: 400;
        margin-bottom: 2rem;
    }

    .feature-card {
        background: rgba(33, 37, 43, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
        margin-bottom: 1.5rem;
        transition: transform 0.3s ease, border-color 0.3s ease;
    }

    .feature-card:hover {
        transform: translateY(-5px);
        border-color: rgba(255, 75, 75, 0.5);
    }

    .feature-icon {
        font-size: 2rem;
        margin-bottom: 1rem;
    }

    .feature-title {
        font-size: 1.2rem;
        font-weight: 600;
        color: #FFFFFF;
        margin-bottom: 0.5rem;
    }

    .feature-desc {
        font-size: 0.95rem;
        color: #90A4AE;
        line-height: 1.4;
    }

    .metric-container {
        display: flex;
        justify-content: space-around;
        margin-top: 2rem;
    }

    .badge {
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Main layout
st.markdown("<div class='main-title'>Multimodal Disaster Response DSS</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Hệ Hỗ Trợ Quyết Định Ứng Phó Thảm Họa Đa Phương Thức Từ Mạng Xã Hội</div>", unsafe_allow_html=True)

# Grid Layout for Intro
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("""
    ### 🚨 Bối cảnh Dự Án
    Trong tình huống thiên tai xảy ra (bão, lũ lụt, động đất, cháy rừng), mạng xã hội (Twitter, Facebook) ngập tràn các bài đăng kêu cứu, báo cáo thiệt hại, quyên góp nhu yếu phẩm hoặc tin tức giả, spam. Các đội ứng cứu khẩn cấp luôn rơi vào tình trạng **quá tải thông tin** và không thể chọn lọc các bài đăng khẩn cấp một cách thủ công.

    Dự án xây dựng một **Hệ hỗ trợ quyết định (Decision Support System - DSS)** giúp tự động hóa việc lọc, phân loại và phân luồng thông tin cứu nạn bằng cách phân tích đồng thời **Văn bản (Text)** và **Hình ảnh (Image)** đi kèm từ CrisisMMD dataset.

    ### ⚙️ Lớp Quyết Định (Decision Support Layer)
    Điểm cốt lõi của hệ thống DSS này là việc chuyển đổi từ kết quả dự báo Học máy (Predictive probabilities) thành **Hành động cụ thể** thông qua:
    1. **Tính toán Điểm rủi ro (Risk Score):** Tích hợp xác suất Informative, trọng số nhóm cứu trợ khẩn cấp, mật độ keyword nguy hiểm, và độ tin cậy của mô hình.
    2. **Đề xuất Mức độ Ưu tiên (Priority Level):** Phân chia thành High, Medium, Low.
    3. **Phân luồng Đội Cứu trợ (Routing Rule):** Chuyển trực tiếp tới Đội khẩn cấp (Emergency), Đội kiểm tra hạ tầng (Infrastructure), Đội phân phối nhu yếu phẩm (Relief), hoặc Đội điều phối chung (Coordination).
    4. **Cơ chế duyệt thủ công (Manual Review / Supervisor Override):** Tự động phát hiện các bài đăng có văn bản và hình ảnh mâu thuẫn (như text kêu cứu nhưng ảnh là meme hài hước) để Supervisor thẩm định lại.
    """)

with col2:
    gallery = os.path.join("reports", "figures", "image_gallery.png")
    if os.path.exists(gallery):
        st.image(
            gallery,
            caption="Real CrisisMMD images used by the pipeline",
            width="stretch",
        )

st.markdown("---")

# Feature Cards representing Pages
st.markdown("### 🛠️ Các Phân Hệ Của Dashboard (Menu bên trái)")

f_col1, f_col2, f_col3 = st.columns(3)

with f_col1:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-icon">📊</div>
        <div class="feature-title">1. Overview</div>
        <div class="feature-desc">Tổng quan thống kê về toàn bộ các bài đăng thiên tai, tỷ lệ thông tin khẩn cấp, phân bố thảm họa và độ ưu tiên phân bổ nguồn lực.</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="feature-card">
        <div class="feature-icon">📝</div>
        <div class="feature-title">2. EDA</div>
        <div class="feature-desc">Phân tích khám phá dữ liệu trực quan: độ dài văn bản, từ khóa khẩn cấp phổ biến nhất, và hiển thị thư viện ảnh thực tế theo nhóm.</div>
    </div>
    """, unsafe_allow_html=True)

with f_col2:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-icon">📈</div>
        <div class="feature-title">3. Model Evaluation</div>
        <div class="feature-desc">Đánh giá hiệu năng và so sánh giữa Text-only model, Image-only model, và Multimodal Fusion model để minh chứng hiệu quả DSS.</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="feature-card">
        <div class="feature-icon">🔍</div>
        <div class="feature-title">4. Single Case Demo</div>
        <div class="feature-desc">Nhập văn bản và tải lên hình ảnh tùy ý để kiểm tra trực tiếp kết quả phân luồng cứu nạn và tính điểm rủi ro thời gian thực.</div>
    </div>
    """, unsafe_allow_html=True)

with f_col3:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-icon">📥</div>
        <div class="feature-title">5. Priority Queue</div>
        <div class="feature-desc">Hàng đợi ưu tiên xử lý hàng loạt bài đăng. Sắp xếp danh sách bài đăng theo Risk Score giảm dần để các đội phân phối nguồn lực cứu trợ hợp lý.</div>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center; color: #90A4AE;'>Multimodal Disaster Response DSS - Đồ án Hệ Hỗ Trợ Quyết Định</p>", unsafe_allow_html=True)
