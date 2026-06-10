import streamlit as st
import os
import re
import tempfile
import pandas as pd
from PIL import Image
from src.fusion import MultimodalFusionPredictor
from src.decision_rules import get_dss_decision
from src.config import CRISIS_KEYWORDS
from src.data_loader import load_dataset
from src.runtime import real_image_corpus_available

st.set_page_config(page_title="Single Demo - Disaster Response DSS", layout="wide")

# Custom Styled HTML Containers
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .decision-container {
        border-radius: 12px;
        padding: 1.8rem;
        color: #FFFFFF;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.25);
        margin-bottom: 2rem;
    }

    .priority-high {
        background: linear-gradient(135deg, #7D0A0A 0%, #B31312 100%);
        border: 2px solid #FF1744;
    }
    .priority-medium {
        background: linear-gradient(135deg, #E65100 0%, #FF9100 100%);
        border: 2px solid #FFAB40;
    }
    .priority-low {
        background: linear-gradient(135deg, #1B5E20 0%, #2E7D32 100%);
        border: 2px solid #00E676;
    }
    .priority-review {
        background: linear-gradient(135deg, #4A148C 0%, #7B1FA2 100%);
        border: 2px solid #E040FB;
    }

    .title-text {
        font-size: 0.95rem;
        text-transform: uppercase;
        letter-spacing: 2px;
        color: rgba(255, 255, 255, 0.8);
        margin-bottom: 0.4rem;
        font-weight: 600;
    }

    .val-text {
        font-size: 2.8rem;
        font-weight: 800;
        margin-bottom: 1rem;
        line-height: 1;
    }

    .detail-item {
        font-size: 1.1rem;
        margin-bottom: 0.6rem;
        line-height: 1.4;
    }
</style>
""", unsafe_allow_html=True)

st.title("🔍 Real-time Disaster Post Decision Simulator")
st.markdown("Nhập văn bản và tải lên hình ảnh để mô phỏng đưa ra quyết định cứu trợ và phân luồng khẩn cấp.")

# Load Predictor
@st.cache_resource
def get_predictor():
    return MultimodalFusionPredictor()

try:
    predictor = get_predictor()

    # Input panel
    col_in1, col_in2 = st.columns([3, 2])

    with col_in1:
        st.subheader("1. Enter Tweet Content")
        tweet_text = st.text_area(
            "Tweet Text:",
            value="Urgent! We are trapped on the second floor of the building due to rising flood waters. Bridge collapsed near Main street. Need immediate rescue!",
            height=150
        )

    with col_in2:
        st.subheader("2. Select or Upload Image")
        image_options = ["Text only", "Upload your own image"]
        if real_image_corpus_available():
            image_options.append("Choose from local CrisisMMD images")
        img_option = st.selectbox(
            "Choose a method:",
            image_options,
        )

        selected_img_path = None
        uploaded_image = None

        if img_option == "Choose from local CrisisMMD images":
            _, _, demo_df, image_base = load_dataset(use_sample_if_missing=False)
            samples = demo_df.groupby("label_top", group_keys=False).head(1)
            category_names = {
                "affected_individuals": "Affected individuals",
                "injured_or_dead_people": "Injured or dead",
                "infrastructure_and_utility_damage": "Infrastructure damage",
                "missing_or_found_people": "Missing or found",
                "not_humanitarian": "Not humanitarian",
                "other_relevant_information": "Other relevant",
                "rescue_volunteering_or_donation_effort": "Rescue or donation",
                "vehicle_damage": "Vehicle damage",
            }
            options = {
                f"{category_names.get(row['label_top'], row['label_top'])} | {str(row['image_id'])[-8:]}": os.path.join(
                    image_base, row["image"]
                )
                for _, row in samples.iterrows()
            }
            selected_label = st.selectbox("Select a real image:", list(options))
            selected_img_path = options[selected_label]
            st.image(Image.open(selected_img_path), caption=selected_label, width=220)
        elif img_option == "Upload your own image":
            uploaded_image = st.file_uploader(
                "Upload Image:",
                type=["png", "jpg", "jpeg"],
            )
            if uploaded_image:
                img = Image.open(uploaded_image).convert("RGB")
                st.image(img, caption="Uploaded image", width=150)
                st.caption(
                    "The first image prediction may take longer while the "
                    "pretrained CLIP feature extractor is loaded."
                )

    # Process prediction on button click
    st.markdown("---")

    if st.button("🚀 Run DSS Analysis", type="primary"):
        if not tweet_text.strip():
            st.warning("Please enter some text to analyze.")
        else:
            temp_path = None
            try:
                if uploaded_image is not None:
                    suffix = os.path.splitext(uploaded_image.name)[1] or ".png"
                    with tempfile.NamedTemporaryFile(
                        suffix=suffix,
                        delete=False,
                    ) as stream:
                        stream.write(uploaded_image.getvalue())
                        temp_path = stream.name
                    selected_img_path = temp_path
                with st.spinner("Đang chạy mô hình hỗ trợ quyết định..."):
                    res = predictor.predict(tweet_text, selected_img_path)
                    dec = get_dss_decision(res, tweet_text)
            finally:
                if temp_path and os.path.exists(temp_path):
                    os.remove(temp_path)

            st.subheader("📢 DSS Recommended Output")

            out1, out2, out3 = st.columns(3)
            out1.metric("Priority", dec["priority"])
            out2.metric("Risk Score", f"{dec['risk_score']} / 100")
            out3.metric("Response Team", dec["assigned_team"])
            st.info(f"**Recommended action:** {dec['recommended_action']}")

            if dec["manual_review"]:
                st.warning(f"🚨 **Supervisor Warning:** {dec['review_reason']}")

            # Detail Analysis Tabs
            tab_anal1, tab_anal2 = st.columns(2)

            with tab_anal1:
                st.subheader("🧠 Evidence & Model Probabilities")

                # Highlight crisis keywords
                words = tweet_text.split()
                highlighted_words = []
                keyword_matches = 0
                for w in words:
                    clean_w = re.sub(r'[^\w]', '', w.lower())
                    if clean_w in CRISIS_KEYWORDS:
                        highlighted_words.append(f"<mark style='background-color: #FF5252; color: white; padding: 2px 5px; border-radius: 3px;'>{w}</mark>")
                        keyword_matches += 1
                    else:
                        highlighted_words.append(w)
                highlighted_text = " ".join(highlighted_words)

                st.markdown(f"**Văn bản đã phân tích (Keyword Highlights):**")
                st.markdown(f"<div style='background-color: rgba(255,255,255,0.05); padding: 15px; border-radius: 8px; border-left: 4px solid #FF5252; font-size: 1.1rem; line-height: 1.5;'>{highlighted_text}</div>", unsafe_allow_html=True)
                st.write("")
                st.write(f"🔍 Tìm thấy **{keyword_matches}** từ khóa khẩn cấp nguy hiểm.")

            with tab_anal2:
                st.subheader("📊 Probability Analysis Matrix")

                # Show Text-only vs Image-only vs Fusion probabilities
                short_category = {
                    "rescue_volunteering_or_donation_effort": "rescue/donation",
                    "infrastructure_and_utility_damage": "infrastructure",
                    "other_relevant_information": "other relevant",
                    "injured_or_dead_people": "injured/dead",
                    "missing_or_found_people": "missing/found",
                    "affected_individuals": "affected people",
                    "vehicle_damage": "vehicle damage",
                    "not_humanitarian": "not humanitarian",
                }
                prob_data = {
                    "Method": ["Text Model", "Image Model" if res['image_present'] else "Image (None)", "Late Fusion DSS"],
                    "P(Informative)": [res['text_informative_prob'], res['image_informative_prob'], res['fused_informative_prob']],
                    "Category": [
                        short_category.get(res['text_category'], res['text_category']),
                        short_category.get(res['image_category'], res['image_category']),
                        short_category.get(res['fused_category'], res['fused_category']),
                    ],
                    "Confidence": [res['text_category_confidence'], res['image_category_confidence'], res['fused_category_confidence']]
                }
                prob_df = pd.DataFrame(prob_data)
                st.dataframe(prob_df, hide_index=True, width="stretch")
                st.metric("Conflict Score (Text vs Image Disagreement)", f"{res['conflict_score']:.3f}")

except Exception as e:
    st.error(f"Error loading models or running prediction: {e}")
    st.info(
        "The deployment bundle must include the five inference model files "
        "listed in `docs/DEPLOY_STREAMLIT.md`."
    )
