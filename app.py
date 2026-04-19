import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import matplotlib.pyplot as plt

# ---------------- PAGE SETUP ----------------
st.set_page_config(
    page_title="AI Vision",
    page_icon="📸",
    layout="centered"
)

# 🎨 Instagram-style header
st.markdown("""
    <h1 style='text-align: center; color: #ffffff;'>
    📸✨ AI Vision Detector
    </h1>
    <p style='text-align: center; color: gray;'>
    Upload a photo and get instant AI analysis
    </p>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL (FIXED) ----------------
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(
        "fake_image_detector.h5",
        compile=False,   # ✅ prevents Keras config error
        safe_mode=False  # ✅ avoids config mismatch crash
    )
    return model

model = load_model()

# ---------------- PREDICT ----------------
def predict(img):
    img = img.convert("RGB")
    img = img.resize((224, 224))
    arr = np.array(img).astype(np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)
    pred = model.predict(arr, verbose=0)[0][0]
    return float(pred)

# ---------------- UPLOAD ----------------
uploaded_file = st.file_uploader("📤 Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file:

    img = Image.open(uploaded_file)

    # 📸 Image display
    st.markdown("### 🖼️ Your Post")
    st.image(img, use_container_width=True)

    # 🧠 prediction
    real = predict(img)
    fake = 1 - real

    # 🎴 STYLE CARDS
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"""
        <div style="
            background-color:#1e1e1e;
            padding:20px;
            border-radius:15px;
            text-align:center;
            box-shadow: 0px 4px 10px rgba(0,0,0,0.3);
        ">
        <h3>😊 Real</h3>
        <h2 style="color:#4CAF50;">{real:.2f}</h2>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div style="
            background-color:#1e1e1e;
            padding:20px;
            border-radius:15px;
            text-align:center;
            box-shadow: 0px 4px 10px rgba(0,0,0,0.3);
        ">
        <h3>🤖 AI / Edited</h3>
        <h2 style="color:#FF9800;">{fake:.2f}</h2>
        </div>
        """, unsafe_allow_html=True)

    # 📊 progress bar
    st.progress(real)

    # 🎯 DECISION LOGIC
    if real >= 0.65:
        st.success("🎉 Looks Real")
        st.balloons()

    elif real <= 0.35:
        st.warning("🤖 Looks AI Generated")
        st.snow()

    else:
        st.info("📊 Mixed Confidence — check values below")

    # 📊 chart
    st.markdown("### 📊 Confidence Breakdown")

    fig, ax = plt.subplots()
    ax.bar(["Real 😊", "AI 🤖"], [real, fake], color=["#4CAF50", "#FF9800"])
    ax.set_ylim(0, 1)

    st.pyplot(fig)
