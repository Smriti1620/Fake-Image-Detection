import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

st.set_page_config(page_title="AI Vision", page_icon="📸", layout="centered")

st.markdown("""
<h1 style='text-align:center;'>📸✨ AI Vision Detector</h1>
<p style='text-align:center;'>Upload a photo and get instant AI analysis</p>
""", unsafe_allow_html=True)

# ✅ SAFE MODEL LOAD (MOST STABLE WAY)
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(
        "fake_image_detector.h5",
        compile=False
    )
    return model

model = load_model()

def predict(img):
    img = img.convert("RGB").resize((224, 224))
    arr = np.array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)
    return float(model.predict(arr, verbose=0)[0][0])

uploaded_file = st.file_uploader("📤 Upload Image", type=["jpg","jpeg","png"])

if uploaded_file:
    img = Image.open(uploaded_file)

    st.image(img, use_container_width=True)

    real = predict(img)
    fake = 1 - real

    st.progress(real)

    col1, col2 = st.columns(2)

    with col1:
        st.metric("😊 Real", f"{real:.2f}")

    with col2:
        st.metric("🤖 AI", f"{fake:.2f}")

    if real > 0.65:
        st.success("🎉 Looks Real")
    elif real < 0.35:
        st.warning("🤖 Looks AI Generated")
    else:
        st.info("📊 Mixed Confidence")

    fig, ax = plt.subplots()
    ax.bar(["Real", "AI"], [real, fake])
    st.pyplot(fig)
