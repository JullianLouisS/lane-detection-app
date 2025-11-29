import streamlit as st
import numpy as np
from PIL import Image, ImageFilter, ImageOps
import io

st.set_page_config(page_title="Lane Detection", page_icon="🛣️", layout="wide")

st.title("🛣️ Lane Detection App")
st.write("Upload a road image to detect lanes")

method = st.sidebar.radio("Detection Method:", ["Edge Detection", "Blur", "Contrast"])

uploaded_file = st.file_uploader("Upload Image:", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    
    if method == "Edge Detection":
        processed = image.filter(ImageFilter.FIND_EDGES)
    elif method == "Blur":
        processed = image.filter(ImageFilter.GaussianBlur(radius=2))
    else:  # Contrast
        processed = ImageOps.autocontrast(image)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original Image")
        st.image(image, use_column_width=True)
    
    with col2:
        st.subheader(f"{method} Result")
        st.image(processed, use_column_width=True)
    
    # Download
    buf = io.BytesIO()
    processed.save(buf, format="PNG")
    buf.seek(0)
    
    st.download_button(
        label="📥 Download Result",
        data=buf.getvalue(),
        file_name=f"lanes_{method}.png",
        mime="image/png"
    )
else:
    st.info("👆 Upload an image to start")
