import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.set_page_config(page_title="Lane Detection", page_icon="🛣️", layout="wide")

st.title("🛣️ Lane Detection App")
st.write("Upload a road image to detect lanes")

method = st.sidebar.radio("Detection Method:", ["Canny", "Sobel", "Laplacian"])

uploaded_file = st.file_uploader("Upload Image:", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    img_array = np.array(image)
    
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array
    
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    if method == "Canny":
        threshold1 = st.sidebar.slider("Threshold 1", 50, 200, 100)
        threshold2 = st.sidebar.slider("Threshold 2", 100, 400, 200)
        edges = cv2.Canny(blurred, threshold1, threshold2)
    
    elif method == "Sobel":
        kernel_size = st.sidebar.slider("Kernel Size", 1, 7, 3)
        kernel_size = kernel_size * 2 + 1
        sobelx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=kernel_size)
        sobely = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=kernel_size)
        edges = np.hypot(sobelx, sobely)
        edges = np.uint8(255 * edges / np.max(edges))
    
    else:
        edges = cv2.Laplacian(blurred, cv2.CV_64F)
        edges = np.uint8(np.absolute(edges))
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original Image")
        st.image(image, use_column_width=True)
    
    with col2:
        st.subheader(f"{method} Detection")
        st.image(edges, use_column_width=True, channels="GRAY")
    
    _, buffer = cv2.imencode('.png', edges)
    st.download_button(
        label="📥 Download Result",
        data=buffer.tobytes(),
        file_name=f"lanes_{method}.png",
        mime="image/png"
    )

else:
    st.info("👆 Upload an image to start")