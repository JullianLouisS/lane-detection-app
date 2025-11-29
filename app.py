import streamlit as st
import numpy as np
from PIL import Image
from skimage import filters, io
import io as io_module

st.set_page_config(page_title="Lane Detection", page_icon="🛣️", layout="wide")

st.title("🛣️ Lane Detection App")
st.write("Upload a road image to detect lanes")

method = st.sidebar.radio("Detection Method:", ["Canny", "Sobel", "Laplacian"])

uploaded_file = st.file_uploader("Upload Image:", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    img_array = np.array(image)
    
    # Convert to grayscale
    if len(img_array.shape) == 3:
        gray = np.mean(img_array, axis=2) / 255.0
    else:
        gray = img_array / 255.0
    
    if method == "Canny":
        threshold1 = st.sidebar.slider("Threshold 1", 0.0, 1.0, 0.1)
        threshold2 = st.sidebar.slider("Threshold 2", 0.0, 1.0, 0.2)
        edges = filters.sobel(gray)
        edges = (edges > threshold2).astype(float)
        
    elif method == "Sobel":
        edges = filters.sobel(gray)
        edges = (edges * 255).astype(np.uint8)
        
    else:  # Laplacian
        edges = filters.laplace(gray)
        edges = np.uint8(np.absolute(edges) * 255 / np.max(np.absolute(edges)))
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original Image")
        st.image(image, use_column_width=True)
    
    with col2:
        st.subheader(f"{method} Detection")
        edges_display = (edges * 255).astype(np.uint8) if edges.max() <= 1 else edges
        st.image(edges_display, use_column_width=True, channels="GRAY")
    
    # Download
    from PIL import Image as PILImage
    result_image = PILImage.fromarray((edges * 255).astype(np.uint8) if edges.max() <= 1 else edges)
    buf = io_module.BytesIO()
    result_image.save(buf, format="PNG")
    buf.seek(0)
    
    st.download_button(
        label="📥 Download Result",
        data=buf.getvalue(),
        file_name=f"lanes_{method}.png",
        mime="image/png"
    )
else:
    st.info("👆 Upload an image to start")
