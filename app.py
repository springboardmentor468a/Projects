import streamlit as st
from PIL import Image
import numpy as np
import cv2
import io

# ------------------- Page Configuration -------------------
st.set_page_config(page_title="OneView Pro", page_icon="🖼️", layout="wide")

# ------------------- Custom CSS -------------------
st.markdown("""
<style>
.main { background-color: #f8f9fa; }
.stApp { background-color: #ffffff; border-radius: 12px; }
h1 { color: #3a0ca3; text-align: center; font-weight: 700; }
h2 { color: #3f37c9; }
.stButton>button { background-color: #3f37c9; color: white; border-radius: 10px; height: 3em; width: 10em; }
.stButton>button:hover { background-color: #4361ee; color: white; }
</style>
""", unsafe_allow_html=True)

# ------------------- Sidebar Navigation -------------------
st.sidebar.title("🧭 Navigation")
page = st.sidebar.radio("Go to:", ["🏠 Home", "🧩 Segmentation", "🎨 Filters", "💾 Export"])

# ------------------- Home Page -------------------
if page == "🏠 Home":
    st.title("🖼️ OneView Pro – Smart Image Segmentation Tool")
    st.write("""
    Welcome to OneView Pro, your all-in-one intelligent image segmentation and enhancement tool.
    - Upload your images.
    - Apply segmentation and visual filters.
    - Download or share your enhanced results.
    """)
    st.image("https://cdn.pixabay.com/photo/2016/11/29/06/15/artificial-intelligence-1867226_1280.jpg", use_container_width=True)

# ------------------- Segmentation Page -------------------
elif page == "🧩 Segmentation":
    st.title("🧩 Image Segmentation")
    uploaded_file = st.file_uploader("Upload your image", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="📸 Original Image", use_container_width=True)

        # Convert to numpy array
        img_array = np.array(image)

        # Example segmentation: grayscale + threshold
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        _, segmented = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)

        st.image(segmented, caption="🧠 Segmented Output", use_container_width=True)
        st.session_state["segmented"] = segmented
    else:
        st.info("Please upload an image to start segmentation.")

# ------------------- Filters Page -------------------
elif page == "🎨 Filters":
    st.title("🎨 Apply Filters")

    if "segmented" in st.session_state:
        segmented = st.session_state["segmented"]
        filter_type = st.selectbox("Choose a filter", ["None", "Blur", "Edge Detection", "Invert Colors"])

        if filter_type == "Blur":
            filtered = cv2.GaussianBlur(segmented, (7, 7), 0)
        elif filter_type == "Edge Detection":
            filtered = cv2.Canny(segmented, 100, 200)
        elif filter_type == "Invert Colors":
            filtered = cv2.bitwise_not(segmented)
        else:
            filtered = segmented

        st.image(filtered, caption=f"🎨 Filter Applied: {filter_type}", use_container_width=True)
        st.session_state["filtered"] = filtered
    else:
        st.warning("Please segment an image first in the 'Segmentation' section.")

# ------------------- Export Page -------------------
elif page == "💾 Export":
    st.title("💾 Download Processed Image")

    if "filtered" in st.session_state:
        result = Image.fromarray(st.session_state["filtered"])
        buf = io.BytesIO()
        result.save(buf, format="PNG")
        byte_im = buf.getvalue()

        st.download_button(
            label="📥 Download Image",
            data=byte_im,
            file_name="processed_image.png",
            mime="image/png"
        )
        st.success("Your image is ready to download!")
    else:
        st.warning("No processed image available. Please segment and apply filters first.")



