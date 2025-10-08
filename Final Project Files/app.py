import io
import os
import streamlit as st
import torch
import torchvision.transforms as T
from torchvision.models.segmentation import deeplabv3_resnet50
from PIL import Image, ImageEnhance
import numpy as np
import cv2

# ---------------- Page Setup ----------------
st.set_page_config(
    page_title="VisionSeg Pro | AI Image Segmentation",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- Custom CSS ----------------
st.markdown("""
<style>
    .main-header {
        font-size: 3.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, #5a67d8 0%, #805ad5 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.8rem;
        letter-spacing: -1px;
    }
    .sub-header {
        font-size: 1.3rem;
        color: #4a5568;
        text-align: center;
        margin-bottom: 2.5rem;
        font-weight: 400;
        line-height: 1.7;
    }
    .feature-card {
        background: rgb(19, 23, 32);
        padding: 2rem;
        border-radius: 14px;
        box-shadow: 0 6px 20px rgba(0,0,0,0.07);
        border-left:5px solid rgb(19, 23, 32);
        margin: 1rem 0;
        text-align: center;
        height: 270px;
        transition: all 0.25s ease;
    }
    .feature-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 10px 28px rgba(0,0,0,0.12);
    }
    .feature-card h3 {
        color: rgb(172, 177, 195);
        font-size: 1.25rem;
        font-weight: 600;
        margin-bottom: 0.8rem;
    }
    .feature-card p {
        color: #718096;
        font-size: 0.95rem;
        line-height: 1.55;
    }
    .feature-icon {
        font-size: 2.2rem;
        margin-bottom: 1.2rem;
    }
    .section-title {
        font-size: 2.1rem;
        font-weight: 700;
        color: #2d3748;
        margin: 2.5rem 0 1.5rem 0;
        text-align: center;
    }
    .metric-card {
        background: #ffffff;
        padding: 1.3rem;
        border-radius: 12px;
        box-shadow: 0 3px 10px rgba(0,0,0,0.05);
        text-align: center;
        border: 1px solid #e2e8f0;
    }
    .metric-card h4 {
        margin-bottom: 0.3rem;
        font-weight: 600;
        font-size: 1.1rem;
    }
    .sample-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
    }
</style>
""", unsafe_allow_html=True)

# ---------------- Header Section ----------------
st.markdown('<h1 class="main-header">VisionSeg Pro</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Pixel-Perfect AI Segmentation</p>', unsafe_allow_html=True)

# ---------------- How It Works Section ----------------
st.markdown("## How It Works")
st.markdown("### Transform your images in three simple steps")

feature_cols = st.columns(3)

with feature_cols[0]:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-icon">📤</div>
        <h3>Upload</h3>
        <p>Upload your image or select from our professional samples.</p>
    </div>
    """, unsafe_allow_html=True)

with feature_cols[1]:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-icon">⚡</div>
        <h3>Process</h3>
        <p>AI analyzes your image with computer vision algorithms.</p>
    </div>
    """, unsafe_allow_html=True)

with feature_cols[2]:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-icon">📥</div>
        <h3>Download</h3>
        <p>Download professional segmented images with custom backgrounds.</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ---------------- Sample Gallery Section ----------------
st.markdown("##  Sample Gallery")
st.markdown("### See our AI in action with professional examples")

# Removed the first sample, keeping only the product sample
sample_images = [
    {
        "original": "samples/sample_product.jpg",
        "segmented": "samples/sample_product_mask.png"
    }
]

with st.container():
    sample_cols = st.columns(2)
    selected_sample = None

    for idx, sample in enumerate(sample_images):
        with sample_cols[idx]:
            with st.container():
                original_exists = os.path.exists(sample["original"])
                segmented_exists = os.path.exists(sample["segmented"])
                
                if original_exists and segmented_exists:
                    original_img = Image.open(sample["original"])
                    segmented_img = Image.open(sample["segmented"])
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.image(original_img, use_container_width=True, caption="Original")
                    with col2:
                        st.image(segmented_img, use_container_width=True, caption="Segmented")
                    
                    if st.button(f"Use This Sample", key=f"sample_{idx}", use_container_width=True):
                        selected_sample = sample
                else:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.image("https://via.placeholder.com/300/667eea/ffffff?text=Original+Image", 
                                use_container_width=True, caption="Original")
                    with col2:
                        st.image("https://via.placeholder.com/300/764ba2/ffffff?text=Segmented+Result", 
                                use_container_width=True, caption="Segmented")
                    
                    st.button(f"Use This Sample", key=f"sample_{idx}", disabled=True, use_container_width=True)

st.markdown("---")

# ---------------- Sidebar ----------------
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    st.markdown("---")
    
    st.markdown("#### Segmentation")
    confidence_threshold = st.slider(
        "Confidence Level",
        min_value=0.1,
        max_value=1.0,
        value=0.7,
        help="Adjust detection sensitivity"
    )
    
    background_mode = st.selectbox(
        "Background",
        ["Black Background", "Blur Background", "Custom Color", "Transparent"],
        help="Select background replacement style"
    )
    
    if background_mode == "Custom Color":
        bg_color = st.color_picker("Background Color", "#000000")
    elif background_mode == "Blur Background":
        blur_strength = st.slider("Blur Intensity", 5, 50, 20)
    
    st.markdown("#### Enhancements")
    enhance_contrast = st.slider("Contrast", 0.5, 2.0, 1.1)
    enhance_sharpness = st.slider("Sharpness", 0.5, 2.0, 1.1)
    
    st.markdown("---")
    st.markdown("#### Platform Info")
    st.info("""
    **VisionSeg Pro**
    
    Professional AI image segmentation
    for enterprise and creative use.
    """)

# ---------------- Load Model ----------------
@st.cache_resource
def load_model():
    path = "model/deeplabv3_segmentation.pth"
    if not os.path.exists(path):
        return None
    model = deeplabv3_resnet50(pretrained=False)
    model.classifier[4] = torch.nn.Conv2d(256, 2, kernel_size=1)
    state_dict = torch.load(path, map_location=torch.device("cpu"))
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model

with st.spinner('🚀 Initializing AI engine...'):
    model = load_model()

if model is not None:
    st.sidebar.success("✓ Model ready")
else:
    st.sidebar.error("Model not available")

# ---------------- Helper Functions ----------------
def preprocess(image):
    transform = T.Compose([
        T.Resize((256, 256)),
        T.ToTensor()
    ])
    return transform(image).unsqueeze(0)

def create_segmented_image(original_image, output, orig_size, background_mode="Black Background", 
                          bg_color="#000000", blur_strength=15, confidence_threshold=0.5):
    probs = torch.softmax(output["out"], dim=1)
    confidence_map = torch.max(probs, dim=1)[0].squeeze().cpu().numpy()
    mask = torch.argmax(probs, dim=1).squeeze().cpu().numpy().astype(np.uint8)
    
    mask[confidence_map < confidence_threshold] = 0
    
    mask_img = Image.fromarray(mask)
    mask_img = mask_img.resize(orig_size, resample=Image.NEAREST)
    mask_array = np.array(mask_img)
    
    original_array = np.array(original_image)
    result_array = original_array.copy()
    
    if background_mode == "Black Background":
        background = np.zeros_like(original_array)
    elif background_mode == "Blur Background":
        background = cv2.GaussianBlur(original_array, (blur_strength, blur_strength), 0)
    elif background_mode == "Custom Color":
        color = np.array([int(bg_color[i:i+2], 16) for i in (1, 3, 5)])
        background = np.full_like(original_array, color)
    elif background_mode == "Transparent":
        rgba_array = np.concatenate([original_array, np.ones((*original_array.shape[:2], 1), dtype=np.uint8) * 255], axis=-1)
        rgba_array[mask_array == 0, 3] = 0
        return Image.fromarray(rgba_array, 'RGBA')
    
    result_array[mask_array == 0] = background[mask_array == 0]
    return Image.fromarray(result_array)

def apply_enhancements(image, contrast=1.0, sharpness=1.0):
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(contrast)
    enhancer = ImageEnhance.Sharpness(image)
    image = enhancer.enhance(sharpness)
    return image

def calculate_metrics(mask_array):
    foreground_pixels = np.sum(mask_array > 0)
    total_pixels = mask_array.size
    coverage = (foreground_pixels / total_pixels) * 100
    return coverage

# ---------------- Upload Section ----------------
st.markdown("## 📤 Upload & Process")
uploaded_file = st.file_uploader(
    "Upload your image for processing",
    type=["jpg", "jpeg", "png", "jfif"],
    help="Supported formats: JPG, PNG, JPEG"
)

# ---------------- Processing Logic ----------------
image_to_process = None
image_source = None

if selected_sample:
    if os.path.exists(selected_sample["original"]):
        image_to_process = Image.open(selected_sample["original"]).convert("RGB")
        image_source = "Selected Sample"

if uploaded_file is not None:
    image_to_process = Image.open(uploaded_file).convert("RGB")
    image_source = "Uploaded Image"

if image_to_process is not None:
    st.markdown("---")
    st.markdown("##  Results")
    
    orig_size = image_to_process.size
    
    with st.spinner('⚡ Processing image...'):
        if model is not None:
            try:
                input_tensor = preprocess(image_to_process)
                with torch.no_grad():
                    output = model(input_tensor)
                
                segmented_image = create_segmented_image(
                    image_to_process, output, orig_size, 
                    background_mode, 
                    bg_color if background_mode == "Custom Color" else "#000000",
                    blur_strength if background_mode == "Blur Background" else 15,
                    confidence_threshold
                )
                
                segmented_image = apply_enhancements(segmented_image, enhance_contrast, enhance_sharpness)
                
            except Exception as e:
                st.error("Processing failed")
                segmented_image = image_to_process
        else:
            segmented_image = image_to_process
    
    result_col1, result_col2 = st.columns(2)
    
    with result_col1:
        st.markdown("#### Original Image")
        st.image(image_to_process, use_container_width=True)
        st.caption(f"Dimensions: {orig_size[0]} × {orig_size[1]}")
    
    with result_col2:
        st.markdown("#### Segmented Result")
        st.image(segmented_image, use_container_width=True)
        st.caption(f"Background: {background_mode}")
        
        mask_array = np.array(segmented_image.convert('L')) if segmented_image.mode != 'RGBA' else np.array(segmented_image)[:, :, 3]
        coverage = calculate_metrics(mask_array)
        
        metric_col1, metric_col2, metric_col3 = st.columns(3)
        with metric_col1:
            st.metric("Foreground", f"{coverage:.1f}%")
        with metric_col2:
            st.metric("Resolution", f"{orig_size[0]}×{orig_size[1]}")
        with metric_col3:
            st.metric("Mode", background_mode.split()[0])
        
        buf = io.BytesIO()
        segmented_image.save(buf, format="PNG")
        file_name = "segmented.png"
        
        st.download_button(
            label="⬇️ Download Result",
            data=buf.getvalue(),
            file_name=file_name,
            mime="image/png",
            use_container_width=True,
            type="primary"
        )

# ---------------- Footer ----------------
st.markdown("---")
footer = """
<div style='text-align: center; color: #718096; font-size: 0.9rem;'>
    <b>VisionSeg Pro</b> — Professional AI Image Segmentation Platform<br>
    © 2024 VisionSeg Pro | support@visionseg.pro
</div>
"""
st.markdown(footer, unsafe_allow_html=True)