import os
import io
from typing import Optional

import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms as T
from torchvision.models.segmentation import deeplabv3_resnet50
from PIL import Image, ImageEnhance
import numpy as np

# ------------------------- PAGE CONFIG & CSS -------------------------
st.set_page_config(page_title="SegmentaAI", layout="wide")
st.markdown("""
<style>
header {visibility: hidden;}
*{background-color: rgb(255, 230, 200);}
body ,li ,p{ color: rgb(41, 84, 29);}
h1,h2,h4,h5,h6 {color: #FFE6C8;}
h3{color: black; background-color:#FFF7DD;}
svg{background-color:#FFF7DD;}
span{background:transparent;}
section.st-emotion-cache-15nprkh{background-color:white;}
div.stButton > button {
    background-color: rgb(213, 232, 197);
    color: black !important;
    width:100%;
    padding:10px;
    font-size:16px;
    border-radius:8px;
    border: 1px solid #555;
    cursor:pointer;
}
section[data-testid="stFileUploader"] > div {
    background-color: violet !important;
    border: 1px solid #555 !important;
    border-radius: 0.5rem;
    padding: 1rem;
    cursor: pointer;
    color: brown
}
section[data-testid="stFileUploader"] > div * {
    color: brown !important;
}
div.stDownloadButton > button {
    background-color: rgb(213, 232, 197);
    color: black;
    padding: 10px;
    font-size: 16px;
    border-radius: 8px;
    border: 1px solid #555;
    cursor: pointer;
}
.main > div {
    display: flex;
    flex-direction: column;
    align-items: center;
}
</style>
""", unsafe_allow_html=True)

# ------------------------- MODEL PATH -------------------------
weights_path = r"D:\C Desktop\VisionExtract\models\deeplabv3_subject_model.pth"

# ------------------------- MODEL CONFIGS -------------------------
image_size = 256
num_classes = 1  # your model was trained for 1 class (subject only)

# ------------------------- BUILD & LOAD MODEL -------------------------
@st.cache_resource
def build_model(num_classes_local: int = 1, device_local: Optional[torch.device] = None):
    device_local = device_local or torch.device("cpu")
    model_local = deeplabv3_resnet50(weights=None, num_classes=num_classes_local)
    model_local.to(device_local)
    model_local.eval()
    return model_local

def load_custom_model(model_local: nn.Module, model_path_local: str, device_local: torch.device):
    if os.path.exists(model_path_local):
        state = torch.load(model_path_local, map_location=device_local)
        model_local.load_state_dict(state, strict=False)
        model_local.to(device_local)
        model_local.eval()
        return model_local
    else:
        raise FileNotFoundError(f"Model file not found at {model_path_local}")

# ------------------------- PREPROCESSING & HELPERS -------------------------
preprocess = T.Compose([T.Resize((image_size, image_size)), T.ToTensor()])

def predict_mask(model_local: nn.Module, pil_img: Image.Image, device_local: torch.device):
    input_tensor = preprocess(pil_img).unsqueeze(0).to(device_local)
    with torch.no_grad():
        out = model_local(input_tensor)
        logits = out["out"] if isinstance(out, dict) and "out" in out else out
        pred = torch.sigmoid(logits).squeeze().cpu().numpy()
        mask = (pred > 0.5).astype(np.uint8)
    return mask

def extract_object_rgb(image_np: np.ndarray, mask_np: np.ndarray):
    obj = np.zeros_like(image_np, dtype=np.uint8)
    if mask_np.shape != image_np.shape[:2]:
        mask_pil = Image.fromarray(mask_np)
        mask_np = np.array(mask_pil.resize((image_np.shape[1], image_np.shape[0]), resample=Image.NEAREST))
    obj[mask_np > 0] = image_np[mask_np > 0]
    return obj

# ------------------------- DEVICE -------------------------
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


# ------------------------- STREAMLIT UI -------------------------
st.markdown("""
<div style="text-align:center;">
    <h1>🤖 Segment-AI</h1>
    <p>✨ Upload an image to instantly isolate the main subject with a black background.</p>
</div>
""", unsafe_allow_html=True)

# ------------------------- SAMPLE IMAGES -------------------------
st.markdown("### 🖼️ Sample Images")
sample_orig_path = r"D:\C Desktop\VisionExtract\samples\Original.png"
sample_seg_path = r"D:\C Desktop\VisionExtract\samples\Segmented.png"

if os.path.exists(sample_orig_path) and os.path.exists(sample_seg_path):
    col1, col2 = st.columns(2)
    col1.image(sample_orig_path, caption="Original", width="stretch")
    col2.image(sample_seg_path, caption="Segmented", width="stretch")

# ------------------------- FILE UPLOADER -------------------------
uploaded_file = st.file_uploader("📸 Upload Image", type=["jpg", "jpeg", "png"], key="uploader")

if 'model_loaded' not in st.session_state:
    st.session_state['model_loaded'] = False


# ------------------------- MAIN LOGIC -------------------------
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    orig_np = np.array(image.resize((image_size, image_size)))
    st.markdown(f"**File:** {uploaded_file.name}")

    if st.button("🚀 Extract Subject"):
        if not st.session_state['model_loaded']:
            model_for_use = build_model(num_classes_local=num_classes, device_local=device)
            model_for_use = load_custom_model(model_for_use, weights_path, device)
            st.session_state['model_obj'] = model_for_use
            st.session_state['model_loaded'] = True

        model_local = st.session_state['model_obj']

        with st.spinner("🧠 Running model..."):
            pred_mask = predict_mask(model_local, image, device)

        pred_obj = extract_object_rgb(orig_np, pred_mask)


        # Display
        c1, c2 = st.columns(2)
        c1.image(orig_np, caption="Original", width="stretch")
        c2.image(pred_obj, caption="Segmented Object", width="stretch")

        # Download button
        buf = io.BytesIO()
        Image.fromarray(pred_obj).save(buf, format="PNG")
        buf.seek(0)
        st.download_button(
            label="📥 Download Segmented Image",
            data=buf,
            file_name="segmented.png",
            mime="image/png"
        )

# ------------------------- FOOTER -------------------------
st.markdown("---")
st.markdown("""
<div style="text-align:center; font-size:14px; padding:10px; color:black;">
    ✨ Developed by <b>Likhitha ❤️</b> ✨
</div>
""", unsafe_allow_html=True)
