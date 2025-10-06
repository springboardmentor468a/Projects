import os
import io
import time
from typing import Optional

import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms as T
from torchvision.models.segmentation import deeplabv3_resnet50
from PIL import Image
import numpy as np
import gdown  # for downloading models from Google Drive

# ------------------------- Page config & CSS -------------------------
st.set_page_config(page_title="AI Image Segmentation", layout="wide")
st.markdown("""
<style>
header {visibility: hidden;}
*{background-color: rgb(213, 232, 197);}
body ,li ,p{ color: rgb(41, 84, 29);}
h1,h2,h4,h5,h6 {color: #6FA4AF;}
h3{color: black; background-color:#FFF7DD;}
svg{background-color:#FFF7DD;}
span{background:transparent;}
section.st-emotion-cache-15nprkh{background-color:white;}
button.st-emotion-cache-jzs692{background-color: rgb(38, 105, 20);color: white;}
svg.st-emotion-cache-6rlrad,svg.st-emotion-cache-4mjat2{background:transparent;}
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

# ------------------------- Google Drive model links -------------------------
best_model_file_id = "1csSCrkz2ErFuPmINlh-p8QDuxnaScPoh"  # infosys_model_best.pth
full_model_file_id = "1oiKsJWZ_Mm2dKQV8BVMGYXYuakBmQZjW"  # infosys_model_full.pth

weights_path = "infosys_model_best.pth"
full_model_path = "infosys_model_full.pth"

# Function to download from Google Drive if not present
def download_model(file_id: str, output_path: str):
    if not os.path.exists(output_path):
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, output_path, quiet=False)

download_model(best_model_file_id, weights_path)
download_model(full_model_file_id, full_model_path)

# ------------------------- Model configs -------------------------
image_size = 256
num_classes = 91

@st.cache_resource
def build_model(num_classes_local: int = 91, pretrained: bool = True, device_local: Optional[torch.device] = None):
    device_local = device_local or torch.device("cpu")
    model_local = deeplabv3_resnet50(pretrained=pretrained)
    try:
        model_local.classifier[4] = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, num_classes_local, kernel_size=1)
        )
    except Exception:
        pass
    model_local.to(device_local)
    model_local.eval()
    return model_local

def load_model_try(weights_path_local: str, full_model_path_local: str, model_local: nn.Module, device_local: torch.device):
    if os.path.exists(weights_path_local):
        try:
            state = torch.load(weights_path_local, map_location=device_local)
            if isinstance(state, dict) and ("state_dict" in state or "model_state_dict" in state):
                key = "state_dict" if "state_dict" in state else "model_state_dict"
                model_local.load_state_dict(state[key], strict=False)
            else:
                model_local.load_state_dict(state, strict=False)
            return model_local
        except Exception:
            pass
    if os.path.exists(full_model_path_local):
        try:
            loaded = torch.load(full_model_path_local, map_location=device_local)
            if isinstance(loaded, nn.Module):
                loaded.to(device_local)
                loaded.eval()
                return loaded
            elif isinstance(loaded, dict) and "state_dict" in loaded:
                model_local.load_state_dict(loaded["state_dict"], strict=False)
                return model_local
        except Exception:
            pass
    raise FileNotFoundError("No valid model file found.")

# ------------------------- Preprocessing & helper functions -------------------------
preprocess = T.Compose([T.Resize((image_size, image_size)), T.ToTensor()])

def predict_mask(model_local: nn.Module, pil_img: Image.Image, device_local: torch.device):
    input_tensor = preprocess(pil_img).unsqueeze(0).to(device_local)
    with torch.no_grad():
        out = model_local(input_tensor)
        logits = out["out"] if isinstance(out, dict) and "out" in out else out
        pred = logits.argmax(1).squeeze(0).cpu().numpy().astype(np.uint8)
    return pred

def extract_object_rgb(image_np: np.ndarray, mask_np: np.ndarray):
    obj = np.zeros_like(image_np, dtype=np.uint8)
    if mask_np.shape != image_np.shape[:2]:
        mask_pil = Image.fromarray(mask_np)
        mask_np = np.array(mask_pil.resize((image_np.shape[1], image_np.shape[0]), resample=Image.NEAREST))
    obj[mask_np > 0] = image_np[mask_np > 0]
    return obj

# ------------------------- Device -------------------------
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# ------------------------- Main Streamlit Page -------------------------
st.title("AI Image Segmentation")

st.subheader("How to Use")
st.markdown("""
1. Upload an image (PNG/JPG).
2. AI will generate a segmented image with black background.
3. Download the result below.
""")

# ------------------------- Sample images -------------------------
st.subheader("Sample Images")
sample_orig_path = "samples/Apple.png"            # put sample images in "samples/" folder
sample_seg_path = "samples/Extracted_Apple.png"  # segmented sample
c1, c2 = st.columns(2)
c1.image(sample_orig_path, caption="Original", use_container_width=True)
c2.image(sample_seg_path, caption="Segmented", use_container_width=True)

# ------------------------- File uploader -------------------------
uploaded_file = st.file_uploader("Upload an image (JPG/PNG)", type=["jpg", "jpeg", "png"], key="uploader")

if 'model_loaded' not in st.session_state:
    st.session_state['model_loaded'] = False

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    orig_np = np.array(image.resize((image_size, image_size)))
    st.markdown(f"**File:** {uploaded_file.name}")

    if st.button("Extract segmented Image"):
        if not st.session_state['model_loaded']:
            model_for_use = build_model(num_classes_local=num_classes, pretrained=True, device_local=device)
            model_for_use = load_model_try(weights_path, full_model_path, model_for_use, device)
            st.session_state['model_obj'] = model_for_use
            st.session_state['model_loaded'] = True

        model_local = st.session_state['model_obj']

        with st.spinner("Running model (inference)..."):
            start = time.time()
            pred_mask = predict_mask(model_local, image, device)
            elapsed = time.time() - start

        pred_obj = extract_object_rgb(orig_np, pred_mask)
        st.success(f"Inference done in {elapsed:.2f}s (device={device})")

        c1, c2 = st.columns(2)
        c1.image(orig_np, caption="Original", use_container_width=True)
        c2.image(pred_obj, caption="Segmented Object", use_container_width=True)

        buf = io.BytesIO()
        Image.fromarray(pred_obj).save(buf, format="PNG")
        buf.seek(0)
        st.download_button(
            label="Download Segmented Image",
            data=buf,
            file_name="segmented.png",
            mime="image/png"
        )
