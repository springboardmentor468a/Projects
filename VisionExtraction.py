import streamlit as st
import torch
import torch.nn as nn
import torchvision.models.segmentation as models
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
from PIL import Image

st.set_page_config(page_title="Vision Extraction", layout="wide")

# Custom pastel background + dark blue text
page_bg = """
<style>
    /* Main app background */
    .stApp {
        background-color: #E0F7FA; /* pastel blue */
    }
    /* Sidebar background */
    section[data-testid="stSidebar"] {
        background-color: #E0F7FA; /* pastel blue */
    }
    /* Dark blue font everywhere */
    h1, h2, h3, h4, h5, h6, p, label, span {
        color: #0D47A1 !important;
    }
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)


# Title with cute symbol
st.markdown("<h1 style='text-align: center;'>🌸 Vision Extraction ✨</h1>", unsafe_allow_html=True)

# Upload image
uploaded_file = st.file_uploader("Upload an image", type=["jpg","png","jpeg"])

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define model architecture
model = models.deeplabv3_resnet50(pretrained=False)
model.classifier[4] = nn.Conv2d(256, 2, kernel_size=1)
state_dict = torch.load("best_model11.pth", map_location=device)
model.load_state_dict(state_dict, strict=False)
model.to(device)
model.eval()

# Albumentations transform
val_transform = A.Compose([
    A.Resize(256, 256),
    A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
    ToTensorV2()
])

def run_inference(img_pil):
    img = np.array(img_pil.convert("RGB"))
    img_resized = cv2.resize(img, (256,256))
    transformed = val_transform(image=img, mask=np.zeros(img.shape[:2], dtype=np.uint8))
    img_tensor = transformed["image"].unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_tensor)["out"]
    pred = torch.argmax(output, dim=1)[0].cpu().numpy()

    object_only = np.zeros_like(img_resized)
    object_only[pred==1] = img_resized[pred==1]

    return img_resized, object_only

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    original, predicted = run_inference(image)

    # Side-by-side layout
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            "<h3 style='text-align:center;'>🌷 Original 🌷</h3>",
            unsafe_allow_html=True
        )
        st.image(original, use_column_width=True)

    with col2:
        st.markdown(
            "<h3 style='text-align:center;'>💖 Predicted 💖</h3>",
            unsafe_allow_html=True
        )
        st.image(predicted, use_column_width=True)
