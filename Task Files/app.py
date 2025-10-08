import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import segmentation_models_pytorch as smp
import numpy as np
import io
import os

@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
    ).to(device)

    model_path = "unet-resnet34_finetuned.pth"

    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("Model loaded successfully.")
    else:
        st.error(f"Model file not found at '{model_path}'")
        return None, None

    model.eval()
    return model, device

IMG_SIZE = 256

preprocess_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

def predict(model, image, device):
    orig_w, orig_h = image.size 
    resized = image.resize((IMG_SIZE, IMG_SIZE))
    input_tensor = preprocess_transform(resized).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        pred_mask = (torch.sigmoid(output) > 0.5).float().cpu().squeeze().numpy()
    mask_resized = Image.fromarray((pred_mask * 255).astype(np.uint8)).resize((orig_w, orig_h), Image.NEAREST)
    mask_np = np.array(mask_resized) / 255.0
    img_np = np.array(image).astype(np.float32) / 255.0
    combined = img_np * np.expand_dims(mask_np, axis=2)
    isolated_subject_image = Image.fromarray((combined * 255).astype(np.uint8))

    return isolated_subject_image
st.set_page_config(page_title="VisionExtract - Background Remover", layout="centered")
st.markdown("<h2 style='text-align: center;'> VisionExtract: Background Remover</h2>", unsafe_allow_html=True)

st.write("Upload an image to automatically remove the background.")

with st.expander("How it Works", expanded=True):
    st.write("""
        This tool uses a deep learning model to automatically identify the main subject in an image and replace the background with black.
        
        1.  **Upload an image** using the button below.
        2.  The model will process the image.
        3.  **Download** your final image.
    """)
    col1_ex, col2_ex = st.columns(2)
    
    try:
        col1_ex.image("example1.png", caption="Example Input: Your original image.", use_column_width=True)
        col2_ex.image("example2.png", caption="Example Output: The isolated subject.", use_column_width=True)
    except Exception as e:
        st.warning(f"Could not load example images. Make sure 'example_input.jpg' and 'example_output.png' are in the same folder as your script.")

st.write("---")

model, device = load_model()

uploaded_file = st.file_uploader("Choose an image to begin", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and model is not None:
    original_image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original Image")
        st.image(original_image, use_column_width=True)

    with col2:
        st.subheader("Isolated Subject")
        with st.spinner('Processing...'):
            isolated_img = predict(model, original_image, device)
            st.image(isolated_img, use_column_width=True)

    buf = io.BytesIO()
    isolated_img.save(buf, format="PNG")
    byte_im = buf.getvalue()
    st.download_button(
        label="Download Isolated Image",
        data=byte_im,
        file_name="isolated_subject.png",
        mime="image/png"
    )