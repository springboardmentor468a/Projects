import streamlit as st
import torch
import torchvision.transforms as T
import torchvision.models.segmentation as models
from PIL import Image, ImageFilter
import numpy as np
import io
import os
import random
import gdown 
# ---------------- Streamlit Page Config ----------------
st.set_page_config(page_title="Image Segmentation App", layout="wide")


# ---------------- Custom CSS ----------------
st.markdown("""
    <style>
        /* Main app background */
        .stApp {
            background: #000000;  /* Solid black background */
            color: #FFFFFF;
        }
        /* Optional: card-style containers for better contrast */
        .stContainer {
            background-color: #2b2b2b;
            padding: 10px;
            border-radius: 10px;
        }
        /* Sidebar Styling */
        [data-testid="stSidebar"] {
            background-color: #1e1e1e;  /* Dark sidebar */
            color: #ffffff;
            padding: 10px;
        }
        [data-testid="stSidebar"] * {
            color: white !important;
        }
        /* Force all sidebar text, labels, and controls to white */
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] span,
        [data-testid="stSidebar"] div,
        [data-testid="stSidebar"] input,
        [data-testid="stSidebar"] select,
        [data-testid="stSidebar"] textarea {
            color: white !important;
        }

        /* Main Page Background */
        .css-18e3th9 {  /* Streamlit main content container */
            background-color: #121212;  /* Change this to your preferred color */
            color: #FCF9EA;
        }

        /* Buttons, text, etc. */
        .try-demo-btn {
            background-color: #4CAF50;
            color: white;
            font-weight: bold;
            padding: 0.5rem 1rem;
            border-radius: 8px;
            border: none;
            transition: 0.3s;
        }
        .try-demo-btn:hover {
            background-color: #45a049;
            cursor: pointer;
            transform: scale(1.05);
        }
        .center-text {
            text-align:center; 
            font-size:20px; 
            color:#FCF9EA; 
            margin-top:-10px; 
            margin-bottom:20px;
        }

        /* Footer Styling (remains same) */
        .footer {
            position: relative;
            left: 0;
            bottom: 0;
            width: 100%;
            background-color: #111;
            color: #fff;
            text-align: center;
            padding: 15px 10px;
            font-size: 14px;
            box-shadow: 0 -2px 10px rgba(0,0,0,0.3);
            margin-top: 50px;
        }
        .footer a {
            color: #4CAF50;
            text-decoration: none;
            margin: 0 10px;
            font-weight: bold;
            transition: all 0.3s ease;
        }
        .footer a:hover {
            color: #ffffff;
            text-shadow: 0 0 5px #4CAF50;
            transform: scale(1.05);
        }
        .footer p {
            margin: 5px 0;
        }
    </style>
""", unsafe_allow_html=True)



# ---------------- Session State ----------------
if "selected_preset" not in st.session_state:
    st.session_state["selected_preset"] = None
if "selected_demo" not in st.session_state:
    st.session_state["selected_demo"] = None

# ---------------- Title / Description ----------------
st.title("🎨 AI-Powered Subject Extraction & Background Editor")
st.markdown('<div class="center-text">Upload your images, try a demo image, or select presets to instantly extract subjects and apply backgrounds.</div>', unsafe_allow_html=True)

# ---------------- Example Before / After ----------------
# before_image_path = r"C:\Users\ASUS\Downloads\download (33).jpg"
# after_image_path = r"C:\Users\ASUS\Downloads\download (33)_black_bg.jpg"  

# example_before = Image.open(before_image_path).convert("RGB") if os.path.exists(before_image_path) else None
# example_after = Image.open(after_image_path).convert("RGBA") if os.path.exists(after_image_path) else None



# --- Download before/after example images from Google Drive if not present ---
EX_BEFORE_PATH = "example_before.jpg"
EX_AFTER_PATH = "example_after.png"
EX_BEFORE_ID = "1qCxGNto7K-JTbyCrRXRp9MlB-0CcuyVn"
EX_AFTER_ID = "1LWM2e2cZYo7FqQfebBG6K_bmHuM70fVA"

def download_example_image(file_id, dest):
    if not os.path.exists(dest):
        url = f"https://drive.google.com/uc?id={file_id}"
        with st.spinner(f"Downloading example image: {dest}..."):
            try:
                gdown.download(url, dest, quiet=True)
            except Exception as e:
                st.warning(f"Could not download {dest}: {e}")

download_example_image(EX_BEFORE_ID, EX_BEFORE_PATH)
download_example_image(EX_AFTER_ID, EX_AFTER_PATH)

example_before = Image.open(EX_BEFORE_PATH).convert("RGB") if os.path.exists(EX_BEFORE_PATH) else None
example_after = Image.open(EX_AFTER_PATH).convert("RGBA") if os.path.exists(EX_AFTER_PATH) else None


if example_before and example_after:
    # st.subheader("📌 Example Before / After")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original Image ")
        st.image(example_before, use_container_width=True)
    with col2:
        st.subheader("Segmented Image ")
        st.image(example_after, use_container_width=True)

# ---------------- How the Tool Works Section ----------------
st.markdown("---")
st.subheader("🛠️ How the Tool Works")
st.markdown("""
- **Upload or Try a Demo Image** — Select a single image, multiple images, or click 'Try Demo Image'.  
- **Automatic AI Segmentation** — The tool detects and extracts the subject with AI precision.  
- **Remove or Replace Backgrounds** — Choose from transparent, colors, blur, or preset backgrounds.  
- **Easy Preview & Adjustments** — See results instantly and tweak detection settings if needed.  
- **Download Your Image** — Save your final image in PNG, JPG, JPEG, or WEBP format.
""")

# ---------------- Sidebar ----------------
st.sidebar.title("⚙️ Detection Threshold")
threshold = st.sidebar.slider("", 0.0, 1.0, 0.5, 0.01)
min_area = st.sidebar.number_input("Minimum Area (pixels)", min_value=0, max_value=1000, value=300, step=10, format="%d")

st.sidebar.title("⚙️ Processing Mode")
processing_mode = st.sidebar.radio("", ("Single Image", "Batch Processing"), horizontal=True)

st.sidebar.title("💾 Export Format")
export_format = st.sidebar.selectbox("Select Output Format", ["PNG", "JPG", "JPEG", "WEBP"])

st.sidebar.title("🎨 Background Options")
bg_options_list = ["Transparent", "White", "Black", "Blur", "Custom Color", "Custom Background", "Preset Backgrounds"]
# Set default to "Black"
bg_option = st.sidebar.selectbox("Choose Background", bg_options_list, index=bg_options_list.index("Black"))
bg_color = None
bg_image_file = None
preset_bg_file = None


if bg_option == "Custom Color":
    bg_color = st.sidebar.color_picker("Pick Background Color", "#ffffff")
elif bg_option == "Custom Background":
    bg_image_file = st.sidebar.file_uploader("Upload Background Image", type=["png", "jpg","jpeg"])


# ---------------- Preset Backgrounds (Google Drive Download) ----------------
PRESET_DIR = "Preset_Backgrounds"
PRESET_FOLDER_ID = "128tYF4FPqkaHx_4Y2NiPiefDKZF_QIt8"

def download_preset_backgrounds():
    need_download = not os.path.exists(PRESET_DIR) or not any(f.lower().endswith((".png", ".jpg", ".jpeg")) for f in os.listdir(PRESET_DIR))
    if need_download:
        with st.spinner("Downloading preset backgrounds from Google Drive..."):
            try:
                gdown.download_folder(f"https://drive.google.com/drive/folders/{PRESET_FOLDER_ID}", output=PRESET_DIR, quiet=True)
                st.success("Preset backgrounds downloaded successfully!")
            except Exception as e:
                st.error(f"Failed to download preset backgrounds: {e}")

download_preset_backgrounds()

preset_files = [f for f in os.listdir(PRESET_DIR) if f.lower().endswith((".png", ".jpg", ".jpeg"))] if os.path.exists(PRESET_DIR) else []

if bg_option == "Preset Backgrounds":
    st.sidebar.write("Select a preset background by clicking a thumbnail:")
    if preset_files:
        cols = st.sidebar.columns(3)
        for idx, file_name in enumerate(preset_files):
            img_path = os.path.join(PRESET_DIR, file_name)
            try:
                thumb = Image.open(img_path).convert("RGBA").resize((80, 80))
            except:
                continue
            col = cols[idx % 3]
            if col.button("", key=f"preset_btn_{idx}"):
                st.session_state["selected_preset"] = img_path
            col.image(thumb, use_column_width=True, caption=file_name)
    else:
        st.sidebar.warning("No preset backgrounds found. Please check your Google Drive folder or try reloading the app.")

if st.session_state["selected_preset"]:
    preset_bg_file = st.session_state["selected_preset"]


# ---------------- Google Drive Model Download (Option-A) ----------------
MODEL_DIR = "model_files"
MODEL_PATH = os.path.join(MODEL_DIR, "best_model (5).pth")

def download_model(file_id: str, dest: str):
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if not os.path.exists(dest):
        with st.spinner("Downloading model from Google Drive..."):
            url = f"https://drive.google.com/uc?id={file_id}"
            gdown.download(url, dest, quiet=False)
    else:
        st.info("Model already downloaded.")

# Get Drive file ID from Streamlit secrets
# file_id = st.secrets.get("DRIVE_FILE_ID")
# if not file_id:
#     st.error("Missing DRIVE_FILE_ID in Streamlit secrets.")
#     st.stop()

file_id = st.secrets.get("DRIVE_FILE_ID", "1BW7ZpdGILFiDjnvEb0V1S4q7ZBYcwiI8")  # optional fallback

if file_id == "1BW7ZpdGILFiDjnvEb0V1S4q7ZBYcwiI8":
    st.warning("⚠️ Using default file ID (local test mode). Add DRIVE_FILE_ID in Streamlit secrets for cloud use.")


download_model(file_id, MODEL_PATH)



# ---------------- Load Custom Model ----------------
# @st.cache_resource
# def load_model():
#     model = models.deeplabv3_resnet50(weights=None)
#     model.classifier[4] = torch.nn.Conv2d(256, 2, kernel_size=1)
#     model.aux_classifier = None
#     checkpoint = torch.load(r"C:\Users\ASUS\Downloads\best_model (5).pth", map_location=torch.device("cpu"))
#     state_dict = {k: v for k, v in checkpoint.items() if k in model.state_dict()}
#     model.load_state_dict(state_dict, strict=False)
#     model.eval()
#     return model

# model = load_model()


@st.cache_resource
def load_model():
    model = models.deeplabv3_resnet50(weights=None)
    model.classifier[4] = torch.nn.Conv2d(256, 2, kernel_size=1)
    model.aux_classifier = None
    checkpoint = torch.load(MODEL_PATH, map_location=torch.device("cpu"))
    state_dict = {k: v for k, v in checkpoint.items() if k in model.state_dict()}
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model

model = load_model()


# ---------------- Preprocessing ----------------
def preprocess(image):
    transform = T.Compose([
        T.Resize((520, 520)),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225))
    ])
    return transform(image).unsqueeze(0)

# ---------------- Segmentation / Extraction ----------------
def extract_subject(image_pil):
    input_tensor = preprocess(image_pil)
    with torch.no_grad():
        output = model(input_tensor)["out"][0]
    mask = output.argmax(0).byte().cpu().numpy()
    mask_resized = np.array(Image.fromarray(mask).resize(image_pil.size, resample=Image.NEAREST))
    image_np = np.array(image_pil).astype(np.uint8)
    rgba = np.dstack((image_np, mask_resized * 255))
    return Image.fromarray(rgba, mode="RGBA")

# ---------------- Apply Background ----------------
def apply_background(fg_image, bg_option, bg_color=None, bg_image_file=None, preset_bg_file=None):
    fg = fg_image.convert("RGBA")
    fg_np = np.array(fg)
    alpha_mask = fg_np[:, :, 3] / 255.0
    bg_np = np.zeros_like(fg_np)

    if bg_option == "Transparent":
        return fg
    elif bg_option == "White":
        bg_np = np.ones_like(fg_np) * 255
    elif bg_option == "Black":
        bg_np = np.zeros_like(fg_np)
    elif bg_option == "Blur":
        bg = fg.filter(ImageFilter.GaussianBlur(radius=10))
        bg_np = np.array(bg)
    elif bg_option == "Custom Color" and bg_color is not None:
        bg_np = np.ones_like(fg_np) * np.array(list(int(bg_color[i:i+2],16) for i in (1,3,5)) + [255])
    elif bg_option == "Custom Background" and bg_image_file is not None:
        bg = Image.open(bg_image_file).convert("RGBA").resize(fg.size)
        bg_np = np.array(bg)
    elif bg_option == "Preset Backgrounds" and preset_bg_file is not None:
        bg = Image.open(preset_bg_file).convert("RGBA").resize(fg.size)
        bg_np = np.array(bg)

    combined = fg_np.copy()
    for c in range(3):
        combined[:, :, c] = fg_np[:, :, c] * alpha_mask + bg_np[:, :, c] * (1 - alpha_mask)
    combined[:, :, 3] = 255
    return Image.fromarray(combined.astype(np.uint8), mode="RGBA")

# ---------------- Process & Display ----------------
def process_and_display(image, filename="image"):
    st.info("Extracting subject... ⏳")
    result = extract_subject(image)
    final_result = apply_background(result, bg_option, bg_color, bg_image_file, preset_bg_file)
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original Image")
        st.image(image, use_container_width=True)
    with col2:
        st.subheader("Segmented Image ")
        st.image(final_result, use_container_width=True)

        buf = io.BytesIO()
        save_format = export_format if export_format != "JPG" else "JPEG"
        img_to_save = final_result.convert("RGB") if save_format == "JPEG" else final_result
        img_to_save.save(buf, format=save_format)

        st.download_button(
            label=f"⬇️ Download Extracted Subject ({export_format})",
            data=buf,
            file_name=f"{filename}_subject.{export_format.lower()}",
            mime=f"image/{export_format.lower()}"
        )

# ---------------- Upload Section ----------------
st.subheader("📤 Upload Images for Subject Extraction")
st.write("You can upload a single image or multiple images for batch processing. Choose a background and download the result.")

if processing_mode == "Single Image":
    uploaded_file = st.file_uploader("Upload a Single Image", type=["png","jpg","jpeg"], accept_multiple_files=False)
    uploaded_any = uploaded_file is not None
else:
    uploaded_files = st.file_uploader("Upload Multiple Images", type=["png","jpg","jpeg"], accept_multiple_files=True)
    uploaded_any = uploaded_files is not None and len(uploaded_files) > 0

 # ---------------- Demo Image Section ----------------
DEMO_DIR = "Demo-Image"
DEMO_FOLDER_ID = "1Ibc5YYMM3byWIiKbI3mACzpYbskmgs_L"

def download_demo_images():
    """Downloads the demo image folder from Google Drive if not present or empty."""
    need_download = not os.path.exists(DEMO_DIR) or not any(f.lower().endswith((".png", ".jpg", ".jpeg")) for f in os.listdir(DEMO_DIR))
    if need_download:
        with st.spinner("Downloading demo images from Google Drive..."):
            try:
                gdown.download_folder(f"https://drive.google.com/drive/folders/{DEMO_FOLDER_ID}", output=DEMO_DIR, quiet=True)
                st.success("Demo images downloaded successfully!")
            except Exception as e:
                st.error(f"Failed to download demo images: {e}")

download_demo_images()

demo_files = [f for f in os.listdir(DEMO_DIR) if f.lower().endswith((".png", ".jpg", ".jpeg"))] if os.path.exists(DEMO_DIR) else []

if demo_files:
    st.markdown("---")
    st.subheader("🚀 Try Demo Image")
    st.write("Click a thumbnail to select a demo image, or use the random button below.")
    cols = st.columns(min(4, len(demo_files)))
    for idx, file_name in enumerate(demo_files):
        img_path = os.path.join(DEMO_DIR, file_name)
        try:
            thumb = Image.open(img_path).convert("RGBA").resize((80, 80))
        except Exception:
            continue
        col = cols[idx % len(cols)]
        if col.button("", key=f"demo_btn_{idx}", help=file_name):
            st.session_state["selected_demo"] = img_path
            uploaded_any = False
        col.image(thumb, use_column_width=True, caption=file_name)
    if st.button("🎯 Try Random Demo Image"):
        random_demo = random.choice(demo_files)
        st.session_state["selected_demo"] = os.path.join(DEMO_DIR, random_demo)
        uploaded_any = False
else:
    st.warning("No demo images found. Please check your Google Drive folder or try reloading the app.")

# ---------------- Display Demo Image ----------------
if st.session_state.get("selected_demo") and not uploaded_any:
    demo_img_path = st.session_state["selected_demo"]
    demo_img = Image.open(demo_img_path).convert("RGB")
    process_and_display(demo_img, os.path.basename(demo_img_path))

# ---------------- Display Uploaded Images ----------------
if processing_mode == "Single Image" and uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    process_and_display(image, uploaded_file.name)
elif processing_mode == "Batch Processing" and uploaded_files:
    st.subheader("📂 Batch Subject Extraction")
    for f in uploaded_files:
        image = Image.open(f).convert("RGB")
        process_and_display(image, f.name)
elif not st.session_state.get("selected_demo"):
    st.info("No images uploaded or selected yet. Please upload images or try a demo image.")

footer = """
<style>
.footer {
    position: relative;  /* Changed from fixed to relative */
    left: 0;
    bottom: 0;
    width: 100%;
    background-color: #111;
    color: #fff;
    text-align: center;
    padding: 15px 10px;
    font-size: 14px;
    box-shadow: 0 -2px 10px rgba(0,0,0,0.3);
    margin-top: 50px;  /* space from content above */
}

.footer a {
    color: #4CAF50;
    text-decoration: none;
    margin: 0 10px;
    font-weight: bold;
    transition: all 0.3s ease;
}

.footer a:hover {
    color: #ffffff;
    text-shadow: 0 0 5px #4CAF50;
    transform: scale(1.05);
}

.footer p {
    margin: 5px 0;
}
</style>

<div class="footer">
    <p>Developed with ❤ by 
    <a href="#" target="_blank">Satyam Jaiswal</a> | 
    <a href="https://github.com/Satyamjai1003" target="_blank">GitHub</a> |  
    <a href="#">Contact</a>
    </p>
</div>
"""

st.markdown(footer, unsafe_allow_html=True)