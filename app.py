
import streamlit as st
import torch
import torchvision
from torchvision import transforms as T
from PIL import Image
import numpy as np
import io, tempfile

st.set_page_config(page_title="VisionExtract - Subject Isolation", layout="wide")
st.title("VisionExtract — Subject Isolation (Streamlit demo)")
st.write("Upload an image and a trained model (optional). The app predicts a subject mask and returns the subject with a black background.")

@st.cache(allow_output_mutation=True)
def load_demo_model(device):
    model = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=True)
    model.eval().to(device)
    return model

@st.cache(allow_output_mutation=True)
def load_model_from_path(path, device):
    try:
        loaded = torch.load(path, map_location=device)
        if isinstance(loaded, torch.nn.Module):
            loaded.eval().to(device)
            return loaded
    except Exception:
        pass
    try:
        model = torch.jit.load(path, map_location=device)
        model.eval().to(device)
        return model
    except Exception:
        pass
    try:
        state = torch.load(path, map_location=device)
        model = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=False, num_classes=1)
        model.load_state_dict(state)
        model.eval().to(device)
        return model
    except Exception as e:
        raise RuntimeError("Failed to load model: " + str(e))

def image_to_tensor(pil_img, target_size=None):
    transforms = []
    if target_size: transforms.append(T.Resize(target_size))
    transforms.extend([T.ToTensor(),
                       T.Normalize(mean=[0.485,0.456,0.406],
                                   std=[0.229,0.224,0.225])])
    return T.Compose(transforms)(pil_img).unsqueeze(0)

def predict_mask(model, pil_img, device, threshold=0.5):
    model.eval()
    w, h = pil_img.size
    short = min(w, h)
    scale = 512 / short if short < 512 else 1
    tw, th = int(w*scale), int(h*scale)
    x = image_to_tensor(pil_img, (th, tw)).to(device)
    with torch.no_grad():
        out = model(x)
    if isinstance(out, dict): out = out.get('out', list(out.values())[0])
    out = out.cpu()
    if out.shape[1] == 1:
        mask = (torch.sigmoid(out)[0,0].numpy() > threshold).astype('uint8')*255
    else:
        mask = (out.argmax(1)[0].numpy() != 0).astype('uint8')*255
    return Image.fromarray(mask).convert("L").resize((w,h), Image.NEAREST)

def apply_mask_to_image(pil_img, mask_pil):
    img_np = np.array(pil_img.convert("RGB"))
    m = np.array(mask_pil) > 127
    out = np.zeros_like(img_np)
    out[m] = img_np[m]
    return Image.fromarray(out)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
st.sidebar.header("Model / settings")
source = st.sidebar.selectbox("Model source",
    ("Demo pre-trained model","Upload model file","Load model from Drive path"))
model = None

if source == "Upload model file":
    f = st.sidebar.file_uploader("Upload model (.pth/.pt)", type=["pth","pt"])
    if f:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pth")
        tmp.write(f.read()); tmp.flush()
        model = load_model_from_path(tmp.name, device)
        st.sidebar.success("Model loaded.")
elif source == "Load model from Drive path":
    p = st.sidebar.text_input("Path in Drive (e.g. /content/drive/MyDrive/model.pth)")
    if p:
        model = load_model_from_path(p, device)
        st.sidebar.success("Model loaded.")
else:
    model = load_demo_model(device)
    st.sidebar.success("Demo model ready.")

threshold = st.sidebar.slider("Mask threshold", 0.01, 0.99, 0.5, 0.01)
overlay = st.sidebar.checkbox("Show mask overlay", True)

img_file = st.file_uploader("Upload an image", type=["png","jpg","jpeg"])
if img_file:
    img = Image.open(img_file).convert("RGB")
    st.image(img, caption="Original image", use_column_width=True)
    if st.button("Run inference"):
        if model is None:
            st.error("No model loaded.")
        else:
            with st.spinner("Running model..."):
                mask = predict_mask(model, img, device, threshold)
                result = apply_mask_to_image(img, mask)
                c1, c2, c3 = st.columns([1,1,1])
                with c1: st.subheader("Original"); st.image(img, use_column_width=True)
                with c2: st.subheader("Mask"); st.image(mask, use_column_width=True)
                with c3: st.subheader("Subject isolated"); st.image(result, use_column_width=True)
                if overlay:
                    o = np.array(img)
                    m = np.array(mask)==0
                    o[m] = (o[m]*0.2).astype(np.uint8)
                    st.subheader("Overlay"); st.image(Image.fromarray(o), use_column_width=True)
                buf = io.BytesIO(); result.save(buf, format="PNG"); buf.seek(0)
                st.download_button("Download isolated subject (PNG)",
                                   data=buf, file_name="subject_isolated.png", mime="image/png")
else:
    st.info("Upload an image to start.")
