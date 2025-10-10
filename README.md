# 🧠 AI Vision Extract

AI Vision Extract is a deep learning-based web application designed to **automatically extract objects from images** using **semantic segmentation**.  
Built with **PyTorch** on the backend and an interactive **Streamlit** frontend, this system enables users to upload any image, segment objects in real time, and download the extracted results easily.

---

## 🚀 Features

- 🎯 **DeepLabV3-ResNet50 Model** (PyTorch)
  - Trained on the **COCO dataset** (91 object classes)
  - Includes feature refinement layers for high-accuracy segmentation
- ⚡ **GPU Acceleration**
  - Automatically detects CUDA for faster inference
- 🖼️ **Streamlit-based Web UI**
  - Simple, interactive interface for uploading, previewing, and downloading segmented images
- ☁️ **Google Drive Integration**
  - Automatically downloads model weights if missing
- 🧩 **Post-Processing**
  - Smooths mask edges and enhances visual clarity

---

## 🧰 Tech Stack

| Layer | Technologies Used |
|-------|--------------------|
| **Frontend** | Streamlit, HTML/CSS styling |
| **Backend** | Python, PyTorch, Torchvision |
| **Model** | DeepLabV3-ResNet50 (Pretrained on COCO) |
| **Data** | COCO Dataset (Common Objects in Context) |
| **Utilities** | NumPy, PIL, gdown, matplotlib |

---

## ⚙️ Project Link

[🚀 Live Demo](https://ai-image-segmentation-gxd8tax5pcmlnfpjlur.streamlit.app/)


