# 🖼️ Segment-AI

A simple **AI-powered image segmentation app** that removes backgrounds and keeps only the **main subject**.  
Built with Streamlit and DeepLabV3.

## ✨ Features
- Upload an image and instantly segment it.  
- Real-time **Original vs Segmented** comparison.  
- Clean and minimal interface.  
- Download segmented output.

## 🧰 Tech Stack
- Python 3.10+  
- Streamlit  
- PyTorch (DeepLabV3)  
- Pillow, OpenCV

## 📂 Project Structure
├── models/
│ └── deeplabv3_subject_model.pth
├── app.py
├── requirements.txt
├── assets/
│ └── sample_image.jpg
└── README.md

⚡ How to Run
```bash
pip install -r requirements.txt
streamlit run app.py
