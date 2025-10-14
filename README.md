## 🧠 AI-Powered Subject Extraction & Background Editor

### 🌐 [Live App → Vision Extraction](https://image-segment.streamlit.app/)

---

## 🖼️ Overview

**Vision Extraction** is a cutting-edge web application that performs **semantic image segmentation** using the **DeepLabV3 (ResNet50)** deep learning model.  
It allows users to **automatically detect, extract, and edit subjects** from images with ease.

The app supports **transparent, blurred, or custom backgrounds**, and even **batch processing** for multiple images — all within an interactive Streamlit interface.

---

## 🚀 Key Features

✅ **Automatic Object Extraction** – Uses a pre-trained **DeepLabV3 model** for precise segmentation.  
✅ **AI Background Removal** – Isolates the main subject from the background intelligently.  
✅ **Multiple Background Options** – Transparent, solid colors, blur, or custom image backgrounds.  
✅ **Preset Backgrounds** – Choose from preset images directly from the sidebar.  
✅ **Batch Processing** – Process multiple images with the same settings in one go.  
✅ **Side-by-Side Display** – Compare **original vs segmented output** instantly.  
✅ **Interactive Web App** – Built using **Streamlit** with real-time previews.  
✅ **Modern UI** – Animated gradient background and clean layout for a professional look.  
✅ **Multi-format Support** – Upload `.jpg`, `.jpeg`, `.png` images.  
✅ **Download Options** – Save results easily in your desired format.  

---

## 🧩 Tech Stack

| Component | Technology |
|------------|-------------|
| **Frontend** | Streamlit |
| **Model** | DeepLabV3 (ResNet50) |
| **Frameworks** | PyTorch, TorchVision |
| **Image Processing** | OpenCV, PIL |
| **Utilities** | NumPy, gdown, os |
| **Deployment** | Streamlit Cloud |

---

## 🧠 How It Works

1. **Upload Image(s):** Choose one or more images to process.  
2. **Segmentation:** The DeepLabV3 model predicts a **pixel-wise mask** separating foreground and background.  
3. **Background Editing:** Apply effects such as blur, solid colors, transparency, or preset backgrounds.  
4. **Preview & Download:** Instantly view the comparison and export the processed image.

---

## ⚙️ Installation & Setup


```bash
1️⃣ Clone the Repository
git clone https://github.com/Satyamjai1003/Image_Segmentation_App.git
cd Image_Segmentation_App

2️⃣ Install Dependencies
pip install -r requirements.txt

3️⃣ Add Model Files

Ensure the following files are present in the project directory:

best_model (5).pth
Explainer.png

▶️ Usage

Run the Streamlit app:

streamlit run Frontend.py
```

---
### Steps:
1. Upload your image or try Demo image.
2. Choose your desired background type.
3. Apply effects and preview.
4. Download the final output.
---
## ⚡ Quick Start
``` bash 
pip install -r requirements.txt
# Place your trained model file in the root folder
streamlit run Fontend.py
```

--- 

## 🧾 Notes

* The model currently predicts two classes: background and object.

* Ensure model weights (.pth) are placed correctly.

* You can modify preprocessing or model path for your custom setup.

* Works best with clean, high-resolution images.

---

## 💡 Future Enhancements

* Support for multi-class segmentation

* Integration with custom-trained models

* Option to auto-save session history

* Add edge refinement and mask smoothing

---

## 🏆 Acknowledgments

***PyTorch***
 * For model training and inference

***Streamlit***
 * For the beautiful web interface

***TorchVision***
* For pre-trained segmentation models

* OpenCV & PIL for advanced image manipulation

---

⭐ If you like this project, don’t forget to give it a star on GitHub!
Let’s make AI-powered image editing smarter and simpler 🚀


---
## 👨‍💻 Author

Developed by: ***Satyam Jaiswal***

📧 For collaboration or queries: open an issue on GitHub!

