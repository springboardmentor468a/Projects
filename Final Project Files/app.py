import gradio as gr
import torch
import numpy as np
import cv2
from PIL import Image, ImageFilter, ImageEnhance
import os
import tempfile
import base64
import segmentation_models_pytorch as smp
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import random

# Set device for Hugging Face deployment
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Using device: {device}")

class VisionExtractApp:
    def __init__(self, model_path):
        self.device = device
        print(f"🚀 Initializing VisionExtract on {self.device}...")
        
        # Load DeepLabV3Plus model
        try:
            self.model = smp.DeepLabV3Plus(
                encoder_name="resnet101",
                encoder_weights=None,
                in_channels=3,
                classes=1
            )
            
            # Load your trained checkpoint - using relative path for Hugging Face
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            if "model_state" in checkpoint:
                self.model.load_state_dict(checkpoint["model_state"])
            else:
                self.model.load_state_dict(checkpoint)
            
            self.model.to(device=self.device)
            self.model.eval()
            print("✅ DeepLabV3Plus model loaded successfully!")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
        
        # Initialize variables
        self.original_image = None
        self.mask = None
        self.extracted_subject = None
        
        # Image normalization
        self.imagenet_mean = [0.485, 0.456, 0.406]
        self.imagenet_std = [0.229, 0.224, 0.225]
        self.confidence_threshold = 0.3

    def prepare_image_simple(self, img_pil):
        """Simple preprocessing that WORKS - same as your training"""
        w, h = img_pil.size
        
        # Pad to square (like in your training)
        s = max(w, h)
        pad_w = s - w
        pad_h = s - h
        padding = (pad_w // 2, pad_h // 2, pad_w - pad_w // 2, pad_h - pad_h // 2)
        img_pil = TF.pad(img_pil, padding, fill=0)
        
        # Resize to 512x512
        img_pil = img_pil.resize((512, 512), Image.BILINEAR)
        
        # Convert to tensor and normalize
        tensor = T.ToTensor()(img_pil)
        tensor = T.Normalize(mean=self.imagenet_mean, std=self.imagenet_std)(tensor)
        
        return tensor.unsqueeze(0).to(self.device), padding, (w, h)

    def remove_padding_simple(self, mask_np, padding, original_size):
        """Remove padding to get back to original dimensions"""
        pad_left, pad_top, pad_right, pad_bottom = padding
        h, w = mask_np.shape
        
        # Remove padding
        if pad_top + pad_bottom > 0:
            mask_np = mask_np[pad_top:h-pad_bottom] if pad_bottom > 0 else mask_np[pad_top:]
        if pad_left + pad_right > 0:
            mask_np = mask_np[:, pad_left:w-pad_right] if pad_right > 0 else mask_np[:, pad_left:]
        
        # Resize to original size
        original_w, original_h = original_size
        mask_original = Image.fromarray(mask_np)
        mask_original = mask_original.resize((original_w, original_h), Image.NEAREST)
        
        return np.array(mask_original)

    def clean_mask_smart(self, mask_np, min_area_ratio=0.005):
        """Smart mask cleaning - keep main object only"""
        if mask_np.sum() == 0:
            return mask_np
        
        # Find connected components
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_np.astype(np.uint8), connectivity=8)
        
        if num_labels <= 1:
            return mask_np
        
        # Calculate minimum area
        total_pixels = mask_np.shape[0] * mask_np.shape[1]
        min_area = total_pixels * min_area_ratio
        
        # Keep only components larger than minimum area
        cleaned = np.zeros_like(mask_np)
        
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= min_area:
                cleaned[labels == i] = 1
        
        # If nothing meets threshold, keep the largest one
        if cleaned.sum() == 0:
            largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            cleaned[labels == largest_label] = 1
        
        return cleaned

    def smooth_edges_optimized(self, mask_np):
        """Optimized edge smoothing - simple but effective"""
        if mask_np.sum() == 0:
            return mask_np
        
        # Step 1: Light morphological closing to fill small gaps
        kernel = np.ones((3, 3), np.uint8)
        closed = cv2.morphologyEx(mask_np, cv2.MORPH_CLOSE, kernel)
        
        # Step 2: Gentle Gaussian blur for smooth edges
        mask_float = closed.astype(np.float32)
        smoothed = cv2.GaussianBlur(mask_float, (5, 5), 0.8)
        
        # Step 3: Re-threshold
        smoothed_binary = (smoothed > 0.4).astype(np.uint8)
        
        return smoothed_binary

    def remove_small_holes(self, mask_np, max_hole_size=500):
        """Remove small holes in the mask"""
        if mask_np.sum() == 0:
            return mask_np
        
        # Invert mask to find holes
        inverted = 1 - mask_np
        
        # Find connected components in inverted mask (holes)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(inverted.astype(np.uint8), connectivity=8)
        
        if num_labels <= 1:
            return mask_np
        
        # Fill small holes
        filled_mask = mask_np.copy()
        
        for i in range(1, num_labels):  # Skip background
            if stats[i, cv2.CC_STAT_AREA] <= max_hole_size:
                filled_mask[labels == i] = 1
        
        return filled_mask

    def extract_subject(self, image):
        """Extract subject using DeepLabV3Plus - MATCHING YOUR NOTEBOOK"""
        if image is None:
            return None, None
            
        try:
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Store original image
            self.original_image = np.array(image)
            
            # Preprocess (with padding) - USING YOUR WORKING NOTEBOOK CODE
            tensor, padding, original_size = self.prepare_image_simple(image)
            
            # Predict
            with torch.no_grad():
                output = self.model(tensor)
                pred_mask = torch.sigmoid(output)[0, 0].cpu().numpy()
            
            # Apply confidence threshold (lower for better edges)
            pred_binary = (pred_mask > self.confidence_threshold).astype(np.uint8)
            
            # Remove padding
            pred_original = self.remove_padding_simple(pred_binary, padding, original_size)
            
            # OPTIMIZED Post-processing Pipeline (3 simple steps) - FROM YOUR NOTEBOOK
            # Step 1: Clean mask (remove small noise)
            cleaned_mask = self.clean_mask_smart(pred_original)
            
            # Step 2: Fill small holes
            filled_mask = self.remove_small_holes(cleaned_mask)
            
            # Step 3: Smooth edges gently
            final_mask = self.smooth_edges_optimized(filled_mask)
            
            # Store the final mask
            self.mask = final_mask.astype(bool)
            
            # Extract subject - SIMPLE AND RELIABLE APPROACH
            extracted_subject = self.original_image.copy()
            
            # Create proper 3D mask for RGB image
            mask_3d = self.mask[:, :, np.newaxis]  # Shape: (H, W, 1)
            extracted_subject = extracted_subject * mask_3d  # Element-wise multiplication
            
            self.extracted_subject = extracted_subject
            
            return Image.fromarray(extracted_subject), self.mask
            
        except Exception as e:
            print(f"❌ Error in extraction: {e}")
            return None, None

    def apply_background(self, background_mode, custom_color=None, custom_bg_image=None, blur_strength=10):
        """Apply different background modes - SIMPLIFIED AND FIXED"""
        if self.original_image is None or self.mask is None:
            return None
            
        try:
            # Start with original image
            result = self.original_image.copy()
            
            # Create proper mask for background replacement
            background_area = ~self.mask
            
            if background_mode == "Transparent":
                # Create RGBA image with transparency
                rgba = np.dstack((self.original_image, self.mask.astype(np.uint8) * 255))
                return Image.fromarray(rgba, 'RGBA')
                
            elif background_mode == "White":
                result[background_area] = [255, 255, 255]
                
            elif background_mode == "Black":
                result[background_area] = [0, 0, 0]
                
            elif background_mode == "Blur":
                # Apply Gaussian blur to background only
                blurred_bg = cv2.GaussianBlur(self.original_image, (blur_strength*2+1, blur_strength*2+1), 0)
                # Replace only the background area with blurred version
                for c in range(3):  # For each color channel
                    result_channel = result[:, :, c]
                    blurred_channel = blurred_bg[:, :, c]
                    result_channel[background_area] = blurred_channel[background_area]
                    result[:, :, c] = result_channel
                
            elif background_mode == "Custom Color" and custom_color:
                # Convert hex color to RGB
                if custom_color.startswith('#'):
                    color_rgb = tuple(int(custom_color[i:i+2], 16) for i in (1, 3, 5))
                else:
                    color_rgb = (255, 255, 255)  # Default white
                # Apply color to background
                result[background_area] = color_rgb
                
            elif background_mode == "Custom Image" and custom_bg_image is not None:
                bg_array = np.array(custom_bg_image)
                # Resize background to match original image
                if len(bg_array.shape) == 2:  # Grayscale background
                    bg_array = cv2.cvtColor(bg_array, cv2.COLOR_GRAY2RGB)
                bg_resized = cv2.resize(bg_array, (result.shape[1], result.shape[0]))
                # Replace background
                result[background_area] = bg_resized[background_area]
            
            return Image.fromarray(result)
            
        except Exception as e:
            print(f"❌ Error applying background: {e}")
            return None

    # ARTISTIC FILTERS FROM YOUR NOTEBOOK
    def apply_cartoon_effect(self, image_pil):
        """Apply cartoon effect to image"""
        img_np = np.array(image_pil)
        
        # Convert to grayscale
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        
        # Apply bilateral filter to smooth while preserving edges
        smooth = cv2.bilateralFilter(img_np, 9, 75, 75)
        
        # Detect edges
        edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                     cv2.THRESH_BINARY, 9, 2)
        
        # Convert edges to 3 channels
        edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        
        # Combine smooth image with edges
        cartoon = cv2.bitwise_and(smooth, edges_colored)
        
        return Image.fromarray(cartoon)

    def apply_pencil_sketch_effect(self, image_pil):
        """Apply pencil sketch effect"""
        img_np = np.array(image_pil)
        
        # Convert to grayscale
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        
        # Invert the grayscale image
        inverted = 255 - gray
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(inverted, (21, 21), 0)
        
        # Invert the blurred image
        inverted_blur = 255 - blurred
        
        # Create pencil sketch by dividing gray by inverted blur
        pencil_sketch = cv2.divide(gray, inverted_blur, scale=256.0)
        
        # Convert back to 3 channels
        pencil_sketch_colored = cv2.cvtColor(pencil_sketch, cv2.COLOR_GRAY2RGB)
        
        return Image.fromarray(pencil_sketch_colored)

    def apply_sepia_effect(self, image_pil):
        """Apply sepia tone effect"""
        img_np = np.array(image_pil)
        
        # Sepia filter matrix
        sepia_filter = np.array([[0.393, 0.769, 0.189],
                                [0.349, 0.686, 0.168],
                                [0.272, 0.534, 0.131]])
        
        # Apply sepia filter
        sepia_img = cv2.transform(img_np, sepia_filter)
        
        # Clip values to valid range
        sepia_img = np.clip(sepia_img, 0, 255).astype(np.uint8)
        
        return Image.fromarray(sepia_img)

    def apply_hdr_effect(self, image_pil):
        """Apply HDR-like effect (enhanced contrast and saturation)"""
        # Enhance contrast
        contrast_enhancer = ImageEnhance.Contrast(image_pil)
        enhanced = contrast_enhancer.enhance(1.5)
        
        # Enhance color saturation
        color_enhancer = ImageEnhance.Color(enhanced)
        enhanced = color_enhancer.enhance(1.3)
        
        # Enhance sharpness
        sharpness_enhancer = ImageEnhance.Sharpness(enhanced)
        enhanced = sharpness_enhancer.enhance(1.2)
        
        return enhanced

    def apply_color_glitch_effect(self, image_pil):
        """Apply color glitch effect"""
        img_np = np.array(image_pil)
        
        # Split channels
        r, g, b = img_np[:, :, 0], img_np[:, :, 1], img_np[:, :, 2]
        
        # Randomly shift channels
        shift_x = random.randint(2, 5)
        shift_y = random.randint(2, 5)
        
        # Shift red channel
        r_shifted = np.roll(r, shift_x, axis=1)
        r_shifted = np.roll(r_shifted, shift_y, axis=0)
        
        # Shift blue channel in opposite direction
        b_shifted = np.roll(b, -shift_x, axis=1)
        b_shifted = np.roll(b_shifted, -shift_y, axis=0)
        
        # Combine channels
        glitched = np.stack([r_shifted, g, b_shifted], axis=2)
        
        return Image.fromarray(glitched)

    def apply_painting_effect(self, image_pil):
        """Apply painting-like effect"""
        img_np = np.array(image_pil)
        
        # Apply bilateral filter for oil painting effect
        painted = cv2.stylization(img_np, sigma_s=60, sigma_r=0.6)
        
        return Image.fromarray(painted)

    def apply_filter(self, filter_name, image_to_filter=None):
        """Apply artistic filters - USING YOUR NOTEBOOK FILTERS"""
        if image_to_filter is None:
            return None
            
        try:
            if filter_name == "None":
                return image_to_filter
            elif filter_name == "Cartoon":
                return self.apply_cartoon_effect(image_to_filter)
            elif filter_name == "Pencil":
                return self.apply_pencil_sketch_effect(image_to_filter)
            elif filter_name == "HDR":
                return self.apply_hdr_effect(image_to_filter)
            elif filter_name == "Sepia":
                return self.apply_sepia_effect(image_to_filter)
            elif filter_name == "Painting":
                return self.apply_painting_effect(image_to_filter)
            elif filter_name == "Blur":
                return image_to_filter.filter(ImageFilter.GaussianBlur(5))
            elif filter_name == "Gray":
                return image_to_filter.convert('L').convert('RGB')
            elif filter_name == "Glitch":
                return self.apply_color_glitch_effect(image_to_filter)
            
            return image_to_filter
            
        except Exception as e:
            print(f"❌ Error applying filter: {e}")
            return image_to_filter

    def apply_filter_to_subject_only(self, filter_name, extracted_image):
        """Apply filter only to the subject area - SIMPLIFIED"""
        if extracted_image is None or self.mask is None:
            return extracted_image
            
        try:
            # Apply filter to the entire extracted image
            filtered_img = self.apply_filter(filter_name, extracted_image)
            if filtered_img is None:
                return extracted_image
                
            return filtered_img
            
        except Exception as e:
            print(f"❌ Error applying filter to subject: {e}")
            return extracted_image

    def crop_image(self, image, aspect_ratio, scale_factor=1.0):
        """Crop image based on aspect ratio"""
        if image is None:
            return None
            
        try:
            img_array = np.array(image)
            h, w = img_array.shape[:2]
            
            # Calculate target dimensions based on aspect ratio
            if aspect_ratio == "Free":
                new_w, new_h = w, h
            elif aspect_ratio == "Square (1:1)":
                size = min(w, h)
                new_w, new_h = size, size
            elif aspect_ratio == "Instagram (4:5)":
                new_h = h
                new_w = int(h * 4/5)
                if new_w > w:
                    new_w = w
                    new_h = int(w * 5/4)
            elif aspect_ratio == "Portrait (3:4)":
                new_h = h
                new_w = int(h * 3/4)
                if new_w > w:
                    new_w = w
                    new_h = int(w * 4/3)
            
            # Apply scale factor
            new_w = int(new_w * scale_factor)
            new_h = int(new_h * scale_factor)
            
            # Ensure dimensions are at least 1
            new_w = max(1, new_w)
            new_h = max(1, new_h)
            
            # Center crop
            start_x = max(0, (w - new_w) // 2)
            start_y = max(0, (h - new_h) // 2)
            cropped = img_array[start_y:start_y+new_h, start_x:start_x+new_w]
            
            return Image.fromarray(cropped)
            
        except Exception as e:
            print(f"❌ Error cropping image: {e}")
            return None

    def create_comparison(self, final_image):
        """Create side-by-side comparison"""
        if self.original_image is None or final_image is None:
            return None
            
        try:
            original_pil = Image.fromarray(self.original_image)
            final_pil = final_image
            
            # Resize to same height
            height = min(original_pil.height, final_pil.height, 400)
            orig_resized = original_pil.resize((int(original_pil.width * height / original_pil.height), height))
            final_resized = final_pil.resize((int(final_pil.width * height / final_pil.height), height))
            
            # Create comparison image
            total_width = orig_resized.width + final_resized.width + 20
            comparison = Image.new('RGB', (total_width, height), (240, 240, 240))
            
            comparison.paste(orig_resized, (0, 0))
            comparison.paste(final_resized, (orig_resized.width + 20, 0))
            
            return comparison
            
        except Exception as e:
            print(f"❌ Error creating comparison: {e}")
            return None

# Initialize the app with relative path for Hugging Face
MODEL_PATH = "resumed_best_epoch30.pth"  # Changed to relative path
app = VisionExtractApp(MODEL_PATH)

# CSS for modern UI
css = """
.gradio-container {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    min-height: 100vh;
}
.header {
    text-align: center;
    padding: 25px;
    background: linear-gradient(135deg, rgba(102,126,234,0.9) 0%, rgba(118,75,162,0.9) 100%);
    border-radius: 20px;
    margin: 15px;
    backdrop-filter: blur(10px);
    border: 2px solid rgba(255,255,255,0.3);
    box-shadow: 0 8px 32px rgba(0,0,0,0.2);
}
.header h1 {
    color: white;
    font-size: 3em;
    margin: 0;
    text-shadow: 2px 2px 8px rgba(0,0,0,0.3);
    font-weight: 700;
}
.header p {
    color: #f0f0f0;
    font-size: 1.3em;
    margin: 10px 0 0 0;
    font-weight: 300;
}
.demo-collage {
    text-align: center;
    margin: 25px 0;
    padding: 20px;
    background: rgba(255,255,255,0.15);
    border-radius: 20px;
    backdrop-filter: blur(5px);
    border: 1px solid rgba(255,255,255,0.2);
}
.collage-image {
    border-radius: 15px;
    box-shadow: 0 12px 40px rgba(0,0,0,0.4);
    max-width: 85%;
    border: 4px solid white;
    transition: transform 0.3s ease;
}
.collage-image:hover {
    transform: scale(1.02);
}
.collage-caption {
    color: white;
    font-size: 1.2em;
    margin-top: 15px;
    font-weight: 600;
    text-shadow: 1px 1px 3px rgba(0,0,0,0.3);
}
.upload-box {
    border: 3px dashed rgba(255,255,255,0.7);
    border-radius: 15px;
    padding: 30px;
    text-align: center;
    background: linear-gradient(135deg, rgba(74,144,226,0.2) 0%, rgba(142,45,226,0.2) 100%);
    backdrop-filter: blur(5px);
    transition: all 0.3s ease;
    margin: 10px 0;
}
.upload-box:hover {
    border-color: rgba(255,255,255,0.9);
    background: linear-gradient(135deg, rgba(74,144,226,0.3) 0%, rgba(142,45,226,0.3) 100%);
}
.upload-box h3 {
    color: white;
    font-size: 1.5em;
    margin: 0 0 10px 0;
}
.upload-box p {
    color: rgba(255,255,255,0.9);
    margin: 0;
}
.section-title {
    color: white;
    font-size: 1.4em;
    font-weight: 600;
    margin: 20px 0 15px 0;
    text-shadow: 1px 1px 3px rgba(0,0,0,0.3);
    padding: 10px 15px;
    border-radius: 10px;
    background: rgba(255,255,255,0.15);
}
.background-title {
    background: linear-gradient(135deg, rgba(255,107,107,0.3) 0%, rgba(255,142,83,0.3) 100%);
    border-left: 4px solid #FF6B6B;
}
.filter-title {
    background: linear-gradient(135deg, rgba(78,205,196,0.3) 0%, rgba(68,160,141,0.3) 100%);
    border-left: 4px solid #4ECDC4;
}
.crop-title {
    background: linear-gradient(135deg, rgba(158,103,255,0.3) 0%, rgba(255,103,165,0.3) 100%);
    border-left: 4px solid #9E67FF;
}
.option-group {
    background: rgba(255,255,255,0.12);
    border-radius: 15px;
    padding: 20px;
    margin: 15px 0;
    backdrop-filter: blur(5px);
    border: 1px solid rgba(255,255,255,0.2);
}
.process-btn {
    background: linear-gradient(135deg, #FF6B6B 0%, #FF8E53 100%) !important;
    border: none !important;
    color: white !important;
    font-weight: 600 !important;
    font-size: 1.2em !important;
    padding: 15px 30px !important;
    border-radius: 12px !important;
    transition: all 0.3s ease !important;
    width: 100%;
    margin: 20px 0;
    box-shadow: 0 4px 15px rgba(255,107,107,0.3);
}
.process-btn:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(255,107,107,0.4);
}
.download-btn {
    background: linear-gradient(135deg, #4ECDC4 0%, #44A08D 100%) !important;
    border: none !important;
    color: white !important;
    font-weight: 600 !important;
    border-radius: 10px !important;
    padding: 12px 20px !important;
    margin: 5px;
}
.results-section {
    background: rgba(255,255,255,0.12);
    border-radius: 20px;
    padding: 25px;
    margin: 20px 0;
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255,255,255,0.2);
    text-align: center;
}
.tab-button {
    background: rgba(255,255,255,0.15) !important;
    color: white !important;
    border: 1px solid rgba(255,255,255,0.3) !important;
    border-radius: 10px !important;
    margin: 5px !important;
    transition: all 0.3s ease !important;
}
.tab-button.selected {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    border: 1px solid rgba(255,255,255,0.5) !important;
    box-shadow: 0 4px 15px rgba(102,126,234,0.4);
}
.main-container {
    max-width: 1400px !important;
    margin: 0 auto !important;
}
.label-text {
    color: white !important;
    font-weight: 600 !important;
}
.radio-group .gr-form {
    background: rgba(255,255,255,0.1) !important;
    border-radius: 10px !important;
    padding: 10px !important;
}
"""

# Function to encode image to base64
def image_to_base64(image_path):
    """Convert image to base64 for embedding"""
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except:
        return ""

# Create the modern interface
with gr.Blocks(css=css, theme=gr.themes.Soft()) as demo:
    
    gr.HTML("""
    <div class="main-container">
        <div class="header">
            <h1>🎯 VisionExtract</h1>
            <p>AI-Powered Subject Isolation & Background Replacement</p>
        </div>
    """)
    
    # Demo collage section
    with gr.Row():
        with gr.Column():
            # Use relative path for Hugging Face deployment
            try:
                collage_base64 = image_to_base64("download.png")  # Changed to relative path
                if collage_base64:
                    gr.HTML(f"""
                    <div class="demo-collage">
                        <h3 style="color: white; text-align: center; margin-bottom: 20px;">✨ See What You Can Create</h3>
                        <img src="data:image/png;base64,{collage_base64}" class="collage-image" alt="Feature Demo Collage">
                        <div class="collage-caption">🎭 Background Replacement • 🎨 Artistic Filters • ✂️ Smart Cropping</div>
                    </div>
                    """)
                else:
                    gr.HTML("""
                    <div class="demo-collage">
                        <h3 style="color: white; text-align: center;">✨ AI-Powered Image Editing</h3>
                        <div class="collage-caption">Upload an image to see the magic! 🎩✨</div>
                    </div>
                    """)
            except:
                gr.HTML("""
                <div class="demo-collage">
                    <h3 style="color: white; text-align: center;">✨ AI-Powered Image Editing</h3>
                    <div class="collage-caption">Upload an image to see the magic! 🎩✨</div>
                </div>
                """)
    
    with gr.Row(equal_height=True):
        with gr.Column(scale=1, min_width=400):
            # Upload section with colorful styling
            with gr.Group():
                gr.HTML("""
                <div class="upload-box">
                    <h3>📤 Upload Your Image</h3>
                    <p>Drag & drop or click to browse • JPG, PNG, JPEG</p>
                </div>
                """)
                input_image = gr.Image(
                    label="", 
                    type="pil", 
                    height=200, 
                    show_label=False,
                    sources=["upload"]
                )
            
            # Background Options with colorful header
            with gr.Group():
                gr.Markdown("<div class='section-title background-title'>🎭 Background Options</div>")
                with gr.Group(elem_classes="option-group"):
                    background_mode = gr.Radio(
                        choices=["Transparent", "White", "Black", "Blur", "Custom Color", "Custom Image"],
                        label="Select Background Style",
                        value="White",
                        info="Choose how to handle the background"
                    )
                    with gr.Row():
                        custom_color = gr.ColorPicker(label="Custom Color", visible=False)
                        blur_strength = gr.Slider(1, 20, value=10, label="Blur Strength", visible=False)
                    custom_bg = gr.Image(
                        label="Custom Background Image", 
                        type="pil", 
                        visible=False, 
                        height=150,
                        sources=["upload"]
                    )
            
            # Filters with colorful header
            with gr.Group():
                gr.Markdown("<div class='section-title filter-title'>🎨 Artistic Filters</div>")
                with gr.Group(elem_classes="option-group"):
                    filter_type = gr.Radio(
                        choices=["None", "Cartoon", "Pencil", "HDR", "Sepia", "Painting", "Blur", "Gray", "Glitch"],
                        label="Choose Artistic Filter",
                        value="None",
                        info="Apply creative effects to your image"
                    )
            
            # Crop options with colorful header
            with gr.Group():
                gr.Markdown("<div class='section-title crop-title'>✂️ Crop & Resize</div>")
                with gr.Group(elem_classes="option-group"):
                    aspect_ratio = gr.Radio(
                        choices=["Free", "Square (1:1)", "Instagram (4:5)", "Portrait (3:4)"],
                        label="Aspect Ratio",
                        value="Free",
                        info="Crop for different platforms"
                    )
                    scale_factor = gr.Slider(0.1, 2.0, value=1.0, label="Scale Factor", info="Resize the output image")
            
            # Process button
            process_btn = gr.Button("🚀 Process Image", elem_classes="process-btn", size="lg")
        
        with gr.Column(scale=2, min_width=600):
            # Results area
            with gr.Group(elem_classes="results-section"):
                gr.Markdown("<div class='section-title' style='text-align: center; font-size: 1.6em;'>📊 Results & Export</div>")
                
                with gr.Tabs() as tabs:
                    with gr.TabItem("🎯 Extracted Subject", elem_classes="tab-button"):
                        extracted_output = gr.Image(
                            label="AI-Extracted Subject",
                            interactive=False,
                            height=400,
                            show_download_button=False
                        )
                    
                    with gr.TabItem("✨ Final Result", elem_classes="tab-button"):
                        final_output = gr.Image(
                            label="Final Processed Image", 
                            interactive=False,
                            height=400,
                            show_download_button=False
                        )
                    
                    with gr.TabItem("🔄 Comparison", elem_classes="tab-button"):
                        comparison_output = gr.Image(
                            label="Before & After Comparison",
                            interactive=False, 
                            height=400,
                            show_download_button=False
                        )
                
                # Download buttons
                with gr.Row():
                    with gr.Column(scale=1):
                        download_final = gr.DownloadButton(
                            "📥 Download Final Image", 
                            visible=False,
                            elem_classes="download-btn"
                        )
                    with gr.Column(scale=1):
                        download_comparison = gr.DownloadButton(
                            "📥 Download Comparison", 
                            visible=False,
                            elem_classes="download-btn"
                        )
    
    gr.HTML("</div>")  # Close main-container
    
    # Show/hide background options based on selection
    def update_background_visibility(background_mode):
        return (
            gr.update(visible=background_mode == "Custom Color"),
            gr.update(visible=background_mode == "Custom Image"),
            gr.update(visible=background_mode == "Blur")
        )
    
    background_mode.change(
        update_background_visibility,
        inputs=[background_mode],
        outputs=[custom_color, custom_bg, blur_strength]
    )
    
    # SIMPLIFIED AND RELIABLE PROCESSING FUNCTION
    def process_and_update(input_image, background_mode, custom_color, custom_bg, blur_strength, 
                          filter_type, aspect_ratio, scale_factor):
        # Reset outputs initially
        extracted, final, comparison = None, None, None
        final_file, comparison_file = None, None
        
        if input_image is not None:
            try:
                # Extract subject
                extracted, mask = app.extract_subject(input_image)
                
                if extracted is not None:
                    # Apply filter to extracted subject if needed
                    if filter_type != "None":
                        extracted = app.apply_filter_to_subject_only(filter_type, extracted)
                    
                    # Apply background
                    final = app.apply_background(background_mode, custom_color, custom_bg, blur_strength)
                    
                    # Apply cropping
                    if final is not None and aspect_ratio != "Free":
                        final = app.crop_image(final, aspect_ratio, scale_factor)
                    
                    # Create comparison
                    comparison = app.create_comparison(final)
                    
                    # Prepare download files
                    if final:
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as f:
                            final.save(f.name, "PNG")
                            final_file = f.name
                    
                    if comparison:
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as f:
                            comparison.save(f.name, "PNG")
                            comparison_file = f.name
                            
            except Exception as e:
                print(f"❌ Error in processing pipeline: {e}")
        
        return extracted, final, comparison, final_file, comparison_file
    
    # Connect process button
    process_btn.click(
        process_and_update,
        inputs=[input_image, background_mode, custom_color, custom_bg, blur_strength,
                filter_type, aspect_ratio, scale_factor],
        outputs=[extracted_output, final_output, comparison_output, 
                download_final, download_comparison]
    )
    
    # Update download button visibility
    def update_download_visibility(final, comparison):
        return (
            gr.update(visible=final is not None), 
            gr.update(visible=comparison is not None)
        )
    
    final_output.change(
        update_download_visibility,
        inputs=[final_output, comparison_output],
        outputs=[download_final, download_comparison]
    )

# Launch the app
if __name__ == "__main__":
    demo.launch(share=True, server_port=7860)
