# import os
# os.environ["STREAMLIT_SERVER_PORT"] = os.environ.get("PORT", "8501")
# os.environ["STREAMLIT_SERVER_ADDRESS"] = "0.0.0.0"

# import streamlit as st
# import torch
# import torch.nn as nn
# import numpy as np
# from PIL import Image

# st.set_page_config(page_title="Digit Generator", layout="wide")

# class Generator(nn.Module):
#     def __init__(self, latent_dim, img_shape):
#         super(Generator, self).__init__()
#         self.img_shape = img_shape
#         self.label_emb = nn.Embedding(10, 10)

#         self.model = nn.Sequential(
#             nn.Linear(latent_dim + 10, 256),
#             nn.ReLU(True),
#             nn.Linear(256, 512),
#             nn.ReLU(True),
#             nn.Linear(512, 784),
#             nn.Tanh()
#         )

#     def forward(self, noise, labels):
#         gen_input = torch.cat((self.label_emb(labels), noise), -1)
#         img = self.model(gen_input)
#         img = img.view(img.size(0), *self.img_shape)
#         return img

# latent_dim = 64
# img_shape = (1, 28, 28)
# device = torch.device('cpu')

# @st.cache_resource
# def load_model():
#     generator = Generator(latent_dim, img_shape)
#     try:
#         generator.load_state_dict(torch.load("generator.pth", map_location=device))
#         generator.eval()
#         return generator, True
#     except Exception as e:
#         st.error(f"❌ Model load failed: {e}")
#         return None, False

# def generate_digit_images(generator, digit):
#     if generator is None:
#         return None

#     with torch.no_grad():
#         z = torch.randn(5, latent_dim, device=device)
#         digit_labels = torch.full((5,), digit, device=device, dtype=torch.long)
#         generated_imgs = generator(z, digit_labels)
#         generated_imgs = generated_imgs.cpu().numpy()
#         generated_imgs = (generated_imgs + 1) / 2
#         generated_imgs = np.clip(generated_imgs, 0, 1)

#         images = []
#         for img_array in generated_imgs:
#             img_array = np.squeeze(img_array) * 255
#             img_array = img_array.astype(np.uint8)
#             img = Image.fromarray(img_array, mode='L')
#             img = img.resize((112, 112), Image.NEAREST)
#             images.append(img)

#         return images

# # App layout
# st.title("🔢 Handwritten Digit Generator")
# st.markdown("Generate synthetic MNIST-like handwritten digits using a trained Conditional GAN")

# # Load model
# generator, model_loaded = load_model()

# if model_loaded:
#     st.success("✅ Model loaded successfully!")
# else:
#     st.error("❌ Model not loaded. Please ensure generator.pth is in the app directory.")

# # Controls
# col1, col2, col3 = st.columns([1, 1, 2])

# with col1:
#     digit = st.selectbox("Select Digit (0-9):", options=list(range(10)), index=2)

# with col2:
#     generate_btn = st.button("🎨 Generate Images", type="primary")

# # Generation and display
# if generate_btn:
#     if not model_loaded:
#         st.error("❌ Cannot generate images - model not loaded")
#     else:
#         with st.spinner(f"Generating 5 images of digit {digit}..."):
#             images = generate_digit_images(generator, digit)

#             if images:
#                 st.success(f"✅ Generated 5 images of digit {digit}!")
#                 st.markdown(f"### 🖼️ Generated Images of Digit {digit}")
#                 cols = st.columns(5)
#                 for i, img in enumerate(images):
#                     with cols[i]:
#                         st.image(img, caption=f"Sample {i+1}", use_column_width=True)
#             else:
#                 st.error("❌ Failed to generate images")

# # Model info
# st.markdown("---")
# st.markdown("### ℹ️ Model Information")
# col_info1, col_info2 = st.columns(2)

# with col_info1:
#     st.markdown("""
#     **Architecture:** Conditional GAN  
#     **Dataset:** MNIST handwritten digits  
#     **Training:** ~15 minutes on CPU or GPU  
#     """)

# with col_info2:
#     st.markdown("""
#     **Output:** 28×28 grayscale images  
#     **Framework:** PyTorch  
#     **Display:** 112×112 pixels  
#     """)

# # Model uploader
# st.markdown("---")
# st.markdown("### 📁 Upload Model (Alternative)")
# uploaded_file = st.file_uploader("Upload generator.pth if not found", type=['pth'])

# if uploaded_file is not None:
#     with open("generator.pth", "wb") as f:
#         f.write(uploaded_file.getbuffer())
#     st.success("✅ Model uploaded! Please refresh the page.")
#     st.rerun()


import os
os.environ["STREAMLIT_SERVER_PORT"] = os.environ.get("PORT", "8501")
os.environ["STREAMLIT_SERVER_ADDRESS"] = "0.0.0.0"

import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from PIL import Image

st.set_page_config(page_title="Digit Generator", layout="wide")

# Lightweight but improved Generator (compatible with Render's limits)
class Generator(nn.Module):
    def __init__(self, latent_dim, img_shape):
        super(Generator, self).__init__()
        self.img_shape = img_shape
        self.label_emb = nn.Embedding(10, 10)

        # Improved but lightweight architecture
        self.model = nn.Sequential(
            nn.Linear(latent_dim + 10, 512),  # Increased from 256
            nn.ReLU(True),
            nn.Dropout(0.2),  # Light dropout
            
            nn.Linear(512, 1024),  # Increased from 512
            nn.ReLU(True),
            nn.Dropout(0.2),
            
            nn.Linear(1024, 784),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        # Set to eval mode during inference to disable dropout
        if not self.training:
            self.eval()
            
        gen_input = torch.cat((self.label_emb(labels), noise), -1)
        img = self.model(gen_input)
        img = img.view(img.size(0), *self.img_shape)
        return img

latent_dim = 64
img_shape = (1, 28, 28)
device = torch.device('cpu')  # Force CPU for stability on Render

@st.cache_resource
def load_model():
    """Load model with better error handling"""
    try:
        generator = Generator(latent_dim, img_shape)
        
        # Try loading the model with more specific error handling
        if os.path.exists("generator.pth"):
            checkpoint = torch.load("generator.pth", map_location=device)
            generator.load_state_dict(checkpoint)
            generator.eval()  # Set to evaluation mode
            return generator, True, "Model loaded successfully!"
        else:
            return None, False, "generator.pth file not found"
            
    except RuntimeError as e:
        if "size mismatch" in str(e):
            # Try loading with the old simple architecture
            try:
                class SimpleGenerator(nn.Module):
                    def __init__(self, latent_dim, img_shape):
                        super(SimpleGenerator, self).__init__()
                        self.img_shape = img_shape
                        self.label_emb = nn.Embedding(10, 10)
                        self.model = nn.Sequential(
                            nn.Linear(latent_dim + 10, 256),
                            nn.ReLU(True),
                            nn.Linear(256, 512),
                            nn.ReLU(True),
                            nn.Linear(512, 784),
                            nn.Tanh()
                        )
                    def forward(self, noise, labels):
                        gen_input = torch.cat((self.label_emb(labels), noise), -1)
                        img = self.model(gen_input)
                        img = img.view(img.size(0), *self.img_shape)
                        return img
                
                simple_gen = SimpleGenerator(latent_dim, img_shape)
                checkpoint = torch.load("generator.pth", map_location=device)
                simple_gen.load_state_dict(checkpoint)
                simple_gen.eval()
                return simple_gen, True, "Loaded with simple architecture (fallback)"
                
            except Exception as fallback_error:
                return None, False, f"Architecture mismatch. Try retraining: {fallback_error}"
        else:
            return None, False, f"Model loading error: {e}"
    except Exception as e:
        return None, False, f"Unexpected error: {e}"

def generate_digit_images(generator, digit):
    """Generate images with better error handling"""
    if generator is None:
        return None, "No model loaded"

    try:
        generator.eval()  # Ensure eval mode
        with torch.no_grad():
            z = torch.randn(5, latent_dim, device=device)
            digit_labels = torch.full((5,), digit, device=device, dtype=torch.long)
            
            # Generate images
            generated_imgs = generator(z, digit_labels)
            
            # Process images
            generated_imgs = generated_imgs.cpu().numpy()
            generated_imgs = (generated_imgs + 1) / 2  # Denormalize
            generated_imgs = np.clip(generated_imgs, 0, 1)

            images = []
            for img_array in generated_imgs:
                img_array = np.squeeze(img_array) * 255
                img_array = img_array.astype(np.uint8)
                img = Image.fromarray(img_array, mode='L')
                img = img.resize((112, 112), Image.NEAREST)
                images.append(img)

            return images, "Success"
            
    except Exception as e:
        return None, f"Generation error: {e}"

# App layout
st.title("🔢 Handwritten Digit Generator")
st.markdown("Generate synthetic MNIST-like handwritten digits using a trained Conditional GAN")

# Load model with detailed status
with st.spinner("Loading model..."):
    generator, model_loaded, load_message = load_model()

if model_loaded:
    st.success(f"✅ {load_message}")
else:
    st.error(f"❌ {load_message}")
    
    # Helpful troubleshooting info
    st.info("""
    **Troubleshooting:**
    - Make sure `generator.pth` is uploaded to your app
    - If you see architecture mismatch, retrain your model
    - For Render deployment, the model file should be in the root directory
    """)

# Controls
col1, col2 = st.columns([1, 1])

with col1:
    digit = st.selectbox("Select Digit (0-9):", options=list(range(10)), index=2)

with col2:
    generate_btn = st.button("🎨 Generate Images", type="primary")

# Generation and display
if generate_btn:
    if not model_loaded:
        st.error("❌ Cannot generate images - model not loaded")
    else:
        with st.spinner(f"Generating 5 images of digit {digit}..."):
            images, status = generate_digit_images(generator, digit)

            if images and status == "Success":
                st.success(f"✅ Generated 5 images of digit {digit}!")
                st.markdown(f"### 🖼️ Generated Images of Digit {digit}")
                
                cols = st.columns(5)
                for i, img in enumerate(images):
                    with cols[i]:
                        st.image(img, caption=f"Sample {i+1}", use_column_width=True)
            else:
                st.error(f"❌ {status}")

# Model info
st.markdown("---")
st.markdown("### ℹ️ Model Information")
col_info1, col_info2 = st.columns(2)

with col_info1:
    st.markdown("""
    **Architecture:** Conditional GAN  
    **Dataset:** MNIST handwritten digits  
    **Training:** Improved architecture for better quality  
    """)

with col_info2:
    st.markdown("""
    **Output:** 28×28 grayscale images  
    **Framework:** PyTorch  
    **Display:** 112×112 pixels  
    """)

# Debug info (for troubleshooting)
with st.expander("🔧 Debug Information"):
    st.write("**System Info:**")
    st.write(f"- PyTorch version: {torch.__version__}")
    st.write(f"- Device: {device}")
    st.write(f"- Model loaded: {model_loaded}")
    
    if os.path.exists("generator.pth"):
        file_size = os.path.getsize("generator.pth") / (1024 * 1024)
        st.write(f"- Model file size: {file_size:.2f} MB")
    else:
        st.write("- Model file: NOT FOUND")

# Model uploader
st.markdown("---")
st.markdown("### 📁 Upload Model")
uploaded_file = st.file_uploader("Upload generator.pth", type=['pth'])

if uploaded_file is not None:
    with open("generator.pth", "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success("✅ Model uploaded! Please refresh the page.")
    st.rerun()