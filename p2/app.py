import os
os.environ["STREAMLIT_SERVER_PORT"] = os.environ.get("PORT", "8501")
os.environ["STREAMLIT_SERVER_ADDRESS"] = "0.0.0.0"

import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from PIL import Image

st.set_page_config(page_title="Digit Generator", layout="wide")

class Generator(nn.Module):
    def __init__(self, latent_dim, img_shape):
        super(Generator, self).__init__()
        self.img_shape = img_shape
        self.latent_dim = latent_dim
        self.label_emb = nn.Embedding(10, 10)

        self.conv_blocks = nn.Sequential(
            nn.ConvTranspose2d(latent_dim + 10, 128, kernel_size=3, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(True),

            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, output_padding=1),
            nn.Tanh()
        )

        self.fc = nn.Linear(latent_dim + 10, (latent_dim + 10) * 1 * 1)

    def forward(self, noise, labels):
        label_embedding = self.label_emb(labels)
        gen_input = torch.cat((label_embedding, noise), -1)

        out = self.fc(gen_input)
        out = out.view(out.size(0), self.latent_dim + 10, 1, 1)
        img = self.conv_blocks(out)
        return img

latent_dim = 64
img_shape = (1, 28, 28)
device = torch.device('cpu')

@st.cache_resource
def load_model():
    generator = Generator(latent_dim, img_shape)
    try:
        generator.load_state_dict(torch.load('generator.pth', map_location=device))
        generator.eval()
        return generator, True
    except Exception as e:
        st.error(f"❌ Model load failed: {e}")
        return None, False

def generate_digit_images(generator, digit):
    if generator is None:
        return None

    with torch.no_grad():
        z = torch.randn(5, latent_dim, device=device)
        digit_labels = torch.full((5,), digit, device=device, dtype=torch.long)
        generated_imgs = generator(z, digit_labels)
        generated_imgs = generated_imgs.cpu().numpy()
        generated_imgs = (generated_imgs + 1) / 2
        generated_imgs = np.clip(generated_imgs, 0, 1)

        images = []
        for img_array in generated_imgs:
            img_array = np.squeeze(img_array) * 255
            img_array = img_array.astype(np.uint8)
            img = Image.fromarray(img_array, mode='L')
            img = img.resize((112, 112), Image.NEAREST)
            images.append(img)

        return images

# App layout
st.title("🔢 Handwritten Digit Generator")
st.markdown("Generate synthetic MNIST-like handwritten digits using a trained Conditional GAN")

# Load model
generator, model_loaded = load_model()

if model_loaded:
    st.success("✅ Model loaded successfully!")
else:
    st.error("❌ Model not loaded. Please ensure generator.pth is in the app directory.")

# Controls
col1, col2, col3 = st.columns([1, 1, 2])

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
            images = generate_digit_images(generator, digit)

            if images:
                st.success(f"✅ Generated 5 images of digit {digit}!")

                # Display images
                st.markdown(f"### 🖼️ Generated Images of Digit {digit}")
                cols = st.columns(5)

                for i, img in enumerate(images):
                    with cols[i]:
                        st.image(img, caption=f"Sample {i+1}", use_column_width=True)
            else:
                st.error("❌ Failed to generate images")

# Model info
st.markdown("---")
st.markdown("### ℹ️ Model Information")
col_info1, col_info2 = st.columns(2)

with col_info1:
    st.markdown("""
    **Architecture:** Conditional GAN  
    **Dataset:** MNIST handwritten digits  
    **Training:** ~15-20 minutes on T4 GPU  
    """)

with col_info2:
    st.markdown("""
    **Output:** 28×28 grayscale images  
    **Framework:** PyTorch  
    **Display:** 112×112 pixels  
    """)

# File uploader (alternative)
st.markdown("---")
st.markdown("### 📁 Upload Model (Alternative)")
uploaded_file = st.file_uploader("Upload generator.pth if not found", type=['pth'])

if uploaded_file is not None:
    with open("generator.pth", "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success("✅ Model uploaded! Please refresh the page.")
    st.rerun()
