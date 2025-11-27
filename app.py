import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import io
from streamlit_image_comparison import image_comparison

# ==========================================
# 1. PAGE CONFIGURATION & CSS
# ==========================================
st.set_page_config(
    page_title="DeBlurX - AI Restoration",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a "Tech" look
st.markdown("""
    <style>
    /* Main Background */
    .stApp {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    /* Sidebar Background */
    [data-testid="stSidebar"] {
        background-color: #262730;
    }
    /* Buttons */
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 8px;
        height: 3em;
        width: 100%;
        font-weight: bold;
    }
    /* Headers */
    h1, h2, h3 {
        color: #00ADB5 !important;
        font-family: 'Segoe UI', sans-serif;
    }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# 2. MODEL ARCHITECTURE (Must match training)
# ==========================================
class FFTBlock(nn.Module):
    def __init__(self, channels):
        super(FFTBlock, self).__init__()
        self.conv_freq = nn.Conv2d(channels * 2, channels * 2, 1) 
    def forward(self, x):
        _, _, H, W = x.shape
        fft = torch.fft.rfft2(x, norm='backward') 
        fft_cat = torch.cat([fft.real, fft.imag], dim=1)
        freq_feats = self.conv_freq(fft_cat)
        real, imag = torch.chunk(freq_feats, 2, dim=1)
        fft_new = torch.complex(real, imag)
        output = torch.fft.irfft2(fft_new, s=(H, W), norm='backward')
        return output + x

class ResBlock(nn.Module):
    def __init__(self, channels):
        super(ResBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1))
    def forward(self, x): return x + self.conv(x)

class DeBlurXNet(nn.Module):
    def __init__(self):
        super(DeBlurXNet, self).__init__()
        self.enc1 = nn.Sequential(nn.Conv2d(3, 32, 3, padding=1), nn.ReLU())
        self.enc2 = nn.Sequential(nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU())
        self.enc3 = nn.Sequential(nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU())
        self.bottleneck_res = nn.Sequential(*[ResBlock(128) for _ in range(4)])
        self.bottleneck_fft = FFTBlock(128)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = nn.Sequential(nn.Conv2d(128, 64, 3, padding=1), nn.ReLU())
        self.up2 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec2 = nn.Sequential(nn.Conv2d(64, 32, 3, padding=1), nn.ReLU())
        self.final = nn.Conv2d(32, 3, 3, padding=1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        e1 = self.enc1(x); e2 = self.enc2(e1); e3 = self.enc3(e2)
        d1 = self.up1(self.bottleneck_fft(self.bottleneck_res(e3)))
        d1 = self.dec1(torch.cat([d1, e2], dim=1))
        d2 = self.up2(d1)
        d2 = self.dec2(torch.cat([d2, e1], dim=1))
        return self.sigmoid(self.final(d2))

# ==========================================
# 3. APP LOGIC
# ==========================================

# Sidebar
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2313/2313449.png", width=80)
    st.title("DeBlurX Controls")
    st.markdown("---")
    uploaded_file = st.file_uploader("📂 Upload Blurred Image", type=["jpg", "png", "jpeg"])
    
    st.markdown("### Model Details")
    st.info("**Architecture:** Hybrid CNN + FFT")
    st.info("**Backbone:** ResNet-UNet")
    st.info("**Trained on:** GoPro Dataset")
    st.markdown("---")
    st.write("Developed by 1MS23IS140 & 1MS23IS149")

# Main Section
st.title("👁️ DeBlurX")
st.subheader("Image Reconstruction using Computer Vision")
st.markdown("Reconstruct fine details and remove motion blur using Frequency-Domain Deep Learning.")

@st.cache_resource
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = DeBlurXNet().to(device)
    try:
        model.load_state_dict(torch.load("deblurx_epoch_50.pth", map_location=device))
        model.eval()
        return model, device
    except:
        return None, None

model, device = load_model()

if uploaded_file is None:
    # Show a placeholder when no image is uploaded
    st.markdown("""
    <div style='background-color: #262730; padding: 20px; border-radius: 10px; text-align: center;'>
        <h3>👋 Welcome!</h3>
        <p>Please upload a blurred image from the sidebar to begin.</p>
    </div>
    """, unsafe_allow_html=True)

elif model is None:
    st.error("Model weights not found! Please ensure 'deblurx_epoch_50.pth' is in the folder.")

else:
    # Process Image
    col1, col2 = st.columns([1, 4])
    
    with col1:
        st.write("Processing...")
        image = Image.open(uploaded_file).convert('RGB')
        
        # Inference
        with st.spinner("🤖 AI is enhancing..."):
            img_np = np.array(image) / 255.0
            img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).float().unsqueeze(0).to(device)
            
            with torch.no_grad():
                output_tensor = model(img_tensor)
            
            output_np = output_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
            output_np = np.clip(output_np, 0, 1)
            
            # Convert back to PIL for display
            output_pil = Image.fromarray((output_np * 255).astype(np.uint8))

    # The Comparison Slider
    with col2:
        st.success("Reconstruction Successful!")
        image_comparison(
            img1=image,
            img2=output_pil,
            label1="Original Blur",
            label2="DeBlurX Output",
            width=700,
            starting_position=50,
            show_labels=True,
            make_responsive=True,
            in_memory=True
        )
    
    # Download Button
    st.markdown("---")
    btn_col1, btn_col2, _ = st.columns([1, 1, 3])
    with btn_col1:
        # Convert output to bytes for download
        buf = io.BytesIO()
        output_pil.save(buf, format="PNG")
        byte_im = buf.getvalue()
        
        st.download_button(
            label="⬇️ Download Result",
            data=byte_im,
            file_name="deblurx_result.png",
            mime="image/png"
        )