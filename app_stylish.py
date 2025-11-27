import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import io
import streamlit as st

from streamlit_image_comparison import image_comparison

# ==========================================
# 1. PAGE CONFIGURATION & FIXES
# ==========================================
st.set_page_config(
    page_title="DeBlurX - AI Restoration",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS - FIXED TEXT COLORS
st.markdown("""
<style>

    /* ==========================
       GLOBAL FONT & TRANSITION
       ========================== */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;900&display=swap');
    * {
        font-family: "Poppins", sans-serif;
        transition: 0.35s ease;
    }

    /* ==========================
       STARFIELD BACKGROUND CANVAS
       ========================== */
    #starfield {
        position: fixed;
        top: 0;
        left: 0;
        pointer-events: none;
        z-index: 1;
    }

    /* ==========================
       APP BACKGROUND
       ========================== */
    .stApp {
        position: relative;
        overflow-x: hidden;
        z-index: 2;
        min-height: 100vh;
    }


    /* ==========================
       TITLE
       ========================== */
    .title-text {
        font-size: 3.8rem;
        font-weight: 900;
        text-align: center;
        margin-bottom: 25px;
        letter-spacing: -1px;
        padding-top: 18px;
    }


    /* ==========================
       GLASS CARDS
       ========================== */
    .glass-card {
        padding: 28px;
        border-radius: 18px;
        margin-bottom: 22px;
        backdrop-filter: blur(12px);
        transform: translateY(0);
        transition: transform .3s ease, box-shadow .3s ease;
    }



    .glass-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 22px rgba(0,0,0,0.14);
    }

    /* ==========================
       CUSTOM SCROLLBAR
       ========================== */
    ::-webkit-scrollbar {
        width: 7px;
    }
    ::-webkit-scrollbar-thumb {
        border-radius: 10px;
    }
   

    /* ==========================
       SIDEBAR
       ========================== */
    [data-testid="stSidebar"] {
        backdrop-filter: blur(16px);
        z-index: 4;
    }
  
    [data-testid="stSidebar"] * {
        font-weight: 500;
    }

   

    /* ==========================
       BUTTONS
       ========================== */
    .stButton>button {
        border: none;
        padding: 10px 30px;
        border-radius: 12px;
        font-weight: 600;
        cursor: pointer;
        position: relative;
        overflow: hidden;
    }

 

    /* AI breathing glow */
    .stButton>button {
        animation: breathe 3s ease-in-out infinite;
    }

    @keyframes breathe {
        0% { box-shadow: 0 0 10px rgba(0,150,255,0.22); }
        50% { box-shadow: 0 0 22px rgba(0,220,255,0.35); }
        100% { box-shadow: 0 0 10px rgba(0,150,255,0.22); }
    }

    /* ==========================
       IMAGE SHIMMER EFFECT
       ========================== */
    .ai-image {
        position: relative;
        border-radius: 12px;
        overflow: hidden;
    }

    .ai-image::after {
        content: "";
        position: absolute;
        top: 0;
        left: -120%;
        width: 80%;
        height: 100%;
        background: linear-gradient(120deg, transparent 0%, rgba(255,255,255,0.4) 50%, transparent 100%);
        animation: shine 2.8s linear infinite;
    }

    @keyframes shine {
        0% { left: -80%; }
        100% { left: 140%; }
    }

    /* ==========================
       INFERENCE GLOW
       ========================== */
    #processing-glow {
        width: 260px;
        height: 260px;
        border-radius: 50%;
        background: radial-gradient(circle, rgba(0,200,255,0.35), transparent 70%);
        position: fixed;
        top: 50%;
        left: 50%;
        transform: translate(-50%,-50%) scale(0.85);
        opacity: 0;
        pointer-events: none;
        filter: blur(22px);
        z-index: 3;
        transition: opacity .45s ease;
    }

    /* ==========================
       TEXT COLORS
       ========================== */
.stApp {
   background: linear-gradient(130deg, #0c111b, #121a29, #0a0f1b);
}


</style>
""", unsafe_allow_html=True)



# ==========================================
# 2. MODEL ARCHITECTURE
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
# Sidebar
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3233/3233483.png", width=70) 
    st.markdown("### ⚙️ Control Panel")

    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])


# Main Header
st.markdown('<h1 class="title-text">DeBlurX Studio ✨</h1>', unsafe_allow_html=True)
st.markdown("""
<div class="glass-card">
    <h3 style="margin-top: 0;">Hi there</h3>    <p style="color: #4a4a4a; font-size: 1.1em;">
    Welcome to the DeBlurX inference engine. Upload a blurry image from the sidebar to instantly restore high-frequency details using our Hybrid Attention Network.
    </p>
</div>
""", unsafe_allow_html=True)

# Load Model
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
    # Empty State (Dark text for visibility)
    st.markdown("""
    <div style="text-align: center; margin-top: 50px; color: #4a4a4a;">
        <h1 style="font-size: 5em;">🖼️</h1>
        <h3 style="color: inherit;">No Image Uploaded</h3>
        <p>Please upload a file from the control panel to start processing.</p>
    </div>
    """, unsafe_allow_html=True)

elif model is None:
    st.error("❌ Model weights not found! Check your folder.")

else:
    # Process
    with st.spinner("✨ Enhancing Image..."):
        image = Image.open(uploaded_file).convert('RGB')
        
        # Inference
        img_np = np.array(image) / 255.0
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).float().unsqueeze(0).to(device)
        
        with torch.no_grad():
            output_tensor = model(img_tensor)
        
        output_np = output_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
        output_np = np.clip(output_np, 0, 1)
        output_pil = Image.fromarray((output_np * 255).astype(np.uint8))

    # Result Section
    st.markdown("<h3 style='color: inherit;'>🔍 Result Analysis</h3>", unsafe_allow_html=True)
    
    # Comparison Slider
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    image_comparison(
        img1=image,
        img2=output_pil,
        label1="Original Blur",
        label2="DeBlurX Enhanced",
        width=800,
        starting_position=50,
        show_labels=True,
        make_responsive=True,
        in_memory=True
    )
    st.markdown('</div>', unsafe_allow_html=True)

    # Download Section
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        buf = io.BytesIO()
        output_pil.save(buf, format="PNG")
        byte_im = buf.getvalue()
        
        st.download_button(
            label="⬇️ Download Enhanced Image",
            data=byte_im,
            file_name="deblurx_enhanced.png",
            mime="image/png",
            use_container_width=True
        )