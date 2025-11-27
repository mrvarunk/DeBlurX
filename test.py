import torch
import torch.nn as nn
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import random

# ==========================================
# 1. CONFIGURATION
# ==========================================
MODEL_PATH = "deblurx_epoch_50.pth"  # The file you just saved
TEST_DIR = "./test"                  # Your test dataset folder
RESULT_DIR = "./results"             # Where to save outputs
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==========================================
# 2. MODEL ARCHITECTURE (Must match train.py)
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
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1)
        )
    def forward(self, x):
        return x + self.conv(x)

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
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        res_out = self.bottleneck_res(e3)
        fft_out = self.bottleneck_fft(res_out)
        d1 = self.up1(fft_out)
        d1 = torch.cat([d1, e2], dim=1)
        d1 = self.dec1(d1)
        d2 = self.up2(d1)
        d2 = torch.cat([d2, e1], dim=1)
        d2 = self.dec2(d2)
        out = self.final(d2)
        return self.sigmoid(out)

# ==========================================
# 3. INFERENCE FUNCTION
# ==========================================
def deblur_random_image():
    # 1. Setup
    os.makedirs(RESULT_DIR, exist_ok=True)
    model = DeBlurXNet().to(DEVICE)
    
    # 2. Load Weights
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file {MODEL_PATH} not found!")
        return
    
    print(f"Loading model from {MODEL_PATH}...")
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    
    # 3. Find a random image in test folder
    test_scenes = os.listdir(TEST_DIR)
    if not test_scenes:
        print("Test folder is empty!")
        return

    # Pick random scene and random image
    random_scene = random.choice(test_scenes)
    blur_folder = os.path.join(TEST_DIR, random_scene, 'blur')
    sharp_folder = os.path.join(TEST_DIR, random_scene, 'sharp')
    
    if not os.path.exists(blur_folder):
        print(f"Blur folder not found in {random_scene}")
        return
        
    img_name = random.choice(os.listdir(blur_folder))
    blur_path = os.path.join(blur_folder, img_name)
    sharp_path = os.path.join(sharp_folder, img_name)
    
    print(f"Testing on: {blur_path}")

    # 4. Preprocess
    img_bgr = cv2.imread(blur_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_tensor = torch.from_numpy(img_rgb / 255.0).permute(2, 0, 1).float().unsqueeze(0).to(DEVICE)
    
    # 5. Inference
    with torch.no_grad():
        output_tensor = model(img_tensor)
    
    # 6. Postprocess
    output_rgb = output_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
    
    # Load GT for comparison
    gt_img = cv2.cvtColor(cv2.imread(sharp_path), cv2.COLOR_BGR2RGB) / 255.0

    # 7. Plot and Save
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(img_rgb / 255.0)
    plt.title("Blurred Input")
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(output_rgb)
    plt.title("DeBlurX Output")
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(gt_img)
    plt.title("Ground Truth (Sharp)")
    plt.axis('off')
    
    save_path = os.path.join(RESULT_DIR, "final_result.png")
    plt.savefig(save_path)
    print(f"Result saved to: {save_path}")
    plt.close()

if __name__ == "__main__":
    deblur_random_image()