import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. CONFIGURATION
# ==========================================
class Config:
    TRAIN_DIR = "./train"   # Fixed path
    VAL_DIR = "./test"      # Fixed path
    BATCH_SIZE = 8          # Keep this. If OOM error, change to 4.
    LEARNING_RATE = 1e-4
    NUM_EPOCHS = 50
    IMAGE_SIZE = 256             
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==========================================
# 2. DATASET LOADER
# ==========================================
class GoProDataset(Dataset):
    def __init__(self, root_dir, is_train=True):
        self.root_dir = root_dir
        self.is_train = is_train
        self.blur_paths = []
        self.sharp_paths = []
        
        if not os.path.exists(root_dir):
            print(f"WARNING: Directory not found: {root_dir}")
            return

        for scene in os.listdir(root_dir):
            scene_path = os.path.join(root_dir, scene)
            if os.path.isdir(scene_path):
                blur_dir = os.path.join(scene_path, 'blur')
                sharp_dir = os.path.join(scene_path, 'sharp')
                
                # Check for standard naming or gamma variants
                if os.path.exists(blur_dir) and os.path.exists(sharp_dir):
                    imgs = sorted(os.listdir(blur_dir))
                    for img_name in imgs:
                        self.blur_paths.append(os.path.join(blur_dir, img_name))
                        self.sharp_paths.append(os.path.join(sharp_dir, img_name))

    def __len__(self):
        return len(self.blur_paths)

    def __getitem__(self, idx):
        # Load images
        try:
            blur_img = cv2.imread(self.blur_paths[idx])
            sharp_img = cv2.imread(self.sharp_paths[idx])
            
            blur_img = cv2.cvtColor(blur_img, cv2.COLOR_BGR2RGB)
            sharp_img = cv2.cvtColor(sharp_img, cv2.COLOR_BGR2RGB)
        except:
            # Fallback for bad image paths
            return torch.zeros(3, 256, 256), torch.zeros(3, 256, 256)

        blur_img = blur_img / 255.0
        sharp_img = sharp_img / 255.0

        if self.is_train:
            h, w, _ = blur_img.shape
            crop_size = Config.IMAGE_SIZE
            if h > crop_size and w > crop_size:
                start_h = np.random.randint(0, h - crop_size)
                start_w = np.random.randint(0, w - crop_size)
                blur_img = blur_img[start_h:start_h+crop_size, start_w:start_w+crop_size, :]
                sharp_img = sharp_img[start_h:start_h+crop_size, start_w:start_w+crop_size, :]
            
            if np.random.rand() > 0.5:
                blur_img = np.flip(blur_img, axis=1).copy()
                sharp_img = np.flip(sharp_img, axis=1).copy()

        blur_tensor = torch.from_numpy(blur_img).permute(2, 0, 1).float()
        sharp_tensor = torch.from_numpy(sharp_img).permute(2, 0, 1).float()

        return blur_tensor, sharp_tensor

# ==========================================
# 3. MODEL: DeBlurX (CNN + Attention + FFT)
# ==========================================
class FFTBlock(nn.Module):
    def __init__(self, channels):
        super(FFTBlock, self).__init__()
        self.conv_freq = nn.Conv2d(channels * 2, channels * 2, 1) 

    def forward(self, x):
        _, _, H, W = x.shape
        # FFT in float32
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
# 4. TRAINING UTILS
# ==========================================
def save_sample_images(model, loader, epoch, device):
    model.eval()
    with torch.no_grad():
        try:
            blur, sharp = next(iter(loader))
            blur = blur.to(device)
            output = model(blur)
            
            b = blur[0].cpu().permute(1, 2, 0).numpy()
            s = sharp[0].cpu().permute(1, 2, 0).numpy()
            o = output[0].cpu().permute(1, 2, 0).numpy()
            
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            axes[0].imshow(b); axes[0].set_title("Blurred Input")
            axes[1].imshow(o); axes[1].set_title(f"DeBlurX Output (Epoch {epoch})")
            axes[2].imshow(s); axes[2].set_title("Ground Truth")
            plt.savefig(f"epoch_{epoch}_result.png")
            plt.close()
        except Exception as e:
            print(f"Could not save image: {e}")
    model.train()

# ==========================================
# 5. MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    print(f"Starting DeBlurX on device: {Config.DEVICE} (Full Precision)")
    
    if not os.path.exists(Config.TRAIN_DIR):
        print(f"ERROR: Dataset path not found: {Config.TRAIN_DIR}")
        exit()

    train_dataset = GoProDataset(Config.TRAIN_DIR, is_train=True)
    if len(train_dataset) == 0:
        print("ERROR: No images found.")
        exit()
        
    print(f"Found {len(train_dataset)} image pairs for training.")
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=2)

    model = DeBlurXNet().to(Config.DEVICE)
    criterion = nn.L1Loss() 
    optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)

    print("Training started...")
    for epoch in range(Config.NUM_EPOCHS):
        total_loss = 0
        for i, (blur, sharp) in enumerate(train_loader):
            blur, sharp = blur.to(Config.DEVICE), sharp.to(Config.DEVICE)
            
            optimizer.zero_grad()
            
            # --- Standard Full Precision Training (No autocast) ---
            outputs = model(blur)
            loss = criterion(outputs, sharp)
            
            # Frequency Loss
            fft_out = torch.fft.rfft2(outputs, norm='backward')
            fft_target = torch.fft.rfft2(sharp, norm='backward')
            loss += 0.1 * criterion(torch.abs(fft_out), torch.abs(fft_target))

            loss.backward()
            
            # Gradient Clipping to prevent explosion
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            
            if i % 10 == 0:
                print(f"Epoch [{epoch+1}/{Config.NUM_EPOCHS}], Step [{i}/{len(train_loader)}], Loss: {loss.item():.4f}")

        print(f"Epoch {epoch+1} Average Loss: {total_loss/len(train_loader):.4f}")
        
        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), f"deblurx_epoch_{epoch+1}.pth")
            save_sample_images(model, train_loader, epoch+1, Config.DEVICE)

    print("Training Complete. Model Saved.")