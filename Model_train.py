import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import random

# ─── 1. DATASET DEFINITION (WITH AUGMENTATION & CROPPING) ───────────────────

class KvasirDataset(Dataset):
    def __init__(self, img_dir, mask_dir, img_size=256, is_train=True):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_size = img_size
        self.is_train = is_train
        self.images = [f for f in os.listdir(img_dir) if f.endswith(('.png', '.jpg'))]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.img_dir, img_name)
        
        base_name = os.path.splitext(img_name)[0]
        mask_path = os.path.join(self.mask_dir, base_name + '.jpg')
        if not os.path.exists(mask_path):
            mask_path = os.path.join(self.mask_dir, base_name + '.png')

        # Load Images
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # --- A. SMART CROP (Remove Black Borders) ---
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        _, thresh = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            c = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(c)
            # Safety check: only crop if the bounding box makes sense
            if w > 50 and h > 50: 
                img = img[y:y+h, x:x+w]
                mask = mask[y:y+h, x:x+w]

        # --- B. RESIZE ---
        img = cv2.resize(img, (self.img_size, self.img_size))
        mask = cv2.resize(mask, (self.img_size, self.img_size))

        # --- C. DATA AUGMENTATION ---
        if self.is_train:
            # Random Horizontal Flip
            if random.random() > 0.5:
                img = cv2.flip(img, 1)
                mask = cv2.flip(mask, 1)
            
            # Random Vertical Flip
            if random.random() > 0.5:
                img = cv2.flip(img, 0)
                mask = cv2.flip(mask, 0)
                
            # Random 90/180/270 degree rotation
            k = random.randint(0, 3)
            if k > 0:
                img = np.rot90(img, k).copy()
                mask = np.rot90(mask, k).copy()

        # --- D. NORMALIZE TO TENSORS ---
        img = img.astype(np.float32) / 255.0 
        img = np.transpose(img, (2, 0, 1))   

        mask = mask.astype(np.float32) / 255.0
        mask = np.expand_dims(mask, axis=0)  

        return torch.tensor(img), torch.tensor(mask)


# ─── 2. U-NET ARCHITECTURE ──────────────────────────────────────────────────

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.pool = nn.MaxPool2d(2, 2)
        
        self.down1 = DoubleConv(3, 32)
        self.down2 = DoubleConv(32, 64)
        self.down3 = DoubleConv(64, 128)
        
        self.bottleneck = DoubleConv(128, 256)
        
        self.up1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv1 = DoubleConv(256, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(128, 64)
        self.up3 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.conv3 = DoubleConv(64, 32)
        
        self.out = nn.Conv2d(32, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(self.pool(d1))
        d3 = self.down3(self.pool(d2))
        
        b = self.bottleneck(self.pool(d3))
        
        u1 = self.up1(b)
        u1 = torch.cat([u1, d3], dim=1) 
        u1 = self.conv1(u1)
        
        u2 = self.up2(u1)
        u2 = torch.cat([u2, d2], dim=1) 
        u2 = self.conv2(u2)
        
        u3 = self.up3(u2)
        u3 = torch.cat([u3, d1], dim=1) 
        u3 = self.conv3(u3)
        
        return self.sigmoid(self.out(u3))


# ─── 3. TRAINING LOOP ───────────────────────────────────────────────────────

def train_model():
    BASE_DIR = os.path.join(os.path.dirname(__file__), '..', 'dataset', 'kvasir-seg')    
    IMG_DIR = os.path.join(BASE_DIR, 'images')
    MASK_DIR = os.path.join(BASE_DIR, 'masks')
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on: {device.type.upper()}")

    # Dataset now uses augmentation
    dataset = KvasirDataset(IMG_DIR, MASK_DIR, img_size=256, is_train=True)
    loader = DataLoader(dataset, batch_size=4, shuffle=True)

    model = UNet().to(device)
    criterion = nn.BCELoss() 
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Increased to 50 Epochs
    epochs = 50 
    best_loss = float('inf')

    print(f"Starting training loop for {epochs} epochs...")
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        
        for batch_idx, (images, masks) in enumerate(loader):
            images, masks = images.to(device), masks.to(device)
            
            optimizer.zero_grad()
            predictions = model(images)
            loss = criterion(predictions, masks)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            if batch_idx % 25 == 0:
                print(f"Epoch [{epoch+1}/{epochs}] Batch [{batch_idx}/{len(loader)}] Loss: {loss.item():.4f}")

        avg_loss = epoch_loss / len(loader)
        print(f"--- Epoch {epoch+1} Complete | Average Loss: {avg_loss:.4f} ---")

        # Save the model if it's the best one we've seen so far
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_model_path = 'unet_kvasir_best.pth'
            torch.save(model.state_dict(), best_model_path)
            print(f"[*] New best model saved! (Loss dropped to {best_loss:.4f})")

    # Save the final epoch just in case, but rely on the 'best' one for inference
    torch.save(model.state_dict(), 'unet_kvasir_final.pth')
    print("\nTraining finished!")
    print("Use 'unet_kvasir_best.pth' in your inference script for the best results.")

if __name__ == '__main__':
    train_model()