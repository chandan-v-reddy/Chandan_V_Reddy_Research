"""
============================================================
Origin Medical - AI Research Intern Role Challenge
Part A: Landmark Detection-Based Approach
Trainer Script
============================================================
Task: Detect 4 landmark points (BPD x2, OFD x2) in fetal
      axial ultrasound images using a CNN-based heatmap
      regression approach.
============================================================
"""

import os
import random
import numpy as np
import pandas as pd
from PIL import Image
import cv2

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# ─────────────────────────────────────────────
# 0. Reproducibility
# ─────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# ─────────────────────────────────────────────
# 1. Configuration
# ─────────────────────────────────────────────
CONFIG = {
    "image_dir":    "dataset/images",
    "csv_path":     "dataset/role_challenge_dataset_ground_truth.csv",
    "img_size":     256,          # Resize all images to 256x256
    "heatmap_size": 64,           # Heatmap output size (256/4)
    "sigma":        5,            # Gaussian sigma for heatmap generation
    "num_landmarks": 4,           # ofd1, ofd2, bpd1, bpd2
    "batch_size":   8,
    "num_epochs":   60,
    "lr":           1e-4,
    "weight_decay": 1e-5,
    "val_split":    0.2,
    "save_dir":     "Model_Weights",
    "device":       "cuda" if torch.cuda.is_available() else "cpu",
}

print(f"[INFO] Using device: {CONFIG['device']}")
os.makedirs(CONFIG["save_dir"], exist_ok=True)


# ─────────────────────────────────────────────
# 2. Heatmap Generation Utility
# ─────────────────────────────────────────────
def generate_heatmap(size, center_x, center_y, sigma=5):
    """Generate a 2D Gaussian heatmap for a single landmark."""
    x = np.arange(0, size, 1, dtype=np.float32)
    y = np.arange(0, size, 1, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)
    heatmap = np.exp(-((xx - center_x) ** 2 + (yy - center_y) ** 2) / (2 * sigma ** 2))
    return heatmap


# ─────────────────────────────────────────────
# 3. Dataset
# ─────────────────────────────────────────────
class FetalLandmarkDataset(Dataset):
    """
    Dataset for fetal ultrasound landmark detection.
    Loads images and generates Gaussian heatmaps for each
    of the 4 landmark points (ofd1, ofd2, bpd1, bpd2).
    """
    def __init__(self, df, image_dir, img_size=256, heatmap_size=64,
                 sigma=5, augment=False):
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.img_size = img_size
        self.heatmap_size = heatmap_size
        self.sigma = sigma
        self.augment = augment
        self.scale = heatmap_size / img_size  # coordinate scaling factor

        self.img_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.df)

    def _load_image(self, path):
        img = Image.open(path).convert("RGB")
        orig_w, orig_h = img.size
        return img, orig_w, orig_h

    def _scale_coords(self, x, y, orig_w, orig_h):
        """Scale original pixel coords → heatmap coords."""
        hx = (x / orig_w) * self.heatmap_size
        hy = (y / orig_h) * self.heatmap_size
        return hx, hy

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.image_dir, row["image_name"])
        img, orig_w, orig_h = self._load_image(img_path)

        # Raw landmark coords
        landmarks = [
            (row["ofd_1_x"], row["ofd_1_y"]),
            (row["ofd_2_x"], row["ofd_2_y"]),
            (row["bpd_1_x"], row["bpd_1_y"]),
            (row["bpd_2_x"], row["bpd_2_y"]),
        ]

        # ── Data Augmentation ──────────────────────────────
        if self.augment:
            # Horizontal flip (50%)
            if random.random() > 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                landmarks = [(orig_w - x, y) for x, y in landmarks]
            # Brightness / contrast jitter
            jitter = transforms.ColorJitter(brightness=0.3, contrast=0.3)
            img = jitter(img)

        # Build heatmaps
        heatmaps = []
        for (x, y) in landmarks:
            hx, hy = self._scale_coords(x, y, orig_w, orig_h)
            hm = generate_heatmap(self.heatmap_size, hx, hy, self.sigma)
            heatmaps.append(hm)
        heatmaps = np.stack(heatmaps, axis=0)  # (4, H, W)

        img_tensor = self.img_transform(img)
        heatmap_tensor = torch.tensor(heatmaps, dtype=torch.float32)

        return img_tensor, heatmap_tensor


# ─────────────────────────────────────────────
# 4. Model Architecture
# ─────────────────────────────────────────────
class LandmarkDetector(nn.Module):
    """
    Hypothesis 1 (Baseline):
    ResNet-18 encoder + lightweight decoder for heatmap regression.

    The encoder extracts rich visual features from ultrasound images.
    The decoder upsamples back to heatmap resolution using transposed
    convolutions, outputting one channel per landmark.
    """
    def __init__(self, num_landmarks=4, pretrained=True):
        super().__init__()
        # ── Encoder: ResNet-18 backbone ────────────────────
        resnet = models.resnet18(
            weights=models.ResNet18_Weights.DEFAULT if pretrained else None
        )
        self.encoder = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1,   # 64 ch
            resnet.layer2,   # 128 ch
            resnet.layer3,   # 256 ch
            resnet.layer4,   # 512 ch  →  8x8 for 256 input
        )
        # ── Decoder: upsample to heatmap_size (64x64) ─────
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),  # 16
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # 32
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64,  kernel_size=4, stride=2, padding=1),  # 64
            nn.BatchNorm2d(64),  nn.ReLU(inplace=True),
            nn.Conv2d(64, num_landmarks, kernel_size=1),                        # 64
        )

    def forward(self, x):
        features = self.encoder(x)
        heatmaps = self.decoder(features)
        return heatmaps


class LandmarkDetectorV2(nn.Module):
    """
    Hypothesis 2:
    ResNet-34 encoder (deeper) + decoder with skip connections.
    More capacity to capture fine-grained ultrasound texture.
    """
    def __init__(self, num_landmarks=4, pretrained=True):
        super().__init__()
        resnet = models.resnet34(
            weights=models.ResNet34_Weights.DEFAULT if pretrained else None
        )
        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1 = resnet.layer1   # 64
        self.layer2 = resnet.layer2   # 128
        self.layer3 = resnet.layer3   # 256
        self.layer4 = resnet.layer4   # 512

        self.up1 = nn.ConvTranspose2d(512, 256, 4, 2, 1)
        self.up2 = nn.ConvTranspose2d(256 + 256, 128, 4, 2, 1)
        self.up3 = nn.ConvTranspose2d(128 + 128, 64,  4, 2, 1)
        self.out = nn.Conv2d(64, num_landmarks, 1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x0 = self.layer0(x)
        x1 = self.layer1(x0)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        d1 = self.relu(self.up1(x4))
        d1 = torch.cat([d1, x3], dim=1)
        d2 = self.relu(self.up2(d1))
        d2 = torch.cat([d2, x2], dim=1)
        d3 = self.relu(self.up3(d2))
        out = self.out(d3)
        return out


class LandmarkDetectorV3(nn.Module):
    """
    Hypothesis 3:
    MobileNetV2 encoder (lightweight, fast) + decoder.
    Efficient model — good for deployment on edge devices.
    """
    def __init__(self, num_landmarks=4, pretrained=True):
        super().__init__()
        mobilenet = models.mobilenet_v2(
            weights=models.MobileNet_V2_Weights.DEFAULT if pretrained else None
        )
        self.encoder = mobilenet.features   # output: 1280 ch
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(1280, 256, 4, 2, 1),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, num_landmarks, 1),
        )

    def forward(self, x):
        f = self.encoder(x)
        return self.decoder(f)


# ─────────────────────────────────────────────
# 5. Loss Function
# ─────────────────────────────────────────────
class HeatmapLoss(nn.Module):
    """Mean Squared Error on heatmap predictions."""
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, pred, target):
        return self.mse(pred, target)


def heatmap_to_coords(heatmap):
    """Convert predicted heatmap → (x, y) pixel coordinates (argmax)."""
    B, C, H, W = heatmap.shape
    flat = heatmap.view(B, C, -1)
    idx = flat.argmax(dim=-1)
    y = (idx // W).float()
    x = (idx  % W).float()
    return x, y  # both (B, C)


def compute_mre(pred_hm, gt_hm, scale_factor):
    """
    Mean Radial Error: average Euclidean distance between
    predicted and ground-truth landmark coordinates (in pixels).
    """
    px, py = heatmap_to_coords(pred_hm)
    gx, gy = heatmap_to_coords(gt_hm)
    dist = torch.sqrt((px - gx) ** 2 + (py - gy) ** 2)
    return (dist / scale_factor).mean().item()


# ─────────────────────────────────────────────
# 6. Training Loop
# ─────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for imgs, hms in tqdm(loader, desc="  Train", leave=False):
        imgs, hms = imgs.to(device), hms.to(device)
        optimizer.zero_grad()
        preds = model(imgs)
        # Resize preds to match heatmap size if needed
        if preds.shape[-1] != hms.shape[-1]:
            preds = nn.functional.interpolate(
                preds, size=hms.shape[-2:], mode="bilinear", align_corners=False
            )
        loss = criterion(preds, hms)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def validate(model, loader, criterion, device, scale_factor):
    model.eval()
    total_loss, total_mre = 0, 0
    with torch.no_grad():
        for imgs, hms in tqdm(loader, desc="  Val  ", leave=False):
            imgs, hms = imgs.to(device), hms.to(device)
            preds = model(imgs)
            if preds.shape[-1] != hms.shape[-1]:
                preds = nn.functional.interpolate(
                    preds, size=hms.shape[-2:], mode="bilinear", align_corners=False
                )
            loss = criterion(preds, hms)
            total_loss += loss.item()
            total_mre  += compute_mre(preds, hms, scale_factor)
    return total_loss / len(loader), total_mre / len(loader)


def train_model(model, train_loader, val_loader, config, model_name="hypothesis_1"):
    device = config["device"]
    model = model.to(device)
    criterion = HeatmapLoss()
    optimizer = optim.Adam(model.parameters(),
                           lr=config["lr"],
                           weight_decay=config["weight_decay"])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["num_epochs"])
    scale = config["heatmap_size"] / config["img_size"]

    best_val_loss = float("inf")
    history = {"train_loss": [], "val_loss": [], "val_mre": []}

    print(f"\n{'='*55}")
    print(f"  Training: {model_name}")
    print(f"{'='*55}")

    for epoch in range(1, config["num_epochs"] + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_mre = validate(model, val_loader, criterion, device, scale)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_mre"].append(val_mre)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = os.path.join(config["save_dir"], f"{model_name}_best_saved_model.pt")
            torch.save(model.state_dict(), save_path)

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch [{epoch:3d}/{config['num_epochs']}] "
                  f"Train Loss: {train_loss:.4f}  "
                  f"Val Loss: {val_loss:.4f}  "
                  f"Val MRE: {val_mre:.2f}px")

    # Save final model
    final_path = os.path.join(config["save_dir"], f"{model_name}_final_saved_model.pt")
    torch.save(model.state_dict(), final_path)
    print(f"\n  Best model saved → {save_path}")
    print(f"  Final model saved → {final_path}")

    return history


# ─────────────────────────────────────────────
# 7. Plot Training Curves
# ─────────────────────────────────────────────
def plot_history(history, model_name):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history["train_loss"], label="Train Loss")
    axes[0].plot(history["val_loss"],   label="Val Loss")
    axes[0].set_title(f"{model_name} — Loss")
    axes[0].set_xlabel("Epoch"); axes[0].legend()

    axes[1].plot(history["val_mre"], color="orange", label="Val MRE (px)")
    axes[1].set_title(f"{model_name} — Mean Radial Error")
    axes[1].set_xlabel("Epoch"); axes[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(CONFIG["save_dir"], f"{model_name}_training_curve.png"))
    plt.close()
    print(f"  Training curve saved.")


# ─────────────────────────────────────────────
# 8. Main
# ─────────────────────────────────────────────
def main():
    # ── Load CSV ──────────────────────────────
    df = pd.read_csv(CONFIG["csv_path"])
    print(f"[INFO] Total samples: {len(df)}")

    # ── Filter to only images that exist ──────
    df = df[df["image_name"].apply(
        lambda n: os.path.exists(os.path.join(CONFIG["image_dir"], n))
    )].reset_index(drop=True)
    print(f"[INFO] Samples with images found: {len(df)}")

    # ── Train / Val Split ─────────────────────
    train_df, val_df = train_test_split(df, test_size=CONFIG["val_split"],
                                        random_state=SEED)
    print(f"[INFO] Train: {len(train_df)} | Val: {len(val_df)}")

    # ── Datasets & Loaders ────────────────────
    train_ds = FetalLandmarkDataset(train_df, CONFIG["image_dir"],
                                    CONFIG["img_size"], CONFIG["heatmap_size"],
                                    CONFIG["sigma"], augment=True)
    val_ds   = FetalLandmarkDataset(val_df,   CONFIG["image_dir"],
                                    CONFIG["img_size"], CONFIG["heatmap_size"],
                                    CONFIG["sigma"], augment=False)

    train_loader = DataLoader(train_ds, batch_size=CONFIG["batch_size"],
                              shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=CONFIG["batch_size"],
                              shuffle=False, num_workers=2, pin_memory=True)

    # ── Train 3 Hypotheses ────────────────────

    # Hypothesis 1: ResNet-18 baseline
    model1 = LandmarkDetector(num_landmarks=CONFIG["num_landmarks"], pretrained=True)
    h1 = train_model(model1, train_loader, val_loader, CONFIG, "hypothesis_1")
    plot_history(h1, "hypothesis_1")

    # Hypothesis 2: ResNet-34 with skip connections
    model2 = LandmarkDetectorV2(num_landmarks=CONFIG["num_landmarks"], pretrained=True)
    h2 = train_model(model2, train_loader, val_loader, CONFIG, "hypothesis_2")
    plot_history(h2, "hypothesis_2")

    # Hypothesis 3: MobileNetV2 lightweight
    model3 = LandmarkDetectorV3(num_landmarks=CONFIG["num_landmarks"], pretrained=True)
    h3 = train_model(model3, train_loader, val_loader, CONFIG, "hypothesis_3")
    plot_history(h3, "hypothesis_3")

    print("\n[DONE] All 3 hypotheses trained successfully!")
    print(f"Model weights saved in: {CONFIG['save_dir']}/")


if __name__ == "__main__":
    main()
