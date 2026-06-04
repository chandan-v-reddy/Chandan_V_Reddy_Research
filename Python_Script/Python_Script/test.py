"""
============================================================
Origin Medical - AI Research Intern Role Challenge
Part A: Landmark Detection — Tester Script
============================================================
Usage:
    python test.py --image_dir dataset/images \
                   --csv_path  dataset/role_challenge_dataset_ground_truth.csv \
                   --weights   Model_Weights/hypothesis_1_best_saved_model.pt \
                   --model     h1

    Model choices: h1 (ResNet-18), h2 (ResNet-34+skip), h3 (MobileNetV2)
============================================================
"""

import os
import argparse
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tqdm import tqdm


# ─────────────────────────────────────────────
# 1. Re-import model definitions
#    (same as train.py — kept here for standalone use)
# ─────────────────────────────────────────────
class LandmarkDetector(nn.Module):
    def __init__(self, num_landmarks=4, pretrained=False):
        super().__init__()
        resnet = models.resnet18(weights=None)
        self.encoder = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4,
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.ConvTranspose2d(128, 64,  4, 2, 1), nn.BatchNorm2d(64),  nn.ReLU(True),
            nn.Conv2d(64, num_landmarks, 1),
        )
    def forward(self, x):
        return self.decoder(self.encoder(x))


class LandmarkDetectorV2(nn.Module):
    def __init__(self, num_landmarks=4, pretrained=False):
        super().__init__()
        resnet = models.resnet34(weights=None)
        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.up1 = nn.ConvTranspose2d(512, 256, 4, 2, 1)
        self.up2 = nn.ConvTranspose2d(512, 128, 4, 2, 1)
        self.up3 = nn.ConvTranspose2d(256, 64,  4, 2, 1)
        self.out = nn.Conv2d(64, num_landmarks, 1)
        self.relu = nn.ReLU(True)
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
        return self.out(d3)


class LandmarkDetectorV3(nn.Module):
    def __init__(self, num_landmarks=4, pretrained=False):
        super().__init__()
        mobilenet = models.mobilenet_v2(weights=None)
        self.encoder = mobilenet.features
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(1280, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.ConvTranspose2d(256,  128, 4, 2, 1), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.ConvTranspose2d(128,  64,  4, 2, 1), nn.BatchNorm2d(64),  nn.ReLU(True),
            nn.Conv2d(64, num_landmarks, 1),
        )
    def forward(self, x):
        return self.decoder(self.encoder(x))


# ─────────────────────────────────────────────
# 2. Utilities
# ─────────────────────────────────────────────
IMG_SIZE = 256
HM_SIZE  = 64
DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"

TRANSFORM = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

LANDMARK_NAMES = ["OFD-1", "OFD-2", "BPD-1", "BPD-2"]
COLORS = ["red", "blue", "green", "orange"]


def heatmap_to_coords(heatmap):
    """heatmap: (C, H, W) tensor → list of (x, y) in heatmap space"""
    coords = []
    for c in range(heatmap.shape[0]):
        hm = heatmap[c].cpu().numpy()
        idx = np.unravel_index(np.argmax(hm), hm.shape)
        coords.append((idx[1], idx[0]))  # (x, y)
    return coords


def scale_coords_to_image(coords_hm, orig_w, orig_h):
    """Scale heatmap-space coords back to original image resolution."""
    return [
        (int(x / HM_SIZE * orig_w), int(y / HM_SIZE * orig_h))
        for (x, y) in coords_hm
    ]


def euclidean_dist(p1, p2):
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


# ─────────────────────────────────────────────
# 3. Single Image Inference
# ─────────────────────────────────────────────
def predict_image(model, img_path, device=DEVICE):
    """Run inference on a single image. Returns predicted coords in orig res."""
    img = Image.open(img_path).convert("RGB")
    orig_w, orig_h = img.size
    inp = TRANSFORM(img).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        pred_hm = model(inp)
        if pred_hm.shape[-1] != HM_SIZE:
            pred_hm = nn.functional.interpolate(
                pred_hm, size=(HM_SIZE, HM_SIZE), mode="bilinear", align_corners=False
            )
    coords_hm = heatmap_to_coords(pred_hm[0])
    coords_img = scale_coords_to_image(coords_hm, orig_w, orig_h)
    return coords_img, img


# ─────────────────────────────────────────────
# 4. Visualisation
# ─────────────────────────────────────────────
def visualize_prediction(img, pred_coords, gt_coords=None,
                          img_name="", save_path=None):
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.imshow(img)
    for i, (px, py) in enumerate(pred_coords):
        ax.plot(px, py, "o", color=COLORS[i], markersize=10,
                label=f"{LANDMARK_NAMES[i]} pred")
    if gt_coords:
        for i, (gx, gy) in enumerate(gt_coords):
            ax.plot(gx, gy, "x", color=COLORS[i], markersize=10,
                    markeredgewidth=2, label=f"{LANDMARK_NAMES[i]} GT")
            ax.plot([pred_coords[i][0], gx], [pred_coords[i][1], gy],
                    "--", color=COLORS[i], alpha=0.5)
    ax.set_title(img_name)
    ax.legend(loc="upper right", fontsize=7)
    ax.axis("off")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=100)
        plt.close()
    else:
        plt.show()


# ─────────────────────────────────────────────
# 5. Full Test Evaluation
# ─────────────────────────────────────────────
def evaluate(model, df, image_dir, device=DEVICE, output_dir="test_outputs"):
    os.makedirs(output_dir, exist_ok=True)
    results = []
    per_landmark_errors = {name: [] for name in LANDMARK_NAMES}

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating"):
        img_path = os.path.join(image_dir, row["image_name"])
        if not os.path.exists(img_path):
            continue

        pred_coords, img = predict_image(model, img_path, device)
        orig_w, orig_h = img.size

        gt_coords = [
            (row["ofd_1_x"], row["ofd_1_y"]),
            (row["ofd_2_x"], row["ofd_2_y"]),
            (row["bpd_1_x"], row["bpd_1_y"]),
            (row["bpd_2_x"], row["bpd_2_y"]),
        ]

        errors = [euclidean_dist(pred_coords[i], gt_coords[i]) for i in range(4)]
        mre = np.mean(errors)

        for i, name in enumerate(LANDMARK_NAMES):
            per_landmark_errors[name].append(errors[i])

        results.append({
            "image": row["image_name"],
            "MRE": mre,
            **{f"{LANDMARK_NAMES[i]}_err": errors[i] for i in range(4)},
        })

        # Save visualisation for first 10 images
        if len(results) <= 10:
            vis_path = os.path.join(output_dir,
                                    row["image_name"].replace(".png", "_pred.png"))
            visualize_prediction(img, pred_coords, gt_coords,
                                 img_name=row["image_name"], save_path=vis_path)

    # ── Summary ───────────────────────────────
    results_df = pd.DataFrame(results)
    print("\n" + "=" * 50)
    print("  EVALUATION RESULTS")
    print("=" * 50)
    print(f"  Overall Mean Radial Error (MRE): {results_df['MRE'].mean():.2f} px")
    print(f"  Std Dev:                         {results_df['MRE'].std():.2f} px")
    print()
    for name in LANDMARK_NAMES:
        errs = per_landmark_errors[name]
        print(f"  {name:8s}  MRE: {np.mean(errs):.2f} ± {np.std(errs):.2f} px")
    print("=" * 50)

    results_df.to_csv(os.path.join(output_dir, "test_results.csv"), index=False)
    print(f"\n  Results CSV saved → {output_dir}/test_results.csv")
    print(f"  Visualisations  saved → {output_dir}/")
    return results_df


# ─────────────────────────────────────────────
# 6. Main
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Fetal Landmark Detection — Tester")
    parser.add_argument("--image_dir", default="dataset/images")
    parser.add_argument("--csv_path",  default="dataset/role_challenge_dataset_ground_truth.csv")
    parser.add_argument("--weights",   default="Model_Weights/hypothesis_1_best_saved_model.pt")
    parser.add_argument("--model",     default="h1", choices=["h1", "h2", "h3"])
    parser.add_argument("--output_dir",default="test_outputs")
    args = parser.parse_args()

    # Load model
    model_map = {"h1": LandmarkDetector, "h2": LandmarkDetectorV2, "h3": LandmarkDetectorV3}
    model = model_map[args.model](num_landmarks=4, pretrained=False).to(DEVICE)
    model.load_state_dict(torch.load(args.weights, map_location=DEVICE))
    print(f"[INFO] Loaded weights: {args.weights}")
    print(f"[INFO] Model: {args.model} | Device: {DEVICE}")

    # Load CSV
    df = pd.read_csv(args.csv_path)
    print(f"[INFO] Test samples: {len(df)}")

    # Evaluate
    evaluate(model, df, args.image_dir, DEVICE, args.output_dir)


if __name__ == "__main__":
    main()
