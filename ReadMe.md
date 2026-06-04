# Origin Medical — AI Research Intern Role Challenge
## ReadMe

---

## Folder Structure

```
Chandan_[Lastname]_Research/
├── Model_Weights/
│   ├── hypothesis_1_best_saved_model.pt     ← ResNet-18 best checkpoint
│   ├── hypothesis_1_final_saved_model.pt    ← ResNet-18 final epoch
│   ├── hypothesis_2_best_saved_model.pt     ← ResNet-34 best checkpoint
│   ├── hypothesis_2_final_saved_model.pt    ← ResNet-34 final epoch
│   ├── hypothesis_3_best_saved_model.pt     ← MobileNetV2 best checkpoint
│   ├── hypothesis_3_final_saved_model.pt    ← MobileNetV2 final epoch
│   └── hypothesis_*_training_curve.png      ← Loss/MRE plots
│
├── Python_Script/
│   ├── train.py                             ← Full training pipeline (Part A)
│   └── test.py                              ← Full testing/evaluation script
│
├── Report/
│   └── Report.pdf                           ← Report following Appendix A format
│
└── ReadMe.md                                ← This file
```

---

## Task Overview

**Goal:** Detect 4 landmark points in fetal axial ultrasound images:
- **BPD points** (A & C): Biparietal Diameter — 2 landmark points
- **OFD points** (B & D): Occipitofrontal Diameter — 2 landmark points

---

## Requirements

```
Python >= 3.8
torch >= 1.12
torchvision >= 0.13
numpy
pandas
opencv-python
Pillow
matplotlib
scikit-learn
tqdm
```

Install all:
```bash
pip install torch torchvision numpy pandas opencv-python Pillow matplotlib scikit-learn tqdm
```

---

## Dataset

Place dataset files as follows:
```
dataset/
├── images/          ← All .png fetal ultrasound images
└── role_challenge_dataset_ground_truth.csv
```

CSV format:
```
image_name, ofd_1_x, ofd_1_y, ofd_2_x, ofd_2_y, bpd_1_x, bpd_1_y, bpd_2_x, bpd_2_y
```

---

## How to Train

```bash
cd Python_Script
python train.py
```

This trains **3 hypotheses** (models) automatically and saves all weights to `Model_Weights/`.

---

## How to Test

```bash
python test.py \
  --image_dir dataset/images \
  --csv_path  dataset/role_challenge_dataset_ground_truth.csv \
  --weights   Model_Weights/hypothesis_1_best_saved_model.pt \
  --model     h1
```

**Model options:**
| Flag | Model |
|------|-------|
| `h1` | ResNet-18 + Decoder (Baseline) |
| `h2` | ResNet-34 + Skip Connections |
| `h3` | MobileNetV2 + Decoder (Lightweight) |

---

## Approach Summary

### Part A — Landmark Detection (Heatmap Regression)
- Input images resized to **256×256**
- Model predicts **4 Gaussian heatmaps** (one per landmark)
- Landmark coordinates extracted via **argmax** on predicted heatmaps
- Loss: **MSE on heatmaps**
- Metric: **Mean Radial Error (MRE)** in pixels

### Hypotheses Tested
| # | Model | Key Idea |
|---|-------|----------|
| H1 | ResNet-18 + Decoder | Lightweight baseline with pretrained ImageNet weights |
| H2 | ResNet-34 + Skip Connections | Deeper encoder + U-Net style skip connections for better spatial detail |
| H3 | MobileNetV2 + Decoder | Efficient depthwise separable convolutions; fast inference |

### Data Augmentation
- Horizontal flip (50% probability)
- Brightness & contrast jitter
- Coordinate labels updated consistently with flips

---

## Training Config

| Parameter | Value |
|-----------|-------|
| Image size | 256 × 256 |
| Heatmap size | 64 × 64 |
| Gaussian sigma | 5 |
| Batch size | 8 |
| Epochs | 60 |
| Optimizer | Adam |
| LR | 1e-4 |
| Scheduler | Cosine Annealing |
| Val split | 20% |

---

## Contact

For questions regarding this submission, please contact via the email provided in the application.
