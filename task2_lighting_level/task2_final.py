"""
AIIJC 2026 - Task 2: Lighting Level Classification
3 classes: dark=0, normal=1, bright=2 | Metric: Accuracy
Strategy:
  - EfficientNet-B0 with FULL unfreeze + differential LR
  - Brightness baseline (image mean luminance)
  - 5-fold CV, TTA x3, early stopping
  - CPU optimised
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)

import numpy as np
import pandas as pd
from PIL import Image
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
def find_data_dir():
    for name in os.listdir('d:/AIJC'):
        p = os.path.join('d:/AIJC', name)
        if os.path.isdir(p) and 'data' in name.lower() and name not in ['catboost_info']:
            if os.path.isdir(os.path.join(p, 'train')):
                return p
    raise RuntimeError("Data directory not found!")

DATA_DIR  = find_data_dir()
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
TEST_DIR  = os.path.join(DATA_DIR, 'test')
OUT_DIR   = 'd:/AIJC'

CLASS_MAP   = {'dark': 0, 'normal': 1, 'bright': 2}
CLASS_NAMES = ['dark', 'normal', 'bright']
NUM_CLASSES = 3

IMG_SIZE     = 224
BATCH_TRAIN  = 16
BATCH_INFER  = 32
N_FOLDS      = 5
EPOCHS       = 40
LR_HEAD      = 3e-4   # learning rate for classification head
LR_BACKBONE  = 1e-5   # learning rate for pretrained backbone
MIN_LR       = 1e-7
WEIGHT_DECAY = 1e-4
PATIENCE     = 10
LABEL_SMOOTH = 0.05
SEED         = 42

torch.manual_seed(SEED)
np.random.seed(SEED)

print("=================================================================")
print("TASK 2: LIGHTING LEVEL CLASSIFICATION")
print("=================================================================")
print(f"Data : {DATA_DIR}")
print(f"Device: CPU | IMG: {IMG_SIZE}x{IMG_SIZE}")
print(f"Folds: {N_FOLDS} | Epochs: {EPOCHS} | BS: {BATCH_TRAIN}")
print(f"LR head: {LR_HEAD} | LR backbone: {LR_BACKBONE}")

# ─────────────────────────────────────────────
# DATA
# ─────────────────────────────────────────────
train_paths, train_labels = [], []
for cls_name, cls_id in CLASS_MAP.items():
    cls_dir = os.path.join(TRAIN_DIR, cls_name)
    for fn in sorted(os.listdir(cls_dir)):
        if fn.lower().endswith(('.png', '.jpg', '.jpeg')):
            train_paths.append(os.path.join(cls_dir, fn))
            train_labels.append(cls_id)

train_labels = np.array(train_labels)

test_paths, test_ids = [], []
for fn in sorted(os.listdir(TEST_DIR)):
    if fn.lower().endswith(('.png', '.jpg', '.jpeg')):
        test_paths.append(os.path.join(TEST_DIR, fn))
        test_ids.append(os.path.splitext(fn)[0])

print(f"\nTrain: {len(train_paths)} | Dist: {Counter(train_labels)}")
print(f"Test : {len(test_paths)}")

# ─────────────────────────────────────────────
# BRIGHTNESS BASELINE
# ─────────────────────────────────────────────
print("\n--- Brightness Baseline ---")

def get_brightness(path):
    """Mean luminance in HSV space — perfect for lighting classification."""
    img = Image.open(path).convert('RGB')
    arr = np.array(img, dtype=np.float32) / 255.0
    # Luminance = 0.299*R + 0.587*G + 0.114*B
    lum = 0.299 * arr[:,:,0] + 0.587 * arr[:,:,1] + 0.114 * arr[:,:,2]
    return float(lum.mean()), float(lum.std()), float(np.percentile(lum, 10)), float(np.percentile(lum, 90))

print("Computing brightness features for train...")
train_brightness = np.array([get_brightness(p) for p in train_paths])  # (N, 4)
print("Computing brightness features for test...")
test_brightness  = np.array([get_brightness(p) for p in test_paths])

# Show class-wise brightness
for cls_id, cls_name in enumerate(CLASS_NAMES):
    mask = train_labels == cls_id
    mean_b = train_brightness[mask, 0].mean()
    print(f"  {cls_name:6s} (id={cls_id}): mean_luminance={mean_b:.4f}")

# Simple threshold classifier based on brightness
# Sort thresholds from training data
bright_means = train_brightness[:, 0]
class_means = [bright_means[train_labels == i].mean() for i in range(NUM_CLASSES)]
print(f"\nClass mean luminances: dark={class_means[0]:.4f}, normal={class_means[1]:.4f}, bright={class_means[2]:.4f}")

# Threshold-based prediction
sorted_classes = sorted(range(NUM_CLASSES), key=lambda i: class_means[i])
print(f"Brightness order: {[CLASS_NAMES[i] for i in sorted_classes]}")

# ─────────────────────────────────────────────
# TRANSFORMS
# ─────────────────────────────────────────────
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

train_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE + 24, IMG_SIZE + 24)),
    transforms.RandomCrop(IMG_SIZE),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.2),
    transforms.RandomRotation(degrees=10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

val_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

tta_tfs = [
    transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ]),
    transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ]),
    transforms.Compose([
        transforms.Resize((IMG_SIZE + 24, IMG_SIZE + 24)),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ]),
]


# ─────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────
class ImgDataset(Dataset):
    def __init__(self, paths, labels=None, tf=None):
        self.paths  = paths
        self.labels = labels
        self.tf     = tf

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert('RGB')
        if self.tf:
            img = self.tf(img)
        if self.labels is not None:
            return img, int(self.labels[idx])
        return img


# ─────────────────────────────────────────────
# MODEL — FULL UNFREEZE + DIFFERENTIAL LR
# ─────────────────────────────────────────────
def build_model():
    """EfficientNet-B0, ALL layers trainable. Returns model + param groups."""
    m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)

    # ALL weights trainable
    for p in m.parameters():
        p.requires_grad = True

    # Replace classifier head
    in_feat = m.classifier[1].in_features
    m.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(in_feat, NUM_CLASSES),
    )

    # Differential LR: backbone gets lower LR
    backbone_params = list(m.features.parameters())
    head_params     = list(m.classifier.parameters()) + list(m.avgpool.parameters())

    param_groups = [
        {'params': backbone_params, 'lr': LR_BACKBONE},
        {'params': head_params,     'lr': LR_HEAD},
    ]
    return m, param_groups


# ─────────────────────────────────────────────
# TRAIN / EVAL
# ─────────────────────────────────────────────
def train_epoch(model, loader, crit, opt):
    model.train()
    loss_sum, correct, total = 0.0, 0, 0
    for imgs, lbls in loader:
        opt.zero_grad()
        out  = model(imgs)
        loss = crit(out, lbls)
        loss.backward()
        opt.step()
        loss_sum += loss.item() * imgs.size(0)
        correct  += out.argmax(1).eq(lbls).sum().item()
        total    += imgs.size(0)
    return loss_sum / total, correct / total


@torch.no_grad()
def eval_probs(model, loader):
    model.eval()
    probs_list, preds_list, lbls_list = [], [], []
    has_labels = False
    for batch in loader:
        if isinstance(batch, (list, tuple)):
            imgs, lbls = batch
            lbls_list.extend(lbls.tolist())
            has_labels = True
        else:
            imgs = batch
        out   = model(imgs)
        probs = torch.softmax(out, dim=1)
        probs_list.append(probs.cpu().numpy())
        preds_list.extend(out.argmax(1).cpu().tolist())
    all_probs = np.vstack(probs_list)
    acc = accuracy_score(lbls_list, preds_list) if has_labels else None
    return preds_list, all_probs, acc


@torch.no_grad()
def tta_predict(model, paths, bs=32):
    model.eval()
    collected = []
    for tf in tta_tfs:
        ds = ImgDataset(paths, tf=tf)
        dl = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=0)
        fold_probs = []
        for imgs in dl:
            fold_probs.append(torch.softmax(model(imgs), dim=1).cpu().numpy())
        collected.append(np.vstack(fold_probs))
    return np.mean(collected, axis=0)


# ─────────────────────────────────────────────
# 5-FOLD TRAINING
# ─────────────────────────────────────────────
print("\n=================================================================")
print("5-FOLD STRATIFIED CV TRAINING  (Full fine-tuning)")
print("=================================================================")

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

oof_probs_cnn  = np.zeros((len(train_paths), NUM_CLASSES), dtype=np.float32)
test_probs_cnn = np.zeros((len(test_paths),  NUM_CLASSES), dtype=np.float32)
fold_accs = []

for fold, (tr_idx, val_idx) in enumerate(skf.split(train_paths, train_labels)):
    print(f"\n[Fold {fold+1}/{N_FOLDS}] ----------------------------------------")

    tr_paths  = [train_paths[i] for i in tr_idx]
    tr_labels = train_labels[tr_idx]
    vl_paths  = [train_paths[i] for i in val_idx]
    vl_labels = train_labels[val_idx]

    tr_dl = DataLoader(ImgDataset(tr_paths, tr_labels, train_tf),
                       batch_size=BATCH_TRAIN, shuffle=True, num_workers=0, drop_last=True)
    vl_dl = DataLoader(ImgDataset(vl_paths, vl_labels, val_tf),
                       batch_size=BATCH_INFER, shuffle=False, num_workers=0)

    model, param_groups = build_model()
    crit  = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTH)
    opt   = optim.AdamW(param_groups, weight_decay=WEIGHT_DECAY)
    sched = optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=EPOCHS, eta_min=MIN_LR
    )

    best_acc   = 0.0
    best_state = None
    no_imp     = 0

    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc = train_epoch(model, tr_dl, crit, opt)
        _, _, vl_acc    = eval_probs(model, vl_dl)
        sched.step()

        if vl_acc > best_acc:
            best_acc   = vl_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_imp     = 0
        else:
            no_imp += 1

        if epoch % 5 == 0 or epoch == 1:
            print(f"  ep {epoch:2d}/{EPOCHS} | loss={tr_loss:.4f} "
                  f"tr={tr_acc:.4f} vl={vl_acc:.4f} best={best_acc:.4f}")

        if no_imp >= PATIENCE:
            print(f"  Early stop ep={epoch}")
            break

    model.load_state_dict(best_state)

    _, oof_fold, _ = eval_probs(model, vl_dl)
    oof_probs_cnn[val_idx] = oof_fold

    test_probs_cnn += tta_predict(model, test_paths, bs=BATCH_INFER) / N_FOLDS

    fold_accs.append(best_acc)
    print(f"  Fold {fold+1} DONE | best val acc = {best_acc:.4f}")


# ─────────────────────────────────────────────
# BRIGHTNESS-BASED ENSEMBLE
# ─────────────────────────────────────────────
print("\n=================================================================")
print("BLENDING: CNN + BRIGHTNESS")
print("=================================================================")

# Convert brightness to soft probs via nearest-centroid-like approach
# Use training brightness stats per class
class_bright_mean = np.array([train_brightness[train_labels == i, 0].mean()
                               for i in range(NUM_CLASSES)])
class_bright_std  = np.array([train_brightness[train_labels == i, 0].std()
                               for i in range(NUM_CLASSES)]) + 1e-6

def brightness_probs(brights):
    """Convert mean luminance to class probabilities via Gaussian likelihood."""
    N = len(brights)
    probs = np.zeros((N, NUM_CLASSES), dtype=np.float32)
    for i in range(NUM_CLASSES):
        diff = (brights[:, 0] - class_bright_mean[i]) / class_bright_std[i]
        probs[:, i] = np.exp(-0.5 * diff**2)
    probs /= probs.sum(axis=1, keepdims=True) + 1e-9
    return probs

train_bright_probs = brightness_probs(train_brightness)
test_bright_probs  = brightness_probs(test_brightness)

# OOF CNN score
oof_cnn_acc = accuracy_score(train_labels, oof_probs_cnn.argmax(1))
# OOF brightness score
oof_bright_acc = accuracy_score(train_labels, train_bright_probs.argmax(1))
print(f"OOF CNN accuracy         : {oof_cnn_acc:.4f}")
print(f"OOF Brightness accuracy  : {oof_bright_acc:.4f}")

# Find best blend weight
best_blend_acc = 0.0
best_alpha = 0.0
for alpha in np.arange(0.0, 1.01, 0.05):
    blend = alpha * oof_probs_cnn + (1 - alpha) * train_bright_probs
    acc   = accuracy_score(train_labels, blend.argmax(1))
    if acc > best_blend_acc:
        best_blend_acc = acc
        best_alpha     = alpha

print(f"Best blend alpha (CNN)   : {best_alpha:.2f}  -> OOF acc = {best_blend_acc:.4f}")

oof_final  = best_alpha * oof_probs_cnn  + (1 - best_alpha) * train_bright_probs
test_final = best_alpha * test_probs_cnn + (1 - best_alpha) * test_bright_probs

oof_preds  = oof_final.argmax(1)
oof_acc    = accuracy_score(train_labels, oof_preds)
points     = max(0.0, oof_acc - 0.40) / 0.60

print(f"\nFinal OOF Accuracy : {oof_acc:.4f}")
print(f"Estimated Points   : {points:.4f}")
print(f"Per-fold accs      : {[f'{a:.4f}' for a in fold_accs]}")
print(f"Mean fold acc      : {np.mean(fold_accs):.4f}")
print()
print(classification_report(train_labels, oof_preds, target_names=CLASS_NAMES))

# ─────────────────────────────────────────────
# SUBMISSION
# ─────────────────────────────────────────────
test_preds = test_final.argmax(1)
sub = pd.DataFrame({'id': test_ids, 'label': test_preds})
out_path = os.path.join(OUT_DIR, 'submission_task2.csv')
sub.to_csv(out_path, index=False)

print("=================================================================")
print(f"Submission -> {out_path}")
print(f"Shape : {sub.shape}  | Dist: {Counter(test_preds.tolist())}")
print(sub.head(10).to_string(index=False))
print("=================================================================")
print("DONE!")
