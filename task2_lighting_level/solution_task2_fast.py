"""
AIIJC 2026 — Задача 2: «Уровень освещённости» — БЫСТРЫЙ вариант
Одна модель EfficientNet-B0, 3-fold, 15 эпох — для быстрого baseline на CPU
Расчётное время: ~30-60 минут
"""

import os
import warnings
warnings.filterwarnings('ignore')

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

# =============================================================================
# КОНФИГУРАЦИЯ
# =============================================================================
DATA_DIR = 'data'

TRAIN_DIR = os.path.join(DATA_DIR, 'train')
TEST_DIR = os.path.join(DATA_DIR, 'test')
OUTPUT_DIR = 'd:/AIJC'

CLASS_MAP = {'dark': 0, 'normal': 1, 'bright': 2}
CLASS_NAMES = ['dark', 'normal', 'bright']

IMG_SIZE = 224
BATCH_SIZE = 16
N_FOLDS = 3
EPOCHS = 15
LR = 1e-4
SEED = 42
NUM_WORKERS = 0

torch.manual_seed(SEED)
np.random.seed(SEED)

print("=" * 70)
print("ЗАДАЧА 2: БЫСТРЫЙ BASELINE")
print("=" * 70)

# =============================================================================
# ДАННЫЕ
# =============================================================================
train_images = []
train_labels = []

for class_name, class_id in CLASS_MAP.items():
    class_dir = os.path.join(TRAIN_DIR, class_name)
    if os.path.isdir(class_dir):
        for img_name in sorted(os.listdir(class_dir)):
            if img_name.endswith(('.png', '.jpg')):
                train_images.append(os.path.join(class_dir, img_name))
                train_labels.append(class_id)

train_labels = np.array(train_labels)

test_images = []
test_ids = []
for img_name in sorted(os.listdir(TEST_DIR)):
    if img_name.endswith(('.png', '.jpg')):
        test_images.append(os.path.join(TEST_DIR, img_name))
        test_ids.append(os.path.splitext(img_name)[0])

print(f"Train: {len(train_images)}, Test: {len(test_images)}")

# =============================================================================
# TRANSFORMS
# =============================================================================
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),
    transforms.RandomCrop(IMG_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

tta_transforms = [
    val_transform,
    transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ]),
    transforms.Compose([
        transforms.Resize((IMG_SIZE + 16, IMG_SIZE + 16)),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ]),
]


class ImageDataset(Dataset):
    def __init__(self, paths, labels=None, transform=None):
        self.paths = paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        if self.labels is not None:
            return img, self.labels[idx]
        return img


# =============================================================================
# ОБУЧЕНИЕ
# =============================================================================
print("\n" + "=" * 70)
print("ОБУЧЕНИЕ EfficientNet-B0")
print("=" * 70)

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
oof_probs = np.zeros((len(train_images), 3))
test_probs = np.zeros((len(test_images), 3))

for fold, (train_idx, val_idx) in enumerate(skf.split(train_images, train_labels)):
    print(f"\n--- Fold {fold+1}/{N_FOLDS} ---")
    
    train_ds = ImageDataset([train_images[i] for i in train_idx], train_labels[train_idx], train_transform)
    val_ds = ImageDataset([train_images[i] for i in val_idx], train_labels[val_idx], val_transform)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    
    # EfficientNet-B0
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    in_feat = model.classifier[1].in_features
    model.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(in_feat, 3))
    
    # Freeze early layers
    for i, layer in enumerate(model.features):
        if i < 5:
            for p in layer.parameters():
                p.requires_grad = False
    
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
    
    best_acc = 0
    best_state = None
    
    for epoch in range(EPOCHS):
        # Train
        model.train()
        for images, labels in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
        scheduler.step()
        
        # Val
        model.eval()
        all_preds, all_true = [], []
        with torch.no_grad():
            for images, labels in val_loader:
                preds = model(images).argmax(1)
                all_preds.extend(preds.numpy())
                all_true.extend(labels.numpy())
        
        acc = accuracy_score(all_true, all_preds)
        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        
        if (epoch + 1) % 3 == 0:
            print(f"  Epoch {epoch+1:2d}/{EPOCHS} | Val Acc: {acc:.4f} (best: {best_acc:.4f})")
    
    # Best model
    model.load_state_dict(best_state)
    model.eval()
    
    # OOF
    val_probs_list = []
    with torch.no_grad():
        for images, _ in val_loader:
            val_probs_list.append(torch.softmax(model(images), dim=1).numpy())
    oof_probs[val_idx] = np.vstack(val_probs_list)
    
    # Test with TTA
    for tta_t in tta_transforms:
        test_ds = ImageDataset(test_images, transform=tta_t)
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
        probs_list = []
        with torch.no_grad():
            for images in test_loader:
                probs_list.append(torch.softmax(model(images), dim=1).numpy())
        test_probs += np.vstack(probs_list) / (N_FOLDS * len(tta_transforms))
    
    print(f"  Fold {fold+1} best acc: {best_acc:.4f}")

# =============================================================================
# РЕЗУЛЬТАТЫ
# =============================================================================
oof_preds = np.argmax(oof_probs, axis=1)
oof_acc = accuracy_score(train_labels, oof_preds)
points = max(0, oof_acc - 0.40) / 0.60

print(f"\n{'='*70}")
print(f"OOF Accuracy: {oof_acc:.4f}")
print(f"Points: {points:.5f}")
print(f"{'='*70}")
print(classification_report(train_labels, oof_preds, target_names=CLASS_NAMES))

# Submission
test_preds = np.argmax(test_probs, axis=1)
submission = pd.DataFrame({'id': test_ids, 'label': test_preds})
submission.to_csv(os.path.join(OUTPUT_DIR, 'submission_task2_fast.csv'), index=False)
print(f"Submission saved: submission_task2_fast.csv")
print(f"Distribution: {Counter(test_preds)}")
print(submission.head(10))
print("\nГОТОВО!")
