"""
AIIJC 2026 — Задача 3: «За окном шумно» — классификатор погодных условий
3 класса: 0=rain, 1=fog, 2=snow | Метрика: Macro F1

Структура данных (после скачивания):
  D:\AIJC\data\train\0\*.png  (rain)
  D:\AIJC\data\train\1\*.png  (fog)
  D:\AIJC\data\train\2\*.png  (snow)
  D:\AIJC\data\test\*.png
  D:\AIJC\data\train.csv      (id, label)
  D:\AIJC\data\test.csv       (id)
  D:\AIJC\data\sample_submission.csv (id, label)
"""

import os
import sys
import time
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, models
from PIL import Image
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

# =============================================================================
# CONFIG
# =============================================================================
DATA_DIR = Path('data')
TRAIN_DIR = DATA_DIR / 'train'
TEST_DIR = DATA_DIR / 'test'
IMG_SIZE = 224
BATCH_SIZE = 32
N_EPOCHS = 15
LR = 3e-4
WEIGHT_DECAY = 1e-4
N_FOLDS = 5
SEED = 42
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
NUM_CLASSES = 3
CLASS_NAMES = {0: 'rain', 1: 'fog', 2: 'snow'}

print(f"Device: {DEVICE}")
print(f"PyTorch: {torch.__version__}")


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


set_seed(SEED)


# =============================================================================
# DATASET
# =============================================================================
class WeatherDataset(Dataset):
    def __init__(self, image_paths, labels=None, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        if self.labels is not None:
            return image, self.labels[idx]
        return image


# =============================================================================
# TRANSFORMS
# =============================================================================
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.3),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.2, scale=(0.02, 0.15)),
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

test_transform = val_transform


def get_tta_transforms(n_aug=5):
    """Test-Time Augmentation transforms."""
    tta_transforms = [val_transform]
    for i in range(n_aug):
        tta_transforms.append(transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2 * (i + 1) / n_aug,
                                   contrast=0.2 * (i + 1) / n_aug),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]))
    return tta_transforms


# =============================================================================
# MODEL
# =============================================================================
def create_model(model_name='resnet18', num_classes=3, pretrained=True):
    if model_name == 'resnet18':
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
        model.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(model.fc.in_features, num_classes)
        )
    elif model_name == 'resnet34':
        model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT if pretrained else None)
        model.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(model.fc.in_features, num_classes)
        )
    elif model_name == 'efficientnet_b0':
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT if pretrained else None)
        model.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(model.classifier[1].in_features, num_classes)
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")

    return model


# =============================================================================
# TRAINING
# =============================================================================
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    return total_loss / total, 100.0 * correct / total


def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    f1 = f1_score(all_labels, all_preds, average='macro')
    acc = (all_preds == all_labels).mean()

    return total_loss / len(all_labels), f1, acc


def predict(model, loader, device, tta_transforms=None):
    """Predict with optional TTA."""
    model.eval()
    all_probs = []

    if tta_transforms is None:
        tta_transforms = [test_transform]

    with torch.no_grad():
        for images in loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            all_probs.append(probs.cpu().numpy())

    all_probs = np.concatenate(all_probs, axis=0)

    if len(tta_transforms) > 1:
        # Average across TTA augmentations
        all_probs_tta = all_probs.copy()
        for aug_transform in tta_transforms[1:]:
            aug_probs_list = []
            # Need to re-create dataset with different transform
            for images in loader:
                images = images.to(device)
                outputs = model(images)
                probs = torch.softmax(outputs, dim=1)
                aug_probs_list.append(probs.cpu().numpy())
            aug_probs = np.concatenate(aug_probs_list, axis=0)
            all_probs_tta += aug_probs
        all_probs = all_probs_tta / len(tta_transforms)

    return all_probs


# =============================================================================
# MAIN
# =============================================================================
def main():
    print("=" * 70)
    print("LOADING DATA")
    print("=" * 70)

    # Load CSVs
    train_csv = pd.read_csv(DATA_DIR / 'train.csv')
    test_csv = pd.read_csv(DATA_DIR / 'test.csv')
    sample_sub = pd.read_csv(DATA_DIR / 'sample_submission.csv')

    print(f"Train samples: {len(train_csv)}")
    print(f"Test samples: {len(test_csv)}")
    print(f"Class distribution:")
    print(train_csv['label'].value_counts().sort_index())

    # Build image paths
    train_paths = []
    train_labels = []
    for _, row in train_csv.iterrows():
        img_path = TRAIN_DIR / str(row['label']) / f"{row['id']}.png"
        if not img_path.exists():
            # Try .jpg
            img_path = TRAIN_DIR / str(row['label']) / f"{row['id']}.jpg"
        if img_path.exists():
            train_paths.append(str(img_path))
            train_labels.append(row['label'])
        else:
            print(f"  WARNING: {img_path} not found, skipping")

    train_paths = np.array(train_paths)
    train_labels = np.array(train_labels)

    test_paths = []
    for _, row in test_csv.iterrows():
        img_path = TEST_DIR / f"{row['id']}.png"
        if not img_path.exists():
            img_path = TEST_DIR / f"{row['id']}.jpg"
        if img_path.exists():
            test_paths.append(str(img_path))
        else:
            print(f"  WARNING: {img_path} not found")

    test_paths = np.array(test_paths)
    print(f"Loaded: {len(train_paths)} train, {len(test_paths)} test")

    # Class weights for imbalanced data
    class_counts = Counter(train_labels)
    total = len(train_labels)
    class_weights = {c: total / (len(class_counts) * count) for c, count in class_counts.items()}
    sample_weights = [class_weights[label] for label in train_labels]

    # =============================================================================
    # CROSS-VALIDATION
    # =============================================================================
    print("\n" + "=" * 70)
    print("CROSS-VALIDATION TRAINING")
    print("=" * 70)

    models_to_try = ['resnet18', 'resnet34', 'efficientnet_b0']
    all_results = {}

    for model_name in models_to_try:
        print(f"\n--- {model_name} ---")
        set_seed(SEED)

        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
        fold_f1s = []
        fold_preds_all = np.zeros((len(test_paths), NUM_CLASSES))

        for fold, (tr_idx, va_idx) in enumerate(skf.split(train_paths, train_labels)):
            print(f"\n  Fold {fold + 1}/{N_FOLDS}")

            tr_paths, va_paths = train_paths[tr_idx], train_paths[va_idx]
            tr_labels, va_labels = train_labels[tr_idx], train_labels[va_idx]

            # Datasets
            tr_dataset = WeatherDataset(tr_paths, tr_labels, transform=train_transform)
            va_dataset = WeatherDataset(va_paths, va_labels, transform=val_transform)
            te_dataset = WeatherDataset(test_paths, transform=test_transform)

            # Weighted sampler for training
            tr_weights = [class_weights[l] for l in tr_labels]
            sampler = WeightedRandomSampler(tr_weights, len(tr_weights), replacement=True)

            tr_loader = DataLoader(tr_dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=0, pin_memory=True)
            va_loader = DataLoader(va_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
            te_loader = DataLoader(te_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

            # Model
            model = create_model(model_name, NUM_CLASSES, pretrained=True).to(DEVICE)

            # Freeze early layers for first few epochs
            if model_name.startswith('resnet'):
                for param in list(model.parameters())[:-20]:
                    param.requires_grad = False

            criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor([class_weights[c] for c in sorted(class_weights)]).to(DEVICE))
            optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR, weight_decay=WEIGHT_DECAY)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=N_EPOCHS, eta_min=1e-6)

            best_f1 = 0
            best_state = None
            patience = 5
            patience_counter = 0

            for epoch in range(N_EPOCHS):
                # Unfreeze after 3 epochs
                if epoch == 3:
                    for param in model.parameters():
                        param.requires_grad = True
                    optimizer = optim.AdamW(model.parameters(), lr=LR * 0.5, weight_decay=WEIGHT_DECAY)
                    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=N_EPOCHS - 3, eta_min=1e-6)

                train_loss, train_acc = train_epoch(model, tr_loader, criterion, optimizer, DEVICE)
                val_loss, val_f1, val_acc = validate(model, va_loader, criterion, DEVICE)
                scheduler.step()

                print(f"    Epoch {epoch + 1}/{N_EPOCHS} | "
                      f"Train Loss: {train_loss:.4f} Acc: {train_acc:.1f}% | "
                      f"Val Loss: {val_loss:.4f} F1: {val_f1:.4f} Acc: {val_acc:.1f}%")

                if val_f1 > best_f1:
                    best_f1 = val_f1
                    best_state = {k: v.clone() for k, v in model.state_dict().items()}
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"    Early stopping at epoch {epoch + 1}")
                        break

            # Load best model and predict
            model.load_state_dict(best_state)
            test_probs = predict(model, te_loader, DEVICE)
            fold_preds_all += test_probs / N_FOLDS
            fold_f1s.append(best_f1)

            print(f"    Fold {fold + 1} best F1: {best_f1:.4f}")

        avg_f1 = np.mean(fold_f1s)
        std_f1 = np.std(fold_f1s)
        print(f"\n  {model_name}: CV Macro F1 = {avg_f1:.4f} ± {std_f1:.4f}")
        all_results[model_name] = {
            'f1': avg_f1,
            'oof_f1s': fold_f1s,
            'test_probs': fold_preds_all,
        }

    # =============================================================================
    # ENSEMBLE
    # =============================================================================
    print("\n" + "=" * 70)
    print("ENSEMBLE & SUBMISSION")
    print("=" * 70)

    # Method 1: Simple average of all models
    ensemble_probs = np.mean([r['test_probs'] for r in all_results.values()], axis=0)
    ensemble_preds = np.argmax(ensemble_probs, axis=1)

    # Method 2: Weighted average by F1
    weights = {name: r['f1'] for name, r in all_results.items()}
    total_weight = sum(weights.values())
    weighted_probs = np.zeros_like(ensemble_probs)
    for name, r in all_results.items():
        weighted_probs += (weights[name] / total_weight) * r['test_probs']
    weighted_preds = np.argmax(weighted_probs, axis=1)

    # Method 3: Best single model
    best_model_name = max(all_results, key=lambda k: all_results[k]['f1'])
    best_single_probs = all_results[best_model_name]['test_probs']
    best_single_preds = np.argmax(best_single_probs, axis=1)

    # Save submissions
    test_ids = test_csv['id'].values[:len(ensemble_preds)]

    # Main submission (weighted ensemble)
    submission = pd.DataFrame({'id': test_ids, 'label': weighted_preds})
    submission.to_csv('submission_task3.csv', index=False)

    # Alternatives
    pd.DataFrame({'id': test_ids, 'label': ensemble_preds}).to_csv('submission_task3_simple_avg.csv', index=False)
    pd.DataFrame({'id': test_ids, 'label': best_single_preds}).to_csv(f'submission_task3_{best_model_name}.csv', index=False)

    # Print results
    print(f"\nAll model results:")
    for name, r in sorted(all_results.items(), key=lambda x: -x[1]['f1']):
        print(f"  {name:20s}: CV F1 = {r['f1']:.4f}")

    print(f"\nBest single model: {best_model_name} (F1={all_results[best_model_name]['f1']:.4f})")
    print(f"\nSubmission saved: submission_task3.csv")
    print(f"Shape: {submission.shape}")
    print(f"Class distribution in predictions:")
    print(pd.Series(ensemble_preds).value_counts().sort_index())
    print(f"\n{'='*70}")
    print("DONE! Upload submission_task3.csv to the platform.")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
