"""
AIIJC 2026 — Задача 2: «Уровень освещённости»
Классификация изображений (3 класса: dark/normal/bright) | Метрика: Accuracy
Подход: EfficientNet-B0 + ResNet18 fine-tuning с ансамблем, TTA
CPU-оптимизированный (нет CUDA)
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from PIL import Image
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms, models
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report

# =============================================================================
# КОНФИГУРАЦИЯ
# =============================================================================
# Путь к данным (читаем сохранённый путь)
DATA_DIR = 'data'

TRAIN_DIR = os.path.join(DATA_DIR, 'train')
TEST_DIR = os.path.join(DATA_DIR, 'test')
OUTPUT_DIR = 'd:/AIJC'

# Маппинг классов
CLASS_MAP = {'dark': 0, 'normal': 1, 'bright': 2}
CLASS_NAMES = ['dark', 'normal', 'bright']

# Гиперпараметры
IMG_SIZE = 224
BATCH_SIZE = 16  # CPU-friendly
N_FOLDS = 5
EPOCHS = 25       # Достаточно для fine-tuning
LR = 1e-4
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 0   # Windows + CPU
SEED = 42
TTA_TRANSFORMS = 5  # Количество TTA аугментаций

# Воспроизводимость
torch.manual_seed(SEED)
np.random.seed(SEED)

print("=" * 70)
print("ЗАДАЧА 2: Уровень освещённости")
print("=" * 70)
print(f"Data dir: {DATA_DIR}")
print(f"Device: CPU")
print(f"Image size: {IMG_SIZE}x{IMG_SIZE}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Epochs: {EPOCHS}")
print(f"Folds: {N_FOLDS}")

# =============================================================================
# 1. ЗАГРУЗКА И ПОДГОТОВКА ДАННЫХ
# =============================================================================
print("\n" + "=" * 70)
print("ЗАГРУЗКА ДАННЫХ")
print("=" * 70)

# Собираем train данные
train_images = []
train_labels = []

for class_name, class_id in CLASS_MAP.items():
    class_dir = os.path.join(TRAIN_DIR, class_name)
    if os.path.isdir(class_dir):
        for img_name in sorted(os.listdir(class_dir)):
            if img_name.endswith('.png') or img_name.endswith('.jpg'):
                train_images.append(os.path.join(class_dir, img_name))
                train_labels.append(class_id)

train_labels = np.array(train_labels)
print(f"Train: {len(train_images)} images")
print(f"Distribution: {Counter(train_labels)}")

# Собираем test данные
test_images = []
test_ids = []

for img_name in sorted(os.listdir(TEST_DIR)):
    if img_name.endswith('.png') or img_name.endswith('.jpg'):
        test_images.append(os.path.join(TEST_DIR, img_name))
        # id = имя файла без расширения
        test_ids.append(os.path.splitext(img_name)[0])

print(f"Test: {len(test_images)} images")

# =============================================================================
# 2. DATASET И TRANSFORMS
# =============================================================================

# Нормализация ImageNet
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Train аугментации — агрессивные для маленького датасета
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),
    transforms.RandomCrop(IMG_SIZE),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.3),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
    transforms.RandomGrayscale(p=0.05),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    transforms.RandomErasing(p=0.1, scale=(0.02, 0.1)),
])

# Val/Test — чистый resize + нормализация
val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

# TTA трансформы — разные аугментации для предсказания
tta_transforms = [
    # Оригинал
    val_transform,
    # HFlip
    transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]),
    # Slight rotation
    transforms.Compose([
        transforms.Resize((IMG_SIZE + 16, IMG_SIZE + 16)),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]),
    # Brightness up
    transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ColorJitter(brightness=0.15),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]),
    # VFlip
    transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomVerticalFlip(p=1.0),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]),
]


class ImageDataset(Dataset):
    def __init__(self, image_paths, labels=None, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        
        if self.labels is not None:
            return img, self.labels[idx]
        return img


# =============================================================================
# 3. МОДЕЛИ
# =============================================================================

def create_model(model_name='resnet18', num_classes=3):
    """Создание предобученной модели с новой головой."""
    if model_name == 'resnet18':
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, num_classes)
        )
    elif model_name == 'resnet34':
        model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, num_classes)
        )
    elif model_name == 'efficientnet_b0':
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, num_classes)
        )
    elif model_name == 'mobilenet_v3':
        model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        in_features = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(in_features, num_classes)
    elif model_name == 'convnext_tiny':
        model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        in_features = model.classifier[2].in_features
        model.classifier[2] = nn.Linear(in_features, num_classes)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    return model


def freeze_backbone(model, model_name, unfreeze_last_n=2):
    """Замораживает backbone, оставляя последние N слоёв обучаемыми."""
    if 'resnet' in model_name:
        layers = [model.conv1, model.bn1, model.layer1, model.layer2, model.layer3, model.layer4]
        for layer in layers[:-unfreeze_last_n]:
            for param in layer.parameters():
                param.requires_grad = False
    elif 'efficientnet' in model_name:
        features = list(model.features.children())
        for layer in features[:-unfreeze_last_n]:
            for param in layer.parameters():
                param.requires_grad = False
    elif 'mobilenet' in model_name:
        features = list(model.features.children())
        for layer in features[:-unfreeze_last_n]:
            for param in layer.parameters():
                param.requires_grad = False
    elif 'convnext' in model_name:
        features = list(model.features.children())
        for layer in features[:-unfreeze_last_n]:
            for param in layer.parameters():
                param.requires_grad = False


# =============================================================================
# 4. TRAIN & EVALUATION
# =============================================================================

def train_one_epoch(model, loader, criterion, optimizer, scheduler=None):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (images, labels) in enumerate(loader):
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    if scheduler:
        scheduler.step()
    
    return total_loss / total, correct / total


def evaluate(model, loader):
    model.eval()
    all_preds = []
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for batch in loader:
            if isinstance(batch, (list, tuple)) and len(batch) == 2:
                images, labels = batch
                all_labels.extend(labels.numpy())
            else:
                images = batch
            
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.numpy())
            all_probs.append(probs.numpy())
    
    all_probs = np.vstack(all_probs)
    
    if all_labels:
        acc = accuracy_score(all_labels, all_preds)
        return all_preds, all_probs, acc
    return all_preds, all_probs, None


def predict_with_tta(model, image_paths, tta_transforms_list, batch_size=16):
    """Предсказание с Test-Time Augmentation."""
    model.eval()
    all_probs = []
    
    for tta_idx, transform in enumerate(tta_transforms_list):
        dataset = ImageDataset(image_paths, labels=None, transform=transform)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS)
        
        probs_list = []
        with torch.no_grad():
            for images in loader:
                outputs = model(images)
                probs = torch.softmax(outputs, dim=1)
                probs_list.append(probs.numpy())
        
        probs = np.vstack(probs_list)
        all_probs.append(probs)
    
    # Усреднение TTA
    avg_probs = np.mean(all_probs, axis=0)
    return avg_probs


# =============================================================================
# 5. ОСНОВНОЙ ПАЙПЛАЙН: K-FOLD + MULTI-MODEL ENSEMBLE
# =============================================================================
print("\n" + "=" * 70)
print("ОБУЧЕНИЕ МОДЕЛЕЙ")
print("=" * 70)

# Модели для ансамбля
MODEL_CONFIGS = [
    {'name': 'efficientnet_b0', 'epochs': EPOCHS, 'lr': 1e-4, 'unfreeze': 3},
    {'name': 'resnet18',        'epochs': EPOCHS, 'lr': 1e-4, 'unfreeze': 2},
    {'name': 'resnet34',        'epochs': EPOCHS, 'lr': 5e-5, 'unfreeze': 2},
    {'name': 'mobilenet_v3',    'epochs': EPOCHS, 'lr': 1e-4, 'unfreeze': 3},
]

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

# Хранилище для ансамбля
all_oof_probs = {}   # {model_name: oof_probs array (n_samples, 3)}
all_test_probs = {}  # {model_name: test_probs array (n_test, 3)}

for config in MODEL_CONFIGS:
    model_name = config['name']
    epochs = config['epochs']
    lr = config['lr']
    unfreeze = config['unfreeze']
    
    print(f"\n{'='*50}")
    print(f"MODEL: {model_name}")
    print(f"{'='*50}")
    
    oof_probs = np.zeros((len(train_images), 3))
    test_probs_folds = np.zeros((len(test_images), 3))
    fold_accs = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(train_images, train_labels)):
        print(f"\n  Fold {fold+1}/{N_FOLDS}")
        
        # Datasets
        train_paths_fold = [train_images[i] for i in train_idx]
        train_labels_fold = train_labels[train_idx]
        val_paths_fold = [train_images[i] for i in val_idx]
        val_labels_fold = train_labels[val_idx]
        
        train_dataset = ImageDataset(train_paths_fold, train_labels_fold, train_transform)
        val_dataset = ImageDataset(val_paths_fold, val_labels_fold, val_transform)
        
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
                                  num_workers=NUM_WORKERS, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, 
                                num_workers=NUM_WORKERS)
        
        # Model
        model = create_model(model_name, num_classes=3)
        freeze_backbone(model, model_name, unfreeze_last_n=unfreeze)
        
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                lr=lr, weight_decay=WEIGHT_DECAY)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
        
        # Training
        best_val_acc = 0
        best_model_state = None
        patience = 7
        no_improve = 0
        
        for epoch in range(epochs):
            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, scheduler)
            _, _, val_acc = evaluate(model, val_loader)
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
            
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"    Epoch {epoch+1:2d}/{epochs} | "
                      f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
                      f"Val Acc: {val_acc:.4f} (best: {best_val_acc:.4f})")
            
            if no_improve >= patience:
                print(f"    Early stopping at epoch {epoch+1}")
                break
        
        # Загружаем лучшую модель
        model.load_state_dict(best_model_state)
        
        # OOF предсказания
        _, val_probs, val_acc = evaluate(model, val_loader)
        oof_probs[val_idx] = val_probs
        fold_accs.append(best_val_acc)
        print(f"  Fold {fold+1} best val acc: {best_val_acc:.4f}")
        
        # Test предсказания с TTA
        test_probs_fold = predict_with_tta(model, test_images, tta_transforms, batch_size=BATCH_SIZE)
        test_probs_folds += test_probs_fold / N_FOLDS
        
        # Сохраняем модель fold'а
        model_save_path = os.path.join(OUTPUT_DIR, f'{model_name}_fold{fold}.pt')
        torch.save(best_model_state, model_save_path)
    
    # Итоги модели
    oof_preds = np.argmax(oof_probs, axis=1)
    oof_acc = accuracy_score(train_labels, oof_preds)
    mean_fold_acc = np.mean(fold_accs)
    
    print(f"\n  {model_name} OOF Accuracy: {oof_acc:.4f}")
    print(f"  {model_name} Mean fold Accuracy: {mean_fold_acc:.4f}")
    print(f"  {model_name} Per-fold: {[f'{a:.4f}' for a in fold_accs]}")
    
    all_oof_probs[model_name] = oof_probs
    all_test_probs[model_name] = test_probs_folds

# =============================================================================
# 6. АНСАМБЛИРОВАНИЕ
# =============================================================================
print("\n" + "=" * 70)
print("АНСАМБЛИРОВАНИЕ")
print("=" * 70)

# Метод 1: Простое усреднение
ensemble_oof = np.mean([all_oof_probs[m] for m in all_oof_probs], axis=0)
ensemble_test = np.mean([all_test_probs[m] for m in all_test_probs], axis=0)

ensemble_oof_preds = np.argmax(ensemble_oof, axis=1)
ensemble_acc = accuracy_score(train_labels, ensemble_oof_preds)
print(f"Simple average ensemble OOF Accuracy: {ensemble_acc:.4f}")

# Метод 2: Взвешенное усреднение (перебор весов)
model_names = list(all_oof_probs.keys())
best_weighted_acc = 0
best_weights = None

if len(model_names) == 4:
    for w1 in np.arange(0.1, 0.6, 0.05):
        for w2 in np.arange(0.1, 0.6, 0.05):
            for w3 in np.arange(0.1, 0.6, 0.05):
                w4 = 1 - w1 - w2 - w3
                if w4 < 0.05 or w4 > 0.5:
                    continue
                weights = [w1, w2, w3, w4]
                blend = sum(w * all_oof_probs[m] for w, m in zip(weights, model_names))
                acc = accuracy_score(train_labels, np.argmax(blend, axis=1))
                if acc > best_weighted_acc:
                    best_weighted_acc = acc
                    best_weights = weights

    print(f"Weighted ensemble OOF Accuracy: {best_weighted_acc:.4f}")
    print(f"Weights: {dict(zip(model_names, [f'{w:.2f}' for w in best_weights]))}")
    
    weighted_test = sum(w * all_test_probs[m] for w, m in zip(best_weights, model_names))
else:
    best_weighted_acc = ensemble_acc
    weighted_test = ensemble_test

# Метод 3: Majority voting
voting_oof = np.zeros_like(ensemble_oof)
for m in model_names:
    preds = np.argmax(all_oof_probs[m], axis=1)
    for i, p in enumerate(preds):
        voting_oof[i, p] += 1
voting_preds = np.argmax(voting_oof, axis=1)
voting_acc = accuracy_score(train_labels, voting_preds)
print(f"Majority voting OOF Accuracy: {voting_acc:.4f}")

# Выбор лучшего метода
results = {
    'simple_avg': (ensemble_acc, ensemble_test),
    'weighted': (best_weighted_acc, weighted_test),
}

best_method = max(results, key=lambda k: results[k][0])
best_acc = results[best_method][0]
best_test_probs = results[best_method][1]
best_test_preds = np.argmax(best_test_probs, axis=1)

print(f"\nЛучший метод: {best_method}")
print(f"OOF Accuracy: {best_acc:.4f}")
points = max(0, best_acc - 0.40) / (1 - 0.40)
print(f"Points: {points:.5f}")

# =============================================================================
# 7. ГЕНЕРАЦИЯ SUBMISSIONS
# =============================================================================
print("\n" + "=" * 70)
print("ГЕНЕРАЦИЯ SUBMISSIONS")
print("=" * 70)

# Основной submission
submission = pd.DataFrame({
    'id': test_ids,
    'label': best_test_preds
})
submission.to_csv(os.path.join(OUTPUT_DIR, 'submission_task2.csv'), index=False)
print(f"Main submission saved: submission_task2.csv")
print(f"Shape: {submission.shape}")
print(f"Distribution: {Counter(best_test_preds)}")
print(f"First 10 rows:")
print(submission.head(10))

# Альтернативные submissions
for method_name, (acc, test_p) in results.items():
    preds = np.argmax(test_p, axis=1)
    sub = pd.DataFrame({'id': test_ids, 'label': preds})
    sub.to_csv(os.path.join(OUTPUT_DIR, f'submission_task2_{method_name}.csv'), index=False)
    print(f"  {method_name}: Acc={acc:.4f}, saved")

# Отдельные модели
for m in model_names:
    preds = np.argmax(all_test_probs[m], axis=1)
    oof_acc = accuracy_score(train_labels, np.argmax(all_oof_probs[m], axis=1))
    sub = pd.DataFrame({'id': test_ids, 'label': preds})
    sub.to_csv(os.path.join(OUTPUT_DIR, f'submission_task2_{m}.csv'), index=False)
    print(f"  {m}: OOF Acc={oof_acc:.4f}, saved")

print("\n" + "=" * 70)
print("CLASSIFICATION REPORT (OOF)")
print("=" * 70)
print(classification_report(train_labels, np.argmax(ensemble_oof, axis=1), 
                            target_names=CLASS_NAMES))

print("\n" + "=" * 70)
print(f"ГОТОВО! Лучший submission: submission_task2.csv")
print(f"OOF Accuracy: {best_acc:.4f}, Points: {points:.5f}")
print("=" * 70)
