import os
import sys
import time
import io
import warnings
warnings.filterwarnings('ignore')

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
    raise RuntimeError("Data dir not found")

DATA_DIR  = find_data_dir()
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
TEST_DIR  = os.path.join(DATA_DIR, 'test')
CLASS_MAP = {'dark': 0, 'normal': 1, 'bright': 2}
CLASS_NAMES = ['dark', 'normal', 'bright']
NUM_CLASSES = 3

IMG_SIZE     = 224
BATCH_TRAIN  = 16
BATCH_INFER  = 32
N_FOLDS      = 5
EPOCHS       = 20
LR           = 2e-4
SEED         = 42

torch.manual_seed(SEED)
np.random.seed(SEED)

print("=================================================================", flush=True)
print("CNN TRAINING (NO COLOR JITTER / PRESERVED EXPOSURE)", flush=True)
print("=================================================================", flush=True)

# Collect paths
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

# Transforms: ONLY GEOMETRIC (No ColorJitter!)
train_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.ToTensor(),
])

val_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])

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

def build_resnet18():
    m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    in_feat = m.fc.in_features
    m.fc = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(in_feat, NUM_CLASSES)
    )
    return m

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
def eval_model(model, loader):
    model.eval()
    all_probs, all_preds, all_lbls = [], [], []
    has_labels = False
    for batch in loader:
        if isinstance(batch, (list, tuple)):
            imgs, lbls = batch
            all_lbls.extend(lbls.tolist())
            has_labels = True
        else:
            imgs = batch
        out   = model(imgs)
        probs = torch.softmax(out, dim=1)
        all_probs.append(probs.cpu().numpy())
        all_preds.extend(out.argmax(1).cpu().tolist())
    all_probs = np.vstack(all_probs)
    acc = accuracy_score(all_lbls, all_preds) if has_labels else None
    return all_preds, all_probs, acc

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
oof_probs  = np.zeros((len(train_paths), NUM_CLASSES), dtype=np.float32)
test_probs = np.zeros((len(test_paths),  NUM_CLASSES), dtype=np.float32)
fold_accs  = []

for fold, (tr_idx, val_idx) in enumerate(skf.split(train_paths, train_labels)):
    print(f"\n--- Fold {fold+1}/{N_FOLDS} ---", flush=True)

    tr_ds = ImgDataset([train_paths[i] for i in tr_idx], train_labels[tr_idx], train_tf)
    vl_ds = ImgDataset([train_paths[i] for i in val_idx], train_labels[val_idx], val_tf)

    tr_dl = DataLoader(tr_ds, batch_size=BATCH_TRAIN, shuffle=True, num_workers=0, drop_last=True)
    vl_dl = DataLoader(vl_ds, batch_size=BATCH_INFER, shuffle=False, num_workers=0)

    model = build_resnet18()
    crit  = nn.CrossEntropyLoss()
    opt   = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

    best_acc   = 0.0
    best_state = None

    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc = train_epoch(model, tr_dl, crit, opt)
        _, _, vl_acc    = eval_model(model, vl_dl)

        if vl_acc > best_acc:
            best_acc   = vl_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if epoch % 3 == 0 or epoch == 1:
            print(f"  ep {epoch:2d}/{EPOCHS} | loss={tr_loss:.4f} tr={tr_acc:.4f} vl={vl_acc:.4f} best={best_acc:.4f}", flush=True)

    model.load_state_dict(best_state)
    _, val_p, _ = eval_model(model, vl_dl)
    oof_probs[val_idx] = val_p

    ts_dl = DataLoader(ImgDataset(test_paths, tf=val_tf), batch_size=BATCH_INFER, shuffle=False, num_workers=0)
    _, ts_p, _ = eval_model(model, ts_dl)
    test_probs += ts_p / N_FOLDS

    fold_accs.append(best_acc)
    print(f"  Fold {fold+1} BEST VAL ACC = {best_acc:.4f}", flush=True)

oof_preds_labels = oof_probs.argmax(axis=1)
oof_acc = accuracy_score(train_labels, oof_preds_labels)
pts     = max(0.0, oof_acc - 0.40) / 0.60

print("\n" + "=" * 65, flush=True)
print(f"ResNet18 (NO ColorJitter) OOF Accuracy : {oof_acc:.4f}", flush=True)
print(f"Estimated Points                       : {pts:.4f}", flush=True)
print(f"Per-fold Accuracies                    : {[f'{a:.4f}' for a in fold_accs]}", flush=True)
print("=" * 65, flush=True)

# Save submission
test_preds_labels = test_probs.argmax(axis=1)
sub = pd.DataFrame({'id': test_ids, 'label': test_preds_labels})
out_path = 'd:/AIJC/submission_task2_resnet18_clean.csv'
sub.to_csv(out_path, index=False)
print(f"Saved submission to {out_path}", flush=True)
