import os
import sys
import time
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from PIL import Image
from collections import Counter
import torch
from torchvision import transforms, models

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif

from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier

# ─────────────────────────────────────────────
# 1. LOAD TABULAR ILLUMINATION FEATURES
# ─────────────────────────────────────────────
tab_data = np.load('d:/AIJC/features_tabular.npz')
X_tr_tab  = tab_data['X_train']
y_train   = tab_data['y_train']
X_te_tab  = tab_data['X_test']
test_ids  = tab_data['test_ids']

print("=================================================================", flush=True)
print("HYBRID CNN + TABULAR FEATURE FUSION EXPERIMENT", flush=True)
print("=================================================================", flush=True)
print(f"Tabular features shape: Train {X_tr_tab.shape}, Test {X_te_tab.shape}", flush=True)

# ─────────────────────────────────────────────
# 2. EXTRACT DEEP CNN EMBEDDINGS (EfficientNet-B0)
# ─────────────────────────────────────────────
def find_data_dir():
    for name in os.listdir('d:/AIJC'):
        p = os.path.join('d:/AIJC', name)
        if os.path.isdir(p) and 'data' in name.lower() and name not in ['catboost_info']:
            if os.path.isdir(os.path.join(p, 'train')):
                return p

DATA_DIR = find_data_dir()
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
TEST_DIR  = os.path.join(DATA_DIR, 'test')
CLASS_MAP = {'dark': 0, 'normal': 1, 'bright': 2}

train_paths, test_paths = [], []
for cls_name in ['dark', 'normal', 'bright']:
    cls_dir = os.path.join(TRAIN_DIR, cls_name)
    for fn in sorted(os.listdir(cls_dir)):
        if fn.lower().endswith(('.png', '.jpg', '.jpeg')):
            train_paths.append(os.path.join(cls_dir, fn))

for fn in sorted(os.listdir(TEST_DIR)):
    if fn.lower().endswith(('.png', '.jpg', '.jpeg')):
        test_paths.append(os.path.join(TEST_DIR, fn))

tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

print("Extracting EfficientNet-B0 deep features...", flush=True)
cnn_model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
cnn_model.classifier = torch.nn.Identity()
cnn_model.eval()

def extract_cnn_feats(paths, bs=32):
    feats = []
    for i in range(0, len(paths), bs):
        batch = [tf(Image.open(p).convert('RGB')) for p in paths[i:i+bs]]
        batch = torch.stack(batch)
        with torch.no_grad():
            out = cnn_model(batch)
        feats.append(out.numpy())
    return np.vstack(feats)

X_tr_cnn = extract_cnn_feats(train_paths)
X_te_cnn = extract_cnn_feats(test_paths)

print(f"CNN embeddings shape: Train {X_tr_cnn.shape}, Test {X_te_cnn.shape}", flush=True)

# ─────────────────────────────────────────────
# 3. FUSE FEATURES (CNN + TABULAR)
# ─────────────────────────────────────────────
scaler_tab = RobustScaler()
X_tr_tab_s = scaler_tab.fit_transform(X_tr_tab)
X_te_tab_s = scaler_tab.transform(X_te_tab)

scaler_cnn = StandardScaler()
X_tr_cnn_s = scaler_cnn.fit_transform(X_tr_cnn)
X_te_cnn_s = scaler_cnn.transform(X_te_cnn)

# Concatenate normalized CNN features and Tabular illumination features
X_train_fused = np.hstack([X_tr_cnn_s, X_tr_tab_s])
X_test_fused  = np.hstack([X_te_cnn_s, X_te_tab_s])

print(f"Fused feature matrix shape: Train {X_train_fused.shape}, Test {X_test_fused.shape}", flush=True)

# ─────────────────────────────────────────────
# 4. CROSS-VALIDATION OF HYBRID MODELS
# ─────────────────────────────────────────────
print("\n" + "=" * 65, flush=True)
print("EVALUATING HYBRID MODELS (5-FOLD CV)", flush=True)
print("=" * 65, flush=True)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

models = {
    'CatBoost': CatBoostClassifier(iterations=400, learning_rate=0.03, depth=5, verbose=0, random_state=42),
    'LightGBM': LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=15, n_jobs=4, random_state=42, verbose=-1),
    'ExtraTrees': ExtraTreesClassifier(n_estimators=300, max_depth=12, n_jobs=-1, random_state=42),
    'Ridge': RidgeClassifier(alpha=100.0, random_state=42),
    'LogReg': LogisticRegression(C=0.05, max_iter=500, random_state=42),
    'MLP': MLPClassifier(hidden_layer_sizes=(256, 64), alpha=0.01, max_iter=300, random_state=42)
}

oof_preds = {}
test_preds = {}

for name, model in models.items():
    oof = np.zeros((len(y_train), 3), dtype=np.float32)
    tst = np.zeros((len(X_test_fused), 3), dtype=np.float32)
    
    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train_fused, y_train)):
        model.fit(X_train_fused[tr_idx], y_train[tr_idx])
        if hasattr(model, 'predict_proba'):
            oof[val_idx] = model.predict_proba(X_train_fused[val_idx])
            tst += model.predict_proba(X_test_fused) / 5.0
        else:
            # Ridge decision function -> softmax
            dec_val = model.decision_function(X_train_fused[val_idx])
            dec_tst = model.decision_function(X_test_fused)
            exp_v = np.exp(dec_val - np.max(dec_val, axis=1, keepdims=True))
            exp_t = np.exp(dec_tst - np.max(dec_tst, axis=1, keepdims=True))
            oof[val_idx] = exp_v / exp_v.sum(axis=1, keepdims=True)
            tst += (exp_t / exp_t.sum(axis=1, keepdims=True)) / 5.0
            
    acc = accuracy_score(y_train, oof.argmax(axis=1))
    pts = max(0.0, acc - 0.40) / 0.60
    print(f"{name:15s} | OOF Accuracy: {acc:.4f} | Points: {pts:.4f}", flush=True)
    
    oof_preds[name]  = oof
    test_preds[name] = tst

# Blend top hybrid models
blend_prob = (oof_preds['LogReg'] + oof_preds['Ridge'] + oof_preds['CatBoost'] + oof_preds['ExtraTrees']) / 4.0
blend_acc  = accuracy_score(y_train, blend_prob.argmax(axis=1))
blend_pts  = max(0.0, blend_acc - 0.40) / 0.60
print("\n" + "=" * 65, flush=True)
print(f"HYBRID ENSEMBLE OOF ACCURACY : {blend_acc:.4f} | POINTS: {blend_pts:.4f}", flush=True)
print("=" * 65, flush=True)

# Generate submission
final_test = (test_preds['LogReg'] + test_preds['Ridge'] + test_preds['CatBoost'] + test_preds['ExtraTrees']) / 4.0
sub = pd.DataFrame({'id': test_ids, 'label': final_test.argmax(axis=1)})
sub.to_csv('d:/AIJC/submission_task2_hybrid_fusion.csv', index=False)
print("Saved d:/AIJC/submission_task2_hybrid_fusion.csv", flush=True)
print(f"Distribution: {Counter(sub['label'].tolist())}", flush=True)
print(sub.head(10).to_string(index=False), flush=True)
