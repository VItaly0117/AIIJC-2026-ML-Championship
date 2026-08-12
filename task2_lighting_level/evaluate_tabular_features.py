import os
import sys
import time
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from collections import Counter
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import RobustScaler

from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression

# ─────────────────────────────────────────────
# LOAD PRE-EXTRACTED FEATURES
# ─────────────────────────────────────────────
feat_path = 'd:/AIJC/features_tabular.npz'
data = np.load(feat_path)
X_train_raw = data['X_train']
y_train     = data['y_train']
X_test_raw  = data['X_test']
test_ids    = data['test_ids']

print("=================================================================", flush=True)
print("TABULAR ILLUMINATION FEATURE ML EVALUATION", flush=True)
print("=================================================================", flush=True)
print(f"X_train shape : {X_train_raw.shape}", flush=True)
print(f"X_test shape  : {X_test_raw.shape}", flush=True)
print(f"Target count  : {Counter(y_train)}", flush=True)

# Preprocessing
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train_raw)
X_test_scaled  = scaler.transform(X_test_raw)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

models = {
    'ExtraTrees': ExtraTreesClassifier(n_estimators=300, max_depth=12, n_jobs=-1, random_state=42),
    'RandomForest': RandomForestClassifier(n_estimators=300, max_depth=12, n_jobs=-1, random_state=42),
    'HistGB': HistGradientBoostingClassifier(max_iter=200, random_state=42),
    'LightGBM': LGBMClassifier(n_estimators=200, learning_rate=0.03, num_leaves=15, n_jobs=4, random_state=42, verbose=-1),
    'CatBoost': CatBoostClassifier(iterations=300, learning_rate=0.03, depth=5, thread_count=4, verbose=0, random_state=42),
    'LogReg': LogisticRegression(C=0.1, max_iter=500, random_state=42)
}

oof_preds = {}
test_preds = {}

for name, model in models.items():
    oof = np.zeros((len(y_train), 3), dtype=np.float32)
    tst = np.zeros((len(X_test_scaled), 3), dtype=np.float32)
    
    X_tr = X_train_scaled if name == 'LogReg' else X_train_raw
    X_te = X_test_scaled  if name == 'LogReg' else X_test_raw
    
    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_tr, y_train)):
        model.fit(X_tr[tr_idx], y_train[tr_idx])
        oof[val_idx] = model.predict_proba(X_tr[val_idx])
        tst += model.predict_proba(X_te) / 5.0
        
    acc = accuracy_score(y_train, oof.argmax(axis=1))
    pts = max(0.0, acc - 0.40) / 0.60
    print(f"{name:15s} | OOF Accuracy: {acc:.4f} | Points: {pts:.4f}", flush=True)
    
    oof_preds[name]  = oof
    test_preds[name] = tst

# ─────────────────────────────────────────────
# OPTIMAL BLENDING
# ─────────────────────────────────────────────
print("\n" + "=" * 65, flush=True)
print("OPTIMAL ENSEMBLE BLENDING", flush=True)
print("=" * 65, flush=True)

top_models = ['ExtraTrees', 'RandomForest', 'HistGB', 'LightGBM', 'CatBoost']

# Simple average
avg_oof = np.mean([oof_preds[m] for m in top_models], axis=0)
avg_acc = accuracy_score(y_train, avg_oof.argmax(axis=1))
avg_pts = max(0.0, avg_acc - 0.40) / 0.60
print(f"Simple Average Ensemble OOF Accuracy: {avg_acc:.4f} | Points: {avg_pts:.4f}", flush=True)

# Rank blend
rank_oof = np.zeros_like(avg_oof)
for m in top_models:
    for c in range(3):
        rank_oof[:, c] += pd.Series(oof_preds[m][:, c]).rank(pct=True).values
rank_acc = accuracy_score(y_train, rank_oof.argmax(axis=1))
rank_pts = max(0.0, rank_acc - 0.40) / 0.60
print(f"Rank-based Blend OOF Accuracy   : {rank_acc:.4f} | Points: {rank_pts:.4f}", flush=True)

# Best probability blend (ExtraTrees + RandomForest + CatBoost)
final_test_prob = (test_preds['ExtraTrees'] + test_preds['RandomForest'] + test_preds['CatBoost']) / 3.0
final_preds = final_test_prob.argmax(axis=1)

sub = pd.DataFrame({'id': test_ids, 'label': final_preds})
out_csv = 'd:/AIJC/submission_task2_tabular_opt.csv'
sub.to_csv(out_csv, index=False)

print("\n" + "=" * 65, flush=True)
print(f"Saved submission to {out_csv}", flush=True)
print(f"Prediction distribution: {Counter(final_preds.tolist())}", flush=True)
print(sub.head(10).to_string(index=False), flush=True)
print("=" * 65, flush=True)
