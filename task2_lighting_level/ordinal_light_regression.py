import os
import sys
import time
import io
import warnings
warnings.filterwarnings('ignore')

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)

import numpy as np
import pandas as pd
from collections import Counter
from scipy.optimize import minimize
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler, RobustScaler

from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import BayesianRidge

# ─────────────────────────────────────────────
# 1. LOAD PRE-EXTRACTED FEATURES
# ─────────────────────────────────────────────
feat_path = 'd:/AIJC/features_tabular.npz'
data = np.load(feat_path)
X_train_raw = data['X_train']
y_train     = data['y_train'].astype(np.float32)
X_test_raw  = data['X_test']
test_ids    = data['test_ids']

print("=================================================================", flush=True)
print("PHASE 1: FAST ORDINAL REGRESSION & THRESHOLD CALIBRATION", flush=True)
print("=================================================================", flush=True)

scaler = RobustScaler()
X_train_s = scaler.fit_transform(X_train_raw)
X_test_s  = scaler.transform(X_test_raw)

def discretize(scores, t1, t2):
    preds = np.zeros(len(scores), dtype=int)
    preds[(scores >= t1) & (scores < t2)] = 1
    preds[scores >= t2] = 2
    return preds

def find_best_thresholds(oof_scores, y_true):
    best_acc = 0.0
    best_t1, best_t2 = 0.5, 1.5
    # Fast vectorized search over quantiles
    q_candidates = np.percentile(oof_scores, np.linspace(15, 85, 71))
    for i in range(len(q_candidates)):
        t1 = q_candidates[i]
        for j in range(i+1, min(i+40, len(q_candidates))):
            t2 = q_candidates[j]
            preds = discretize(oof_scores, t1, t2)
            acc = (preds == y_true).mean()
            if acc > best_acc:
                best_acc = acc
                best_t1, best_t2 = t1, t2
    return best_t1, best_t2, best_acc

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

models = {
    'CatBoost_MAE':  CatBoostRegressor(iterations=400, learning_rate=0.03, depth=5, loss_function='MAE', verbose=0, random_state=42),
    'LightGBM_Hub':  LGBMRegressor(n_estimators=300, learning_rate=0.03, objective='huber', num_leaves=15, n_jobs=4, random_state=42, verbose=-1),
    'XGB_Huber':     XGBRegressor(n_estimators=300, learning_rate=0.03, max_depth=4, objective='reg:pseudohubererror', random_state=42),
    'ExtraTrees':    ExtraTreesRegressor(n_estimators=300, max_depth=12, n_jobs=-1, random_state=42),
    'RandomForest':  RandomForestRegressor(n_estimators=300, max_depth=12, n_jobs=-1, random_state=42),
    'BayesianRidge': BayesianRidge()
}

oof_scores = {}
test_scores = {}

for name, model in models.items():
    oof = np.zeros(len(y_train), dtype=np.float32)
    tst = np.zeros(len(X_test_s), dtype=np.float32)
    
    X_tr = X_train_s if name == 'BayesianRidge' else X_train_raw
    X_te = X_test_s  if name == 'BayesianRidge' else X_test_raw
    
    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_tr, y_train.astype(int))):
        model.fit(X_tr[tr_idx], y_train[tr_idx])
        oof[val_idx] = model.predict(X_tr[val_idx])
        tst += model.predict(X_te) / 5.0
        
    t1, t2, acc = find_best_thresholds(oof, y_train.astype(int))
    pts = max(0.0, acc - 0.40) / 0.60
    print(f"{name:15s} | Opt Thresholds: [{t1:.3f}, {t2:.3f}] | OOF Accuracy: {acc:.4f} | Points: {pts:.4f}", flush=True)
    
    oof_scores[name]  = oof
    test_scores[name] = tst

print("\n" + "=" * 65, flush=True)
print("ENSEMBLE OF ORDINAL REGRESSORS", flush=True)
print("=" * 65, flush=True)

# Rank-based score normalization before averaging
ranked_oof = np.zeros(len(y_train), dtype=np.float32)
ranked_test = np.zeros(len(X_test_s), dtype=np.float32)

weights = {'CatBoost_MAE': 0.2, 'LightGBM_Hub': 0.2, 'XGB_Huber': 0.15, 'ExtraTrees': 0.2, 'RandomForest': 0.15, 'BayesianRidge': 0.1}

for name, w in weights.items():
    s_tr = oof_scores[name]
    s_te = test_scores[name]
    # Standardize scores
    mean, std = s_tr.mean(), s_tr.std() + 1e-5
    ranked_oof  += w * ((s_tr - mean) / std)
    ranked_test += w * ((s_te - mean) / std)

t1_ens, t2_ens, acc_ens = find_best_thresholds(ranked_oof, y_train.astype(int))
pts_ens = max(0.0, acc_ens - 0.40) / 0.60

print(f"Weighted Standardized Ordinal Ensemble | Thresholds: [{t1_ens:.3f}, {t2_ens:.3f}]", flush=True)
print(f"OOF Accuracy: {acc_ens:.4f} | Points: {pts_ens:.4f}", flush=True)

final_preds = discretize(ranked_test, t1_ens, t2_ens)
sub = pd.DataFrame({'id': test_ids, 'label': final_preds})
out_csv = 'd:/AIJC/submission_task2_ordinal_regression.csv'
sub.to_csv(out_csv, index=False)

print("\n" + "=" * 65, flush=True)
print(f"Saved submission to {out_csv}", flush=True)
print(f"Prediction distribution: {Counter(final_preds.tolist())}", flush=True)
print(sub.head(10).to_string(index=False), flush=True)
print("=" * 65, flush=True)
