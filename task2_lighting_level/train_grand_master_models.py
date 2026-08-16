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
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler, RobustScaler, QuantileTransformer

from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier

# ─────────────────────────────────────────────
# 1. LOAD 366-DIM GRAND MASTER FEATURES
# ─────────────────────────────────────────────
data = np.load('d:/AIJC/features_grand_master.npz')
X_train_raw = data['X_train']
y_train     = data['y_train']
X_test_raw  = data['X_test']
test_ids    = data['test_ids']

print("=================================================================", flush=True)
print("TRAINING 366-DIM GRAND MASTER PHYSICS + WAVELET MODEL ZOO", flush=True)
print("=================================================================", flush=True)
print(f"Grand Master feature matrix: Train {X_train_raw.shape}, Test {X_test_raw.shape}", flush=True)
print(f"Target distribution        : {Counter(y_train.tolist())}", flush=True)

# Preprocessing & Scaling
scaler = RobustScaler()
X_train_s = scaler.fit_transform(X_train_raw)
X_test_s  = scaler.transform(X_test_raw)

# Clean any NaNs/Infs
X_train_s = np.nan_to_num(X_train_s, nan=0.0, posinf=1.0, neginf=-1.0)
X_test_s  = np.nan_to_num(X_test_s,  nan=0.0, posinf=1.0, neginf=-1.0)

# ─────────────────────────────────────────────
# 2. 10-FOLD STRATIFIED CROSS VALIDATION
# ─────────────────────────────────────────────
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

models = {
    'CatBoost':   CatBoostClassifier(iterations=700, learning_rate=0.025, depth=5, l2_leaf_reg=4.0, verbose=0, random_state=42),
    'LightGBM':   LGBMClassifier(n_estimators=500, learning_rate=0.02, num_leaves=20, min_child_samples=20, n_jobs=4, random_state=42, verbose=-1),
    'XGBoost':    XGBClassifier(n_estimators=500, learning_rate=0.02, max_depth=4, subsample=0.8, colsample_bytree=0.8, random_state=42, eval_metric='mlogloss'),
    'ExtraTrees': ExtraTreesClassifier(n_estimators=500, max_depth=16, min_samples_split=4, n_jobs=-1, random_state=42),
    'RandomForest': RandomForestClassifier(n_estimators=500, max_depth=16, min_samples_split=4, n_jobs=-1, random_state=42),
    'HistGB':     HistGradientBoostingClassifier(max_iter=350, learning_rate=0.025, max_leaf_nodes=22, random_state=42),
    'MLP':        MLPClassifier(hidden_layer_sizes=(128, 64), alpha=0.05, max_iter=400, random_state=42),
    'LogReg':     LogisticRegression(C=0.1, max_iter=500, random_state=42),
    'Ridge':      RidgeClassifier(alpha=50.0, random_state=42)
}

oof_probs = {}
test_probs = {}

print("\n--- 10-FOLD CROSS VALIDATION ---", flush=True)
for name, model in models.items():
    oof = np.zeros((len(y_train), 3), dtype=np.float32)
    tst = np.zeros((len(X_test_s), 3), dtype=np.float32)
    
    X_tr = X_train_s if name in ['MLP', 'LogReg', 'Ridge'] else X_train_raw
    X_te = X_test_s  if name in ['MLP', 'LogReg', 'Ridge'] else X_test_raw
    
    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_tr, y_train)):
        model.fit(X_tr[tr_idx], y_train[tr_idx])
        
        if hasattr(model, 'predict_proba'):
            oof[val_idx] = model.predict_proba(X_tr[val_idx])
            tst += model.predict_proba(X_te) / 10.0
        else:
            # Ridge decision function -> softmax
            dec_val = model.decision_function(X_tr[val_idx])
            dec_tst = model.decision_function(X_te)
            exp_v = np.exp(dec_val - np.max(dec_val, axis=1, keepdims=True))
            exp_t = np.exp(dec_tst - np.max(dec_tst, axis=1, keepdims=True))
            oof[val_idx] = exp_v / exp_v.sum(axis=1, keepdims=True)
            tst += (exp_t / exp_t.sum(axis=1, keepdims=True)) / 10.0
            
    acc = accuracy_score(y_train, oof.argmax(axis=1))
    pts = max(0.0, acc - 0.40) / 0.60
    print(f"{name:15s} | 10-Fold OOF Accuracy: {acc:.4f} | Points: {pts:.4f}", flush=True)
    
    oof_probs[name]  = oof
    test_probs[name] = tst

# ─────────────────────────────────────────────
# 3. OPTIMAL ENSEMBLE BLEND
# ─────────────────────────────────────────────
print("\n" + "=" * 65, flush=True)
print("FINDING OPTIMAL GRAND MASTER ENSEMBLE BLEND", flush=True)
print("=" * 65, flush=True)

top_names = ['CatBoost', 'LightGBM', 'XGBoost', 'ExtraTrees', 'RandomForest', 'LogReg', 'Ridge']

# Equal weight baseline
eq_oof = np.mean([oof_probs[n] for n in top_names], axis=0)
eq_acc = accuracy_score(y_train, eq_oof.argmax(axis=1))
print(f"Equal-Weight Grand Ensemble OOF Accuracy: {eq_acc:.4f} | Points: {max(0.0, eq_acc-0.4)/0.6:.4f}", flush=True)

# Grid search optimal weights
best_acc = eq_acc
best_weights = {n: 1.0/len(top_names) for n in top_names}

import itertools
candidate_weights = [0.0, 0.1, 0.2, 0.3, 0.4]
for w in itertools.product(candidate_weights, repeat=len(top_names)):
    s = sum(w)
    if s < 0.5 or s > 1.5: continue
    w_arr = np.array(w) / s
    blend_oof = sum(w_i * oof_probs[n] for w_i, n in zip(w_arr, top_names))
    acc = (blend_oof.argmax(axis=1) == y_train).mean()
    if acc > best_acc:
        best_acc = acc
        best_weights = dict(zip(top_names, [round(float(x), 3) for x in w_arr]))

best_pts = max(0.0, best_acc - 0.40) / 0.60
print(f"Weighted Grand Master Ensemble OOF Acc : {best_acc:.4f} | Points: {best_pts:.4f}", flush=True)
print(f"Optimal Weights: {best_weights}", flush=True)

# Final test prediction
final_test_prob = sum(best_weights[n] * test_probs[n] for n in top_names)
final_preds = final_test_prob.argmax(axis=1)

sub = pd.DataFrame({'id': test_ids, 'label': final_preds})
out_path = 'd:/AIJC/submission_task2_grand_master.csv'
sub.to_csv(out_path, index=False)

print("\n" + "=" * 65, flush=True)
print(f"Saved submission to {out_path}", flush=True)
print(f"Prediction distribution: {Counter(final_preds.tolist())}", flush=True)
print(sub.head(10).to_string(index=False), flush=True)
print("=" * 65, flush=True)
