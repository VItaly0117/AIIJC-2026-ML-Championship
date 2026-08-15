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
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import RobustScaler

from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

print("=================================================================", flush=True)
print("PHASE 3: SELF-TRAINING & PSEUDO-LABELING ON TEST SET", flush=True)
print("=================================================================", flush=True)

# 1. Load Master Features
data = np.load('d:/AIJC/features_master_combined.npz')
X_train_raw = data['X_train']
y_train     = data['y_train']
X_test_raw  = data['X_test']
test_ids    = data['test_ids']

scaler = RobustScaler()
X_tr_s = scaler.fit_transform(X_train_raw)
X_te_s = scaler.transform(X_test_raw)

X_tr_s = np.nan_to_num(X_tr_s, nan=0.0, posinf=1.0, neginf=-1.0)
X_te_s = np.nan_to_num(X_te_s, nan=0.0, posinf=1.0, neginf=-1.0)

# 2. Compare Predictions across top models to extract High-Confidence Pseudo-Labels
sub_physics = pd.read_csv('d:/AIJC/submission_task2_physics_blend.csv')
sub_mobile  = pd.read_csv('d:/AIJC/submission_task2_mobilenet_v3.csv')
sub_master  = pd.read_csv('d:/AIJC/submission_task2_master.csv')

p_phys = sub_physics['label'].values
p_mobl = sub_mobile['label'].values
p_mast = sub_master['label'].values

# Find samples where all 3 independent models 100% agree
consensus_mask = (p_phys == p_mobl) & (p_mobl == p_mast)
consensus_idx  = np.where(consensus_mask)[0]
pseudo_labels  = p_phys[consensus_idx]

print(f"Total test samples           : {len(test_ids)}", flush=True)
print(f"High-confidence consensus    : {len(consensus_idx)} ({len(consensus_idx)/len(test_ids)*100:.1f}%)", flush=True)
print(f"Pseudo-label distribution    : {Counter(pseudo_labels.tolist())}", flush=True)

# 3. Augment Training Set with High-Confidence Pseudo-Labeled Test Samples
X_pseudo = X_te_s[consensus_idx]
y_pseudo = pseudo_labels

X_train_aug = np.vstack([X_tr_s, X_pseudo])
y_train_aug = np.concatenate([y_train, y_pseudo])

print(f"Augmented Train Shape        : {X_train_aug.shape}", flush=True)
print(f"Augmented Target Distribution: {Counter(y_train_aug.tolist())}", flush=True)

# 4. Train 2nd Generation Models with Augmented Data (5-Fold CV on original train)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

gen2_models = {
    'CatBoost_Gen2': CatBoostClassifier(iterations=600, learning_rate=0.025, depth=5, l2_leaf_reg=4.0, verbose=0, random_state=42),
    'LightGBM_Gen2': LGBMClassifier(n_estimators=400, learning_rate=0.025, num_leaves=18, min_child_samples=25, n_jobs=4, random_state=42, verbose=-1),
    'XGBoost_Gen2':  XGBClassifier(n_estimators=400, learning_rate=0.025, max_depth=4, subsample=0.8, colsample_bytree=0.8, random_state=42, eval_metric='mlogloss'),
    'ExtraTrees_Gen2': ExtraTreesClassifier(n_estimators=400, max_depth=14, min_samples_split=4, n_jobs=-1, random_state=42),
    'LogReg_Gen2':   LogisticRegression(C=0.1, max_iter=500, random_state=42)
}

gen2_oof_probs = {}
gen2_test_probs = {}

print("\n--- 2ND GENERATION SELF-TRAINING 5-FOLD CV ---", flush=True)
for name, model in gen2_models.items():
    oof = np.zeros((len(y_train), 3), dtype=np.float32)
    tst = np.zeros((len(X_te_s), 3), dtype=np.float32)
    
    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_tr_s, y_train)):
        # Train on Fold Train + ALL Pseudo Labels
        X_tr_fold = np.vstack([X_tr_s[tr_idx], X_pseudo])
        y_tr_fold = np.concatenate([y_train[tr_idx], y_pseudo])
        
        model.fit(X_tr_fold, y_tr_fold)
        
        # Validate purely on un-augmented validation fold
        oof[val_idx] = model.predict_proba(X_tr_s[val_idx])
        tst += model.predict_proba(X_te_s) / 5.0
        
    acc = accuracy_score(y_train, oof.argmax(axis=1))
    pts = max(0.0, acc - 0.40) / 0.60
    print(f"{name:16s} | OOF Accuracy: {acc:.4f} | Points: {pts:.4f}", flush=True)
    
    gen2_oof_probs[name]  = oof
    gen2_test_probs[name] = tst

# Blend 2nd Gen Models
gen2_blend_oof = (gen2_oof_probs['CatBoost_Gen2']*0.3 + 
                  gen2_oof_probs['LightGBM_Gen2']*0.3 + 
                  gen2_oof_probs['XGBoost_Gen2']*0.2 + 
                  gen2_oof_probs['ExtraTrees_Gen2']*0.1 + 
                  gen2_oof_probs['LogReg_Gen2']*0.1)

gen2_blend_acc = accuracy_score(y_train, gen2_blend_oof.argmax(axis=1))
gen2_blend_pts = max(0.0, gen2_blend_acc - 0.40) / 0.60

print("\n" + "=" * 65, flush=True)
print(f"2ND GENERATION SELF-TRAINING BLEND OOF ACC: {gen2_blend_acc:.4f} | POINTS: {gen2_blend_pts:.4f}", flush=True)
print("=" * 65, flush=True)

gen2_test_final = (gen2_test_probs['CatBoost_Gen2']*0.3 + 
                   gen2_test_probs['LightGBM_Gen2']*0.3 + 
                   gen2_test_probs['XGBoost_Gen2']*0.2 + 
                   gen2_test_probs['ExtraTrees_Gen2']*0.1 + 
                   gen2_test_probs['LogReg_Gen2']*0.1)

sub_gen2 = pd.DataFrame({'id': test_ids, 'label': gen2_test_final.argmax(axis=1)})
sub_gen2.to_csv('d:/AIJC/submission_task2_self_training.csv', index=False)

print(f"Saved d:/AIJC/submission_task2_self_training.csv", flush=True)
print(f"Prediction distribution: {Counter(sub_gen2['label'].tolist())}", flush=True)
print(sub_gen2.head(10).to_string(index=False), flush=True)
