import os
import sys
import time
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from PIL import Image
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, HistGradientBoostingClassifier

# ─────────────────────────────────────────────
# CONFIG & PATHS
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

# Collect file paths
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

print(f"Loaded dataset: Train={len(train_paths)}, Test={len(test_paths)}")
print(f"Class counts: {Counter(train_labels)}")

# ─────────────────────────────────────────────
# FEATURE EXTRACTION FUNCTION
# ─────────────────────────────────────────────
def extract_single_image_features(path):
    # Read image with PIL to support Cyrillic paths perfectly
    img_pil = Image.open(path).convert('RGB')
    img_rgb = np.array(img_pil, dtype=np.float32) # (H, W, 3)
    
    # Color spaces via PIL
    img_hsv_arr = np.array(img_pil.convert('HSV'), dtype=np.float32)
    img_l_arr   = np.array(img_pil.convert('L'), dtype=np.float32)
    
    feats = []
    
    # Helper to compute channel statistics
    def add_channel_stats(ch):
        mean_val = np.mean(ch)
        std_val  = np.std(ch)
        p1, p5, p10, p25, p50, p75, p90, p95, p99 = np.percentile(ch, [1, 5, 10, 25, 50, 75, 90, 95, 99])
        over_240 = np.mean(ch > 240)
        over_250 = np.mean(ch > 250)
        under_15 = np.mean(ch < 15)
        under_5  = np.mean(ch < 5)
        feats.extend([mean_val, std_val, p1, p5, p10, p25, p50, p75, p90, p95, p99, over_240, over_250, under_15, under_5])

    # 1. Stats for RGB
    for c in range(3):
        add_channel_stats(img_rgb[:, :, c])
        
    # 2. Stats for HSV (Hue, Saturation, Value)
    for c in range(3):
        add_channel_stats(img_hsv_arr[:, :, c])

    # 3. Luminance Y = 0.299R + 0.587G + 0.114B
    lum = 0.299 * img_rgb[:,:,0] + 0.587 * img_rgb[:,:,1] + 0.114 * img_rgb[:,:,2]
    add_channel_stats(lum)

    # 4. Color Channel Ratios
    r, g, b = img_rgb[:,:,0], img_rgb[:,:,1], img_rgb[:,:,2]
    feats.append((r / (g + 1e-5)).mean())
    feats.append((b / (g + 1e-5)).mean())
    feats.append((r / (b + 1e-5)).mean())
    
    # 5. Histograms (16 bins) for R, G, B, Lum, Saturation, Value
    for data in [r, g, b, lum, img_hsv_arr[:,:,1], img_hsv_arr[:,:,2]]:
        hist, _ = np.histogram(data, bins=16, range=(0, 256), density=True)
        feats.extend(hist)

    # 6. Spatial Grid Stats (3x3 grid) for Luminance & Saturation
    H, W = lum.shape
    h_step, w_step = H // 3, W // 3
    for i in range(3):
        for j in range(3):
            cell_lum = lum[i*h_step:(i+1)*h_step, j*w_step:(j+1)*w_step]
            cell_sat = img_hsv_arr[i*h_step:(i+1)*h_step, j*w_step:(j+1)*w_step, 1]
            feats.extend([cell_lum.mean(), cell_lum.std(), cell_sat.mean()])

    # 7. Center vs Border Luminance Ratio
    center_cell = lum[h_step:2*h_step, w_step:2*w_step]
    border_mask = np.ones_like(lum, dtype=bool)
    border_mask[h_step:2*h_step, w_step:2*w_step] = False
    center_mean = center_cell.mean()
    border_mean = lum[border_mask].mean()
    feats.extend([center_mean, border_mean, center_mean / (border_mean + 1e-5)])

    return feats

if __name__ == '__main__':
    print("Starting parallel extraction of features on CPU...")
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        X_train_list = list(executor.map(extract_single_image_features, train_paths))
        X_test_list  = list(executor.map(extract_single_image_features, test_paths))

    X_train = np.array(X_train_list, dtype=np.float32)
    X_test  = np.array(X_test_list, dtype=np.float32)
    print(f"Parallel feature extraction done in {time.time()-t0:.2f}s! Feature shape: {X_train.shape}")

    # ─────────────────────────────────────────────
    # MODEL EVALUATION & BLENDING
    # ─────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("EVALUATING TABULAR MODELS (5-FOLD CV)")
    print("=" * 65)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    models = {
        'CatBoost': CatBoostClassifier(iterations=600, learning_rate=0.03, depth=6, verbose=0, random_state=42),
        'LightGBM': LGBMClassifier(n_estimators=400, learning_rate=0.03, num_leaves=31, random_state=42, verbose=-1),
        'XGBoost': XGBClassifier(n_estimators=400, learning_rate=0.03, max_depth=5, random_state=42, eval_metric='mlogloss'),
        'ExtraTrees': ExtraTreesClassifier(n_estimators=400, max_depth=15, random_state=42),
        'RandomForest': RandomForestClassifier(n_estimators=400, max_depth=15, random_state=42),
        'HistGB': HistGradientBoostingClassifier(max_iter=300, random_state=42)
    }

    oof_predictions = {}
    test_predictions = {}

    for name, model in models.items():
        oof_prob = np.zeros((len(train_labels), 3), dtype=np.float32)
        test_prob = np.zeros((len(test_paths), 3), dtype=np.float32)
        
        for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train, train_labels)):
            model.fit(X_train[tr_idx], train_labels[tr_idx])
            oof_prob[val_idx] = model.predict_proba(X_train[val_idx])
            test_prob += model.predict_proba(X_test) / 5.0
            
        oof_acc = accuracy_score(train_labels, oof_prob.argmax(axis=1))
        pts = max(0.0, oof_acc - 0.40) / 0.60
        print(f"{name:15s} | OOF Accuracy: {oof_acc:.4f} | Points: {pts:.4f}")
        
        oof_predictions[name] = oof_prob
        test_predictions[name] = test_prob

    # ─────────────────────────────────────────────
    # OPTIMAL BLEND
    # ─────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("FINDING OPTIMAL TABULAR BLEND")
    print("=" * 65)

    names = ['CatBoost', 'LightGBM', 'XGBoost', 'ExtraTrees', 'RandomForest', 'HistGB']
    best_w_acc = 0.0
    best_weights = None

    for w1 in np.linspace(0.1, 0.4, 4):
        for w2 in np.linspace(0.1, 0.4, 4):
            for w3 in np.linspace(0.0, 0.3, 4):
                for w4 in np.linspace(0.0, 0.3, 4):
                    for w5 in np.linspace(0.0, 0.3, 4):
                        w6 = round(1.0 - w1 - w2 - w3 - w4 - w5, 2)
                        if w6 < 0: continue
                        weights = [w1, w2, w3, w4, w5, w6]
                        blend_prob = sum(w * oof_predictions[n] for w, n in zip(weights, names))
                        acc = accuracy_score(train_labels, blend_prob.argmax(axis=1))
                        if acc > best_w_acc:
                            best_w_acc = acc
                            best_weights = weights

    best_w_pts = max(0.0, best_w_acc - 0.40) / 0.60
    print(f"Weighted Ensemble OOF Accuracy      : {best_w_acc:.4f} | Points: {best_w_pts:.4f}")
    print(f"Optimal weights: {dict(zip(names, [round(w, 2) for w in best_weights]))}")

    # Generate submission
    final_test_prob = sum(w * test_predictions[n] for w, n in zip(best_weights, names))
    final_preds = final_test_prob.argmax(axis=1)

    sub = pd.DataFrame({'id': test_ids, 'label': final_preds})
    sub.to_csv('d:/AIJC/submission_task2_tabular_blend.csv', index=False)
    print("\n" + "=" * 65)
    print("Submission saved to d:/AIJC/submission_task2_tabular_blend.csv")
    print(f"Distribution: {Counter(final_preds.tolist())}")
    print(sub.head(10).to_string(index=False))
    print("=" * 65)
