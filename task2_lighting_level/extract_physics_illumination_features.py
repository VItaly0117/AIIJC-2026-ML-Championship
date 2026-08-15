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
import cv2
from concurrent.futures import ProcessPoolExecutor

print("=================================================================", flush=True)
print("PHASE 2: EXTRACTING PHYSICS-BASED ILLUMINATION DESCRIPTORS", flush=True)
print("  - Multi-Scale Retinex (MSR) Illumination Maps (sigma=15, 80, 250)", flush=True)
print("  - Dark Channel Prior (DCP) Shadow & Haze Levels", flush=True)
print("  - Color Constancy & Illuminant Temperature Vectors (Gray-World, Max-RGB, p=6)", flush=True)
print("=================================================================", flush=True)

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

def extract_single_image_physics(path):
    # Load image safely via PIL then to numpy float32 in [0, 1]
    img = Image.open(path).convert('RGB')
    arr = np.array(img, dtype=np.float32) / 255.0 # (H, W, 3)
    
    r, g, b = arr[:,:,0], arr[:,:,1], arr[:,:,2]
    gray = 0.299 * r + 0.587 * g + 0.114 * b
    
    feats = []
    
    # 1. Multi-Scale Retinex Illumination Maps
    # L_sigma = GaussianBlur(I)
    # R_sigma = log(I + eps) - log(L_sigma + eps)
    eps = 1e-4
    sigmas = [15, 80, 250]
    for s in sigmas:
        # Illumination estimate L
        L = cv2.GaussianBlur(gray, (0, 0), sigmaX=s, sigmaY=s)
        # Illumination statistics
        feats.extend([
            float(L.mean()),
            float(L.std()),
            float(np.percentile(L, 5)),
            float(np.percentile(L, 50)),
            float(np.percentile(L, 95)),
            float(np.percentile(L, 99)),
            float((L < 0.15).mean()), # low illumination fraction
            float((L > 0.85).mean()), # high illumination fraction
        ])
        # Reflectance statistics R = log(I) - log(L)
        R = np.log(gray + eps) - np.log(L + eps)
        feats.extend([
            float(R.mean()),
            float(R.std()),
            float(np.percentile(R, 10)),
            float(np.percentile(R, 90)),
        ])
        
    # 2. Dark Channel Prior (DCP)
    # min across RGB channels
    min_rgb = np.minimum(np.minimum(r, g), b)
    # patch min (shadow depth)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    dark_channel = cv2.erode(min_rgb, kernel)
    feats.extend([
        float(dark_channel.mean()),
        float(dark_channel.std()),
        float(np.percentile(dark_channel, 1)),
        float(np.percentile(dark_channel, 10)),
        float(np.percentile(dark_channel, 50)),
        float(np.percentile(dark_channel, 90)),
        float(np.percentile(dark_channel, 99)),
        float((dark_channel < 0.05).mean()), # pure shadow pixels
    ])
    
    # 3. Bright Channel Prior (BCP) - specular reflection / glare
    max_rgb = np.maximum(np.maximum(r, g), b)
    bright_channel = cv2.dilate(max_rgb, kernel)
    feats.extend([
        float(bright_channel.mean()),
        float(bright_channel.std()),
        float(np.percentile(bright_channel, 50)),
        float(np.percentile(bright_channel, 90)),
        float(np.percentile(bright_channel, 99)),
        float((bright_channel > 0.95).mean()), # pure glare pixels
    ])
    
    # 4. Color Constancy & Illuminant Temperature Vectors
    # Gray-World: e_GW
    mean_r, mean_g, mean_b = float(r.mean()), float(g.mean()), float(b.mean())
    norm_gw = np.sqrt(mean_r**2 + mean_g**2 + mean_b**2) + 1e-6
    e_gw_r, e_gw_g, e_gw_b = mean_r/norm_gw, mean_g/norm_gw, mean_b/norm_gw
    
    # Max-RGB: e_Max
    max_r, max_g, max_b = float(r.max()), float(g.max()), float(b.max())
    norm_max = np.sqrt(max_r**2 + max_g**2 + max_b**2) + 1e-6
    e_mx_r, e_mx_g, e_mx_b = max_r/norm_max, max_g/norm_max, max_b/norm_max
    
    # Shades of Gray (Minkowski p=6)
    p = 6
    sog_r = float((r**p).mean() ** (1/p))
    sog_g = float((g**p).mean() ** (1/p))
    sog_b = float((b**p).mean() ** (1/p))
    norm_sog = np.sqrt(sog_r**2 + sog_g**2 + sog_b**2) + 1e-6
    e_sg_r, e_sg_g, e_sg_b = sog_r/norm_sog, sog_g/norm_sog, sog_b/norm_sog
    
    # Neutral Daylight deviation (angle to [1/sqrt(3), 1/sqrt(3), 1/sqrt(3)])
    neutral = 1.0 / np.sqrt(3.0)
    dev_gw  = (e_gw_r - neutral)**2 + (e_gw_g - neutral)**2 + (e_gw_b - neutral)**2
    dev_sog = (e_sg_r - neutral)**2 + (e_sg_g - neutral)**2 + (e_sg_b - neutral)**2
    
    feats.extend([
        e_gw_r, e_gw_g, e_gw_b, dev_gw,
        e_mx_r, e_mx_g, e_mx_b,
        e_sg_r, e_sg_g, e_sg_b, dev_sog
    ])
    
    # 5. Local Contrast & Weber Contrast Ratio
    # Local RMS contrast
    local_mean = cv2.blur(gray, (21, 21))
    local_sq_mean = cv2.blur(gray**2, (21, 21))
    local_var = np.maximum(0.0, local_sq_mean - local_mean**2)
    local_rms = np.sqrt(local_var)
    feats.extend([
        float(local_rms.mean()),
        float(local_rms.std()),
        float(np.percentile(local_rms, 10)),
        float(np.percentile(local_rms, 90)),
        float(np.percentile(local_rms, 99))
    ])
    
    return feats

if __name__ == '__main__':
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

    print(f"Dataset size: Train={len(train_paths)}, Test={len(test_paths)}", flush=True)
    t0 = time.time()
    
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        X_tr_physics = list(executor.map(extract_single_image_physics, train_paths))
        X_te_physics = list(executor.map(extract_single_image_physics, test_paths))
        
    X_tr_physics = np.array(X_tr_physics, dtype=np.float32)
    X_te_physics = np.array(X_te_physics, dtype=np.float32)
    
    print(f"Physics features extracted in {time.time()-t0:.2f}s! Shape: Train {X_tr_physics.shape}, Test {X_te_physics.shape}", flush=True)
    
    # Save standalone physics features
    np.savez('d:/AIJC/features_physics.npz',
             X_train=X_tr_physics,
             y_train=train_labels,
             X_test=X_te_physics,
             test_ids=np.array(test_ids))
             
    # Combine with existing 268 tabular features
    old_data = np.load('d:/AIJC/features_tabular.npz')
    X_tr_combined = np.hstack([old_data['X_train'], X_tr_physics])
    X_te_combined = np.hstack([old_data['X_test'], X_te_physics])
    
    print(f"Combined Master Feature Matrix: Train {X_tr_combined.shape}, Test {X_te_combined.shape}", flush=True)
    np.savez('d:/AIJC/features_master_combined.npz',
             X_train=X_tr_combined,
             y_train=train_labels,
             X_test=X_te_combined,
             test_ids=np.array(test_ids))
    print("Saved d:/AIJC/features_master_combined.npz successfully!", flush=True)
