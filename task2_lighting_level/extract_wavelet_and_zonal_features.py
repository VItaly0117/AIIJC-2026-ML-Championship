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

def haar_dwt_2d(gray):
    # Vectorized 2D Haar Wavelet Decomposition
    H, W = gray.shape
    # Ensure even dimensions
    if H % 2 != 0: gray = gray[:-1, :]
    if W % 2 != 0: gray = gray[:, :-1]
    
    # Row-wise transform
    L_row = (gray[:, 0::2] + gray[:, 1::2]) / np.sqrt(2.0)
    H_row = (gray[:, 0::2] - gray[:, 1::2]) / np.sqrt(2.0)
    
    # Column-wise transform
    LL = (L_row[0::2, :] + L_row[1::2, :]) / np.sqrt(2.0)
    LH = (L_row[0::2, :] - L_row[1::2, :]) / np.sqrt(2.0) # Horizontal edges
    HL = (H_row[0::2, :] + H_row[1::2, :]) / np.sqrt(2.0) # Vertical edges / glare
    HH = (H_row[0::2, :] - H_row[1::2, :]) / np.sqrt(2.0) # High-frequency sensor grain & noise
    
    return LL, LH, HL, HH

def extract_single_image_wavelet_zonal(path):
    img = Image.open(path).convert('RGB')
    arr = np.array(img, dtype=np.float32) / 255.0 # (H, W, 3)
    
    r, g, b = arr[:,:,0], arr[:,:,1], arr[:,:,2]
    gray = 0.299 * r + 0.587 * g + 0.114 * b
    
    hsv = cv2.cvtColor((arr * 255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32) / 255.0
    sat = hsv[:,:,1]
    
    feats = []
    
    # ─────────────────────────────────────────────
    # 1. 2D HAAR WAVELET SUB-BAND DECOMPOSITION
    # ─────────────────────────────────────────────
    LL, LH, HL, HH = haar_dwt_2d(gray)
    
    # Sub-band Energies & Variances
    e_ll = float((LL**2).mean())
    e_lh = float((LH**2).mean())
    e_hl = float((HL**2).mean())
    e_hh = float((HH**2).mean()) # Sensor ISO noise
    
    v_lh = float(LH.var())
    v_hl = float(HL.var())
    v_hh = float(HH.var())
    
    # High-to-Low frequency energy ratios
    noise_to_signal = float((e_hh + e_lh + e_hl) / (e_ll + 1e-6))
    grain_ratio     = float(e_hh / (e_lh + e_hl + 1e-6))
    
    # Level 2 Wavelet Decomposition (on LL)
    LL2, LH2, HL2, HH2 = haar_dwt_2d(LL)
    e_ll2 = float((LL2**2).mean())
    e_hh2 = float((HH2**2).mean())
    
    feats.extend([
        e_ll, e_lh, e_hl, e_hh,
        v_lh, v_hl, v_hh,
        noise_to_signal, grain_ratio,
        e_ll2, e_hh2
    ])
    
    # ─────────────────────────────────────────────
    # 2. ZONAL DECOMPOSITION (Sky, Road, Center, Borders)
    # ─────────────────────────────────────────────
    H, W = gray.shape
    
    # Sky (Top 25%)
    sky_gray = gray[:int(H*0.25), :]
    sky_sat  = sat[:int(H*0.25), :]
    sky_mean = float(sky_gray.mean())
    sky_std  = float(sky_gray.std())
    sky_p90  = float(np.percentile(sky_gray, 90))
    sky_p99  = float(np.percentile(sky_gray, 99))
    sky_over = float((sky_gray > 0.90).mean())
    sky_sat_mean = float(sky_sat.mean())
    
    # Road / Ground (Bottom 25%)
    road_gray = gray[int(H*0.75):, :]
    road_sat  = sat[int(H*0.75):, :]
    road_mean = float(road_gray.mean())
    road_std  = float(road_gray.std())
    road_p10  = float(np.percentile(road_gray, 10))
    road_p50  = float(np.percentile(road_gray, 50))
    road_under = float((road_gray < 0.10).mean())
    road_sat_mean = float(road_sat.mean())
    
    # Center Region (Middle 50%)
    center_gray = gray[int(H*0.25):int(H*0.75), int(W*0.25):int(W*0.75)]
    center_mean = float(center_gray.mean())
    center_std  = float(center_gray.std())
    center_dyn  = float(np.percentile(center_gray, 90) - np.percentile(center_gray, 10))
    
    # Border Mask (Outer perimeter)
    border_mask = np.ones((H, W), dtype=bool)
    border_mask[int(H*0.25):int(H*0.75), int(W*0.25):int(W*0.75)] = False
    border_gray = gray[border_mask]
    border_mean = float(border_gray.mean())
    
    # Cross-Zone Contrast and Illumination Ratios
    sky_road_ratio   = float(sky_mean / (road_mean + 1e-5))
    sky_road_diff    = float(sky_mean - road_mean)
    sky_road_sat_rat = float(sky_sat_mean / (road_sat_mean + 1e-5))
    center_border_rat= float(center_mean / (border_mean + 1e-5))
    center_border_diff= float(center_mean - border_mean)
    
    feats.extend([
        sky_mean, sky_std, sky_p90, sky_p99, sky_over, sky_sat_mean,
        road_mean, road_std, road_p10, road_p50, road_under, road_sat_mean,
        center_mean, center_std, center_dyn, border_mean,
        sky_road_ratio, sky_road_diff, sky_road_sat_rat,
        center_border_rat, center_border_diff
    ])
    
    return feats

if __name__ == '__main__':
    print("=================================================================", flush=True)
    print("EXTRACTING 2D WAVELET ISO-NOISE & ZONAL SKY/ROAD DESCRIPTORS", flush=True)
    print("=================================================================", flush=True)

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

    print(f"Dataset: Train={len(train_paths)}, Test={len(test_paths)}", flush=True)
    t0 = time.time()
    
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        X_tr_wz = list(executor.map(extract_single_image_wavelet_zonal, train_paths))
        X_te_wz = list(executor.map(extract_single_image_wavelet_zonal, test_paths))
        
    X_tr_wz = np.array(X_tr_wz, dtype=np.float32)
    X_te_wz = np.array(X_te_wz, dtype=np.float32)
    
    print(f"Wavelet & Zonal features extracted in {time.time()-t0:.2f}s! Shape: Train {X_tr_wz.shape}, Test {X_te_wz.shape}", flush=True)
    
    # Save standalone wavelet/zonal features
    np.savez('d:/AIJC/features_wavelet_zonal.npz',
             X_train=X_tr_wz,
             y_train=train_labels,
             X_test=X_te_wz,
             test_ids=np.array(test_ids))
             
    # Combine with 334 physics + photometric features -> 366 Grand Master Matrix
    old_data = np.load('d:/AIJC/features_master_combined.npz')
    X_tr_grand = np.hstack([old_data['X_train'], X_tr_wz])
    X_te_grand = np.hstack([old_data['X_test'], X_te_wz])
    
    print(f"Grand Master Feature Matrix: Train {X_tr_grand.shape}, Test {X_te_grand.shape}", flush=True)
    np.savez('d:/AIJC/features_grand_master.npz',
             X_train=X_tr_grand,
             y_train=train_labels,
             X_test=X_te_grand,
             test_ids=np.array(test_ids))
    print("Saved d:/AIJC/features_grand_master.npz successfully!", flush=True)
