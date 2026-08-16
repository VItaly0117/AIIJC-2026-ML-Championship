import os
import glob
import numpy as np
import pandas as pd
from collections import Counter

print("=================================================================", flush=True)
print("BUILDING ULTIMATE GRAND MASTER ENSEMBLE FOR TASK 2", flush=True)
print("=================================================================", flush=True)

submission_files = [
    'd:/AIJC/submission_task2_grand_master.csv',
    'd:/AIJC/submission_task2_self_training.csv',
    'd:/AIJC/submission_task2_physics_blend.csv',
    'd:/AIJC/submission_task2_mobilenet_v3.csv',
    'd:/AIJC/submission_task2_resnet18_clean.csv',
    'd:/AIJC/submission_task2_tabular_blend.csv',
    'd:/AIJC/submission_task2_hybrid_fusion.csv'
]

valid_subs = []
for p in submission_files:
    if os.path.exists(p):
        df = pd.read_csv(p)
        print(f"Found {os.path.basename(p):45s} | Shape: {df.shape} | Dist: {Counter(df['label'].tolist())}", flush=True)
        valid_subs.append(df)

if len(valid_subs) == 0:
    raise RuntimeError("No submission files found!")

# Majority voting across all 7 state-of-the-art model families
all_labels = np.array([df['label'].values for df in valid_subs]) # (7, 300)

master_labels = []
for i in range(all_labels.shape[1]):
    preds_for_item = all_labels[:, i]
    counts = Counter(preds_for_item)
    most_common = counts.most_common(1)[0][0]
    master_labels.append(most_common)

master_df = pd.DataFrame({
    'id': valid_subs[0]['id'],
    'label': master_labels
})

out_path = 'd:/AIJC/submission_task2_master.csv'
master_df.to_csv(out_path, index=False)

# Update primary deliverable
primary_path = 'd:/AIJC/submission_task2.csv'
master_df.to_csv(primary_path, index=False)

print("\n" + "=" * 65, flush=True)
print(f"ULTIMATE GRAND MASTER ENSEMBLE DELIVERABLE GENERATED -> {primary_path}", flush=True)
print(f"Final Prediction Distribution: {Counter(master_df['label'].tolist())}", flush=True)
print(master_df.head(15).to_string(index=False), flush=True)
print("=" * 65, flush=True)
