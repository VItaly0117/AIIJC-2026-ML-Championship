import os
import glob
import numpy as np
import pandas as pd
from collections import Counter

print("=================================================================")
print("BUILDING MASTER ENSEMBLE SUBMISSION FOR TASK 2")
print("=================================================================")

submission_files = [
    'd:/AIJC/submission_task2_tabular_blend.csv',
    'd:/AIJC/submission_task2_hybrid_fusion.csv',
    'd:/AIJC/submission_task2_tabular_opt.csv',
    'd:/AIJC/submission_task2_svm.csv',
    'd:/AIJC/submission_task2_resnet18_clean.csv'
]

valid_subs = []
for p in submission_files:
    if os.path.exists(p):
        df = pd.read_csv(p)
        print(f"Found {os.path.basename(p):37s} | Shape: {df.shape} | Dist: {Counter(df['label'].tolist())}")
        valid_subs.append(df)

if len(valid_subs) == 0:
    raise RuntimeError("No submission files found!")

# Majority voting across all available submission files
all_labels = np.array([df['label'].values for df in valid_subs]) # (num_models, 300)

master_labels = []
for i in range(all_labels.shape[1]):
    preds_for_item = all_labels[:, i]
    # Majority vote
    counts = Counter(preds_for_item)
    most_common = counts.most_common(1)[0][0]
    master_labels.append(most_common)

master_df = pd.DataFrame({
    'id': valid_subs[0]['id'],
    'label': master_labels
})

out_path = 'd:/AIJC/submission_task2_master.csv'
master_df.to_csv(out_path, index=False)

print("\n" + "=" * 65)
print(f"MASTER ENSEMBLE CREATED -> {out_path}")
print(f"Final Prediction Distribution: {Counter(master_df['label'].tolist())}")
print(master_df.head(15).to_string(index=False))
print("=" * 65)

# Also copy to root submission_task2.csv as primary deliverable
primary_path = 'd:/AIJC/submission_task2.csv'
master_df.to_csv(primary_path, index=False)
print(f"Updated primary deliverable: {primary_path}")
