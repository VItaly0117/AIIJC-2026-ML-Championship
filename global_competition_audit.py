import os
import pandas as pd
import numpy as np
from collections import Counter

print("=================================================================")
print("GLOBAL COMPETITION SUBMISSION AUDIT")
print("=================================================================")

# 1. TASK 1 AUDIT
t1_path = 'd:/AIJC/submission.csv'
assert os.path.exists(t1_path), "Task 1 submission not found!"
df1 = pd.read_csv(t1_path)
print("\n--- TASK 1: USER RETENTION ---")
print(f"File Path    : {t1_path}")
print(f"Shape        : {df1.shape}")
print(f"Columns      : {list(df1.columns)}")
print(f"Missing (NaN): {df1.isnull().sum().to_dict()}")
print(f"Value Range  : [{df1['retention'].min():.4f}, {df1['retention'].max():.4f}]")
assert df1.shape == (2532, 2), f"Expected (2532, 2), got {df1.shape}"
assert list(df1.columns) == ['id', 'retention'], f"Expected [id, retention], got {list(df1.columns)}"
assert df1.isnull().sum().sum() == 0, "Found NaNs in Task 1"
print(">>> TASK 1 STATUS: 100% VERIFIED & VALID")

# 2. TASK 2 AUDIT
t2_path = 'd:/AIJC/submission_task2.csv'
assert os.path.exists(t2_path), "Task 2 submission not found!"
df2 = pd.read_csv(t2_path)
print("\n--- TASK 2: LIGHTING LEVEL ---")
print(f"File Path    : {t2_path}")
print(f"Shape        : {df2.shape}")
print(f"Columns      : {list(df2.columns)}")
print(f"Missing (NaN): {df2.isnull().sum().to_dict()}")
print(f"Class Dist   : {Counter(df2['label'].tolist())}")
assert df2.shape == (300, 2), f"Expected (300, 2), got {df2.shape}"
assert list(df2.columns) == ['id', 'label'], f"Expected [id, label], got {list(df2.columns)}"
assert df2.isnull().sum().sum() == 0, "Found NaNs in Task 2"
assert set(df2['label'].unique()).issubset({0, 1, 2}), f"Invalid labels: {df2['label'].unique()}"
print(">>> TASK 2 STATUS: 100% VERIFIED & VALID")

print("\n" + "=" * 65)
print("ALL TASKS VERIFIED WITH ZERO ERRORS!")
print("=================================================================")
