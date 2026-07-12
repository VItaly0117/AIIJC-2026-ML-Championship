"""
AIIJC 2026 — Задача 1 v3
Исправления: feature selection, больше diversity, pseudo-labeling, robust blending
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import (
    RandomForestClassifier, ExtraTreesClassifier,
    HistGradientBoostingClassifier
)
from sklearn.neural_network import MLPClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score
from sklearn.feature_selection import mutual_info_classif
from scipy.stats import rankdata

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

N_FOLDS = 10
SEEDS = [42, 2024, 7, 13, 99]
TARGET = 'retention'
ID_COL = 'id'

# =============================================================================
# 1. LOAD DATA
# =============================================================================
print("=" * 70)
print("LOADING DATA")
print("=" * 70)

train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')
print(f"Train: {train.shape}, Test: {test.shape}")
print(f"Target balance: {train[TARGET].mean():.4f}")

# =============================================================================
# 2. FEATURE ENGINEERING
# =============================================================================
print("\n" + "=" * 70)
print("FEATURE ENGINEERING")
print("=" * 70)

def create_features(df):
    d = df.copy()

    d['sessions_per_active_day'] = d['sessions_count'] / (d['active_days'] + 1)
    d['purchases_per_session'] = d['purchases_count'] / (d['sessions_count'] + 1)
    d['purchases_per_active_day'] = d['purchases_count'] / (d['active_days'] + 1)

    d['total_session_time'] = d['avg_session_time'] * d['sessions_count']
    d['total_spend'] = d['avg_purchase_value'] * d['purchases_count']
    d['spend_per_session'] = d['total_spend'] / (d['sessions_count'] + 1)
    d['spend_per_active_day'] = d['total_spend'] / (d['active_days'] + 1)
    d['time_per_active_day'] = d['total_session_time'] / (d['active_days'] + 1)

    d['inactivity_ratio'] = d['days_since_last_activity'] / (d['active_days'] + 1)
    d['activity_gap'] = d['days_since_last_activity'] - d['active_days']
    d['recency_score'] = 1.0 / (d['days_since_last_activity'] + 1)
    d['activity_density'] = d['active_days'] / (d['active_days'] + d['days_since_last_activity'] + 1)

    d['session_cv'] = d['session_std'] / (d['avg_session_time'] + 1e-6)
    d['session_stability'] = 1 / (d['session_std'] + 1)

    d['engagement_score'] = d['sessions_count'] * d['active_days'] / (d['days_since_last_activity'] + 1)
    d['purchase_intensity'] = d['purchases_count'] * d['avg_purchase_value'] / (d['days_since_last_activity'] + 1)
    d['monetary_engagement'] = d['total_spend'] * d['recency_score']

    d['rfm_r'] = 1.0 / (d['days_since_last_activity'] + 1)
    d['rfm_f'] = d['sessions_count'] + d['purchases_count']
    d['rfm_m'] = d['total_spend']
    d['rfm_combined'] = d['rfm_r'] * d['rfm_f'] * np.log1p(d['rfm_m'])

    for feat in ['total_session_time', 'total_spend', 'avg_purchase_value',
                 'sessions_count', 'avg_session_time', 'session_std',
                 'engagement_score', 'purchase_intensity', 'rfm_combined']:
        d[f'log_{feat}'] = np.log1p(d[feat].clip(lower=0))

    d['sessions_count_sq'] = d['sessions_count'] ** 2
    d['days_since_sq'] = d['days_since_last_activity'] ** 2
    d['active_days_sq'] = d['active_days'] ** 2

    d['session_x_purchase'] = d['avg_session_time'] * d['avg_purchase_value']
    d['session_x_days_since'] = d['avg_session_time'] * d['days_since_last_activity']
    d['active_x_sessions'] = d['active_days'] * d['sessions_count']
    d['purchase_x_weekend'] = d['purchases_count'] * d['is_weekend_user']
    d['session_x_weekend'] = d['sessions_count'] * d['is_weekend_user']
    d['days_x_weekend'] = d['days_since_last_activity'] * d['is_weekend_user']
    d['active_x_weekend'] = d['active_days'] * d['is_weekend_user']
    d['spend_x_weekend'] = d['total_spend'] * d['is_weekend_user']
    d['engagement_x_weekend'] = d['engagement_score'] * d['is_weekend_user']
    d['std_x_sessions'] = d['session_std'] * d['sessions_count']
    d['purchase_val_x_count'] = d['avg_purchase_value'] * d['sessions_count']

    d['sessions_minus_purchases'] = d['sessions_count'] - d['purchases_count']
    d['active_minus_days_since'] = d['active_days'] - d['days_since_last_activity']
    d['sessions_minus_active'] = d['sessions_count'] - d['active_days']

    d['sin_sessions'] = np.sin(d['sessions_count'] * np.pi / 15)
    d['cos_sessions'] = np.cos(d['sessions_count'] * np.pi / 15)
    d['sin_active'] = np.sin(d['active_days'] * np.pi / 15)
    d['cos_active'] = np.cos(d['active_days'] * np.pi / 15)

    for feat in ['sessions_count', 'avg_session_time', 'purchases_count',
                 'avg_purchase_value', 'active_days', 'days_since_last_activity',
                 'session_std', 'total_spend', 'engagement_score']:
        d[f'rank_{feat}'] = d[feat].rank(pct=True)

    d['is_high_spender'] = (d['total_spend'] > d['total_spend'].median()).astype(int)
    d['is_frequent_user'] = (d['sessions_count'] > d['sessions_count'].median()).astype(int)
    d['is_recent_user'] = (d['days_since_last_activity'] <= 7).astype(int)
    d['is_loyal'] = (d['active_days'] > d['active_days'].median()).astype(int)
    d['has_many_purchases'] = (d['purchases_count'] > d['purchases_count'].median()).astype(int)
    d['is_stable_user'] = (d['session_cv'] < d['session_cv'].median()).astype(int)
    d['no_recent_activity'] = (d['days_since_last_activity'] > 30).astype(int)

    return d

train['_source'] = 'train'
test['_source'] = 'test'
combined = pd.concat([train, test], ignore_index=True)
combined_fe = create_features(combined)

train_fe = combined_fe[combined_fe['_source'] == 'train'].drop('_source', axis=1).copy()
test_fe = combined_fe[combined_fe['_source'] == 'test'].drop('_source', axis=1).copy()

all_feature_cols = [c for c in train_fe.columns if c not in [ID_COL, TARGET]]
print(f"All features: {len(all_feature_cols)}")

X_all = train_fe[all_feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0).values
y = train_fe[TARGET].values
X_test_all = test_fe[all_feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0).values
test_ids = test_fe[ID_COL].values

# =============================================================================
# 3. FEATURE SELECTION — remove noise
# =============================================================================
print("\n" + "=" * 70)
print("FEATURE SELECTION")
print("=" * 70)

# Mutual information
mi_scores = mutual_info_classif(X_all, y, random_state=42, n_neighbors=5)
mi_ranking = np.argsort(mi_scores)[::-1]

# LightGBM importance (quick model)
lgb_quick = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, max_depth=6,
                                 num_leaves=31, min_child_samples=20,
                                 subsample=0.8, colsample_bytree=0.8,
                                 random_state=42, verbosity=-1, n_jobs=-1)
lgb_quick.fit(X_all, y)
lgb_importance = lgb_quick.feature_importances_

# Combine rankings
mi_rank = np.argsort(mi_ranking)  # rank of each feature by MI
lgb_rank = np.argsort(np.argsort(-lgb_importance))  # rank by LGB importance
combined_rank = mi_rank + lgb_rank
top_feature_indices = np.argsort(combined_rank)[:60]  # top 60 features
selected_features = [all_feature_cols[i] for i in top_feature_indices]

X = X_all[:, top_feature_indices]
X_test = X_test_all[:, top_feature_indices]
print(f"Selected {len(selected_features)} features (from {len(all_feature_cols)})")

# Print top features
print("\nTop 15 features by MI:")
for i in mi_ranking[:15]:
    print(f"  {all_feature_cols[i]:40s} MI={mi_scores[i]:.4f}")

# =============================================================================
# 4. OPTUNA OPTIMIZATION
# =============================================================================
print("\n" + "=" * 70)
print("OPTUNA OPTIMIZATION")
print("=" * 70)

# LightGBM
print("\n--- LightGBM ---")
def lgb_objective(trial):
    params = {
        'objective': 'binary', 'metric': 'auc', 'verbosity': -1, 'n_jobs': -1,
        'random_state': 42,
        'n_estimators': trial.suggest_int('n_estimators', 200, 2000),
        'learning_rate': trial.suggest_float('lr', 0.005, 0.15, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'num_leaves': trial.suggest_int('num_leaves', 15, 127),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
    }
    scores = []
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    for tr_idx, va_idx in skf.split(X, y):
        m = lgb.LGBMClassifier(**params)
        m.fit(X[tr_idx], y[tr_idx], eval_set=[(X[va_idx], y[va_idx])],
              callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
        scores.append(roc_auc_score(y[va_idx], m.predict_proba(X[va_idx])[:, 1]))
    return np.mean(scores)

study_lgb = optuna.create_study(direction='maximize')
study_lgb.optimize(lgb_objective, n_trials=100, show_progress_bar=False)
best_lgb_params = study_lgb.best_params
if 'lr' in best_lgb_params:
    best_lgb_params['learning_rate'] = best_lgb_params.pop('lr')
print(f"Best: {study_lgb.best_value:.5f}")

# XGBoost
print("\n--- XGBoost ---")
def xgb_objective(trial):
    params = {
        'objective': 'binary:logistic', 'eval_metric': 'auc', 'verbosity': 0,
        'n_jobs': -1, 'random_state': 42,
        'n_estimators': trial.suggest_int('n_estimators', 200, 2000),
        'learning_rate': trial.suggest_float('lr', 0.005, 0.15, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 30),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 1.0),
        'gamma': trial.suggest_float('gamma', 0.0, 5.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
    }
    scores = []
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    for tr_idx, va_idx in skf.split(X, y):
        m = xgb.XGBClassifier(**params)
        m.fit(X[tr_idx], y[tr_idx], eval_set=[(X[va_idx], y[va_idx])], verbose=False)
        scores.append(roc_auc_score(y[va_idx], m.predict_proba(X[va_idx])[:, 1]))
    return np.mean(scores)

study_xgb = optuna.create_study(direction='maximize')
study_xgb.optimize(xgb_objective, n_trials=80, show_progress_bar=False)
best_xgb_params = study_xgb.best_params
if 'lr' in best_xgb_params:
    best_xgb_params['learning_rate'] = best_xgb_params.pop('lr')
print(f"Best: {study_xgb.best_value:.5f}")

# CatBoost
print("\n--- CatBoost ---")
def cat_objective(trial):
    params = {
        'loss_function': 'Logloss', 'eval_metric': 'AUC', 'verbose': 0,
        'random_seed': 42,
        'iterations': trial.suggest_int('iterations', 200, 2000),
        'learning_rate': trial.suggest_float('lr', 0.005, 0.15, log=True),
        'depth': trial.suggest_int('depth', 3, 10),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-3, 10.0, log=True),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 5.0),
        'random_strength': trial.suggest_float('random_strength', 0.0, 5.0),
        'border_count': trial.suggest_int('border_count', 32, 255),
    }
    scores = []
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    for tr_idx, va_idx in skf.split(X, y):
        m = CatBoostClassifier(**params)
        m.fit(X[tr_idx], y[tr_idx], eval_set=(X[va_idx], y[va_idx]), early_stopping_rounds=50)
        scores.append(roc_auc_score(y[va_idx], m.predict_proba(X[va_idx])[:, 1]))
    return np.mean(scores)

study_cat = optuna.create_study(direction='maximize')
study_cat.optimize(cat_objective, n_trials=60, show_progress_bar=False)
best_cat_params = study_cat.best_params
if 'lr' in best_cat_params:
    best_cat_params['learning_rate'] = best_cat_params.pop('lr')
print(f"Best: {study_cat.best_value:.5f}")

# =============================================================================
# 5. MULTI-SEED TRAINING — ALL MODELS
# =============================================================================
print("\n" + "=" * 70)
print("MULTI-SEED TRAINING")
print("=" * 70)

MODEL_NAMES = ['lgb', 'xgb', 'cat', 'et', 'hgb', 'rf', 'lr', 'ridge', 'mlp']
all_oof = {m: [] for m in MODEL_NAMES}
all_test = {m: [] for m in MODEL_NAMES}

for seed in SEEDS:
    print(f"\n--- Seed {seed} ---")
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    scaler = StandardScaler()

    oof = {m: np.zeros(len(X)) for m in MODEL_NAMES}
    test_preds = {m: np.zeros(len(X_test)) for m in MODEL_NAMES}

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
        X_tr, X_va = X[tr_idx], X[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]
        X_tr_sc = scaler.fit_transform(X_tr)
        X_va_sc = scaler.transform(X_va)
        X_test_sc = scaler.transform(X_test)

        # LightGBM
        p = {'objective': 'binary', 'metric': 'auc', 'verbosity': -1, 'n_jobs': -1,
             'random_state': seed, **best_lgb_params}
        m = lgb.LGBMClassifier(**p)
        m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)],
              callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
        oof['lgb'][va_idx] = m.predict_proba(X_va)[:, 1]
        test_preds['lgb'] += m.predict_proba(X_test)[:, 1] / N_FOLDS

        # XGBoost
        p = {'objective': 'binary:logistic', 'eval_metric': 'auc', 'verbosity': 0,
             'n_jobs': -1, 'random_state': seed, **best_xgb_params}
        m = xgb.XGBClassifier(**p)
        m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        oof['xgb'][va_idx] = m.predict_proba(X_va)[:, 1]
        test_preds['xgb'] += m.predict_proba(X_test)[:, 1] / N_FOLDS

        # CatBoost
        p = {'loss_function': 'Logloss', 'eval_metric': 'AUC', 'verbose': 0,
             'random_seed': seed, **best_cat_params}
        m = CatBoostClassifier(**p)
        m.fit(X_tr, y_tr, eval_set=(X_va, y_va), early_stopping_rounds=50)
        oof['cat'][va_idx] = m.predict_proba(X_va)[:, 1]
        test_preds['cat'] += m.predict_proba(X_test)[:, 1] / N_FOLDS

        # ExtraTrees
        m = ExtraTreesClassifier(n_estimators=500, max_depth=None, min_samples_split=5,
                                  min_samples_leaf=2, random_state=seed, n_jobs=-1)
        m.fit(X_tr, y_tr)
        oof['et'][va_idx] = m.predict_proba(X_va)[:, 1]
        test_preds['et'] += m.predict_proba(X_test)[:, 1] / N_FOLDS

        # HistGradientBoosting
        m = HistGradientBoostingClassifier(
            max_iter=500, learning_rate=0.05, max_depth=6,
            min_samples_leaf=20, l2_regularization=1.0,
            random_state=seed)
        m.fit(X_tr, y_tr)
        oof['hgb'][va_idx] = m.predict_proba(X_va)[:, 1]
        test_preds['hgb'] += m.predict_proba(X_test)[:, 1] / N_FOLDS

        # RandomForest
        m = RandomForestClassifier(n_estimators=500, max_depth=None, min_samples_split=5,
                                   min_samples_leaf=2, random_state=seed, n_jobs=-1)
        m.fit(X_tr, y_tr)
        oof['rf'][va_idx] = m.predict_proba(X_va)[:, 1]
        test_preds['rf'] += m.predict_proba(X_test)[:, 1] / N_FOLDS

        # LogisticRegression (scaled)
        m = LogisticRegression(max_iter=5000, C=0.5, random_state=seed, solver='lbfgs')
        m.fit(X_tr_sc, y_tr)
        oof['lr'][va_idx] = m.predict_proba(X_va_sc)[:, 1]
        test_preds['lr'] += m.predict_proba(X_test_sc)[:, 1] / N_FOLDS

        # Ridge (scaled, calibrated for probabilities)
        m = CalibratedClassifierCV(RidgeClassifier(alpha=1.0), cv=3)
        m.fit(X_tr_sc, y_tr)
        oof['ridge'][va_idx] = m.predict_proba(X_va_sc)[:, 1]
        test_preds['ridge'] += m.predict_proba(X_test_sc)[:, 1] / N_FOLDS

        # MLP (scaled)
        m = MLPClassifier(hidden_layer_sizes=(128, 64, 32), activation='relu',
                          max_iter=500, random_state=seed, early_stopping=True,
                          validation_fraction=0.15, alpha=0.01)
        m.fit(X_tr_sc, y_tr)
        oof['mlp'][va_idx] = m.predict_proba(X_va_sc)[:, 1]
        test_preds['mlp'] += m.predict_proba(X_test_sc)[:, 1] / N_FOLDS

    for name in MODEL_NAMES:
        auc = roc_auc_score(y, oof[name])
        all_oof[name].append(oof[name])
        all_test[name].append(test_preds[name])
        print(f"  {name:8s}: {auc:.5f}")

# Average across seeds
avg_oof = {m: np.mean(all_oof[m], axis=0) for m in MODEL_NAMES}
avg_test = {m: np.mean(all_test[m], axis=0) for m in MODEL_NAMES}

print(f"\n--- Final OOF scores ---")
for name in MODEL_NAMES:
    auc = roc_auc_score(y, avg_oof[name])
    print(f"  {name:8s}: {auc:.5f}")

# =============================================================================
# 6. PSEUDO-LABELING
# =============================================================================
print("\n" + "=" * 70)
print("PSEUDO-LABELING")
print("=" * 70)

# Use average of top 3 models for pseudo-labels
top3_avg = (avg_test['lgb'] + avg_test['xgb'] + avg_test['cat']) / 3

# Select high-confidence predictions
high_conf_mask_pos = top3_avg > 0.85
high_conf_mask_neg = top3_avg < 0.15
n_pos = high_conf_mask_pos.sum()
n_neg = high_conf_mask_neg.sum()
print(f"High-confidence positives: {n_pos}, negatives: {n_neg}")

pl_oof = {m: np.zeros(len(X)) for m in ['lgb', 'xgb', 'cat']}
pl_test = {m: np.zeros(len(X_test)) for m in ['lgb', 'xgb', 'cat']}

if n_pos + n_neg > 50:
    pl_indices = np.where(high_conf_mask_pos | high_conf_mask_neg)[0]
    pl_labels = (top3_avg[pl_indices] > 0.5).astype(int)
    print(f"Pseudo-labeled: {len(pl_indices)} samples")

    for seed in SEEDS:
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
        for tr_idx, va_idx in skf.split(X, y):
            X_tr, X_va = X[tr_idx], X[va_idx]
            y_tr, y_va = y[tr_idx], y[va_idx]

            # Combine train fold with pseudo-labels (excluding val fold)
            X_pseudo = np.vstack([X_tr, X_test[pl_indices]])
            y_pseudo = np.concatenate([y_tr, pl_labels])

            # LightGBM
            p = {'objective': 'binary', 'metric': 'auc', 'verbosity': -1, 'n_jobs': -1,
                 'random_state': seed, **best_lgb_params}
            m = lgb.LGBMClassifier(**p)
            m.fit(X_pseudo, y_pseudo, eval_set=[(X_va, y_va)],
                  callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
            pl_oof['lgb'][va_idx] = m.predict_proba(X_va)[:, 1]
            pl_test['lgb'] += m.predict_proba(X_test)[:, 1] / N_FOLDS

            # XGBoost
            p = {'objective': 'binary:logistic', 'eval_metric': 'auc', 'verbosity': 0,
                 'n_jobs': -1, 'random_state': seed, **best_xgb_params}
            m = xgb.XGBClassifier(**p)
            m.fit(X_pseudo, y_pseudo, eval_set=[(X_va, y_va)], verbose=False)
            pl_oof['xgb'][va_idx] = m.predict_proba(X_va)[:, 1]
            pl_test['xgb'] += m.predict_proba(X_test)[:, 1] / N_FOLDS

            # CatBoost
            p = {'loss_function': 'Logloss', 'eval_metric': 'AUC', 'verbose': 0,
                 'random_seed': seed, **best_cat_params}
            m = CatBoostClassifier(**p)
            m.fit(X_pseudo, y_pseudo, eval_set=(X_va, y_va), early_stopping_rounds=50)
            pl_oof['cat'][va_idx] = m.predict_proba(X_va)[:, 1]
            pl_test['cat'] += m.predict_proba(X_test)[:, 1] / N_FOLDS

    for name in ['lgb', 'xgb', 'cat']:
        auc = roc_auc_score(y, pl_oof[name])
        print(f"  PL {name:8s}: {auc:.5f}")
else:
    print("Not enough high-confidence samples for pseudo-labeling")
    pl_oof = avg_oof.copy()
    pl_test = avg_test.copy()

# =============================================================================
# 7. BLENDING — MULTIPLE STRATEGIES
# =============================================================================
print("\n" + "=" * 70)
print("BLENDING")
print("=" * 70)

results = {}

# --- Strategy 1: Simple average of ALL models (no optimization) ---
simple_avg_oof = np.mean([avg_oof[m] for m in MODEL_NAMES], axis=0)
simple_avg_test = np.mean([avg_test[m] for m in MODEL_NAMES], axis=0)
results['simple_all_avg'] = (roc_auc_score(y, simple_avg_oof), simple_avg_test)

# --- Strategy 2: Top-3 simple average ---
top3_oof = (avg_oof['lgb'] + avg_oof['xgb'] + avg_oof['cat']) / 3
top3_test = (avg_test['lgb'] + avg_test['xgb'] + avg_test['cat']) / 3
results['top3_avg'] = (roc_auc_score(y, top3_oof), top3_test)

# --- Strategy 3: Top-3 + diversity models average ---
div_models = ['lgb', 'xgb', 'cat', 'et', 'hgb']
div_oof = np.mean([avg_oof[m] for m in div_models], axis=0)
div_test = np.mean([avg_test[m] for m in div_models], axis=0)
results['div5_avg'] = (roc_auc_score(y, div_oof), div_test)

# --- Strategy 4: Log-odds average (top3) ---
def safe_log_odds(p):
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return np.log(p / (1 - p))

lo_top3_oof = np.mean([safe_log_odds(avg_oof[m]) for m in ['lgb', 'xgb', 'cat']], axis=0)
lo_top3_test = np.mean([safe_log_odds(avg_test[m]) for m in ['lgb', 'xgb', 'cat']], axis=0)
lo_top3_oof_prob = 1 / (1 + np.exp(-lo_top3_oof))
lo_top3_test_prob = 1 / (1 + np.exp(-lo_top3_test))
results['logodds_top3'] = (roc_auc_score(y, lo_top3_oof_prob), lo_top3_test_prob)

# --- Strategy 5: Log-odds average (all models) ---
lo_all_oof = np.mean([safe_log_odds(avg_oof[m]) for m in MODEL_NAMES], axis=0)
lo_all_test = np.mean([safe_log_odds(avg_test[m]) for m in MODEL_NAMES], axis=0)
lo_all_oof_prob = 1 / (1 + np.exp(-lo_all_oof))
lo_all_test_prob = 1 / (1 + np.exp(-lo_all_test))
results['logodds_all'] = (roc_auc_score(y, lo_all_oof_prob), lo_all_test_prob)

# --- Strategy 6: Pseudo-label top3 ---
pl_top3_oof = (pl_oof['lgb'] + pl_oof['xgb'] + pl_oof['cat']) / 3
pl_top3_test = (pl_test['lgb'] + pl_test['xgb'] + pl_test['cat']) / 3
results['pl_top3'] = (roc_auc_score(y, pl_top3_oof), pl_top3_test)

# --- Strategy 7: Pseudo-label log-odds ---
pl_lo_oof = np.mean([safe_log_odds(pl_oof[m]) for m in ['lgb', 'xgb', 'cat']], axis=0)
pl_lo_test = np.mean([safe_log_odds(pl_test[m]) for m in ['lgb', 'xgb', 'cat']], axis=0)
pl_lo_oof_prob = 1 / (1 + np.exp(-pl_lo_oof))
pl_lo_test_prob = 1 / (1 + np.exp(-pl_lo_test))
results['pl_logodds'] = (roc_auc_score(y, pl_lo_oof_prob), pl_lo_test_prob)

# --- Strategy 8: Blend original + pseudo-label predictions ---
blend_oof = 0.5 * top3_oof + 0.5 * pl_top3_oof
blend_test = 0.5 * top3_test + 0.5 * pl_top3_test
results['blend_pl50'] = (roc_auc_score(y, blend_oof), blend_test)

# --- Strategy 9: Proper stacking with CV ---
print("\n--- CV Stacking ---")
stack_features = ['lgb', 'xgb', 'cat', 'et', 'hgb', 'rf', 'lr', 'mlp']
stack_oof = np.column_stack([avg_oof[m] for m in stack_features])
stack_test = np.column_stack([avg_test[m] for m in stack_features])

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
stack_oof_pred = np.zeros(len(X))
stack_test_pred = np.zeros(len(X_test))

for tr_idx, va_idx in skf.split(stack_oof, y):
    meta = LogisticRegression(C=1.0, max_iter=5000, random_state=42)
    meta.fit(stack_oof[tr_idx], y[tr_idx])
    stack_oof_pred[va_idx] = meta.predict_proba(stack_oof[va_idx])[:, 1]
    stack_test_pred += meta.predict_proba(stack_test)[:, 1] / N_FOLDS

results['stacking_all'] = (roc_auc_score(y, stack_oof_pred), stack_test_pred)

# --- Strategy 10: Stacking on top-3 only ---
stack3_features = ['lgb', 'xgb', 'cat']
stack3_oof = np.column_stack([avg_oof[m] for m in stack3_features])
stack3_test = np.column_stack([avg_test[m] for m in stack3_features])

stack3_oof_pred = np.zeros(len(X))
stack3_test_pred = np.zeros(len(X_test))

for tr_idx, va_idx in skf.split(stack3_oof, y):
    meta = LogisticRegression(C=1.0, max_iter=5000, random_state=42)
    meta.fit(stack3_oof[tr_idx], y[tr_idx])
    stack3_oof_pred[va_idx] = meta.predict_proba(stack3_oof[va_idx])[:, 1]
    stack3_test_pred += meta.predict_proba(stack3_test)[:, 1] / N_FOLDS

results['stacking_top3'] = (roc_auc_score(y, stack3_oof_pred), stack3_test_pred)

# --- Strategy 11: Rank-based average (no optimization) ---
rank_avg_oof = np.mean([rankdata(avg_oof[m]) / len(y) for m in MODEL_NAMES], axis=0)
rank_avg_test = np.mean([rankdata(avg_test[m]) / len(X_test) for m in MODEL_NAMES], axis=0)
results['rank_avg_all'] = (roc_auc_score(y, rank_avg_oof), rank_avg_test)

# --- Strategy 12: Top-3 rank average ---
rank_top3_oof = np.mean([rankdata(avg_oof[m]) / len(y) for m in ['lgb', 'xgb', 'cat']], axis=0)
rank_top3_test = np.mean([rankdata(avg_test[m]) / len(X_test) for m in ['lgb', 'xgb', 'cat']], axis=0)
results['rank_avg_top3'] = (roc_auc_score(y, rank_top3_oof), rank_top3_test)

# =============================================================================
# 8. SELECT BEST AND GENERATE SUBMISSION
# =============================================================================
print("\n" + "=" * 70)
print("RESULTS")
print("=" * 70)

best_method = None
best_auc = 0
for name, (auc, preds) in sorted(results.items(), key=lambda x: -x[1][0]):
    points = max(0, auc - 0.60) / 0.40
    marker = " ***" if auc > best_auc else ""
    print(f"  {name:25s}: ROC-AUC={auc:.5f}  points={points:.5f}{marker}")
    if auc > best_auc:
        best_auc = auc
        best_method = name

# Save ALL submissions
for name, (auc, preds) in results.items():
    sub = pd.DataFrame({'id': test_ids, 'retention': preds})
    sub.to_csv(f'submission_{name}.csv', index=False)

# Main submission = best method
best_preds = results[best_method][1]
submission = pd.DataFrame({'id': test_ids, 'retention': best_preds})
submission.to_csv('submission.csv', index=False)

print(f"\n{'='*70}")
print(f"BEST: {best_method}")
print(f"ROC-AUC: {best_auc:.5f}")
print(f"Points: {max(0, best_auc - 0.60) / 0.40:.5f}")
print(f"Saved: submission.csv")
print(f"Shape: {submission.shape}")
print(f"Range: [{submission['retention'].min():.4f}, {submission['retention'].max():.4f}]")
print(f"{'='*70}")
