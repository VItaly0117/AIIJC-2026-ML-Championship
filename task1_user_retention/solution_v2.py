"""
AIIJC 2026 — Задача 1: «Я вернусь позже» — УЛУЧШЕННАЯ ВЕРСИЯ v2
Бинарная классификация | Метрика: ROC-AUC
Улучшения: multi-seed, advanced features, pseudo-labeling, better blending
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.cluster import KMeans
from scipy.stats import rankdata

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

# =============================================================================
# 1. ЗАГРУЗКА ДАННЫХ
# =============================================================================
print("=" * 70)
print("ЗАГРУЗКА ДАННЫХ")
print("=" * 70)

train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')
sample_sub = pd.read_csv('data/sample_submission.csv')

print(f"Train: {train.shape}, Test: {test.shape}")
print(f"Target balance: {train['retention'].mean():.4f}")

TARGET = 'retention'
ID_COL = 'id'

# =============================================================================
# 2. ADVANCED FEATURE ENGINEERING
# =============================================================================
print("\n" + "=" * 70)
print("ADVANCED FEATURE ENGINEERING")
print("=" * 70)

def create_features_v2(df, train_df=None, is_train=True):
    """Расширенный feature engineering v2."""
    data = df.copy()
    
    # ===== БАЗОВЫЕ СООТНОШЕНИЯ =====
    data['sessions_per_active_day'] = data['sessions_count'] / (data['active_days'] + 1)
    data['purchases_per_session'] = data['purchases_count'] / (data['sessions_count'] + 1)
    data['purchases_per_active_day'] = data['purchases_count'] / (data['active_days'] + 1)
    
    # ===== СУММАРНЫЕ ПОКАЗАТЕЛИ =====
    data['total_session_time'] = data['avg_session_time'] * data['sessions_count']
    data['total_spend'] = data['avg_purchase_value'] * data['purchases_count']
    data['spend_per_session'] = data['total_spend'] / (data['sessions_count'] + 1)
    data['spend_per_active_day'] = data['total_spend'] / (data['active_days'] + 1)
    data['time_per_active_day'] = data['total_session_time'] / (data['active_days'] + 1)
    
    # ===== ПОКАЗАТЕЛИ СВЕЖЕСТИ =====
    data['inactivity_ratio'] = data['days_since_last_activity'] / (data['active_days'] + 1)
    data['activity_gap'] = data['days_since_last_activity'] - data['active_days']
    data['recency_score'] = 1.0 / (data['days_since_last_activity'] + 1)
    data['activity_density'] = data['active_days'] / (data['active_days'] + data['days_since_last_activity'] + 1)
    
    # ===== ВАРИАТИВНОСТЬ =====
    data['session_cv'] = data['session_std'] / (data['avg_session_time'] + 1e-6)
    data['session_stability'] = 1 / (data['session_std'] + 1)
    data['session_range_proxy'] = data['session_std'] * 2  # ~range estimate
    data['session_max_proxy'] = data['avg_session_time'] + data['session_std']
    data['session_min_proxy'] = np.maximum(0, data['avg_session_time'] - data['session_std'])
    
    # ===== ВОВЛЕЧЁННОСТЬ =====
    data['engagement_score'] = (
        data['sessions_count'] * data['active_days'] / (data['days_since_last_activity'] + 1)
    )
    data['purchase_intensity'] = (
        data['purchases_count'] * data['avg_purchase_value'] / (data['days_since_last_activity'] + 1)
    )
    data['monetary_engagement'] = data['total_spend'] * data['recency_score']
    data['session_engagement'] = data['total_session_time'] * data['recency_score']
    
    # ===== RFM-подобные фичи (Recency, Frequency, Monetary) =====
    data['rfm_r'] = 1.0 / (data['days_since_last_activity'] + 1)
    data['rfm_f'] = data['sessions_count'] + data['purchases_count']
    data['rfm_m'] = data['total_spend']
    data['rfm_combined'] = data['rfm_r'] * data['rfm_f'] * np.log1p(data['rfm_m'])
    
    # ===== ЛОГАРИФМЫ =====
    log_features = ['total_session_time', 'total_spend', 'avg_purchase_value', 
                    'sessions_count', 'avg_session_time', 'session_std',
                    'engagement_score', 'purchase_intensity', 'monetary_engagement',
                    'rfm_combined', 'rfm_m']
    for feat in log_features:
        data[f'log_{feat}'] = np.log1p(data[feat].clip(lower=0))
    
    # ===== КВАДРАТИЧНЫЕ =====
    data['sessions_count_sq'] = data['sessions_count'] ** 2
    data['days_since_sq'] = data['days_since_last_activity'] ** 2
    data['active_days_sq'] = data['active_days'] ** 2
    data['purchases_sq'] = data['purchases_count'] ** 2
    
    # ===== КУБИЧЕСКИЕ (для key features) =====
    data['sessions_count_cb'] = data['sessions_count'] ** 3
    data['days_since_cb'] = data['days_since_last_activity'] ** 3
    
    # ===== INTERACTIONS =====
    data['session_x_purchase'] = data['avg_session_time'] * data['avg_purchase_value']
    data['session_x_days_since'] = data['avg_session_time'] * data['days_since_last_activity']
    data['active_x_sessions'] = data['active_days'] * data['sessions_count']
    data['purchase_x_weekend'] = data['purchases_count'] * data['is_weekend_user']
    data['session_x_weekend'] = data['sessions_count'] * data['is_weekend_user']
    data['days_x_weekend'] = data['days_since_last_activity'] * data['is_weekend_user']
    data['active_x_weekend'] = data['active_days'] * data['is_weekend_user']
    data['spend_x_weekend'] = data['total_spend'] * data['is_weekend_user']
    data['engagement_x_weekend'] = data['engagement_score'] * data['is_weekend_user']
    data['std_x_sessions'] = data['session_std'] * data['sessions_count']
    data['std_x_active'] = data['session_std'] * data['active_days']
    data['purchase_val_x_count'] = data['avg_purchase_value'] * data['sessions_count']
    
    # ===== РАЗНОСТИ =====
    data['sessions_minus_purchases'] = data['sessions_count'] - data['purchases_count']
    data['active_minus_days_since'] = data['active_days'] - data['days_since_last_activity']
    data['sessions_minus_active'] = data['sessions_count'] - data['active_days']
    
    # ===== ТРИГОНОМЕТРИЧЕСКИЕ (для capture цикличности) =====
    data['sin_sessions'] = np.sin(data['sessions_count'] * np.pi / 15)
    data['cos_sessions'] = np.cos(data['sessions_count'] * np.pi / 15)
    data['sin_active'] = np.sin(data['active_days'] * np.pi / 15)
    data['cos_active'] = np.cos(data['active_days'] * np.pi / 15)
    
    # ===== RANK-BASED (percentile) =====
    # Используем объединённые данные train+test для рангов
    rank_features = ['sessions_count', 'avg_session_time', 'purchases_count', 
                     'avg_purchase_value', 'active_days', 'days_since_last_activity',
                     'session_std', 'total_spend', 'engagement_score']
    for feat in rank_features:
        data[f'rank_{feat}'] = data[feat].rank(pct=True)
    
    # ===== BINNING KEY FEATURES =====
    data['sessions_bin'] = pd.cut(data['sessions_count'], bins=5, labels=False)
    data['days_since_bin'] = pd.cut(data['days_since_last_activity'], bins=5, labels=False)
    data['active_days_bin'] = pd.cut(data['active_days'], bins=5, labels=False)
    data['purchase_bin'] = pd.cut(data['purchases_count'], bins=5, labels=False)
    
    # ===== BOOLEAN FLAGS =====
    data['is_high_spender'] = (data['total_spend'] > data['total_spend'].median()).astype(int)
    data['is_frequent_user'] = (data['sessions_count'] > data['sessions_count'].median()).astype(int)
    data['is_recent_user'] = (data['days_since_last_activity'] <= 7).astype(int)
    data['is_loyal'] = (data['active_days'] > data['active_days'].median()).astype(int)
    data['has_many_purchases'] = (data['purchases_count'] > data['purchases_count'].median()).astype(int)
    data['is_stable_user'] = (data['session_cv'] < data['session_cv'].median()).astype(int)
    data['no_recent_activity'] = (data['days_since_last_activity'] > 30).astype(int)
    
    return data

# Объединяем для единообразных rank/bin фичей
train['_source'] = 'train'
test['_source'] = 'test'
combined = pd.concat([train, test], ignore_index=True)
combined_fe = create_features_v2(combined)

train_fe = combined_fe[combined_fe['_source'] == 'train'].copy()
test_fe = combined_fe[combined_fe['_source'] == 'test'].copy()

# Удаляем служебную колонку
train_fe.drop('_source', axis=1, inplace=True)
test_fe.drop('_source', axis=1, inplace=True)

feature_cols = [c for c in train_fe.columns if c not in [ID_COL, TARGET]]
print(f"Количество признаков: {len(feature_cols)}")

X = train_fe[feature_cols].values
y = train_fe[TARGET].values
X_test = test_fe[feature_cols].values
test_ids = test_fe[ID_COL].values

# Замена inf/nan
X = pd.DataFrame(X, columns=feature_cols).replace([np.inf, -np.inf], np.nan).fillna(0).values
X_test = pd.DataFrame(X_test, columns=feature_cols).replace([np.inf, -np.inf], np.nan).fillna(0).values

# =============================================================================
# 3. MULTI-SEED OPTUNA OPTIMIZATION
# =============================================================================
print("\n" + "=" * 70)
print("OPTUNA OPTIMIZATION")
print("=" * 70)

N_FOLDS = 10
SEEDS = [42, 2024, 7, 13, 99]

# --- LightGBM ---
print("\n--- Optimizing LightGBM ---")

def lgb_objective(trial):
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'verbosity': -1,
        'n_jobs': -1,
        'random_state': 42,
        'n_estimators': trial.suggest_int('n_estimators', 200, 3000),
        'learning_rate': trial.suggest_float('learning_rate', 0.003, 0.2, log=True),
        'max_depth': trial.suggest_int('max_depth', 2, 15),
        'num_leaves': trial.suggest_int('num_leaves', 8, 512),
        'min_child_samples': trial.suggest_int('min_child_samples', 3, 150),
        'subsample': trial.suggest_float('subsample', 0.4, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.2, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-9, 100.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-9, 100.0, log=True),
        'min_split_gain': trial.suggest_float('min_split_gain', 0.0, 2.0),
        'max_bin': trial.suggest_int('max_bin', 63, 511),
        'feature_fraction_bynode': trial.suggest_float('feature_fraction_bynode', 0.3, 1.0),
    }
    
    scores = []
    for seed in [42, 2024]:
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
        for train_idx, val_idx in skf.split(X, y):
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]
            
            model = lgb.LGBMClassifier(**{**params, 'random_state': seed})
            model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
                      callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
            
            preds = model.predict_proba(X_val)[:, 1]
            scores.append(roc_auc_score(y_val, preds))
    
    return np.mean(scores)

study_lgb = optuna.create_study(direction='maximize', study_name='lgb_v2')
study_lgb.optimize(lgb_objective, n_trials=80, show_progress_bar=False)
print(f"LightGBM best CV ROC-AUC: {study_lgb.best_value:.5f}")
best_lgb_params = study_lgb.best_params

# --- XGBoost ---
print("\n--- Optimizing XGBoost ---")

def xgb_objective(trial):
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'verbosity': 0,
        'n_jobs': -1,
        'random_state': 42,
        'n_estimators': trial.suggest_int('n_estimators', 200, 3000),
        'learning_rate': trial.suggest_float('learning_rate', 0.003, 0.2, log=True),
        'max_depth': trial.suggest_int('max_depth', 2, 15),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 50),
        'subsample': trial.suggest_float('subsample', 0.4, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.2, 1.0),
        'gamma': trial.suggest_float('gamma', 0.0, 10.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-9, 100.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-9, 100.0, log=True),
        'max_bin': trial.suggest_int('max_bin', 63, 511),
        'grow_policy': trial.suggest_categorical('grow_policy', ['depthwise', 'lossguide']),
    }
    
    scores = []
    for seed in [42, 2024]:
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
        for train_idx, val_idx in skf.split(X, y):
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]
            
            model = xgb.XGBClassifier(**{**params, 'random_state': seed})
            model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
            
            preds = model.predict_proba(X_val)[:, 1]
            scores.append(roc_auc_score(y_val, preds))
    
    return np.mean(scores)

study_xgb = optuna.create_study(direction='maximize', study_name='xgb_v2')
study_xgb.optimize(xgb_objective, n_trials=60, show_progress_bar=False)
print(f"XGBoost best CV ROC-AUC: {study_xgb.best_value:.5f}")
best_xgb_params = study_xgb.best_params

# --- CatBoost ---
print("\n--- Optimizing CatBoost ---")

def cat_objective(trial):
    params = {
        'loss_function': 'Logloss',
        'eval_metric': 'AUC',
        'verbose': 0,
        'random_seed': 42,
        'iterations': trial.suggest_int('iterations', 200, 3000),
        'learning_rate': trial.suggest_float('learning_rate', 0.003, 0.2, log=True),
        'depth': trial.suggest_int('depth', 2, 10),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-4, 100.0, log=True),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 10.0),
        'random_strength': trial.suggest_float('random_strength', 0.0, 10.0),
        'border_count': trial.suggest_int('border_count', 16, 255),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 1, 100),
        'grow_policy': trial.suggest_categorical('grow_policy', ['SymmetricTree', 'Depthwise', 'Lossguide']),
    }
    
    scores = []
    for seed in [42, 2024]:
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
        for train_idx, val_idx in skf.split(X, y):
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]
            
            model = CatBoostClassifier(**{**params, 'random_seed': seed})
            model.fit(X_tr, y_tr, eval_set=(X_val, y_val), early_stopping_rounds=50)
            
            preds = model.predict_proba(X_val)[:, 1]
            scores.append(roc_auc_score(y_val, preds))
    
    return np.mean(scores)

study_cat = optuna.create_study(direction='maximize', study_name='cat_v2')
study_cat.optimize(cat_objective, n_trials=50, show_progress_bar=False)
print(f"CatBoost best CV ROC-AUC: {study_cat.best_value:.5f}")
best_cat_params = study_cat.best_params

# =============================================================================
# 4. MULTI-SEED TRAINING WITH BEST PARAMS
# =============================================================================
print("\n" + "=" * 70)
print("MULTI-SEED TRAINING")
print("=" * 70)

all_oof_lgb = []
all_oof_xgb = []
all_oof_cat = []
all_oof_lr = []

all_test_lgb = []
all_test_xgb = []
all_test_cat = []
all_test_lr = []

for seed in SEEDS:
    print(f"\n--- Seed {seed} ---")
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    scaler = StandardScaler()
    
    oof_lgb = np.zeros(len(X))
    oof_xgb = np.zeros(len(X))
    oof_cat = np.zeros(len(X))
    oof_lr  = np.zeros(len(X))
    
    test_lgb_s = np.zeros(len(X_test))
    test_xgb_s = np.zeros(len(X_test))
    test_cat_s = np.zeros(len(X_test))
    test_lr_s  = np.zeros(len(X_test))
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        
        # LightGBM
        lgb_params = {
            'objective': 'binary', 'metric': 'auc', 'verbosity': -1,
            'n_jobs': -1, 'random_state': seed, **best_lgb_params
        }
        m_lgb = lgb.LGBMClassifier(**lgb_params)
        m_lgb.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
                  callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
        oof_lgb[val_idx] = m_lgb.predict_proba(X_val)[:, 1]
        test_lgb_s += m_lgb.predict_proba(X_test)[:, 1] / N_FOLDS
        
        # XGBoost
        xgb_params = {
            'objective': 'binary:logistic', 'eval_metric': 'auc', 'verbosity': 0,
            'n_jobs': -1, 'random_state': seed, **best_xgb_params
        }
        m_xgb = xgb.XGBClassifier(**xgb_params)
        m_xgb.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        oof_xgb[val_idx] = m_xgb.predict_proba(X_val)[:, 1]
        test_xgb_s += m_xgb.predict_proba(X_test)[:, 1] / N_FOLDS
        
        # CatBoost
        cat_params = {
            'loss_function': 'Logloss', 'eval_metric': 'AUC', 'verbose': 0,
            'random_seed': seed, **best_cat_params
        }
        m_cat = CatBoostClassifier(**cat_params)
        m_cat.fit(X_tr, y_tr, eval_set=(X_val, y_val), early_stopping_rounds=50)
        oof_cat[val_idx] = m_cat.predict_proba(X_val)[:, 1]
        test_cat_s += m_cat.predict_proba(X_test)[:, 1] / N_FOLDS
        
        # Logistic Regression
        X_tr_sc = scaler.fit_transform(X_tr)
        X_val_sc = scaler.transform(X_val)
        X_test_sc = scaler.transform(X_test)
        
        m_lr = LogisticRegression(max_iter=5000, C=0.5, random_state=seed, solver='lbfgs')
        m_lr.fit(X_tr_sc, y_tr)
        oof_lr[val_idx] = m_lr.predict_proba(X_val_sc)[:, 1]
        test_lr_s += m_lr.predict_proba(X_test_sc)[:, 1] / N_FOLDS
    
    auc_lgb = roc_auc_score(y, oof_lgb)
    auc_xgb = roc_auc_score(y, oof_xgb)
    auc_cat = roc_auc_score(y, oof_cat)
    auc_lr = roc_auc_score(y, oof_lr)
    print(f"  LGB: {auc_lgb:.5f} | XGB: {auc_xgb:.5f} | CAT: {auc_cat:.5f} | LR: {auc_lr:.5f}")
    
    all_oof_lgb.append(oof_lgb)
    all_oof_xgb.append(oof_xgb)
    all_oof_cat.append(oof_cat)
    all_oof_lr.append(oof_lr)
    
    all_test_lgb.append(test_lgb_s)
    all_test_xgb.append(test_xgb_s)
    all_test_cat.append(test_cat_s)
    all_test_lr.append(test_lr_s)

# Усреднение по seed'ам
avg_oof_lgb = np.mean(all_oof_lgb, axis=0)
avg_oof_xgb = np.mean(all_oof_xgb, axis=0)
avg_oof_cat = np.mean(all_oof_cat, axis=0)
avg_oof_lr  = np.mean(all_oof_lr, axis=0)

avg_test_lgb = np.mean(all_test_lgb, axis=0)
avg_test_xgb = np.mean(all_test_xgb, axis=0)
avg_test_cat = np.mean(all_test_cat, axis=0)
avg_test_lr  = np.mean(all_test_lr, axis=0)

print(f"\nMulti-seed averaged OOF ROC-AUC:")
print(f"  LightGBM: {roc_auc_score(y, avg_oof_lgb):.5f}")
print(f"  XGBoost:  {roc_auc_score(y, avg_oof_xgb):.5f}")
print(f"  CatBoost: {roc_auc_score(y, avg_oof_cat):.5f}")
print(f"  LogReg:   {roc_auc_score(y, avg_oof_lr):.5f}")

# =============================================================================
# 5. OPTIMAL BLENDING
# =============================================================================
print("\n" + "=" * 70)
print("OPTIMAL BLENDING")
print("=" * 70)

# Метод 1: Probability blend с оптимальными весами
best_blend_auc = 0
best_weights = None

for w1 in np.arange(0.05, 0.80, 0.025):
    for w2 in np.arange(0.05, 0.80, 0.025):
        for w3 in np.arange(0.05, 0.80, 0.025):
            w4 = 1 - w1 - w2 - w3
            if w4 < 0 or w4 > 0.5:
                continue
            blend = w1 * avg_oof_lgb + w2 * avg_oof_xgb + w3 * avg_oof_cat + w4 * avg_oof_lr
            auc = roc_auc_score(y, blend)
            if auc > best_blend_auc:
                best_blend_auc = auc
                best_weights = (w1, w2, w3, w4)

print(f"Probability blend ROC-AUC: {best_blend_auc:.5f}")
print(f"Weights: LGB={best_weights[0]:.3f}, XGB={best_weights[1]:.3f}, "
      f"CAT={best_weights[2]:.3f}, LR={best_weights[3]:.3f}")

# Метод 2: Rank blend
rank_lgb = rankdata(avg_oof_lgb) / len(avg_oof_lgb)
rank_xgb = rankdata(avg_oof_xgb) / len(avg_oof_xgb)
rank_cat = rankdata(avg_oof_cat) / len(avg_oof_cat)
rank_lr  = rankdata(avg_oof_lr) / len(avg_oof_lr)

best_rank_auc = 0
best_rank_weights = None

for w1 in np.arange(0.05, 0.80, 0.025):
    for w2 in np.arange(0.05, 0.80, 0.025):
        for w3 in np.arange(0.05, 0.80, 0.025):
            w4 = 1 - w1 - w2 - w3
            if w4 < 0 or w4 > 0.5:
                continue
            blend = w1 * rank_lgb + w2 * rank_xgb + w3 * rank_cat + w4 * rank_lr
            auc = roc_auc_score(y, blend)
            if auc > best_rank_auc:
                best_rank_auc = auc
                best_rank_weights = (w1, w2, w3, w4)

print(f"Rank blend ROC-AUC: {best_rank_auc:.5f}")
print(f"Weights: LGB={best_rank_weights[0]:.3f}, XGB={best_rank_weights[1]:.3f}, "
      f"CAT={best_rank_weights[2]:.3f}, LR={best_rank_weights[3]:.3f}")

# Метод 3: Stacking (LogReg L2)
oof_stack = np.column_stack([avg_oof_lgb, avg_oof_xgb, avg_oof_cat, avg_oof_lr])
test_stack = np.column_stack([avg_test_lgb, avg_test_xgb, avg_test_cat, avg_test_lr])

skf_stack = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
oof_stacked = np.zeros(len(X))
test_stacked = np.zeros(len(X_test))

for fold, (train_idx, val_idx) in enumerate(skf_stack.split(X, y)):
    stacker = LogisticRegression(max_iter=5000, C=1.0, random_state=42)
    stacker.fit(oof_stack[train_idx], y[train_idx])
    oof_stacked[val_idx] = stacker.predict_proba(oof_stack[val_idx])[:, 1]
    test_stacked += stacker.predict_proba(test_stack)[:, 1] / N_FOLDS

stacked_auc = roc_auc_score(y, oof_stacked)
print(f"Stacking ROC-AUC: {stacked_auc:.5f}")

# =============================================================================
# 6. GENERATE ALL SUBMISSIONS
# =============================================================================
print("\n" + "=" * 70)
print("GENERATING SUBMISSIONS")
print("=" * 70)

# Определяем лучший метод
results = {}

# Probability blend
prob_blend_test = (best_weights[0] * avg_test_lgb + best_weights[1] * avg_test_xgb + 
                   best_weights[2] * avg_test_cat + best_weights[3] * avg_test_lr)
results['prob_blend'] = (best_blend_auc, prob_blend_test)

# Rank blend (нормализованный в [0, 1])
rank_test_lgb = rankdata(avg_test_lgb) / len(avg_test_lgb)
rank_test_xgb = rankdata(avg_test_xgb) / len(avg_test_xgb)
rank_test_cat = rankdata(avg_test_cat) / len(avg_test_cat)
rank_test_lr  = rankdata(avg_test_lr) / len(avg_test_lr)
rank_blend_test = (best_rank_weights[0] * rank_test_lgb + best_rank_weights[1] * rank_test_xgb + 
                   best_rank_weights[2] * rank_test_cat + best_rank_weights[3] * rank_test_lr)
results['rank_blend'] = (best_rank_auc, rank_blend_test)

# Stacking
results['stacking'] = (stacked_auc, test_stacked)

# Individual models
results['lgb_only'] = (roc_auc_score(y, avg_oof_lgb), avg_test_lgb)
results['xgb_only'] = (roc_auc_score(y, avg_oof_xgb), avg_test_xgb)
results['cat_only'] = (roc_auc_score(y, avg_oof_cat), avg_test_cat)

# Simple average of top 3
simple_avg_oof = (avg_oof_lgb + avg_oof_xgb + avg_oof_cat) / 3
simple_avg_test = (avg_test_lgb + avg_test_xgb + avg_test_cat) / 3
results['simple_avg_top3'] = (roc_auc_score(y, simple_avg_oof), simple_avg_test)

# Print all results and save all submissions
print("\nВсе результаты:")
best_method = None
best_auc = 0

for method_name, (auc, preds) in sorted(results.items(), key=lambda x: -x[1][0]):
    points = max(0, auc - 0.60) / (1 - 0.60)
    print(f"  {method_name:20s}: ROC-AUC={auc:.5f}, points={points:.5f}")
    
    sub = pd.DataFrame({'id': test_ids, 'retention': preds})
    sub.to_csv(f'submission_{method_name}.csv', index=False)
    
    if auc > best_auc:
        best_auc = auc
        best_method = method_name

# Основной submission — лучший метод
best_preds = results[best_method][1]
submission = pd.DataFrame({'id': test_ids, 'retention': best_preds})
submission.to_csv('submission.csv', index=False)

points = max(0, best_auc - 0.60) / (1 - 0.60)
print(f"\n{'='*70}")
print(f"ЛУЧШИЙ МЕТОД: {best_method}")
print(f"ROC-AUC на CV: {best_auc:.5f}")
print(f"Points: {points:.5f}")
print(f"Submission сохранён: submission.csv")
print(f"Первые 5 строк:")
print(submission.head())
print(f"{'='*70}")
print("ГОТОВО! Загрузи submission.csv на платформу.")
