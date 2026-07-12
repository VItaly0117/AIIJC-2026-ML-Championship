"""
AIIJC 2026 — Задача 1: «Я вернусь позже» — предсказание повторного обращения пользователя
Бинарная классификация | Метрика: ROC-AUC
Автор: AI Solution
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

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

print(f"Train shape: {train.shape}")
print(f"Test shape:  {test.shape}")
print(f"Sample submission shape: {sample_sub.shape}")
print(f"\nTarget distribution:\n{train['retention'].value_counts(normalize=True)}")
print(f"\nПропуски в train:\n{train.isnull().sum()}")
print(f"\nПропуски в test:\n{test.isnull().sum()}")

# =============================================================================
# 2. EDA — БАЗОВАЯ СТАТИСТИКА
# =============================================================================
print("\n" + "=" * 70)
print("БАЗОВАЯ СТАТИСТИКА")
print("=" * 70)
print(train.describe())

# =============================================================================
# 3. FEATURE ENGINEERING
# =============================================================================
print("\n" + "=" * 70)
print("FEATURE ENGINEERING")
print("=" * 70)

TARGET = 'retention'
ID_COL = 'id'

def create_features(df):
    """Создание новых признаков из существующих."""
    data = df.copy()
    
    # --- Соотношения ---
    # Интенсивность сессий
    data['sessions_per_active_day'] = data['sessions_count'] / (data['active_days'] + 1)
    # Покупки на сессию
    data['purchases_per_session'] = data['purchases_count'] / (data['sessions_count'] + 1)
    # Покупки на активный день
    data['purchases_per_active_day'] = data['purchases_count'] / (data['active_days'] + 1)
    
    # --- Суммарные показатели ---
    # Общее время в приложении
    data['total_session_time'] = data['avg_session_time'] * data['sessions_count']
    # Общая сумма покупок
    data['total_spend'] = data['avg_purchase_value'] * data['purchases_count']
    # Средняя покупка на сессию
    data['spend_per_session'] = data['total_spend'] / (data['sessions_count'] + 1)
    # Средняя покупка на активный день
    data['spend_per_active_day'] = data['total_spend'] / (data['active_days'] + 1)
    
    # --- Показатели «свежести» ---
    # Чем больше дней с последнего визита vs активных дней — тем хуже
    data['inactivity_ratio'] = data['days_since_last_activity'] / (data['active_days'] + 1)
    # Разница между общим периодом и активными днями
    data['activity_gap'] = data['days_since_last_activity'] - data['active_days']
    
    # --- Вариативность поведения ---
    # Коэффициент вариации длительности сессий
    data['session_cv'] = data['session_std'] / (data['avg_session_time'] + 1e-6)
    # Стабильность сессий (низкий std = стабильный пользователь)
    data['session_stability'] = 1 / (data['session_std'] + 1)
    
    # --- Вовлечённость ---
    # Общий «скор» вовлечённости
    data['engagement_score'] = (
        data['sessions_count'] * data['active_days'] / (data['days_since_last_activity'] + 1)
    )
    # Покупательская активность
    data['purchase_intensity'] = (
        data['purchases_count'] * data['avg_purchase_value'] / (data['days_since_last_activity'] + 1)
    )
    
    # --- Логарифмы для skewed-признаков ---
    log_features = ['total_session_time', 'total_spend', 'avg_purchase_value', 
                    'sessions_count', 'avg_session_time', 'session_std',
                    'engagement_score', 'purchase_intensity']
    for feat in log_features:
        data[f'log_{feat}'] = np.log1p(data[feat].clip(lower=0))
    
    # --- Полиномиальные ---
    data['sessions_count_sq'] = data['sessions_count'] ** 2
    data['days_since_sq'] = data['days_since_last_activity'] ** 2
    data['active_days_sq'] = data['active_days'] ** 2
    
    # --- Interactions ---
    data['session_x_purchase'] = data['avg_session_time'] * data['avg_purchase_value']
    data['session_x_days_since'] = data['avg_session_time'] * data['days_since_last_activity']
    data['active_x_sessions'] = data['active_days'] * data['sessions_count']
    data['purchase_x_weekend'] = data['purchases_count'] * data['is_weekend_user']
    data['session_x_weekend'] = data['sessions_count'] * data['is_weekend_user']
    
    # --- Rank-based features ---
    rank_features = ['sessions_count', 'avg_session_time', 'purchases_count', 
                     'avg_purchase_value', 'active_days', 'days_since_last_activity']
    for feat in rank_features:
        data[f'rank_{feat}'] = data[feat].rank(pct=True)
    
    return data

train_fe = create_features(train)
test_fe = create_features(test)

# Список признаков для моделей
feature_cols = [c for c in train_fe.columns if c not in [ID_COL, TARGET]]
print(f"Количество признаков: {len(feature_cols)}")
print(f"Признаки: {feature_cols}")

X = train_fe[feature_cols].values
y = train_fe[TARGET].values
X_test = test_fe[feature_cols].values
test_ids = test_fe[ID_COL].values

# Замена inf на NaN, затем заполнение медианой
X = pd.DataFrame(X, columns=feature_cols).replace([np.inf, -np.inf], np.nan).fillna(0).values
X_test = pd.DataFrame(X_test, columns=feature_cols).replace([np.inf, -np.inf], np.nan).fillna(0).values

# =============================================================================
# 4. КРОСС-ВАЛИДАЦИЯ И ОБУЧЕНИЕ МОДЕЛЕЙ
# =============================================================================
print("\n" + "=" * 70)
print("ОБУЧЕНИЕ МОДЕЛЕЙ")
print("=" * 70)

N_FOLDS = 10
SEED = 42
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

# --- 4a. Optuna для LightGBM ---
print("\n--- Оптимизация LightGBM с Optuna ---")

def lgb_objective(trial):
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'verbosity': -1,
        'n_jobs': -1,
        'random_state': SEED,
        'n_estimators': trial.suggest_int('n_estimators', 300, 2000),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.15, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'num_leaves': trial.suggest_int('num_leaves', 15, 255),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'min_split_gain': trial.suggest_float('min_split_gain', 0.0, 1.0),
    }
    
    scores = []
    for train_idx, val_idx in skf.split(X, y):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        
        model = lgb.LGBMClassifier(**params)
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
                  callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
        
        preds = model.predict_proba(X_val)[:, 1]
        scores.append(roc_auc_score(y_val, preds))
    
    return np.mean(scores)

study_lgb = optuna.create_study(direction='maximize', study_name='lgb')
study_lgb.optimize(lgb_objective, n_trials=60, show_progress_bar=False)
print(f"LightGBM лучший ROC-AUC на CV: {study_lgb.best_value:.5f}")
best_lgb_params = study_lgb.best_params

# --- 4b. Optuna для XGBoost ---
print("\n--- Оптимизация XGBoost с Optuna ---")

def xgb_objective(trial):
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'verbosity': 0,
        'n_jobs': -1,
        'random_state': SEED,
        'n_estimators': trial.suggest_int('n_estimators', 300, 2000),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.15, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 30),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 1.0),
        'gamma': trial.suggest_float('gamma', 0.0, 5.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
    }
    
    scores = []
    for train_idx, val_idx in skf.split(X, y):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        
        model = xgb.XGBClassifier(**params)
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        
        preds = model.predict_proba(X_val)[:, 1]
        scores.append(roc_auc_score(y_val, preds))
    
    return np.mean(scores)

study_xgb = optuna.create_study(direction='maximize', study_name='xgb')
study_xgb.optimize(xgb_objective, n_trials=50, show_progress_bar=False)
print(f"XGBoost лучший ROC-AUC на CV: {study_xgb.best_value:.5f}")
best_xgb_params = study_xgb.best_params

# --- 4c. Optuna для CatBoost ---
print("\n--- Оптимизация CatBoost с Optuna ---")

def cat_objective(trial):
    params = {
        'loss_function': 'Logloss',
        'eval_metric': 'AUC',
        'verbose': 0,
        'random_seed': SEED,
        'iterations': trial.suggest_int('iterations', 300, 2000),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.15, log=True),
        'depth': trial.suggest_int('depth', 3, 10),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-3, 10.0, log=True),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 5.0),
        'random_strength': trial.suggest_float('random_strength', 0.0, 5.0),
        'border_count': trial.suggest_int('border_count', 32, 255),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 1, 50),
    }
    
    scores = []
    for train_idx, val_idx in skf.split(X, y):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        
        model = CatBoostClassifier(**params)
        model.fit(X_tr, y_tr, eval_set=(X_val, y_val), early_stopping_rounds=50)
        
        preds = model.predict_proba(X_val)[:, 1]
        scores.append(roc_auc_score(y_val, preds))
    
    return np.mean(scores)

study_cat = optuna.create_study(direction='maximize', study_name='cat')
study_cat.optimize(cat_objective, n_trials=40, show_progress_bar=False)
print(f"CatBoost лучший ROC-AUC на CV: {study_cat.best_value:.5f}")
best_cat_params = study_cat.best_params

# =============================================================================
# 5. ФИНАЛЬНОЕ ОБУЧЕНИЕ С ЛУЧШИМИ ПАРАМЕТРАМИ + OOF ПРЕДСКАЗАНИЯ
# =============================================================================
print("\n" + "=" * 70)
print("ФИНАЛЬНОЕ ОБУЧЕНИЕ С ЛУЧШИМИ ПАРАМЕТРАМИ")
print("=" * 70)

# OOF predictions для стекинга
oof_lgb = np.zeros(len(X))
oof_xgb = np.zeros(len(X))
oof_cat = np.zeros(len(X))
oof_lr  = np.zeros(len(X))

test_lgb = np.zeros(len(X_test))
test_xgb = np.zeros(len(X_test))
test_cat = np.zeros(len(X_test))
test_lr  = np.zeros(len(X_test))

# Скейлер для LogReg
scaler = StandardScaler()

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_tr, X_val = X[train_idx], X[val_idx]
    y_tr, y_val = y[train_idx], y[val_idx]
    
    # LightGBM
    lgb_params = {
        'objective': 'binary', 'metric': 'auc', 'verbosity': -1,
        'n_jobs': -1, 'random_state': SEED, **best_lgb_params
    }
    m_lgb = lgb.LGBMClassifier(**lgb_params)
    m_lgb.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
              callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
    oof_lgb[val_idx] = m_lgb.predict_proba(X_val)[:, 1]
    test_lgb += m_lgb.predict_proba(X_test)[:, 1] / N_FOLDS
    
    # XGBoost
    xgb_params = {
        'objective': 'binary:logistic', 'eval_metric': 'auc', 'verbosity': 0,
        'n_jobs': -1, 'random_state': SEED, **best_xgb_params
    }
    m_xgb = xgb.XGBClassifier(**xgb_params)
    m_xgb.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    oof_xgb[val_idx] = m_xgb.predict_proba(X_val)[:, 1]
    test_xgb += m_xgb.predict_proba(X_test)[:, 1] / N_FOLDS
    
    # CatBoost
    cat_params = {
        'loss_function': 'Logloss', 'eval_metric': 'AUC', 'verbose': 0,
        'random_seed': SEED, **best_cat_params
    }
    m_cat = CatBoostClassifier(**cat_params)
    m_cat.fit(X_tr, y_tr, eval_set=(X_val, y_val), early_stopping_rounds=50)
    oof_cat[val_idx] = m_cat.predict_proba(X_val)[:, 1]
    test_cat += m_cat.predict_proba(X_test)[:, 1] / N_FOLDS
    
    # Logistic Regression
    X_tr_sc = scaler.fit_transform(X_tr)
    X_val_sc = scaler.transform(X_val)
    X_test_sc = scaler.transform(X_test)
    
    m_lr = LogisticRegression(max_iter=5000, C=1.0, random_state=SEED, solver='lbfgs')
    m_lr.fit(X_tr_sc, y_tr)
    oof_lr[val_idx] = m_lr.predict_proba(X_val_sc)[:, 1]
    test_lr += m_lr.predict_proba(X_test_sc)[:, 1] / N_FOLDS
    
    print(f"  Fold {fold+1}/{N_FOLDS} | "
          f"LGB: {roc_auc_score(y_val, oof_lgb[val_idx]):.5f} | "
          f"XGB: {roc_auc_score(y_val, oof_xgb[val_idx]):.5f} | "
          f"CAT: {roc_auc_score(y_val, oof_cat[val_idx]):.5f} | "
          f"LR:  {roc_auc_score(y_val, oof_lr[val_idx]):.5f}")

print(f"\nOOF ROC-AUC:")
print(f"  LightGBM:  {roc_auc_score(y, oof_lgb):.5f}")
print(f"  XGBoost:   {roc_auc_score(y, oof_xgb):.5f}")
print(f"  CatBoost:  {roc_auc_score(y, oof_cat):.5f}")
print(f"  LogReg:    {roc_auc_score(y, oof_lr):.5f}")

# =============================================================================
# 6. СТЕКИНГ / БЛЕНДИНГ
# =============================================================================
print("\n" + "=" * 70)
print("СТЕКИНГ / ОПТИМАЛЬНЫЙ БЛЕНДИНГ")
print("=" * 70)

# Метод 1: Стекинг (LogReg 2-го уровня)
oof_stack = np.column_stack([oof_lgb, oof_xgb, oof_cat, oof_lr])
test_stack = np.column_stack([test_lgb, test_xgb, test_cat, test_lr])

stacker = LogisticRegression(max_iter=5000, C=1.0, random_state=SEED)
oof_stacked = np.zeros(len(X))

test_stacked = np.zeros(len(X_test))
for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    stacker.fit(oof_stack[train_idx], y[train_idx])
    oof_stacked[val_idx] = stacker.predict_proba(oof_stack[val_idx])[:, 1]
    test_stacked += stacker.predict_proba(test_stack)[:, 1] / N_FOLDS

stacked_auc = roc_auc_score(y, oof_stacked)
print(f"Стекинг (LogReg L2) ROC-AUC: {stacked_auc:.5f}")

# Метод 2: Оптимальное взвешенное усреднение (перебор весов)
best_blend_auc = 0
best_weights = None

for w1 in np.arange(0.1, 0.7, 0.05):
    for w2 in np.arange(0.1, 0.7, 0.05):
        for w3 in np.arange(0.1, 0.7, 0.05):
            w4 = 1 - w1 - w2 - w3
            if w4 < 0 or w4 > 0.5:
                continue
            blend = w1 * oof_lgb + w2 * oof_xgb + w3 * oof_cat + w4 * oof_lr
            auc = roc_auc_score(y, blend)
            if auc > best_blend_auc:
                best_blend_auc = auc
                best_weights = (w1, w2, w3, w4)

print(f"Оптимальный блендинг ROC-AUC: {best_blend_auc:.5f}")
print(f"Веса: LGB={best_weights[0]:.2f}, XGB={best_weights[1]:.2f}, "
      f"CAT={best_weights[2]:.2f}, LR={best_weights[3]:.2f}")

# Метод 3: Rank-based блендинг
from scipy.stats import rankdata
rank_lgb = rankdata(oof_lgb)
rank_xgb = rankdata(oof_xgb)
rank_cat = rankdata(oof_cat)
rank_lr  = rankdata(oof_lr)

best_rank_auc = 0
best_rank_weights = None

for w1 in np.arange(0.1, 0.7, 0.05):
    for w2 in np.arange(0.1, 0.7, 0.05):
        for w3 in np.arange(0.1, 0.7, 0.05):
            w4 = 1 - w1 - w2 - w3
            if w4 < 0 or w4 > 0.5:
                continue
            blend = w1 * rank_lgb + w2 * rank_xgb + w3 * rank_cat + w4 * rank_lr
            auc = roc_auc_score(y, blend)
            if auc > best_rank_auc:
                best_rank_auc = auc
                best_rank_weights = (w1, w2, w3, w4)

print(f"Rank-based блендинг ROC-AUC: {best_rank_auc:.5f}")
print(f"Веса: LGB={best_rank_weights[0]:.2f}, XGB={best_rank_weights[1]:.2f}, "
      f"CAT={best_rank_weights[2]:.2f}, LR={best_rank_weights[3]:.2f}")

# =============================================================================
# 7. ВЫБОР ЛУЧШЕГО МЕТОДА И ГЕНЕРАЦИЯ SUBMISSION
# =============================================================================
print("\n" + "=" * 70)
print("ГЕНЕРАЦИЯ SUBMISSION")
print("=" * 70)

results = {
    'stacking': (stacked_auc, test_stacked),
    'weighted_blend': (best_blend_auc, 
                       best_weights[0] * test_lgb + best_weights[1] * test_xgb + 
                       best_weights[2] * test_cat + best_weights[3] * test_lr),
    'rank_blend': (best_rank_auc,
                   best_rank_weights[0] * rankdata(test_lgb) + best_rank_weights[1] * rankdata(test_xgb) + 
                   best_rank_weights[2] * rankdata(test_cat) + best_rank_weights[3] * rankdata(test_lr)),
}

best_method = max(results, key=lambda k: results[k][0])
best_auc = results[best_method][0]
best_preds = results[best_method][1]

print(f"Лучший метод: {best_method}")
print(f"ROC-AUC на CV: {best_auc:.5f}")
print(f"Итоговый балл (points): {max(0, best_auc - 0.60) / (1 - 0.60):.5f}")

# Формирование submission
submission = pd.DataFrame({'id': test_ids, 'retention': best_preds})
submission.to_csv('submission.csv', index=False)
print(f"\nSubmission сохранён: submission.csv")
print(f"Форма: {submission.shape}")
print(f"Первые 5 строк:\n{submission.head()}")

# Также сохраним submission от каждого метода для перестраховки
for method_name, (auc, preds) in results.items():
    sub = pd.DataFrame({'id': test_ids, 'retention': preds})
    sub.to_csv(f'submission_{method_name}.csv', index=False)
    print(f"  {method_name}: ROC-AUC={auc:.5f}, saved to submission_{method_name}.csv")

print("\n" + "=" * 70)
print("ГОТОВО! Загрузи submission.csv на платформу.")
print("=" * 70)
