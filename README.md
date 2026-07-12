# 🏆 AIIJC 2026 ML Championship Solutions

Решения задач квалификационного этапа **AI Challenge 2026** (Конкурс по искусственному интеллекту для студентов). 

Этот репозиторий содержит полный набор решений для трех квалификационных задач по машинному обучению и компьютерному зрению. Наша команда разработала пайплайны, включающие продвинутый Feature Engineering, перебор гиперпараметров (Optuna), ансамблирование моделей и Test-Time Augmentation (TTA).

---

## 📁 Структура репозитория

```text
AIIJC-2026-Solutions/
├── task1_user_retention/          # Задача 1: Предсказание повторного обращения пользователя
│   ├── data/                      # Данные (train, test, sample_submission)
│   ├── submissions/               # Лучшие сабмиты
│   ├── solution.py                # Baseline решение
│   ├── solution_v2.py             # Оптимизированная версия с Optuna
│   └── solution_v3.py             # Финальная лучшая модель с Feature Selection
│
├── task2_lighting_level/          # Задача 2: Определение уровня освещенности
│   ├── data/                      # Инструкция и плейсхолдер для изображений
│   ├── solution_task2.py          # Полное решение (Ensemble 4-х сетей + TTA)
│   └── solution_task2_fast.py     # Быстрый baseline (EfficientNet-B0 + TTA)
│
└── task3_weather_classification/  # Задача 3: Классификатор погодных условий
    ├── data/                      # Инструкция и плейсхолдер для изображений
    └── solution_task3.py          # PyTorch модель (ResNet34 + TTA)
```

---

## 📊 Обзор задач и результатов

### [1. Предсказание повторного обращения (User Retention)](./task1_user_retention/)
* **Тип задачи:** Бинарная классификация табличных данных.
* **Метрика:** ROC-AUC.
* **Результат:** 
  * Кросс-валидация (10-fold CV): **0.68731 ROC-AUC**
  * Баллы на платформе: **0.21827 Points** (1-е место на лидерборде в момент отправки).
* **Стек:** LightGBM, XGBoost, CatBoost, Logistic Regression, Optuna (hyperparameter tuning), Rank & Probability Blending, Stacking.

### [2. Определение освещенности по изображениям](./task2_lighting_level/)
* **Тип задачи:** Мультиклассовая классификация изображений (3 класса: *dark*, *normal*, *bright*).
* **Метрика:** Accuracy.
* **Результат:**
  * Кросс-валидация: **~0.92+ Accuracy**
  * Баллы: **~0.86+ Points**
* **Стек:** PyTorch, torchvision, EfficientNet-B0, ResNet18, ResNet34, MobileNetV3, Albumentations/Transforms, TTA (Test-Time Augmentation), Label Smoothing.

### [3. Классификатор погодных условий](./task3_weather_classification/)
* **Тип задачи:** Мультиклассовая классификация изображений (3 класса: *rain*, *fog*, *snow*) в условиях шума и искажений.
* **Метрика:** Macro F1-score.
* **Стек:** PyTorch, ResNet34, WeightedRandomSampler (для балансировки классов), Cosine Annealing scheduler, Test-Time Augmentation (TTA).

---

## ⚙️ Требования и установка

Рекомендуется использовать Python 3.10+. Установка необходимых библиотек:

```bash
pip install pandas numpy scikit-learn xgboost lightgbm catboost optuna torch torchvision pillow scipy
```

Для каждой задачи подробные инструкции по запуску и подготовке данных находятся в соответствующих папках.

---

## 💡 Основные подходы к оптимизации
1. **Продвинутый Feature Engineering:** В табличной задаче размерность признаков была увеличена с 8 до 89 за счет создания RFM-метрик, тригонометрических и нелинейных взаимодействий, что помогло выжать максимум из бустингов.
2. **Feature Selection:** Обрезка малоэффективных фичей позволила сократить переобучение (CV-LB gap).
3. **Ансамблирование:** Использование взвешенного усреднения и стекинга второго уровня (Stacking L2 Meta-Learner) дало прирост к стабильности и качеству.
4. **TTA (Test-Time Augmentation):** Применение флипов и аффинных преобразований во время инференса на изображениях прибавило ~1.5% к Accuracy/F1.

---
*Разработано в рамках квалификационного этапа AIIJC 2026.*
