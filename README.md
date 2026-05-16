[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/kOqwghv0)
# ML Project — Предсказание концентрации PM2.5 в атмосфере Пекина

**Студент:** Иванов В.

**Группа:** БИВ 236


## Описание задачи

Предсказание почасовой концентрации мелкодисперсных частиц PM2.5 (мкг/м³) по метеорологическим и химическим показателям 12 станций мониторинга г. Пекин.

**Задача:** Регрессия

**Датасет:** Beijing Multi-Site Air-Quality Data (UCI / Kaggle)  
Источник: https://www.kaggle.com/datasets/sid321axn/beijing-multisite-airquality-data-set  
420 000+ строк, 12 станций, период 2013–2017.

**Целевая метрика:** RMSE (Root Mean Squared Error)  
Дополнительно: MAE, R²


## Структура репозитория

```
.
├── data
│   ├── processed/              # Обработанные данные (train/val/test.parquet)
│   └── raw/                    # Исходные CSV-файлы по станциям
├── models/                     # Сохранённые модели (.joblib)
├── notebooks/
│   ├── 01_eda.ipynb            # EDA, очистка, feature engineering, сплит
│   ├── 02_baseline.ipynb       # Baseline — Linear Regression (CP1)
│   └── 03_experiments.ipynb    # Эксперименты с 5+ моделями, Optuna, PCA (CP2)
├── presentation/               # Презентация для защиты
├── report/
│   ├── images/                 # Графики для отчёта
│   ├── experiments_table.csv   # Таблица экспериментов CP2
│   └── report.md               # Финальный отчёт
├── src/
│   ├── __init__.py
│   └── preprocessing.py        # Пайплайн предобработки данных
├── tests/
│   └── test.py                 # Тесты пайплайна (pytest)
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```


## Запуск

### Локально

```bash
# 1. Клонировать репозиторий
git clone <url>
cd <repo-name>

# 2. Создать виртуальное окружение
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate    # Windows

# 3. Установить зависимости
pip install -r requirements.txt

# 4. Скачать датасет с Kaggle и распаковать в data/raw/
# kaggle datasets download -d sid321axn/beijing-multisite-airquality-data-set
# unzip *.zip -d data/raw/

# 5. Запустить ноутбуки по порядку
jupyter notebook notebooks/01_eda.ipynb
jupyter notebook notebooks/02_baseline.ipynb
jupyter notebook notebooks/03_experiments.ipynb

# 6. Запустить тесты
pytest tests/
```

### Docker

```bash
docker-compose up
# Jupyter доступен на http://localhost:8888
```


## Данные

- `data/raw/` — исходные CSV-файлы (по одному на станцию)
- `data/processed/` — обработанные данные после `01_eda.ipynb`:
  - `train.parquet` — 2013–2015 (71%)
  - `val.parquet` — 2016 (25%)
  - `test.parquet` — 2017 (4%)


## Результаты

| Модель | RMSE (Val) | MAE (Val) | R² (Val) | Примечание |
|--------|-----------|----------|---------|------------|
| Linear Regression (CP1 baseline) | 26.21 | 17.50 | 0.853 | Сырые фичи, без лагов |
| Linear Regression (все фичи) | 13.39 | 8.06 | 0.9616 | С лагами и rolling |
| Ridge (Optuna) | 13.39 | 8.06 | 0.9616 | alpha≈0.28 |
| LightGBM + PCA (19 comp) | 13.51 | 8.48 | 0.9609 | Эксперимент PCA |
| HistGradientBoosting | 3.23 | 1.35 | 0.9978 | sklearn fast GBM |
| Random Forest (Optuna) | 3.19 | 0.32 | 0.9978 | n_est=200 |
| XGBoost (Optuna) | 3.02 | 0.97 | 0.998 | early stopping |
| Ensemble avg(RF+XGB+LGBM) | 2.57 | 0.51 | 0.9986 | Averaging |
| **LightGBM (Optuna)** | **2.30** | **0.63** | **0.9989** | **Финальная модель** |

Полная таблица экспериментов: [`report/experiments_table.csv`](report/experiments_table.csv)


## Отчёт

Финальный отчёт: [`report/report.md`](report/report.md)
