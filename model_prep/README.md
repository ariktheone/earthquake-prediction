# Earthquake Magnitude Prediction — ML Pipeline

Industry-standard machine learning pipeline for predicting earthquake magnitude from spatial, temporal, and seismological features.

## Dataset

- **Source**: `data/Earthquake_data_processed.csv`
- **Records**: 18,030 California earthquakes (NCSN, 1966–2007)
- **Target**: `Magnitude(ergs)` (range 3.0–7.39)

## Pipeline Modules

| Module | Purpose |
|--------|---------|
| `feature_engineering.py` | Transforms raw CSV → 59 engineered features (temporal, spatial, rolling stats, lag features) |
| `model_pipeline.py` | Trains & evaluates 11 ML algorithms with 70/30 split |
| `hyperparameter_tuning.py` | RandomizedSearchCV on top-K models |
| `run_pipeline.py` | End-to-end orchestrator |

## Algorithms (11)

1. Linear Regression
2. Ridge Regression
3. Lasso Regression
4. ElasticNet
5. Decision Tree
6. Random Forest
7. Gradient Boosting
8. AdaBoost
9. K-Nearest Neighbors
10. Extra Trees
11. XGBoost

## Usage

```bash
cd model_prep
source ../venv/bin/activate
python run_pipeline.py
```

## Output (`output/`)

| File | Description |
|------|-------------|
| `best_model.joblib` | Serialized best pipeline (StandardScaler + model) |
| `model_comparison.csv` | All 11 models with train/test metrics |
| `final_comparison.csv` | Tuned vs initial top models |
| `feature_importance.csv` | Feature importance rankings |
| `features.json` | List of 59 engineered features |
| `pipeline_report.json` | Full run summary (metrics, params, timing) |

## Results

| Model | Test R² | Test MAE | Test RMSE |
|-------|---------|----------|-----------|
| **Extra Trees (tuned)** | **0.3297** | **0.2522** | **0.3463** |
| XGBoost (tuned) | 0.3286 | 0.2547 | 0.3466 |
| Random Forest (tuned) | 0.3158 | 0.2554 | 0.3499 |

Best model: **Extra Trees** with n_estimators=300, max_depth=20, min_samples_split=5, min_samples_leaf=4.

## Feature Engineering Highlights

- **Temporal**: Cyclic sin/cos encoding for month/hour/day, lag features, rolling statistics
- **Spatial**: Distance from center, lat/lon bins, spatial event counts
- **Seismological**: Log depth, stations per gap, RMS log, close-depth interaction
- **59 total features**, all leakage-free
