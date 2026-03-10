"""
Model Training Pipeline
-----------------------
Trains N standard ML algorithms on the engineered features,
evaluates each, and selects the best model.

Algorithms included:
  1. Linear Regression (baseline)
  2. Ridge Regression
  3. Lasso Regression
  4. ElasticNet
  5. Decision Tree Regressor
  6. Random Forest Regressor
  7. Gradient Boosting Regressor
  8. AdaBoost Regressor
  9. K-Nearest Neighbors Regressor
  10. Extra Trees Regressor
  11. XGBoost Regressor

70/30 train-test split. All models evaluated on the same hold-out set.
"""

import time
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    explained_variance_score,
)

# Algorithms
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    AdaBoostRegressor,
    ExtraTreesRegressor,
)
from sklearn.neighbors import KNeighborsRegressor

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

warnings.filterwarnings("ignore", category=UserWarning)


def get_all_models() -> dict:
    """
    Returns a dictionary of {name: estimator} for all candidate models.
    Each estimator is wrapped in a Pipeline with StandardScaler.
    """
    models = {
        "Linear Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LinearRegression()),
        ]),
        "Ridge Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=1.0)),
        ]),
        "Lasso Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", Lasso(alpha=0.01, max_iter=5000)),
        ]),
        "ElasticNet": Pipeline([
            ("scaler", StandardScaler()),
            ("model", ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=5000)),
        ]),
        "Decision Tree": Pipeline([
            ("scaler", StandardScaler()),
            ("model", DecisionTreeRegressor(random_state=42, max_depth=15)),
        ]),
        "Random Forest": Pipeline([
            ("scaler", StandardScaler()),
            ("model", RandomForestRegressor(
                n_estimators=100, max_depth=15, random_state=42, n_jobs=-1
            )),
        ]),
        "Gradient Boosting": Pipeline([
            ("scaler", StandardScaler()),
            ("model", GradientBoostingRegressor(
                n_estimators=50, max_depth=4, learning_rate=0.1, random_state=42
            )),
        ]),
        "AdaBoost": Pipeline([
            ("scaler", StandardScaler()),
            ("model", AdaBoostRegressor(
                n_estimators=50, learning_rate=0.1, random_state=42
            )),
        ]),
        "KNN Regressor": Pipeline([
            ("scaler", StandardScaler()),
            ("model", KNeighborsRegressor(n_neighbors=7, weights="distance", n_jobs=-1)),
        ]),
        "Extra Trees": Pipeline([
            ("scaler", StandardScaler()),
            ("model", ExtraTreesRegressor(
                n_estimators=100, max_depth=15, random_state=42, n_jobs=-1
            )),
        ]),
    }

    if HAS_XGB:
        models["XGBoost"] = Pipeline([
            ("scaler", StandardScaler()),
            ("model", XGBRegressor(
                n_estimators=100, max_depth=6, learning_rate=0.1,
                random_state=42, n_jobs=-1, verbosity=0,
            )),
        ])

    return models


def evaluate_model(y_true, y_pred) -> dict:
    """Compute standard regression metrics."""
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2": r2_score(y_true, y_pred),
        "Explained Variance": explained_variance_score(y_true, y_pred),
        "MAPE (%)": np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100,
    }


def train_and_evaluate_all(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.30,
    random_state: int = 42,
    cv_folds: int = 3,
) -> tuple[pd.DataFrame, dict, dict]:
    """
    70/30 split → train all models → evaluate on hold-out set.

    Returns:
        results_df: Comparison table of all models
        trained_models: dict of {name: fitted pipeline}
        best_info: dict with best model name, pipeline, metrics
    """
    # ---- 70/30 Split ----
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    print(f"Train: {X_train.shape[0]} samples | Test: {X_test.shape[0]} samples")
    print(f"Features: {X_train.shape[1]}\n")

    models = get_all_models()
    results = []
    trained_models = {}

    for name, pipeline in models.items():
        print(f"Training: {name} ...", end=" ", flush=True)
        start = time.time()

        # Fit
        pipeline.fit(X_train, y_train)
        train_time = time.time() - start

        # Predict
        y_pred_train = pipeline.predict(X_train)
        y_pred_test = pipeline.predict(X_test)

        # Metrics
        train_metrics = evaluate_model(y_train.values, y_pred_train)
        test_metrics = evaluate_model(y_test.values, y_pred_test)

        row = {
            "Model": name,
            "Train R²": train_metrics["R2"],
            "Test R²": test_metrics["R2"],
            "Test MAE": test_metrics["MAE"],
            "Test RMSE": test_metrics["RMSE"],
            "Test MAPE (%)": test_metrics["MAPE (%)"],
            "Explained Var": test_metrics["Explained Variance"],
            "Train Time (s)": round(train_time, 2),
        }
        results.append(row)
        trained_models[name] = pipeline

        print(f"R²={test_metrics['R2']:.4f}  MAE={test_metrics['MAE']:.4f}  ({train_time:.1f}s)")

    results_df = pd.DataFrame(results).sort_values("Test R²", ascending=False).reset_index(drop=True)

    # Best model
    best_name = results_df.iloc[0]["Model"]
    best_info = {
        "name": best_name,
        "pipeline": trained_models[best_name],
        "metrics": results_df.iloc[0].to_dict(),
    }

    print(f"\n{'='*60}")
    print(f"BEST MODEL: {best_name} (Test R² = {best_info['metrics']['Test R²']:.4f})")
    print(f"{'='*60}\n")

    return results_df, trained_models, best_info


if __name__ == "__main__":
    from feature_engineering import load_raw_data, engineer_features, get_feature_target_split
    from pathlib import Path

    data_path = Path(__file__).resolve().parent.parent / "data" / "Earthquake_data_processed.csv"
    raw = load_raw_data(data_path)
    engineered = engineer_features(raw)
    X, y = get_feature_target_split(engineered)

    results_df, trained_models, best = train_and_evaluate_all(X, y)
    print("\nModel Comparison Table:")
    print(results_df.to_string(index=False))
