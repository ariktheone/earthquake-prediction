"""
Hyperparameter Tuning Module
-----------------------------
Takes the top-K models from the initial sweep and runs
RandomizedSearchCV / GridSearchCV to find optimal hyperparameters.
"""

import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    ExtraTreesRegressor,
)

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False


# ---- Search Spaces for Top Models ----
PARAM_GRIDS = {
    "Random Forest": {
        "model__n_estimators": [100, 200, 300],
        "model__max_depth": [10, 15, 20, None],
        "model__min_samples_split": [2, 5, 10],
        "model__min_samples_leaf": [1, 2, 4],
    },
    "Gradient Boosting": {
        "model__n_estimators": [100, 200, 300],
        "model__max_depth": [3, 5, 7],
        "model__learning_rate": [0.01, 0.05, 0.1],
        "model__subsample": [0.8, 0.9, 1.0],
        "model__min_samples_leaf": [1, 2, 4],
    },
    "Extra Trees": {
        "model__n_estimators": [100, 200, 300],
        "model__max_depth": [10, 15, 20, None],
        "model__min_samples_split": [2, 5, 10],
        "model__min_samples_leaf": [1, 2, 4],
    },
}

if HAS_XGB:
    PARAM_GRIDS["XGBoost"] = {
        "model__n_estimators": [100, 200, 300],
        "model__max_depth": [3, 5, 7, 9],
        "model__learning_rate": [0.01, 0.05, 0.1],
        "model__subsample": [0.8, 0.9, 1.0],
        "model__colsample_bytree": [0.7, 0.8, 1.0],
        "model__reg_lambda": [0.5, 1.0, 2.0],
    }


def _make_fresh_pipeline(model_name: str) -> Pipeline:
    """Create a fresh (unfitted) pipeline for a given model name."""
    estimators = {
        "Random Forest": RandomForestRegressor(random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42),
        "Extra Trees": ExtraTreesRegressor(random_state=42, n_jobs=-1),
    }
    if HAS_XGB:
        estimators["XGBoost"] = XGBRegressor(random_state=42, n_jobs=-1, verbosity=0)

    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", estimators[model_name]),
    ])


def tune_top_models(
    X_train,
    y_train,
    top_model_names: list[str],
    n_iter: int = 20,
    cv: int = 3,
    scoring: str = "r2",
    random_state: int = 42,
) -> dict:
    """
    Run RandomizedSearchCV on each of the top models.

    Args:
        X_train, y_train: Training data
        top_model_names: List of model names to tune (must be in PARAM_GRIDS)
        n_iter: Number of parameter combinations to try
        cv: Cross-validation folds
        scoring: Metric to optimize
        random_state: Seed

    Returns:
        dict of {model_name: {"best_pipeline": ..., "best_params": ..., "best_score": ...}}
    """
    tuning_results = {}

    for name in top_model_names:
        if name not in PARAM_GRIDS:
            print(f"  Skipping {name} — no param grid defined.")
            continue

        print(f"\n{'─'*50}")
        print(f"Tuning: {name}  ({n_iter} iterations, {cv}-fold CV)")
        print(f"{'─'*50}")

        pipeline = _make_fresh_pipeline(name)
        param_dist = PARAM_GRIDS[name]

        search = RandomizedSearchCV(
            pipeline,
            param_distributions=param_dist,
            n_iter=n_iter,
            cv=cv,
            scoring=scoring,
            random_state=random_state,
            n_jobs=-1,
            verbose=1,
            return_train_score=True,
        )
        search.fit(X_train, y_train)

        print(f"\n  Best CV {scoring}: {search.best_score_:.4f}")
        print(f"  Best params: {search.best_params_}")

        tuning_results[name] = {
            "best_pipeline": search.best_estimator_,
            "best_params": search.best_params_,
            "best_score": search.best_score_,
            "cv_results": search.cv_results_,
        }

    return tuning_results


if __name__ == "__main__":
    from feature_engineering import load_raw_data, engineer_features, get_feature_target_split
    from sklearn.model_selection import train_test_split
    from pathlib import Path

    data_path = Path(__file__).resolve().parent.parent / "data" / "Earthquake_data_processed.csv"
    raw = load_raw_data(data_path)
    engineered = engineer_features(raw)
    X, y = get_feature_target_split(engineered)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42
    )

    top_names = ["Random Forest", "Gradient Boosting", "Extra Trees"]
    if HAS_XGB:
        top_names.append("XGBoost")

    results = tune_top_models(X_train, y_train, top_names, n_iter=30, cv=5)
    for name, info in results.items():
        print(f"\n{name}: Best CV R² = {info['best_score']:.4f}")
