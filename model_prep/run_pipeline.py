"""
Main Pipeline Runner
--------------------
End-to-end execution:
  1. Load data
  2. Feature engineering
  3. Train & evaluate 11 models (70/30 split)
  4. Hyperparameter tune top-3 models
  5. Export best model, comparison report, feature importance
"""

import sys
import time
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

warnings.filterwarnings("ignore")

# Add model_prep to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent))

from feature_engineering import load_raw_data, engineer_features, get_feature_target_split
from model_pipeline import train_and_evaluate_all, evaluate_model
from hyperparameter_tuning import tune_top_models, PARAM_GRIDS

from sklearn.model_selection import train_test_split


def run_pipeline(
    data_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    tune_top_k: int = 3,
    tune_n_iter: int = 20,
    test_size: float = 0.30,
    random_state: int = 42,
):
    """
    Run the full ML pipeline.

    Args:
        data_path: Path to the CSV file
        output_dir: Where to save models and reports
        tune_top_k: How many top models to hyperparameter-tune
        tune_n_iter: RandomizedSearchCV iterations per model
        test_size: Test set proportion (default 30%)
        random_state: Reproducibility seed
    """
    base = Path(__file__).resolve().parent
    if data_path is None:
        data_path = base.parent / "data" / "Earthquake_data_processed.csv"
    if output_dir is None:
        output_dir = base / "output"

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    total_start = time.time()

    # ============================================================
    # STEP 1: Load & Feature Engineering
    # ============================================================
    print("=" * 64)
    print("STEP 1: DATA LOADING & FEATURE ENGINEERING")
    print("=" * 64)
    raw = load_raw_data(data_path)
    print(f"Raw data: {raw.shape[0]} rows × {raw.shape[1]} columns")

    engineered = engineer_features(raw)
    X, y = get_feature_target_split(engineered)
    print(f"Engineered: {X.shape[0]} rows × {X.shape[1]} features")
    print(f"Target: Magnitude(ergs) — min={y.min():.2f}, max={y.max():.2f}, mean={y.mean():.2f}")

    # Save feature list
    feature_list = X.columns.tolist()
    with open(output_dir / "features.json", "w") as f:
        json.dump(feature_list, f, indent=2)
    print(f"Feature list saved → output/features.json ({len(feature_list)} features)\n")

    # ============================================================
    # STEP 2: Train & Evaluate All Models (70/30 Split)
    # ============================================================
    print("=" * 64)
    print(f"STEP 2: TRAINING 11 MODELS (70/30 SPLIT)")
    print("=" * 64)
    results_df, trained_models, best_initial = train_and_evaluate_all(
        X, y, test_size=test_size, random_state=random_state
    )

    # Save comparison
    results_df.to_csv(output_dir / "model_comparison.csv", index=False)
    print("Model comparison saved → output/model_comparison.csv")

    # ============================================================
    # STEP 3: Hyperparameter Tuning (Top K)
    # ============================================================
    print("\n" + "=" * 64)
    print(f"STEP 3: HYPERPARAMETER TUNING (TOP {tune_top_k} MODELS)")
    print("=" * 64)

    # Get top K model names that have param grids
    tunable = [name for name in results_df["Model"] if name in PARAM_GRIDS]
    top_models = tunable[:tune_top_k]
    print(f"Models to tune: {top_models}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    tuning_results = tune_top_models(
        X_train, y_train, top_models,
        n_iter=tune_n_iter, cv=5, random_state=random_state,
    )

    # ============================================================
    # STEP 4: Final Evaluation & Best Model Selection
    # ============================================================
    print("\n" + "=" * 64)
    print("STEP 4: FINAL EVALUATION ON HOLD-OUT TEST SET")
    print("=" * 64)

    final_results = []
    for name, info in tuning_results.items():
        pipeline = info["best_pipeline"]
        y_pred = pipeline.predict(X_test)
        metrics = evaluate_model(y_test.values, y_pred)
        metrics["Model"] = name + " (tuned)"
        metrics["CV R²"] = info["best_score"]
        final_results.append(metrics)
        print(f"  {name} (tuned): R²={metrics['R2']:.4f}  MAE={metrics['MAE']:.4f}  RMSE={metrics['RMSE']:.4f}")

    # Also include the initial best model for comparison
    initial_best_pipeline = best_initial["pipeline"]
    y_pred_init = initial_best_pipeline.predict(X_test)
    init_metrics = evaluate_model(y_test.values, y_pred_init)
    init_metrics["Model"] = best_initial["name"] + " (initial)"
    final_results.append(init_metrics)

    final_df = pd.DataFrame(final_results).sort_values("R2", ascending=False).reset_index(drop=True)
    final_df.to_csv(output_dir / "final_comparison.csv", index=False)
    print("\nFinal comparison saved → output/final_comparison.csv")

    # Select absolute best
    best_row = final_df.iloc[0]
    best_name = best_row["Model"]

    # Find the actual pipeline
    if "(tuned)" in best_name:
        base_name = best_name.replace(" (tuned)", "")
        best_pipeline = tuning_results[base_name]["best_pipeline"]
        best_params = tuning_results[base_name]["best_params"]
    else:
        base_name = best_name.replace(" (initial)", "")
        best_pipeline = trained_models[base_name]
        best_params = {}

    # ============================================================
    # STEP 5: Save Best Model & Report
    # ============================================================
    print("\n" + "=" * 64)
    print("STEP 5: SAVING BEST MODEL & REPORT")
    print("=" * 64)

    # Save model
    model_path = output_dir / "best_model.joblib"
    joblib.dump(best_pipeline, model_path)
    print(f"Best model saved → {model_path}")

    # Feature importance (if tree-based)
    try:
        inner_model = best_pipeline.named_steps["model"]
        if hasattr(inner_model, "feature_importances_"):
            importances = inner_model.feature_importances_
            feat_imp = pd.DataFrame({
                "feature": feature_list,
                "importance": importances,
            }).sort_values("importance", ascending=False).reset_index(drop=True)
            feat_imp.to_csv(output_dir / "feature_importance.csv", index=False)
            print(f"Feature importance saved → output/feature_importance.csv")
            print(f"\nTop 15 Features:")
            for _, row in feat_imp.head(15).iterrows():
                bar = "█" * int(row["importance"] * 100)
                print(f"  {row['feature']:35s} {row['importance']:.4f}  {bar}")
    except Exception:
        pass

    # Save report
    total_time = time.time() - total_start
    cv_r2 = float(best_row.get("CV R²", 0.0)) if "CV R²" in best_row else 0.0
    report = {
        "best_model": best_name,
        "best_params": best_params,
        "test_r2": float(best_row["R2"]),
        "test_mae": float(best_row["MAE"]),
        "test_rmse": float(best_row["RMSE"]),
        "test_mape_pct": float(best_row["MAPE (%)"]),
        "explained_variance": float(best_row["Explained Variance"]),
        "n_features": len(feature_list),
        "n_train_samples": X_train.shape[0],
        "n_test_samples": X_test.shape[0],
        "test_size": test_size,
        "total_pipeline_time_sec": round(total_time, 1),
        "models_evaluated": len(results_df),
        "models_tuned": len(tuning_results),
    }
    with open(output_dir / "pipeline_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nPipeline report saved → output/pipeline_report.json")

    # ---- Final Summary ----
    print(f"\n{'='*64}")
    print(f"  PIPELINE COMPLETE  —  Total time: {total_time:.1f}s")
    print(f"{'='*64}")
    print(f"  Best Model    : {best_name}")
    print(f"  Test R²       : {best_row['R2']:.4f}")
    print(f"  Test MAE      : {best_row['MAE']:.4f}")
    print(f"  Test RMSE     : {best_row['RMSE']:.4f}")
    print(f"  Test MAPE     : {best_row['MAPE (%)']:.2f}%")
    print(f"  Features      : {len(feature_list)}")
    print(f"  Train/Test    : {X_train.shape[0]} / {X_test.shape[0]}")
    print(f"{'='*64}\n")

    return report


if __name__ == "__main__":
    run_pipeline()
