"""
StockSense - Phase 4: Advanced Predictive Machine Learning Modeling
NovaMart Retail Challenge - Day 2

This script trains and evaluates two production-grade machine learning models with
hyperparameter tuning and comprehensive evaluation:
1. Model 1 (Demand Forecasting):
   - Target: 'next_7_day_demand' (continuous)
   - Features: lag_7_demand, rolling_mean_7_demand, promo_active, temp_c, weekend,
     ewma_7_demand, demand_acceleration, promo_x_weekend, heavy_rain
   - Algorithm: XGBoost Regressor (with automatic fallback to RandomForestRegressor)
   - Hyperparameter Tuning: RandomizedSearchCV with 3-fold cross-validation
   - Evaluation Metrics: RMSE, MAE, R², 3-Fold Cross-Validation Score
   - Export: models/demand_model.pkl
2. Model 2 (Stock-out Risk Classification):
   - Target: 'stockout_flag' (binary classification: 1 = stockout, 0 = in-stock)
   - Features: days_of_inventory, reorder_gap, promo_active, lag_1_demand,
     inventory_turnover, stock_velocity, days_since_restock
   - Algorithm: Random Forest Classifier (balanced class weights)
   - Evaluation Metrics: ROC-AUC, Precision-Recall AUC, Classification Report
   - Export: models/risk_model.pkl
3. Feature importance rankings for both models.
4. Model comparison summary table.
5. Exports test set predictions to data/processed/test_set_predictions.csv.
"""

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    classification_report,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
    average_precision_score,
)
from sklearn.model_selection import RandomizedSearchCV, cross_val_score, train_test_split

# Try importing XGBoost; fall back to RandomForestRegressor if not available
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False


def load_model_data(data_path: Path) -> pd.DataFrame:
    """Load model-ready feature dataset."""
    if not data_path.exists():
        raise FileNotFoundError(f"Model-ready data not found at: {data_path}")
    df = pd.read_csv(data_path)
    return df


def print_feature_importances(model, feature_names: list[str], model_name: str) -> None:
    """Print ranked feature importance table for a given model."""
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "best_estimator_") and hasattr(model.best_estimator_, "feature_importances_"):
        importances = model.best_estimator_.feature_importances_
    else:
        print(f"  [INFO] {model_name} does not expose feature importances.")
        return

    feat_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances,
        "Share (%)": (importances * 100).round(2),
    }).sort_values("Importance", ascending=False).reset_index(drop=True)

    print(f"\n  [{model_name} Feature Importance Ranking]")
    print(f"  {'Rank':<6} {'Feature':<30} {'Importance':<12} {'Share (%)'}")
    print(f"  {'-'*6} {'-'*30} {'-'*12} {'-'*10}")
    for idx, row in feat_df.iterrows():
        print(f"  {idx+1:<6} {row['Feature']:<30} {row['Importance']:.6f}    {row['Share (%)']:.2f}%")


def train_demand_model(
    df: pd.DataFrame, models_dir: Path
) -> tuple[object, pd.DataFrame, dict[str, float]]:
    """
    Train Model 1: Demand Forecasting Regressor with hyperparameter tuning.
    Target: next_7_day_demand
    Features: lag_7_demand, rolling_mean_7_demand, promo_active, temp_c, weekend,
              ewma_7_demand, demand_acceleration, promo_x_weekend, heavy_rain
    """
    features = [
        "lag_7_demand", "rolling_mean_7_demand", "promo_active", "temp_c", "weekend",
        "ewma_7_demand", "demand_acceleration", "promo_x_weekend", "heavy_rain",
    ]

    # Filter to features that actually exist in the dataset
    available_features = [f for f in features if f in df.columns]
    missing_features = [f for f in features if f not in df.columns]
    if missing_features:
        print(f"  [WARN] Missing features (will use available): {missing_features}")

    target = "next_7_day_demand"

    X = df[available_features]
    y = df[target]

    # 80/20 train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    # Hyperparameter search grid
    param_grid = {
        "n_estimators": [100, 150, 200],
        "max_depth": [5, 8, 10],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2],
    }

    if HAS_XGBOOST:
        print("  -> Training XGBoost Regressor with RandomizedSearchCV ...")
        base_model = xgb.XGBRegressor(
            learning_rate=0.08,
            random_state=42,
            n_jobs=-1,
        )
        xgb_param_grid = {
            "n_estimators": [100, 150, 200],
            "max_depth": [5, 8, 10],
            "min_child_weight": [1, 3],
            "subsample": [0.8, 1.0],
        }
        search = RandomizedSearchCV(
            base_model, xgb_param_grid, n_iter=8, cv=3,
            scoring="neg_mean_absolute_error", random_state=42, n_jobs=-1, verbose=0,
        )
    else:
        print("  -> Training RandomForestRegressor with RandomizedSearchCV ...")
        base_model = RandomForestRegressor(random_state=42, n_jobs=-1)
        search = RandomizedSearchCV(
            base_model, param_grid, n_iter=8, cv=3,
            scoring="neg_mean_absolute_error", random_state=42, n_jobs=-1, verbose=0,
        )

    search.fit(X_train, y_train)
    model = search.best_estimator_
    print(f"  -> Best hyperparameters: {search.best_params_}")

    y_pred = model.predict(X_test)
    y_pred_clipped = np.maximum(0, y_pred)  # Demand cannot be negative

    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred_clipped)))
    mae = float(mean_absolute_error(y_test, y_pred_clipped))
    r2 = float(r2_score(y_test, y_pred_clipped))

    # 3-Fold Cross-Validation Score on full training set
    cv_scores = cross_val_score(model, X_train, y_train, cv=3, scoring="neg_mean_absolute_error")
    cv_mae_mean = float(-cv_scores.mean())
    cv_mae_std = float(cv_scores.std())

    # Save model
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / "demand_model.pkl"
    joblib.dump(model, model_path)
    print(f"  -> Model saved to: {model_path.resolve()}")

    test_df = df.loc[X_test.index].copy()
    test_df["predicted_7_day_demand"] = y_pred_clipped.round(1)

    # Print feature importances
    print_feature_importances(model, available_features, "Demand Forecasting Model")

    metrics = {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "cv_mae_mean": cv_mae_mean,
        "cv_mae_std": cv_mae_std,
    }
    return model, test_df, metrics


def train_risk_model(
    df: pd.DataFrame, test_df: pd.DataFrame, models_dir: Path
) -> tuple[object, pd.DataFrame, dict[str, float]]:
    """
    Train Model 2: Stock-out Risk Classifier with expanded feature set.
    Target: stockout_flag
    Features: days_of_inventory, reorder_gap, promo_active, lag_1_demand,
              inventory_turnover, stock_velocity, days_since_restock
    """
    features = [
        "days_of_inventory", "reorder_gap", "promo_active", "lag_1_demand",
        "inventory_turnover", "stock_velocity", "days_since_restock",
    ]

    # Filter to features that actually exist in the dataset
    available_features = [f for f in features if f in df.columns]
    missing_features = [f for f in features if f not in df.columns]
    if missing_features:
        print(f"  [WARN] Missing features (will use available): {missing_features}")

    target = "stockout_flag"

    X = df[available_features]
    y = df[target]

    # Use same test indices to ensure aligned predictions across models
    test_indices = test_df.index
    train_indices = df.index.difference(test_indices)

    X_train = X.loc[train_indices]
    y_train = y.loc[train_indices]
    X_test = X.loc[test_indices]
    y_test = y.loc[test_indices]

    print("  -> Training Random Forest Classifier for Stockout Risk ...")
    clf = RandomForestClassifier(
        n_estimators=150,
        max_depth=8,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    # Predict probabilities of stockout (class 1)
    y_prob = clf.predict_proba(X_test)[:, 1]
    y_pred = clf.predict(X_test)

    # Handle single-class edge cases in ROC-AUC
    if len(np.unique(y_test)) > 1:
        roc_auc = float(roc_auc_score(y_test, y_prob))
        pr_auc = float(average_precision_score(y_test, y_prob))
    else:
        roc_auc = 1.0
        pr_auc = 1.0

    clf_report = classification_report(y_test, y_pred, zero_division=0)

    # Save model
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / "risk_model.pkl"
    joblib.dump(clf, model_path)
    print(f"  -> Model saved to: {model_path.resolve()}")

    test_df["stockout_prob"] = y_prob.round(4)
    test_df["stockout_pred_label"] = y_pred

    # Print feature importances
    print_feature_importances(clf, available_features, "Stockout Risk Classifier")

    metrics = {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "classification_report": clf_report,
        "feature_importances": dict(zip(available_features, clf.feature_importances_)),
    }
    return clf, test_df, metrics


def print_model_comparison(demand_metrics: dict, risk_metrics: dict) -> None:
    """Print a formatted side-by-side model comparison summary table."""
    print("\n" + "=" * 80)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 80)
    print(f"{'Metric':<35} {'Demand Forecaster':<22} {'Risk Classifier':<22}")
    print(f"{'-'*35} {'-'*22} {'-'*22}")
    print(f"{'Algorithm':<35} {'RF/XGB Regressor':<22} {'RF Classifier':<22}")
    print(f"{'RMSE':<35} {demand_metrics['rmse']:<22.4f} {'N/A':<22}")
    print(f"{'MAE':<35} {demand_metrics['mae']:<22.4f} {'N/A':<22}")
    print(f"{'R² Score':<35} {demand_metrics['r2']:<22.4f} {'N/A':<22}")
    print(f"{'CV MAE (3-Fold)':<35} {demand_metrics['cv_mae_mean']:<22.4f} {'N/A':<22}")
    print(f"{'ROC-AUC':<35} {'N/A':<22} {risk_metrics['roc_auc']:<22.4f}")
    print(f"{'Precision-Recall AUC':<35} {'N/A':<22} {risk_metrics['pr_auc']:<22.4f}")
    print("=" * 80)


def main():
    print("=" * 80)
    print("StockSense - Phase 4: Advanced Machine Learning Modeling Suite")
    print("=" * 80)

    base_dir = Path(__file__).resolve().parent.parent
    data_path = base_dir / "data" / "processed" / "model_ready_data.csv"
    models_dir = base_dir / "models"
    test_export_path = base_dir / "data" / "processed" / "test_set_predictions.csv"

    print(f"[Step 1] Loading model-ready dataset from: {data_path} ...")
    df = load_model_data(data_path)
    print(f"  Dataset loaded: {df.shape[0]} rows x {df.shape[1]} columns.")

    # 1. Train Demand Model with Hyperparameter Tuning
    print("\n[Step 2] Training Demand Forecasting Model with RandomizedSearchCV (Model 1) ...")
    demand_model, test_df_with_demand, demand_metrics = train_demand_model(df, models_dir)
    print(f"\n  [Demand Model Evaluation Metrics]")
    print(f"  • Root Mean Squared Error (RMSE): {demand_metrics['rmse']:.4f}")
    print(f"  • Mean Absolute Error (MAE)     : {demand_metrics['mae']:.4f}")
    print(f"  • R² Score                      : {demand_metrics['r2']:.4f}")
    print(f"  • 3-Fold CV MAE (mean ± std)    : {demand_metrics['cv_mae_mean']:.4f} ± {demand_metrics['cv_mae_std']:.4f}")

    # 2. Train Stockout Risk Model with Expanded Features
    print("\n[Step 3] Training Stock-out Risk Model with Expanded Features (Model 2) ...")
    risk_model, full_test_df, risk_metrics = train_risk_model(df, test_df_with_demand, models_dir)
    print(f"\n  [Stock-out Risk Model Evaluation Metrics]")
    print(f"  • ROC-AUC Score          : {risk_metrics['roc_auc']:.4f}")
    print(f"  • Precision-Recall AUC   : {risk_metrics['pr_auc']:.4f}")
    print(f"\n  Classification Report:\n{risk_metrics['classification_report']}")

    # 3. Model Comparison Summary
    print_model_comparison(demand_metrics, risk_metrics)

    # 4. Export Interim Test Predictions
    full_test_df.to_csv(test_export_path, index=False)
    print(f"[Step 4] Interim scored test set saved to: {test_export_path.resolve()}")

    print("\n" + "=" * 80)
    print("ADVANCED MACHINE LEARNING MODELING COMPLETED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    main()
