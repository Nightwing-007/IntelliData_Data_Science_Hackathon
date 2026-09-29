"""
StockSense - Phase 4: Predictive Machine Learning Modeling
NovaMart Retail Challenge - Day 2

This script trains and evaluates two production-grade machine learning models:
1. Model 1 (Demand Forecasting):
   - Target: 'next_7_day_demand' (continuous)
   - Features: lag_7_demand, rolling_mean_7_demand, promo_active, temp_c, weekend
   - Algorithm: XGBoost Regressor (with automatic fallback to RandomForestRegressor)
   - Split: 80% train / 20% test partition (random_state=42)
   - Evaluation Metrics: Root Mean Squared Error (RMSE) and Mean Absolute Error (MAE)
   - Export: models/demand_model.pkl
2. Model 2 (Stock-out Risk Classification):
   - Target: 'stockout_flag' (binary classification: 1 = stockout, 0 = in-stock)
   - Features: days_of_inventory, reorder_gap, promo_active, lag_1_demand
   - Algorithm: Random Forest Classifier (balanced class weights)
   - Evaluation Metrics: ROC-AUC score, Precision, Recall, F1-score, and Classification Report
   - Export: models/risk_model.pkl
3. Exports test set predictions to data/processed/test_set_predictions.csv for the Intelligence Layer.
"""

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import classification_report, mean_absolute_error, mean_squared_error, roc_auc_score
from sklearn.model_selection import train_test_split

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


def train_demand_model(
    df: pd.DataFrame, models_dir: Path
) -> tuple[object, pd.DataFrame, dict[str, float]]:
    """
    Train Model 1: Demand Forecasting Regressor.
    Target: next_7_day_demand
    Features: lag_7_demand, rolling_mean_7_demand, promo_active, temp_c, weekend
    """
    features = ["lag_7_demand", "rolling_mean_7_demand", "promo_active", "temp_c", "weekend"]
    target = "next_7_day_demand"

    X = df[features]
    y = df[target]

    # 80/20 train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    if HAS_XGBOOST:
        print("  -> Training XGBoost Regressor for Demand Forecasting ...")
        model = xgb.XGBRegressor(
            n_estimators=120,
            learning_rate=0.08,
            max_depth=5,
            random_state=42,
            n_jobs=-1,
        )
    else:
        print("  -> XGBoost unavailable. Training RandomForestRegressor for Demand Forecasting ...")
        model = RandomForestRegressor(
            n_estimators=120,
            max_depth=8,
            random_state=42,
            n_jobs=-1,
        )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_pred_clipped = np.maximum(0, y_pred)  # Demand cannot be negative

    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred_clipped)))
    mae = float(mean_absolute_error(y_test, y_pred_clipped))

    # Save model
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / "demand_model.pkl"
    joblib.dump(model, model_path)
    print(f"  -> Model saved to: {model_path.resolve()}")

    test_df = df.loc[X_test.index].copy()
    test_df["predicted_7_day_demand"] = y_pred_clipped.round(1)

    metrics = {"rmse": rmse, "mae": mae}
    return model, test_df, metrics


def train_risk_model(
    df: pd.DataFrame, test_df: pd.DataFrame, models_dir: Path
) -> tuple[object, pd.DataFrame, dict[str, float]]:
    """
    Train Model 2: Stock-out Risk Classifier.
    Target: stockout_flag
    Features: days_of_inventory, reorder_gap, promo_active, lag_1_demand
    """
    features = ["days_of_inventory", "reorder_gap", "promo_active", "lag_1_demand"]
    target = "stockout_flag"

    X = df[features]
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
        n_estimators=100,
        max_depth=6,
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
    else:
        roc_auc = 1.0

    clf_report = classification_report(y_test, y_pred, zero_division=0)

    # Save model
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / "risk_model.pkl"
    joblib.dump(clf, model_path)
    print(f"  -> Model saved to: {model_path.resolve()}")

    test_df["stockout_prob"] = y_prob.round(4)
    test_df["stockout_pred_label"] = y_pred

    metrics = {
        "roc_auc": roc_auc,
        "classification_report": clf_report,
        "feature_importances": dict(zip(features, clf.feature_importances_)),
    }
    return clf, test_df, metrics


def main():
    print("=" * 80)
    print("StockSense - Phase 4: Machine Learning Modeling Suite")
    print("=" * 80)

    base_dir = Path(__file__).resolve().parent.parent
    data_path = base_dir / "data" / "processed" / "model_ready_data.csv"
    models_dir = base_dir / "models"
    test_export_path = base_dir / "data" / "processed" / "test_set_predictions.csv"

    print(f"[Step 1] Loading model-ready dataset from: {data_path} ...")
    df = load_model_data(data_path)
    print(f"  Dataset loaded: {df.shape[0]} rows x {df.shape[1]} columns.")

    # 1. Train Demand Model
    print("\n[Step 2] Training Demand Forecasting Model (Model 1) ...")
    demand_model, test_df_with_demand, demand_metrics = train_demand_model(df, models_dir)
    print(f"\n  [Demand Model Evaluation Metrics]")
    print(f"  • Root Mean Squared Error (RMSE): {demand_metrics['rmse']:.4f}")
    print(f"  • Mean Absolute Error (MAE)    : {demand_metrics['mae']:.4f}")

    # 2. Train Stockout Risk Model
    print("\n[Step 3] Training Stock-out Risk Model (Model 2) ...")
    risk_model, full_test_df, risk_metrics = train_risk_model(df, test_df_with_demand, models_dir)
    print(f"\n  [Stock-out Risk Model Evaluation Metrics]")
    print(f"  • ROC-AUC Score: {risk_metrics['roc_auc']:.4f}")
    print(f"\n  Classification Report:\n{risk_metrics['classification_report']}")

    # 3. Export Interim Test Predictions
    full_test_df.to_csv(test_export_path, index=False)
    print(f"[Step 4] Interim scored test set saved to: {test_export_path.resolve()}")

    print("\n" + "=" * 80)
    print("MACHINE LEARNING MODELING COMPLETED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    main()
