"""
StockSense - Phase 5: Replenishment Intelligence & Model Explainability
NovaMart Retail Challenge - Day 2

This script constructs the prescriptive decision layer:
1. Loads test predictions from data/processed/test_set_predictions.csv and models/risk_model.pkl.
2. Computes the prescriptive replenishment formula:
   Recommended_Reorder_Qty = max(0, predicted_7_day_demand - closing_stock)
3. Assigns Operational Risk Categories:
   - High Risk: Stockout Probability >= 0.70
   - Medium Risk: 0.40 <= Stockout Probability < 0.70
   - Low Risk: Stockout Probability < 0.40
4. Extracts feature importances from the Random Forest risk model to identify the
   Top 3 Drivers of Stockout Risk, printing an executive "Explainability Report".
5. Exports the manager-ready dataset to data/processed/scored_predictions.csv.
"""

from pathlib import Path
import joblib
import numpy as np
import pandas as pd


def load_interim_predictions(data_path: Path) -> pd.DataFrame:
    """Load test predictions from Phase 4."""
    if not data_path.exists():
        raise FileNotFoundError(f"Interim test predictions not found at: {data_path}")
    return pd.read_csv(data_path)


def load_risk_model(model_path: Path) -> object:
    """Load trained Random Forest risk classifier."""
    if not model_path.exists():
        raise FileNotFoundError(f"Risk model not found at: {model_path}")
    return joblib.load(model_path)


def compute_replenishment_intelligence(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply prescriptive replenishment logic:
    - Recommended_Reorder_Qty = max(0, predicted_7_day_demand - closing)
    - Risk Tiering: High (>=0.70), Medium (0.40-0.70), Low (<0.40)
    """
    df = df.copy()

    # Recommended reorder calculation
    df["recommended_reorder_qty"] = np.maximum(
        0, df["predicted_7_day_demand"] - df["closing"]
    ).round(0).astype(int)

    # Risk level categorization
    conditions = [
        df["stockout_prob"] >= 0.70,
        (df["stockout_prob"] >= 0.40) & (df["stockout_prob"] < 0.70),
        df["stockout_prob"] < 0.40,
    ]
    choices = ["High", "Medium", "Low"]
    df["risk_level"] = np.select(conditions, choices, default="Low")

    # Clean display columns
    df["Store"] = df["store_id"] + " (" + df["city"] + ")"
    df["Product"] = df["product_id"] + " - " + df["sub_category"]
    df["Current Stock"] = df["closing"].round(0).astype(int)
    df["7-Day Forecast"] = df["predicted_7_day_demand"].round(1)
    df["Stock-out Prob"] = (df["stockout_prob"] * 100).round(1).astype(str) + "%"
    df["Risk Level"] = df["risk_level"]
    df["Recommended Order"] = df["recommended_reorder_qty"]

    # Potential Revenue at Risk (if stockout occurs during high demand)
    df["revenue_at_risk"] = np.where(
        df["risk_level"].isin(["High", "Medium"]),
        (df["predicted_7_day_demand"] * df["mrp"]).round(2),
        0.0,
    )

    return df


def generate_explainability_report(clf: object, feature_names: list[str]) -> pd.DataFrame:
    """Extract and format feature importances from the stockout classifier."""
    importances = clf.feature_importances_
    exp_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances,
        "Relative Share (%)": (importances * 100).round(2),
    }).sort_values("Importance", ascending=False).reset_index(drop=True)

    print("\n" + "=" * 80)
    print("DAY 2 STOCKSENSE - MODEL EXPLAINABILITY REPORT (STOCKOUT DRIVERS)")
    print("=" * 80)
    print("Top Drivers Identified by Random Forest Classifier:")
    print("-" * 80)
    for idx, row in exp_df.iterrows():
        rank = idx + 1
        print(f"  {rank}. {row['Feature'].ljust(22)}: {row['Relative Share (%)']:.2f}% importance")

    top3 = exp_df.head(3)["Feature"].tolist()
    print("-" * 80)
    print(f"Key Takeaway: The top 3 drivers of inventory stockout risk are:")
    print(f"  1. {top3[0]} (Immediate buffer coverage vs current velocity)")
    print(f"  2. {top3[1]} (Proximity to safety reorder threshold)")
    print(f"  3. {top3[2]} (Short-term sales spike velocity)")
    print("=" * 80 + "\n")

    return exp_df


def main():
    print("=" * 80)
    print("StockSense - Phase 5: Replenishment Intelligence & Decision Engine")
    print("=" * 80)

    base_dir = Path(__file__).resolve().parent.parent
    test_data_path = base_dir / "data" / "processed" / "test_set_predictions.csv"
    risk_model_path = base_dir / "models" / "risk_model.pkl"
    scored_export_path = base_dir / "data" / "processed" / "scored_predictions.csv"

    # 1. Load Data & Model
    print(f"[Step 1] Loading test set predictions from: {test_data_path} ...")
    test_df = load_interim_predictions(test_data_path)
    print(f"  Test set size: {test_df.shape[0]} rows.")

    print(f"\n[Step 2] Loading trained stockout risk model from: {risk_model_path} ...")
    risk_clf = load_risk_model(risk_model_path)

    # 2. Compute Reorder Logic & Risk Tiers
    print("\n[Step 3] Computing prescriptive reorder quantities & risk tiers ...")
    scored_df = compute_replenishment_intelligence(test_df)

    risk_counts = scored_df["risk_level"].value_counts().to_dict()
    print(f"  Risk Categorization: High={risk_counts.get('High', 0)}, Medium={risk_counts.get('Medium', 0)}, Low={risk_counts.get('Low', 0)}")
    print(f"  Total Recommended Replenishment Units: {scored_df['recommended_reorder_qty'].sum()} units")
    print(f"  Total Revenue at Risk: ${scored_df['revenue_at_risk'].sum():,.2f}")

    # 3. Model Explainability Report
    feature_names = ["days_of_inventory", "reorder_gap", "promo_active", "lag_1_demand"]
    exp_df = generate_explainability_report(risk_clf, feature_names)

    # 4. Save Final Scored Dataset
    scored_df.to_csv(scored_export_path, index=False)
    print(f"[Step 4] Final scored predictions exported to: {scored_export_path.resolve()}")

    print("\n" + "=" * 80)
    print("REPLENISHMENT DECISION LAYER COMPLETED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    main()
