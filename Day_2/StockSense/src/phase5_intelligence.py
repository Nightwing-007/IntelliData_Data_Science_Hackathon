"""
StockSense - Phase 5: Advanced Replenishment Intelligence & Model Explainability
NovaMart Retail Challenge - Day 2

This script constructs the prescriptive decision layer with safety stock buffering:
1. Loads test predictions from data/processed/test_set_predictions.csv and models/risk_model.pkl.
2. Computes Safety Stock Buffer: safety_stock = 1.65 * rolling_std_demand * sqrt(lead_days)
3. Computes the enhanced prescriptive replenishment formula:
   Recommended_Reorder_Qty = max(0, predicted_7_day_demand + safety_stock - closing_stock)
4. Assigns Operational Risk Categories:
   - High Risk: Stockout Probability >= 0.70
   - Medium Risk: 0.40 <= Stockout Probability < 0.70
   - Low Risk: Stockout Probability < 0.40
5. Computes Reorder Priority Score (0-100): composite metric for urgency ranking.
6. Estimates Days Until Stockout: closing / max(1, lag_1_demand).
7. Calculates Cost Intelligence: overstock savings for low-risk items.
8. Generates Top 10 Urgent Actions summary for immediate manager action.
9. Extracts feature importances from the Random Forest risk model.
10. Exports the manager-ready dataset to data/processed/scored_predictions.csv.
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


def compute_safety_stock(df: pd.DataFrame) -> pd.Series:
    """
    Compute safety stock buffer using the statistical formula:
    safety_stock = z * sigma_demand * sqrt(lead_time)
    where z=1.65 for 95% service level.
    """
    z_score = 1.65  # 95% service level confidence

    # Use rolling std of demand if available, otherwise estimate from lag spread
    if "rolling_std_7_demand" in df.columns:
        std_demand = df["rolling_std_7_demand"].fillna(0)
    elif "lag_1_demand" in df.columns and "lag_7_demand" in df.columns:
        # Approximate std from the spread between recent and weekly lag
        std_demand = (df["lag_1_demand"] - df["lag_7_demand"]).abs().fillna(0)
    else:
        std_demand = pd.Series(0, index=df.index)

    # Use lead_days if available, otherwise assume 3-day average
    if "lead_days" in df.columns:
        lead_days = df["lead_days"].fillna(3)
    else:
        lead_days = pd.Series(3, index=df.index)

    safety_stock = (z_score * std_demand * np.sqrt(lead_days)).round(1)
    return safety_stock


def compute_reorder_priority_score(df: pd.DataFrame) -> pd.Series:
    """
    Compute composite Reorder Priority Score (0-100):
    - stockout_prob * 40 (risk urgency weight)
    - (1 / max(1, days_of_inventory)) * 30 (inventory depletion weight)
    - promo_active * 30 (promotional demand amplification weight)
    Normalized to 0-100 via min-max scaling.
    """
    stockout_component = df["stockout_prob"] * 40

    doi_component = (1.0 / np.maximum(1.0, df["days_of_inventory"])) * 30 if "days_of_inventory" in df.columns else 0

    promo_component = df["promo_active"] * 30 if "promo_active" in df.columns else 0

    raw_score = stockout_component + doi_component + promo_component

    # Min-max normalize to 0-100
    score_min = raw_score.min()
    score_max = raw_score.max()
    if score_max > score_min:
        normalized = ((raw_score - score_min) / (score_max - score_min) * 100).round(1)
    else:
        normalized = pd.Series(50.0, index=df.index)

    return normalized


def compute_replenishment_intelligence(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply enhanced prescriptive replenishment logic:
    - Safety Stock Buffer calculation
    - Recommended_Reorder_Qty = max(0, predicted_7_day_demand + safety_stock - closing)
    - Risk Tiering: High (>=0.70), Medium (0.40-0.70), Low (<0.40)
    - Reorder Priority Score (0-100)
    - Estimated Days Until Stockout
    - Cost savings intelligence
    """
    df = df.copy()

    # -------------------------------------------------------------------------
    # Safety Stock Calculation
    # -------------------------------------------------------------------------
    df["safety_stock"] = compute_safety_stock(df)

    # -------------------------------------------------------------------------
    # Enhanced Reorder Formula (with safety stock buffer)
    # -------------------------------------------------------------------------
    df["recommended_reorder_qty"] = np.maximum(
        0, df["predicted_7_day_demand"] + df["safety_stock"] - df["closing"]
    ).round(0).astype(int)

    # -------------------------------------------------------------------------
    # Risk Level Categorization
    # -------------------------------------------------------------------------
    conditions = [
        df["stockout_prob"] >= 0.70,
        (df["stockout_prob"] >= 0.40) & (df["stockout_prob"] < 0.70),
        df["stockout_prob"] < 0.40,
    ]
    choices = ["High", "Medium", "Low"]
    df["risk_level"] = np.select(conditions, choices, default="Low")

    # -------------------------------------------------------------------------
    # Reorder Priority Score (0-100)
    # -------------------------------------------------------------------------
    df["reorder_priority_score"] = compute_reorder_priority_score(df)

    # -------------------------------------------------------------------------
    # Estimated Days Until Stockout
    # -------------------------------------------------------------------------
    if "lag_1_demand" in df.columns:
        df["est_days_to_stockout"] = (
            df["closing"] / np.maximum(1.0, df["lag_1_demand"])
        ).round(1)
    else:
        df["est_days_to_stockout"] = 99.0

    # -------------------------------------------------------------------------
    # Cost Intelligence: overstock savings estimate
    # -------------------------------------------------------------------------
    if "cost_price" in df.columns:
        df["overstock_savings"] = np.where(
            (df["risk_level"] == "Low") & (df["closing"] > 2 * df["predicted_7_day_demand"]),
            ((df["closing"] - df["predicted_7_day_demand"]) * df["cost_price"]).round(2),
            0.0,
        )
    else:
        df["overstock_savings"] = 0.0

    # -------------------------------------------------------------------------
    # Clean display columns for manager-ready output
    # -------------------------------------------------------------------------
    df["Store"] = df["store_id"] + " (" + df["city"] + ")"
    df["Product"] = df["product_id"] + " - " + df["sub_category"]
    df["Current Stock"] = df["closing"].round(0).astype(int)
    df["7-Day Forecast"] = df["predicted_7_day_demand"].round(1)
    df["Stock-out Prob"] = (df["stockout_prob"] * 100).round(1).astype(str) + "%"
    df["Risk Level"] = df["risk_level"]
    df["Recommended Order"] = df["recommended_reorder_qty"]
    df["Safety Buffer"] = df["safety_stock"].round(0).astype(int)
    df["Reorder Priority"] = df["reorder_priority_score"]
    df["Est. Days to Stockout"] = df["est_days_to_stockout"]

    # Potential Revenue at Risk (if stockout occurs during high demand)
    df["revenue_at_risk"] = np.where(
        df["risk_level"].isin(["High", "Medium"]),
        (df["predicted_7_day_demand"] * df["mrp"]).round(2),
        0.0,
    )

    return df


def generate_top_10_urgent_actions(df: pd.DataFrame) -> None:
    """Print the Top 10 most urgent store-product reorder actions."""
    print("\n" + "=" * 100)
    print("TOP 10 URGENT REORDER ACTIONS (HIGHEST PRIORITY)")
    print("=" * 100)

    top10 = df.nlargest(10, "reorder_priority_score")

    print(f"{'Rank':<6} {'Store':<22} {'Product':<28} {'Priority':<10} {'Risk':<8} {'Days Left':<12} {'Reorder Qty':<12} {'Rev at Risk'}")
    print(f"{'-'*6} {'-'*22} {'-'*28} {'-'*10} {'-'*8} {'-'*12} {'-'*12} {'-'*12}")

    for rank, (_, row) in enumerate(top10.iterrows(), start=1):
        store = str(row.get("Store", "N/A"))[:20]
        product = str(row.get("Product", "N/A"))[:26]
        priority = f"{row['reorder_priority_score']:.0f}/100"
        risk = str(row["risk_level"])
        days_left = f"{row['est_days_to_stockout']:.1f} days"
        reorder = f"{row['recommended_reorder_qty']:,} units"
        rev_risk = f"${row['revenue_at_risk']:,.2f}"
        print(f"  {rank:<4} {store:<22} {product:<28} {priority:<10} {risk:<8} {days_left:<12} {reorder:<12} {rev_risk}")

    print("=" * 100)


def generate_cost_intelligence_summary(df: pd.DataFrame) -> None:
    """Print cost intelligence insights: savings from optimized ordering."""
    total_overstock_savings = df["overstock_savings"].sum()
    overstock_items = (df["overstock_savings"] > 0).sum()

    print("\n" + "=" * 80)
    print("COST INTELLIGENCE SUMMARY")
    print("=" * 80)
    print(f"  • Items with excess inventory (closing > 2x forecast): {overstock_items}")
    print(f"  • Estimated overstock capital tied up (avoidable)    : ${total_overstock_savings:,.2f}")
    print(f"  • Recommendation: Defer reorders for low-risk overstocked items")
    print(f"    to free working capital and reduce spoilage risk.")
    print("=" * 80)


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
        print(f"  {rank}. {row['Feature'].ljust(26)}: {row['Relative Share (%)']:.2f}% importance")

    top3 = exp_df.head(min(3, len(exp_df)))["Feature"].tolist()
    print("-" * 80)
    print(f"Key Takeaway: The top drivers of inventory stockout risk are:")
    for i, feat in enumerate(top3, 1):
        desc_map = {
            "days_of_inventory": "Immediate buffer coverage vs current velocity",
            "reorder_gap": "Proximity to safety reorder threshold",
            "lag_1_demand": "Short-term sales spike velocity",
            "promo_active": "Promotional demand amplification effect",
            "inventory_turnover": "Stock-to-sales conversion efficiency",
            "stock_velocity": "Rate of stock depletion relative to buffer",
            "days_since_restock": "Supply chain replenishment recency",
        }
        desc = desc_map.get(feat, "Contributing operational factor")
        print(f"  {i}. {feat} ({desc})")
    print("=" * 80 + "\n")

    return exp_df


def main():
    print("=" * 80)
    print("StockSense - Phase 5: Advanced Replenishment Intelligence & Decision Engine")
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

    # 2. Compute Enhanced Reorder Logic, Safety Stock & Risk Tiers
    print("\n[Step 3] Computing safety stock buffers, prescriptive reorder quantities & risk tiers ...")
    scored_df = compute_replenishment_intelligence(test_df)

    risk_counts = scored_df["risk_level"].value_counts().to_dict()
    print(f"  Risk Categorization: High={risk_counts.get('High', 0)}, Medium={risk_counts.get('Medium', 0)}, Low={risk_counts.get('Low', 0)}")
    print(f"  Total Recommended Replenishment Units: {scored_df['recommended_reorder_qty'].sum():,} units")
    print(f"  Total Safety Stock Buffer Units      : {scored_df['safety_stock'].sum():,.0f} units")
    print(f"  Total Revenue at Risk                : ${scored_df['revenue_at_risk'].sum():,.2f}")
    print(f"  Average Reorder Priority Score        : {scored_df['reorder_priority_score'].mean():.1f} / 100")
    print(f"  Average Est. Days Until Stockout      : {scored_df['est_days_to_stockout'].mean():.1f} days")

    # 3. Top 10 Urgent Actions
    generate_top_10_urgent_actions(scored_df)

    # 4. Cost Intelligence
    generate_cost_intelligence_summary(scored_df)

    # 5. Model Explainability Report
    # Determine which features the model was actually trained on
    risk_features = [
        "days_of_inventory", "reorder_gap", "promo_active", "lag_1_demand",
        "inventory_turnover", "stock_velocity", "days_since_restock",
    ]
    available_features = [f for f in risk_features if f in test_df.columns]
    # If model has more/fewer features than expected, use what we can
    n_model_features = len(risk_clf.feature_importances_) if hasattr(risk_clf, "feature_importances_") else len(available_features)
    feature_names = available_features[:n_model_features]
    exp_df = generate_explainability_report(risk_clf, feature_names)

    # 6. Save Final Scored Dataset
    scored_df.to_csv(scored_export_path, index=False)
    print(f"[Step 6] Final scored predictions exported to: {scored_export_path.resolve()}")

    print("\n" + "=" * 80)
    print("ADVANCED REPLENISHMENT DECISION LAYER COMPLETED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    main()
