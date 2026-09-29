"""
StockSense - Phase 3: Feature Engineering & Time-Series Target Construction
NovaMart Retail Challenge - Day 2

This script engineers predictive features and target labels from the master dataset:
1. Loads processed/master_analytics_dataset.csv and sorts by [store_id, product_id, date].
2. Creates Calendar/Temporal Features:
   - day_of_week (0-6)
   - is_weekend (binary flag: Saturday/Sunday = 1)
   - month
3. Creates Time-Series Lag & Rolling Window Features (grouped by store and product):
   - lag_1_demand: Previous day units sold (shift 1)
   - lag_7_demand: Same day last week units sold (shift 7)
   - rolling_mean_7_demand: 7-day trailing average demand (excluding current day to prevent leakage)
4. Creates Inventory & Supply Dynamics Features:
   - days_of_inventory: closing stock / rolling_mean_7_demand (handles div/0 safely)
   - reorder_gap: closing stock - reorder_lvl
5. Creates Supervised Machine Learning Target Variables:
   - next_7_day_demand: Cumulative units sold over the upcoming 7 days (rolling 7-day sum shifted -7)
   - stockout_flag: 1 if closing stock == 0 else 0
6. Drops boundary NaN rows from lag/forward windows and saves to data/processed/model_ready_data.csv.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd


def load_master_dataset(data_path: Path) -> pd.DataFrame:
    """Load cleaned master analytics dataset."""
    if not data_path.exists():
        raise FileNotFoundError(f"Master analytics dataset not found at: {data_path}")
    df = pd.read_csv(data_path)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer temporal, lag, rolling, inventory, and target features."""
    # Ensure chronological order per time series entity
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["store_id", "product_id", "date"]).reset_index(drop=True)

    print("  -> Creating Calendar & Temporal Features ...")
    df["day_of_week"] = df["date"].dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["month"] = df["date"].dt.month

    print("  -> Generating Lag and Trailing Window Demand Features ...")
    grp = df.groupby(["store_id", "product_id"])["total_units_sold"]
    df["lag_1_demand"] = grp.shift(1)
    df["lag_7_demand"] = grp.shift(7)
    df["rolling_mean_7_demand"] = grp.transform(lambda s: s.shift(1).rolling(7, min_periods=7).mean())

    print("  -> Engineering Inventory Velocity & Replenishment Features ...")
    # days_of_inventory: safely handle zero trailing demand
    df["days_of_inventory"] = np.where(
        df["rolling_mean_7_demand"] > 0,
        (df["closing"] / df["rolling_mean_7_demand"]).round(2),
        np.where(df["closing"] > 0, 99.0, 0.0),
    )
    df["reorder_gap"] = df["closing"] - df["reorder_lvl"]

    print("  -> Constructing Supervised Forecasting & Stockout Targets ...")
    # Target 1: Cumulative 7-day forward demand (t+1 to t+7)
    df["next_7_day_demand"] = df.groupby(["store_id", "product_id"])["total_units_sold"].transform(
        lambda s: s.rolling(7, min_periods=7).sum().shift(-7)
    )

    # Target 2: Binary stockout event flag
    df["stockout_flag"] = (df["closing"] == 0).astype(int)

    return df


def prune_and_export(df: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    """Filter out boundary NaN records and export model-ready data."""
    initial_rows = len(df)
    model_ready = df.dropna().reset_index(drop=True)
    retained_rows = len(model_ready)
    dropped_rows = initial_rows - retained_rows

    output_path.parent.mkdir(parents=True, exist_ok=True)
    model_ready.to_csv(output_path, index=False)

    print(f"\n[Dataset Filter Summary]")
    print(f"  • Total Initial Records  : {initial_rows}")
    print(f"  • Boundary NaN Rows Dropped: {dropped_rows} (7 initial lag days + 7 target horizon days)")
    print(f"  • Final Model-Ready Rows : {retained_rows}")
    print(f"  • Features in Dataset    : {model_ready.shape[1]} columns")
    print(f"  • Stockout Incidents     : {model_ready['stockout_flag'].sum()} ({model_ready['stockout_flag'].mean()*100:.2f}%)")
    print(f"  • Exported cleanly to    : {output_path.resolve()}")

    return model_ready


def main():
    print("=" * 80)
    print("StockSense - Phase 3: Feature Engineering & Model Preparation")
    print("=" * 80)

    base_dir = Path(__file__).resolve().parent.parent
    master_path = base_dir / "data" / "processed" / "master_analytics_dataset.csv"
    model_ready_path = base_dir / "data" / "processed" / "model_ready_data.csv"

    print(f"[Step 1] Loading master analytics dataset from: {master_path} ...")
    raw_df = load_master_dataset(master_path)
    print(f"  Master dataset loaded: {raw_df.shape[0]} rows x {raw_df.shape[1]} columns.")

    print("\n[Step 2] Executing feature engineering pipeline ...")
    featured_df = engineer_features(raw_df)

    print("\n[Step 3] Finalizing model-ready dataset ...")
    model_ready_df = prune_and_export(featured_df, model_ready_path)

    print("\n" + "=" * 80)
    print("FEATURE ENGINEERING COMPLETED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    main()
