"""
StockSense - Phase 3: Advanced Feature Engineering & Time-Series Target Construction
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
   - ewma_7_demand: Exponentially Weighted Moving Average with span=7 (captures trend momentum)
4. Creates Advanced Demand Dynamics Features:
   - demand_acceleration: lag_1_demand - lag_7_demand (demand acceleration/deceleration)
   - inventory_turnover: total_units_sold / max(1, opening) (stock-to-sales efficiency)
   - promo_x_weekend: promo_active * is_weekend (combined promotional + weekend effects)
   - temp_deviation: temp_c - store mean temp (weather anomaly feature)
   - heavy_rain: 1 if rain_mm > 20 else 0 (demand dampening indicator)
   - days_since_restock: cumulative days since last received > 0 per store-product
   - stock_velocity: lag_1_demand / max(1, days_of_inventory) (stock depletion rate)
5. Creates Inventory & Supply Dynamics Features:
   - days_of_inventory: closing stock / rolling_mean_7_demand (handles div/0 safely)
   - reorder_gap: closing stock - reorder_lvl
6. Creates Supervised Machine Learning Target Variables:
   - next_7_day_demand: Cumulative units sold over the upcoming 7 days (rolling 7-day sum shifted -7)
   - stockout_flag: 1 if closing stock == 0 else 0
7. Drops boundary NaN rows from lag/forward windows and saves to data/processed/model_ready_data.csv.
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
    """Engineer temporal, lag, rolling, inventory, advanced dynamics, and target features."""
    # Ensure chronological order per time series entity
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["store_id", "product_id", "date"]).reset_index(drop=True)

    # -------------------------------------------------------------------------
    # Calendar & Temporal Features
    # -------------------------------------------------------------------------
    print("  -> Creating Calendar & Temporal Features ...")
    df["day_of_week"] = df["date"].dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["month"] = df["date"].dt.month

    # -------------------------------------------------------------------------
    # Lag and Rolling Window Features
    # -------------------------------------------------------------------------
    print("  -> Generating Lag, Rolling, and EWMA Demand Features ...")
    grp = df.groupby(["store_id", "product_id"])["total_units_sold"]
    df["lag_1_demand"] = grp.shift(1)
    df["lag_7_demand"] = grp.shift(7)
    df["rolling_mean_7_demand"] = grp.transform(lambda s: s.shift(1).rolling(7, min_periods=7).mean())

    # Exponentially Weighted Moving Average (captures trend momentum better than simple rolling mean)
    df["ewma_7_demand"] = df.groupby(["store_id", "product_id"])["total_units_sold"].transform(
        lambda s: s.shift(1).ewm(span=7, min_periods=7, adjust=False).mean()
    )

    # Rolling standard deviation for safety stock calculations downstream
    df["rolling_std_7_demand"] = df.groupby(["store_id", "product_id"])["total_units_sold"].transform(
        lambda s: s.shift(1).rolling(7, min_periods=7).std()
    )

    # -------------------------------------------------------------------------
    # Advanced Demand Dynamics Features
    # -------------------------------------------------------------------------
    print("  -> Engineering Advanced Demand Dynamics Features ...")

    # Demand acceleration: positive = demand accelerating, negative = decelerating
    df["demand_acceleration"] = df["lag_1_demand"] - df["lag_7_demand"]

    # Inventory turnover: how efficiently stock converts to sales
    df["inventory_turnover"] = df["total_units_sold"] / np.maximum(1, df["opening"])

    # Promotional-Weekend interaction: captures amplified demand when promotions hit weekends
    df["promo_x_weekend"] = df["promo_active"] * df["is_weekend"]

    # Temperature deviation from per-store mean (weather anomaly indicator)
    if "temp_c" in df.columns:
        store_temp_mean = df.groupby("store_id")["temp_c"].transform("mean")
        df["temp_deviation"] = (df["temp_c"] - store_temp_mean).round(2)
    else:
        df["temp_deviation"] = 0.0

    # Heavy rain indicator (monsoon demand dampening)
    if "rain_mm" in df.columns:
        df["heavy_rain"] = (df["rain_mm"] > 20).astype(int)
    else:
        df["heavy_rain"] = 0

    # Days since last restock per store-product time series
    def _days_since_restock(group):
        received_col = group["received"] if "received" in group.columns else pd.Series(0, index=group.index)
        days_counter = []
        counter = 0
        for val in received_col:
            if val > 0:
                counter = 0
            else:
                counter += 1
            days_counter.append(counter)
        return pd.Series(days_counter, index=group.index)

    df["days_since_restock"] = df.groupby(["store_id", "product_id"], group_keys=False).apply(_days_since_restock)

    # -------------------------------------------------------------------------
    # Inventory Velocity & Replenishment Features
    # -------------------------------------------------------------------------
    print("  -> Engineering Inventory Velocity & Replenishment Features ...")
    # days_of_inventory: safely handle zero trailing demand
    df["days_of_inventory"] = np.where(
        df["rolling_mean_7_demand"] > 0,
        (df["closing"] / df["rolling_mean_7_demand"]).round(2),
        np.where(df["closing"] > 0, 99.0, 0.0),
    )
    df["reorder_gap"] = df["closing"] - df["reorder_lvl"]

    # Stock velocity: how fast stock is depleting relative to buffer
    df["stock_velocity"] = df["lag_1_demand"] / np.maximum(1.0, df["days_of_inventory"])
    df["stock_velocity"] = df["stock_velocity"].round(4)

    # -------------------------------------------------------------------------
    # Supervised Forecasting & Stockout Targets
    # -------------------------------------------------------------------------
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

    # Feature summary
    new_features = [
        "ewma_7_demand", "rolling_std_7_demand", "demand_acceleration",
        "inventory_turnover", "promo_x_weekend", "temp_deviation",
        "heavy_rain", "days_since_restock", "stock_velocity",
    ]
    existing_features = [
        "day_of_week", "is_weekend", "month", "lag_1_demand", "lag_7_demand",
        "rolling_mean_7_demand", "days_of_inventory", "reorder_gap",
    ]

    print(f"\n[Dataset Filter Summary]")
    print(f"  • Total Initial Records  : {initial_rows}")
    print(f"  • Boundary NaN Rows Dropped: {dropped_rows} (7 initial lag days + 7 target horizon days)")
    print(f"  • Final Model-Ready Rows : {retained_rows}")
    print(f"  • Features in Dataset    : {model_ready.shape[1]} columns")
    print(f"  • Base Features          : {len(existing_features)} ({', '.join(existing_features)})")
    print(f"  • New Advanced Features  : {len(new_features)} ({', '.join(new_features)})")
    print(f"  • Stockout Incidents     : {model_ready['stockout_flag'].sum()} ({model_ready['stockout_flag'].mean()*100:.2f}%)")
    print(f"  • Exported cleanly to    : {output_path.resolve()}")

    return model_ready


def main():
    print("=" * 80)
    print("StockSense - Phase 3: Advanced Feature Engineering & Model Preparation")
    print("=" * 80)

    base_dir = Path(__file__).resolve().parent.parent
    master_path = base_dir / "data" / "processed" / "master_analytics_dataset.csv"
    model_ready_path = base_dir / "data" / "processed" / "model_ready_data.csv"

    print(f"[Step 1] Loading master analytics dataset from: {master_path} ...")
    raw_df = load_master_dataset(master_path)
    print(f"  Master dataset loaded: {raw_df.shape[0]} rows x {raw_df.shape[1]} columns.")

    print("\n[Step 2] Executing advanced feature engineering pipeline ...")
    featured_df = engineer_features(raw_df)

    print("\n[Step 3] Finalizing model-ready dataset ...")
    model_ready_df = prune_and_export(featured_df, model_ready_path)

    print("\n" + "=" * 80)
    print("ADVANCED FEATURE ENGINEERING COMPLETED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    main()
