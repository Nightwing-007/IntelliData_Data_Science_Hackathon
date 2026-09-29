"""
StockSense - Synthetic Retail Data Generator
NovaMart Retail Challenge - Day 2 Setup

Generates 5 realistic retail datasets spanning August 2026 (31 days) across 4 stores (S01-S04)
and 10 distinct products (P01-P10), featuring realistic demand dynamics, supply chain lead times,
and intentional anomalies, missing values, and outliers to simulate real-world messy data.
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd


def set_seed(seed: int = 42) -> None:
    """Set random seed for deterministic generation."""
    np.random.seed(seed)


def generate_products() -> pd.DataFrame:
    """
    Generate products.csv with 10 distinct products across categories.
    Schema: product_id, category, sub_category, brand, mrp, cost_price, shelf_life_days, supplier_id
    """
    products_data = [
        {
            "product_id": "P01",
            "category": "Dairy",
            "sub_category": "Fresh Milk",
            "brand": "Amul",
            "mrp": 66.0,
            "cost_price": 52.0,
            "shelf_life_days": 3,
            "supplier_id": "SUP001",
        },
        {
            "product_id": "P02",
            "category": "Dairy",
            "sub_category": "Butter",
            "brand": "Amul",
            "mrp": 275.0,
            "cost_price": 235.0,
            "shelf_life_days": 180,
            "supplier_id": "SUP001",
        },
        {
            "product_id": "P03",
            "category": "Bakery",
            "sub_category": "Whole Wheat Bread",
            "brand": "Britannia",
            "mrp": 45.0,
            "cost_price": 34.0,
            "shelf_life_days": 6,
            "supplier_id": "SUP002",
        },
        {
            "product_id": "P04",
            "category": "Produce",
            "sub_category": "Royal Gala Apples",
            "brand": "FarmFresh",
            "mrp": 180.0,
            "cost_price": 130.0,
            "shelf_life_days": 12,
            "supplier_id": "SUP003",
        },
        {
            "product_id": "P05",
            "category": "Produce",
            "sub_category": "Organic Potatoes",
            "brand": "GreenFields",
            "mrp": 40.0,
            "cost_price": 25.0,
            "shelf_life_days": 25,
            "supplier_id": "SUP003",
        },
        {
            "product_id": "P06",
            "category": "Beverages",
            "sub_category": "Tea Gold",
            "brand": "Tata",
            "mrp": 340.0,
            "cost_price": 270.0,
            "shelf_life_days": 365,
            "supplier_id": "SUP004",
        },
        {
            "product_id": "P07",
            "category": "Personal Care",
            "sub_category": "Antiseptic Soap",
            "brand": "Dettol",
            "mrp": 125.0,
            "cost_price": 95.0,
            "shelf_life_days": 730,
            "supplier_id": "SUP005",
        },
        {
            "product_id": "P08",
            "category": "Staples",
            "sub_category": "Basmati Rice 5kg",
            "brand": "India Gate",
            "mrp": 520.0,
            "cost_price": 415.0,
            "shelf_life_days": 365,
            "supplier_id": "SUP006",
        },
        {
            "product_id": "P09",
            "category": "Staples",
            "sub_category": "Sunflower Oil 1L",
            "brand": "Fortune",
            "mrp": 175.0,
            "cost_price": 140.0,
            "shelf_life_days": 270,
            "supplier_id": "SUP006",
        },
        {
            "product_id": "P10",
            "category": "Snacks",
            "sub_category": "Gold Biscuits",
            "brand": "Parle",
            "mrp": 60.0,
            "cost_price": 45.0,
            "shelf_life_days": 180,
            "supplier_id": "SUP007",
        },
    ]
    df = pd.DataFrame(products_data)

    # Intentional missing values
    # P05 (loose farm potatoes) missing shelf_life_days, P10 missing supplier_id
    df.loc[df["product_id"] == "P05", "shelf_life_days"] = np.nan
    df.loc[df["product_id"] == "P10", "supplier_id"] = np.nan

    return df


def generate_stores() -> pd.DataFrame:
    """
    Generate stores.csv covering 4 stores S01-S04 across distinct regions and formats.
    Schema: store_id, city, store_type, floor_area_sqft, avg_daily_customers, region
    """
    stores_data = [
        {
            "store_id": "S01",
            "city": "Mumbai",
            "store_type": "Hypermarket",
            "floor_area_sqft": 28500,
            "avg_daily_customers": 3400,
            "region": "West",
        },
        {
            "store_id": "S02",
            "city": "Delhi",
            "store_type": "Supermarket",
            "floor_area_sqft": 16200,
            "avg_daily_customers": 2250,
            "region": "North",
        },
        {
            "store_id": "S03",
            "city": "Bengaluru",
            "store_type": "Express",
            "floor_area_sqft": 6400,
            "avg_daily_customers": 1150,
            "region": "South",
        },
        {
            "store_id": "S04",
            "city": "Kolkata",
            "store_type": "Supermarket",
            "floor_area_sqft": 14800,
            "avg_daily_customers": 1850,
            "region": "East",
        },
    ]
    df = pd.DataFrame(stores_data)

    # Intentional missing value: S03 newly renovated store with unrecorded avg_daily_customers
    df.loc[df["store_id"] == "S03", "avg_daily_customers"] = np.nan

    return df


def generate_external_factors(dates: pd.DatetimeIndex, cities: list[str]) -> pd.DataFrame:
    """
    Generate external_factors.csv covering 31 days of August 2026 for all 4 store cities.
    Schema: date, city, temp_c, rain_mm, holiday, festival, weekend, local_event
    """
    records = []
    # August monsoon base temperature and rain probabilities
    city_temp_base = {"Mumbai": 29.5, "Delhi": 33.2, "Bengaluru": 26.4, "Kolkata": 31.0}
    city_rain_prob = {"Mumbai": 0.75, "Delhi": 0.38, "Bengaluru": 0.42, "Kolkata": 0.65}

    for dt in dates:
        date_str = dt.strftime("%Y-%m-%d")
        is_weekend = 1 if dt.dayofweek in [5, 6] else 0
        # Independence Day (Aug 15)
        is_holiday = 1 if (dt.month == 8 and dt.day == 15) else 0
        # Festivals: Independence Day (Aug 15) & Raksha Bandhan (Aug 28)
        is_festival = 1 if (dt.month == 8 and dt.day in [15, 28]) else 0

        for city in cities:
            temp = round(city_temp_base[city] + np.random.normal(0, 1.8), 1)

            # Monsoon precipitation
            has_rain = np.random.rand() < city_rain_prob[city]
            if has_rain:
                rain = round(np.random.exponential(scale=28.0) + 2.0, 1)
            else:
                rain = 0.0

            # Local events (weekend fairs, sports, local promotions)
            local_event = 1 if np.random.rand() < 0.12 else 0

            records.append({
                "date": date_str,
                "city": city,
                "temp_c": temp,
                "rain_mm": rain,
                "holiday": is_holiday,
                "festival": is_festival,
                "weekend": is_weekend,
                "local_event": local_event,
            })

    df = pd.DataFrame(records)

    # Intentional missing values (weather station transmission dropouts)
    missing_idx = np.random.choice(df.index, size=6, replace=False)
    df.loc[missing_idx[:3], "temp_c"] = np.nan
    df.loc[missing_idx[3:], "rain_mm"] = np.nan

    # Intentional outliers (sensor anomalies)
    outlier_idx_temp = np.random.choice(df.index, size=2, replace=False)
    df.loc[outlier_idx_temp[0], "temp_c"] = 88.5  # Extreme high temperature spike
    df.loc[outlier_idx_temp[1], "temp_c"] = -12.0  # Impossible negative temperature for August

    outlier_idx_rain = np.random.choice(df.index, size=1, replace=False)
    df.loc[outlier_idx_rain[0], "rain_mm"] = -25.0  # Impossible negative rainfall

    return df


def generate_transactions(
    dates: pd.DatetimeIndex,
    stores_df: pd.DataFrame,
    products_df: pd.DataFrame,
    n_transactions: int = 4200,
) -> pd.DataFrame:
    """
    Generate transactions.csv spanning August 2026.
    Schema: transaction_id, date, store_id, product_id, quantity, selling_price,
            discount_pct, promotion_flag, customer_id, payment_mode, hour
    """
    store_ids = stores_df["store_id"].tolist()
    store_weights = [0.40, 0.28, 0.12, 0.20]  # S01 (Hypermarket) has highest volume

    product_ids = products_df["product_id"].tolist()
    # High frequency essentials: milk (P01), bread (P03), potatoes (P05)
    prod_weights = [0.22, 0.08, 0.20, 0.09, 0.15, 0.06, 0.05, 0.04, 0.05, 0.06]

    mrp_map = dict(zip(products_df["product_id"], products_df["mrp"]))

    # Date distribution with weekend & festival surges
    day_weights = []
    for dt in dates:
        weight = 1.0
        if dt.dayofweek in [5, 6]:
            weight += 0.45
        if dt.month == 8 and dt.day in [15, 28]:
            weight += 0.65
        day_weights.append(weight)
    day_weights = np.array(day_weights) / sum(day_weights)

    date_strs = [dt.strftime("%Y-%m-%d") for dt in dates]
    chosen_date_strs = np.random.choice(date_strs, size=n_transactions, p=day_weights)
    chosen_stores = np.random.choice(store_ids, size=n_transactions, p=store_weights)
    chosen_products = np.random.choice(product_ids, size=n_transactions, p=prod_weights)

    # Customer loyalty pool + guest shoppers
    cust_pool = [f"CUST_{i:04d}" for i in range(1, 651)]
    chosen_customers = np.random.choice(cust_pool, size=n_transactions)

    payment_modes = ["UPI", "Credit Card", "Debit Card", "Cash", "Digital Wallet"]
    pay_weights = [0.44, 0.23, 0.15, 0.14, 0.04]
    chosen_payments = np.random.choice(payment_modes, size=n_transactions, p=pay_weights)

    # Store operating hours with lunch and evening shopping peaks
    hour_probs = np.zeros(24)
    hour_probs[8:11] = 0.05
    hour_probs[11:14] = 0.10
    hour_probs[14:17] = 0.06
    hour_probs[17:21] = 0.12
    hour_probs[21:23] = 0.04
    hour_probs = hour_probs / hour_probs.sum()
    chosen_hours = np.random.choice(np.arange(24), size=n_transactions, p=hour_probs)

    # Normal basket purchase quantities (1-5 units)
    base_qty = np.random.choice([1, 2, 3, 4, 5], size=n_transactions, p=[0.58, 0.24, 0.11, 0.04, 0.03])

    # Promotional discounts
    discounts = np.random.choice([0.0, 5.0, 10.0, 15.0, 20.0, 25.0], size=n_transactions, p=[0.55, 0.15, 0.13, 0.09, 0.05, 0.03])
    promo_flags = (discounts > 0).astype(int)

    records = []
    for i in range(n_transactions):
        txn_id = f"TXN_{2026080000 + i + 1}"
        d_str = chosen_date_strs[i]
        s_id = chosen_stores[i]
        p_id = chosen_products[i]
        q = int(base_qty[i])
        disc = float(discounts[i])
        promo = int(promo_flags[i])
        mrp = mrp_map[p_id]
        selling_price = round(mrp * (1.0 - disc / 100.0), 2)
        c_id = chosen_customers[i]
        pm = chosen_payments[i]
        hr = int(chosen_hours[i])

        records.append({
            "transaction_id": txn_id,
            "date": d_str,
            "store_id": s_id,
            "product_id": p_id,
            "quantity": q,
            "selling_price": selling_price,
            "discount_pct": disc,
            "promotion_flag": promo,
            "customer_id": c_id,
            "payment_mode": pm,
            "hour": hr,
        })

    df = pd.DataFrame(records)

    # INTENTIONAL OUTLIERS AND MESSY DATA INJECTIONS:
    # 1. Negative quantities (customer returns / order cancellations / register voids)
    neg_indices = np.random.choice(df.index, size=50, replace=False)
    df.loc[neg_indices, "quantity"] = np.random.choice([-1, -2, -3, -5], size=50)

    # 2. Extreme bulk quantities / data keying errors
    bulk_indices = np.random.choice(df.index, size=15, replace=False)
    df.loc[bulk_indices, "quantity"] = np.random.choice([75, 120, 250, 500], size=15)

    # 3. Missing quantities (unrecorded barcode scan error)
    nan_qty_indices = np.random.choice(df.index, size=35, replace=False)
    df.loc[nan_qty_indices, "quantity"] = np.nan

    # 4. Missing customer IDs (walk-in guest checkouts without loyalty registration)
    nan_cust_indices = np.random.choice(df.index, size=280, replace=False)
    df.loc[nan_cust_indices, "customer_id"] = np.nan

    # 5. Missing payment mode (communication timeout with POS terminal)
    nan_pay_indices = np.random.choice(df.index, size=25, replace=False)
    df.loc[nan_pay_indices, "payment_mode"] = np.nan

    # 6. Selling price anomalies (zero price samples, negative prices, missing values)
    price_indices = np.random.choice(df.index, size=30, replace=False)
    df.loc[price_indices[:12], "selling_price"] = np.nan
    df.loc[price_indices[12:20], "selling_price"] = 0.0
    df.loc[price_indices[20:], "selling_price"] = -45.0

    # 7. Discount percentage anomalies (typos like 150% or negative discounts)
    disc_indices = np.random.choice(df.index, size=12, replace=False)
    df.loc[disc_indices[:6], "discount_pct"] = 125.0
    df.loc[disc_indices[6:], "discount_pct"] = -10.0

    # 8. Out-of-bounds hour values (glitched timestamps: 25 or -1)
    hr_indices = np.random.choice(df.index, size=10, replace=False)
    df.loc[hr_indices[:5], "hour"] = 25
    df.loc[hr_indices[5:], "hour"] = -1

    return df


def generate_inventory(
    dates: pd.DatetimeIndex,
    stores_df: pd.DataFrame,
    products_df: pd.DataFrame,
    transactions_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Generate inventory.csv simulating daily stock movements across 31 days.
    Schema: date, store_id, product_id, opening, received, sold, closing, reorder_lvl, lead_days
    """
    # Calculate daily aggregated sales from transactions (treating positive sales)
    valid_txns = transactions_df[transactions_df["quantity"] > 0].copy()
    daily_sales_df = (
        valid_txns.groupby(["date", "store_id", "product_id"])["quantity"]
        .sum()
        .reset_index()
        .rename(columns={"quantity": "actual_sold"})
    )

    sales_lookup = {}
    for _, row in daily_sales_df.iterrows():
        sales_lookup[(row["date"], row["store_id"], row["product_id"])] = int(row["actual_sold"])

    # Base parameters per product: lead_days and reorder_lvl
    product_params = {
        "P01": {"reorder_lvl": 40, "lead_days": 1, "batch": 100, "initial": 90},   # Milk (daily delivery)
        "P02": {"reorder_lvl": 25, "lead_days": 3, "batch": 60, "initial": 70},    # Butter
        "P03": {"reorder_lvl": 45, "lead_days": 1, "batch": 90, "initial": 80},    # Bread
        "P04": {"reorder_lvl": 30, "lead_days": 2, "batch": 70, "initial": 65},    # Apples
        "P05": {"reorder_lvl": 50, "lead_days": 2, "batch": 120, "initial": 110},  # Potatoes
        "P06": {"reorder_lvl": 20, "lead_days": 4, "batch": 50, "initial": 55},    # Tea
        "P07": {"reorder_lvl": 25, "lead_days": 5, "batch": 60, "initial": 60},    # Soap
        "P08": {"reorder_lvl": 20, "lead_days": 4, "batch": 45, "initial": 50},    # Rice
        "P09": {"reorder_lvl": 25, "lead_days": 3, "batch": 55, "initial": 60},    # Oil
        "P10": {"reorder_lvl": 35, "lead_days": 3, "batch": 80, "initial": 75},    # Biscuits
    }

    records = []
    store_ids = stores_df["store_id"].tolist()
    product_ids = products_df["product_id"].tolist()

    # Track pending shipments: list of dicts {'arrival_date': date, 'store_id': s, 'product_id': p, 'qty': batch}
    pending_orders = []

    for s_id in store_ids:
        for p_id in product_ids:
            param = product_params[p_id]
            reorder_lvl = param["reorder_lvl"]
            lead_days = param["lead_days"]
            batch_size = param["batch"]
            curr_stock = param["initial"]

            for day_idx, dt in enumerate(dates):
                date_str = dt.strftime("%Y-%m-%d")

                # Check shipments arriving today
                received_qty = 0
                remaining_orders = []
                for order in pending_orders:
                    if order["store_id"] == s_id and order["product_id"] == p_id and order["arrival_date"] == dt:
                        received_qty += order["qty"]
                    else:
                        remaining_orders.append(order)
                pending_orders = remaining_orders

                opening_stock = curr_stock
                sold_qty = sales_lookup.get((date_str, s_id, p_id), 0)

                # Closing stock calculation
                closing_stock = opening_stock + received_qty - sold_qty

                # If stock falls below reorder level and no pending order exists for this store/product, order more
                has_pending = any(o["store_id"] == s_id and o["product_id"] == p_id for o in pending_orders)
                if closing_stock <= reorder_lvl and not has_pending:
                    arrival_dt = dt + pd.Timedelta(days=lead_days)
                    pending_orders.append({
                        "store_id": s_id,
                        "product_id": p_id,
                        "arrival_date": arrival_dt,
                        "qty": batch_size,
                    })

                records.append({
                    "date": date_str,
                    "store_id": s_id,
                    "product_id": p_id,
                    "opening": opening_stock,
                    "received": received_qty,
                    "sold": sold_qty,
                    "closing": closing_stock,
                    "reorder_lvl": reorder_lvl,
                    "lead_days": lead_days,
                })

                curr_stock = closing_stock

    df = pd.DataFrame(records)

    # INTENTIONAL INVENTORY ANOMALIES & OUTLIERS:
    # 1. Missing values in inventory counts (damaged scanner or unperformed cycle count)
    nan_indices = np.random.choice(df.index, size=24, replace=False)
    df.loc[nan_indices[:8], "opening"] = np.nan
    df.loc[nan_indices[8:16], "closing"] = np.nan
    df.loc[nan_indices[16:], "sold"] = np.nan

    # 2. Phantom negative closing inventory (theft/shrinkage or unrecorded sales backorders)
    neg_closing_indices = np.random.choice(df.index, size=12, replace=False)
    df.loc[neg_closing_indices, "closing"] = np.random.choice([-3, -8, -15], size=12)

    # 3. Data entry extreme outlier (accidental extra zeros entered by warehouse clerk)
    outlier_indices = np.random.choice(df.index, size=4, replace=False)
    df.loc[outlier_indices[:2], "closing"] = 9999
    df.loc[outlier_indices[2:], "opening"] = 8888

    return df


def main():
    set_seed(42)

    # Setup directories
    base_dir = Path(__file__).resolve().parent.parent
    raw_data_dir = base_dir / "data" / "raw"
    processed_data_dir = base_dir / "data" / "processed"

    raw_data_dir.mkdir(parents=True, exist_ok=True)
    processed_data_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("StockSense - NovaMart Retail Challenge Synthetic Data Generator")
    print("=" * 70)
    print(f"Target Output Directory: {raw_data_dir.resolve()}\n")

    # Timeframe: 31 days of August 2026
    dates = pd.date_range(start="2026-08-01", end="2026-08-31", freq="D")
    print(f"Generated Date Range: {dates[0].strftime('%Y-%m-%d')} to {dates[-1].strftime('%Y-%m-%d')} ({len(dates)} days)")

    # 1. Generate Stores
    stores_df = generate_stores()
    stores_path = raw_data_dir / "stores.csv"
    stores_df.to_csv(stores_path, index=False)
    print(f"\n[1/5] Generated stores.csv: {stores_df.shape[0]} rows, {stores_df.shape[1]} columns")
    print(f"      Stores: {', '.join(stores_df['store_id'].tolist())}")

    # 2. Generate Products
    products_df = generate_products()
    products_path = raw_data_dir / "products.csv"
    products_df.to_csv(products_path, index=False)
    print(f"\n[2/5] Generated products.csv: {products_df.shape[0]} rows, {products_df.shape[1]} columns")
    print(f"      Products: {', '.join(products_df['product_id'].tolist())}")

    # 3. Generate External Factors
    cities = stores_df["city"].tolist()
    external_df = generate_external_factors(dates, cities)
    external_path = raw_data_dir / "external_factors.csv"
    external_df.to_csv(external_path, index=False)
    print(f"\n[3/5] Generated external_factors.csv: {external_df.shape[0]} rows, {external_df.shape[1]} columns")
    print(f"      Cities: {', '.join(cities)}")

    # 4. Generate Transactions
    transactions_df = generate_transactions(dates, stores_df, products_df, n_transactions=4500)
    transactions_path = raw_data_dir / "transactions.csv"
    transactions_df.to_csv(transactions_path, index=False)
    print(f"\n[4/5] Generated transactions.csv: {transactions_df.shape[0]} rows, {transactions_df.shape[1]} columns")

    # 5. Generate Inventory
    inventory_df = generate_inventory(dates, stores_df, products_df, transactions_df)
    inventory_path = raw_data_dir / "inventory.csv"
    inventory_df.to_csv(inventory_path, index=False)
    print(f"\n[5/5] Generated inventory.csv: {inventory_df.shape[0]} rows, {inventory_df.shape[1]} columns")

    # Summary of injected messy data / anomalies
    print("\n" + "=" * 70)
    print("DATA QUALITY & MESSINESS DIAGNOSTIC SUMMARY")
    print("=" * 70)

    datasets = {
        "stores.csv": stores_df,
        "products.csv": products_df,
        "external_factors.csv": external_df,
        "transactions.csv": transactions_df,
        "inventory.csv": inventory_df,
    }

    for name, df in datasets.items():
        missing_counts = df.isnull().sum()
        missing_cols = missing_counts[missing_counts > 0].to_dict()
        print(f"\nDataset: {name} (Rows: {len(df)})")
        if missing_cols:
            print(f"  - Missing Values (NaN): {missing_cols}")
        else:
            print("  - Missing Values: None")

    print("\nInjected Outlier Highlights:")
    neg_qty_count = (transactions_df["quantity"] < 0).sum()
    bulk_qty_count = (transactions_df["quantity"] > 50).sum()
    neg_price_count = (transactions_df["selling_price"] < 0).sum()
    invalid_hour_count = ((transactions_df["hour"] < 0) | (transactions_df["hour"] > 23)).sum()
    neg_inventory = (inventory_df["closing"] < 0).sum()
    extreme_inventory = (inventory_df["closing"] > 5000).sum()
    extreme_temp = ((external_df["temp_c"] < 0) | (external_df["temp_c"] > 50)).sum()
    neg_rain = (external_df["rain_mm"] < 0).sum()

    print(f"  - Transactions with negative quantity (returns/errors): {neg_qty_count}")
    print(f"  - Transactions with extreme bulk quantity (>50): {bulk_qty_count}")
    print(f"  - Transactions with invalid negative selling price: {neg_price_count}")
    print(f"  - Transactions with out-of-range hours (<0 or >23): {invalid_hour_count}")
    print(f"  - Inventory records with negative closing stock: {neg_inventory}")
    print(f"  - Inventory records with extreme closing stock (>5000): {extreme_inventory}")
    print(f"  - External factor temperature outliers (<0C or >50C): {extreme_temp}")
    print(f"  - External factor negative rainfall values: {neg_rain}")

    print("\n" + "=" * 70)
    print("ALL 5 CSV FILES SUCCESSFULLY GENERATED AND SAVED TO RAW DIRECTORY!")
    print("=" * 70)


if __name__ == "__main__":
    main()
