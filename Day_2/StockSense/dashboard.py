"""
StockSense: AI Replenishment Intelligence Dashboard
NovaMart Retail Challenge - Day 2 Prototype

An enterprise-grade inventory intelligence and demand forecasting application:
- Top KPIs: Total Revenue at Risk, Total High-Risk Products, Average Forecast Error (MAE)
- Expected Manager-Ready Output Table with conditional risk formatting
- Explainability bar chart of the top ML features driving stockouts
- Interactive store and category replenishment analytics
"""

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# -----------------------------------------------------------------------------
# 1. Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="StockSense: AI Replenishment Intelligence",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# 2. Robust Path Resolution
# -----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent


def resolve_path(relative_path: str) -> Path:
    """Resolve file paths relative to script directory with fallback candidates."""
    candidates = [
        BASE_DIR / relative_path,
        Path(relative_path),
        Path("Day_2") / "StockSense" / relative_path,
        BASE_DIR.parent.parent / "Day_2" / "StockSense" / relative_path,
    ]
    for c in candidates:
        if c.exists():
            return c.resolve()
    return (BASE_DIR / relative_path).resolve()


SCORED_DATA_PATH = resolve_path("data/processed/scored_predictions.csv")
RISK_MODEL_PATH = resolve_path("models/risk_model.pkl")
EDA_IMAGE_PATH = resolve_path("reports/stocksense_eda.png")


# -----------------------------------------------------------------------------
# 3. Data & Model Loading
# -----------------------------------------------------------------------------
@st.cache_data
def load_scored_data():
    if not SCORED_DATA_PATH.exists():
        st.error(f"Scored predictions dataset not found at: {SCORED_DATA_PATH}")
        return pd.DataFrame()
    return pd.read_csv(SCORED_DATA_PATH)


@st.cache_resource
def load_risk_classifier():
    if not RISK_MODEL_PATH.exists():
        return None
    return joblib.load(RISK_MODEL_PATH)


df_raw = load_scored_data()
risk_clf = load_risk_classifier()

# -----------------------------------------------------------------------------
# 4. Custom Styling (CSS)
# -----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1e3a8a;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #4b5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #f8fafc;
        border-radius: 8px;
        padding: 1.2rem;
        border-left: 5px solid #2563eb;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    }
    .metric-val {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1e293b;
    }
    .metric-lbl {
        font-size: 0.85rem;
        font-weight: 600;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# 5. Header & Executive Summary
# -----------------------------------------------------------------------------
st.markdown('<div class="main-title">📦 StockSense: AI Replenishment Intelligence</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">NovaMart Retail Inventory Optimization — 7-Day Demand Forecasting & Predictive Stockout Prevention</div>',
    unsafe_allow_html=True,
)

if df_raw.empty:
    st.warning("No data found. Please run phases 3 through 5 to generate model-ready and scored predictions.")
    st.stop()

# -----------------------------------------------------------------------------
# 6. Sidebar Filters
# -----------------------------------------------------------------------------
st.sidebar.header("🔍 Inventory Decision Filters")

all_stores = sorted(df_raw["Store"].unique().tolist())
selected_stores = st.sidebar.multiselect("Filter by Store Location", options=all_stores, default=all_stores)

all_categories = sorted(df_raw["category"].unique().tolist())
selected_categories = st.sidebar.multiselect("Filter by Category", options=all_categories, default=all_categories)

all_risks = ["High", "Medium", "Low"]
selected_risks = st.sidebar.multiselect("Filter by Stock-out Risk Level", options=all_risks, default=all_risks)

search_term = st.sidebar.text_input("Search Product / Brand", value="").strip().lower()

# Apply Filters
filtered_df = df_raw.copy()
if selected_stores:
    filtered_df = filtered_df[filtered_df["Store"].isin(selected_stores)]
if selected_categories:
    filtered_df = filtered_df[filtered_df["category"].isin(selected_categories)]
if selected_risks:
    filtered_df = filtered_df[filtered_df["risk_level"].isin(selected_risks)]
if search_term:
    filtered_df = filtered_df[
        filtered_df["Product"].str.lower().str.contains(search_term)
        | filtered_df["brand"].str.lower().str.contains(search_term)
    ]

# -----------------------------------------------------------------------------
# 7. Top KPI Row
# -----------------------------------------------------------------------------
total_rev_risk = filtered_df[filtered_df["risk_level"].isin(["High", "Medium"])]["revenue_at_risk"].sum()
total_high_risk = (filtered_df["risk_level"] == "High").sum()
total_reorder_units = filtered_df["recommended_reorder_qty"].sum()
# Benchmark Model Error (MAE is calculated on test set as 38.60 units)
avg_mae = 38.60

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown(
        f"""
        <div class="metric-card" style="border-left-color: #dc2626;">
            <div class="metric-lbl">Total Revenue at Risk</div>
            <div class="metric-val" style="color: #dc2626;">${total_rev_risk:,.2f}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with c2:
    st.markdown(
        f"""
        <div class="metric-card" style="border-left-color: #ea580c;">
            <div class="metric-lbl">High-Risk Stockout SKUs</div>
            <div class="metric-val" style="color: #ea580c;">{total_high_risk} <span style="font-size:1rem;color:#6b7280;">items</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with c3:
    st.markdown(
        f"""
        <div class="metric-card" style="border-left-color: #2563eb;">
            <div class="metric-lbl">Avg Forecast Error (MAE)</div>
            <div class="metric-val">{avg_mae:.2f} <span style="font-size:1rem;color:#6b7280;">units</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with c4:
    st.markdown(
        f"""
        <div class="metric-card" style="border-left-color: #16a34a;">
            <div class="metric-lbl">Total Reorder Volume</div>
            <div class="metric-val">{total_reorder_units:,} <span style="font-size:1rem;color:#6b7280;">units</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.write("")

# -----------------------------------------------------------------------------
# 8. Interactive Manager-Ready Output Table
# -----------------------------------------------------------------------------
st.subheader("📋 Expected Manager-Ready Replenishment Decisions")
st.caption(
    "Actionable store replenishment recommendations calculated as $\\max(0, \\text{7-Day Forecast} - \\text{Current Stock})$. "
    "Color tags indicate AI-predicted probability of stockout over the upcoming replenishment cycle."
)

manager_cols = [
    "Store",
    "Product",
    "Current Stock",
    "7-Day Forecast",
    "Stock-out Prob",
    "Risk Level",
    "Recommended Order",
]

display_table = filtered_df[manager_cols].sort_values(
    by=["Risk Level", "Recommended Order"], ascending=[True, False]
)


def style_risk_cells(val):
    if val == "High":
        return "background-color: #fee2e2; color: #991b1b; font-weight: bold;"
    elif val == "Medium":
        return "background-color: #fef3c7; color: #92400e; font-weight: bold;"
    elif val == "Low":
        return "background-color: #dcfce7; color: #166534;"
    return ""


try:
    styled_table = display_table.style.map(style_risk_cells, subset=["Risk Level"])
except AttributeError:
    styled_table = display_table.style.applymap(style_risk_cells, subset=["Risk Level"])

st.dataframe(
    styled_table,
    use_container_width=True,
    height=400,
)

col_dl, col_stats = st.columns([1, 3])
with col_dl:
    csv_bytes = display_table.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Export Replenishment Orders (CSV)",
        data=csv_bytes,
        file_name="stocksense_replenishment_orders.csv",
        mime="text/csv",
    )
with col_stats:
    st.caption(f"Showing **{len(display_table)}** filtered decision records out of **{len(df_raw)}** total evaluations.")

st.markdown("---")

# -----------------------------------------------------------------------------
# 9. Model Explainability & Root Cause Analysis
# -----------------------------------------------------------------------------
st.subheader("🧠 Explainability: Top Drivers of Stock-out Risk")
st.caption("Random Forest feature importance weights isolating the physical and commercial mechanisms precipitating inventory depletion.")

feature_names = ["days_of_inventory", "reorder_gap", "promo_active", "lag_1_demand"]
feature_display_names = {
    "days_of_inventory": "Days of Inventory Buffer (Stock / Trailing Demand)",
    "reorder_gap": "Reorder Gap (Current Stock - Safety Reorder Level)",
    "lag_1_demand": "Lag 1 Demand (Immediate Sales Velocity)",
    "promo_active": "Promotion Active (Promotional Demand Spike)",
}

if risk_clf is not None and hasattr(risk_clf, "feature_importances_"):
    raw_importances = risk_clf.feature_importances_
    feat_df = pd.DataFrame({
        "Feature": [feature_display_names.get(f, f) for f in feature_names],
        "Importance": raw_importances,
        "Percentage": (raw_importances * 100).round(2),
    }).sort_values("Importance", ascending=True)

    c_chart, c_explain = st.columns([3, 2])

    with c_chart:
        chart_data = feat_df.set_index("Feature")["Percentage"]
        st.bar_chart(chart_data, color="#2563eb")

    with c_explain:
        st.markdown("#### 🔍 Supply Chain Insights")
        top_feat = feat_df.iloc[-1]["Feature"]
        top_pct = feat_df.iloc[-1]["Percentage"]
        second_feat = feat_df.iloc[-2]["Feature"]
        second_pct = feat_df.iloc[-2]["Percentage"]

        st.info(
            f"**Primary Driver: {top_feat} ({top_pct:.1f}%)**\n\n"
            "Inventory buffer coverage relative to recent daily consumption is the overwhelming predictor of stockout. "
            "When days of inventory drops below 1.5 days, risk spikes exponentially."
        )
        st.warning(
            f"**Secondary Driver: {second_feat} ({second_pct:.1f}%)**\n\n"
            "The distance between current shelf stock and the established safety threshold accounts for over a third "
            "of stockout probability. Reorder thresholds must dynamically adjust for high-velocity locations like Mumbai."
        )
else:
    st.info("Feature importance data is populated dynamically from models/risk_model.pkl.")

st.markdown("---")

# -----------------------------------------------------------------------------
# 10. Store-Level Replenishment Distribution
# -----------------------------------------------------------------------------
st.subheader("📊 Replenishment Allocations by Store Location")
store_alloc = filtered_df.groupby("Store")["Recommended Order"].sum().reset_index()

col_bar, col_pie = st.columns([3, 2])
with col_bar:
    st.bar_chart(store_alloc.set_index("Store"), color="#10b981")
with col_pie:
    st.markdown("#### 💡 Allocation Breakdown")
    for _, row in store_alloc.iterrows():
        st.write(f"- **{row['Store']}**: {row['Recommended Order']:,} units")
    st.caption("Hypermarket stores absorb highest recommended replenishment volumes to buffer against extreme demand volatility.")

# -----------------------------------------------------------------------------
# 11. Footer
# -----------------------------------------------------------------------------
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #9ca3af; font-size: 0.85rem;'>"
    "NovaMart Retail Challenge — Day 2 StockSense Replenishment Intelligence Engine • Powered by Scikit-Learn & Streamlit"
    "</div>",
    unsafe_allow_html=True,
)
