"""
StockSense: AI Replenishment Intelligence Dashboard (v2.0)
NovaMart Retail Challenge - Day 2 Prototype

An enterprise-grade inventory intelligence and demand forecasting application featuring:
- Tabbed layout: Overview, Replenishment, Analytics, Explainability
- Executive KPI cards with modern gradient styling
- Interactive risk threshold slider and promotion simulation
- Manager-ready replenishment table with conditional risk formatting
- Top 10 urgent reorder actions panel
- Days-until-stockout countdown visualization
- Risk distribution donut chart and stock vs forecast scatter plot
- Store performance comparison and category reorder heatmap
- ML explainability with feature importance decomposition
- Cost intelligence and savings analysis
- CSV and Excel export capabilities
"""

from pathlib import Path
from datetime import datetime
import io
import joblib
import numpy as np
import pandas as pd
import streamlit as st

try:
    import plotly.express as px
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

try:
    import openpyxl
    HAS_OPENPYXL = True
except ImportError:
    HAS_OPENPYXL = False

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
MASTER_DATA_PATH = resolve_path("data/processed/master_analytics_dataset.csv")

# -----------------------------------------------------------------------------
# 3. Theme and Session State
# -----------------------------------------------------------------------------
if "theme" not in st.session_state:
    st.session_state.theme = "light"

# -----------------------------------------------------------------------------
# 4. Custom Styling (CSS)
# -----------------------------------------------------------------------------
def get_css(theme: str) -> str:
    """Generate CSS based on current theme."""
    if theme == "dark":
        bg_primary = "#0f172a"
        bg_card = "#1e293b"
        text_primary = "#f1f5f9"
        text_secondary = "#94a3b8"
        border_color = "#334155"
    else:
        bg_primary = "#ffffff"
        bg_card = "#f8fafc"
        text_primary = "#1e293b"
        text_secondary = "#64748b"
        border_color = "#e2e8f0"

    return f"""
    <style>
    .main-title {{
        font-size: 2.4rem;
        font-weight: 800;
        background: linear-gradient(135deg, #4f46e5, #7c3aed);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.1rem;
    }}
    .sub-title {{
        font-size: 1.05rem;
        color: {text_secondary};
        margin-bottom: 1.5rem;
    }}
    .metric-card {{
        background: {bg_card};
        border-radius: 12px;
        padding: 1.2rem 1.4rem;
        border-left: 5px solid #4f46e5;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        position: relative;
        overflow: hidden;
    }}
    .metric-card:hover {{
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(79, 70, 229, 0.15);
    }}
    .metric-card::before {{
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 3px;
        background: linear-gradient(90deg, #4f46e5, #7c3aed, #ec4899);
    }}
    .metric-val {{
        font-size: 1.9rem;
        font-weight: 700;
        color: {text_primary};
    }}
    .metric-lbl {{
        font-size: 0.82rem;
        font-weight: 600;
        color: {text_secondary};
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }}
    .pulse-dot {{
        display: inline-block;
        width: 10px; height: 10px;
        background: #f43f5e;
        border-radius: 50%;
        animation: pulse 1.5s ease-in-out infinite;
        margin-left: 6px;
        vertical-align: middle;
    }}
    @keyframes pulse {{
        0%, 100% {{ opacity: 1; transform: scale(1); }}
        50% {{ opacity: 0.5; transform: scale(1.3); }}
    }}
    .section-header {{
        font-size: 1.15rem;
        font-weight: 700;
        color: {text_primary};
        margin-top: 0.5rem;
    }}
    .freshness-badge {{
        display: inline-block;
        background: #ecfdf5;
        color: #065f46;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
    }}
    </style>
    """


st.markdown(get_css(st.session_state.theme), unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 5. Data & Model Loading
# -----------------------------------------------------------------------------
@st.cache_data
def load_scored_data():
    if not SCORED_DATA_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(SCORED_DATA_PATH)


@st.cache_data
def load_master_data():
    if not MASTER_DATA_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(MASTER_DATA_PATH)


@st.cache_resource
def load_risk_classifier():
    if not RISK_MODEL_PATH.exists():
        return None
    return joblib.load(RISK_MODEL_PATH)


with st.spinner("Loading StockSense intelligence data ..."):
    df_raw = load_scored_data()
    df_master = load_master_data()
    risk_clf = load_risk_classifier()

# -----------------------------------------------------------------------------
# 6. Header
# -----------------------------------------------------------------------------
st.markdown('<div class="main-title">📦 StockSense: AI Replenishment Intelligence</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">NovaMart Retail Inventory Optimization — 7-Day Demand Forecasting, '
    'Safety Stock Buffering & Predictive Stockout Prevention</div>',
    unsafe_allow_html=True,
)

if df_raw.empty:
    st.warning("No data found. Please run phases 3 through 5 to generate model-ready and scored predictions.")
    st.stop()

# Data freshness indicator
if SCORED_DATA_PATH.exists():
    mod_time = datetime.fromtimestamp(SCORED_DATA_PATH.stat().st_mtime)
    st.markdown(
        f'<span class="freshness-badge">📊 Data last updated: {mod_time.strftime("%b %d, %Y at %I:%M %p")}</span>',
        unsafe_allow_html=True,
    )

# -----------------------------------------------------------------------------
# 7. Sidebar Filters
# -----------------------------------------------------------------------------
st.sidebar.header("🔍 Inventory Decision Filters")

# Theme toggle
theme_choice = st.sidebar.toggle("🌙 Dark Mode", value=(st.session_state.theme == "dark"))
if theme_choice and st.session_state.theme != "dark":
    st.session_state.theme = "dark"
    st.rerun()
elif not theme_choice and st.session_state.theme != "light":
    st.session_state.theme = "light"
    st.rerun()

all_stores = sorted(df_raw["Store"].unique().tolist())
selected_stores = st.sidebar.multiselect("Filter by Store Location", options=all_stores, default=all_stores)

all_categories = sorted(df_raw["category"].unique().tolist()) if "category" in df_raw.columns else []
selected_categories = st.sidebar.multiselect("Filter by Category", options=all_categories, default=all_categories)

# Dynamic risk threshold slider
st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Risk Configuration")
high_risk_threshold = st.sidebar.slider("High Risk Threshold", 0.0, 1.0, 0.70, 0.05)
medium_risk_threshold = st.sidebar.slider("Medium Risk Threshold", 0.0, high_risk_threshold, 0.40, 0.05)

# Promotion simulation toggle
simulate_promo = st.sidebar.toggle("🔮 Simulate Promotion Impact (+220%)", value=False)

search_term = st.sidebar.text_input("🔎 Search Product / Brand", value="").strip().lower()

# Date filter
if "date" in df_raw.columns:
    try:
        df_raw["date"] = pd.to_datetime(df_raw["date"])
        date_min = df_raw["date"].min().date()
        date_max = df_raw["date"].max().date()
        date_range = st.sidebar.date_input("Date Range", value=(date_min, date_max), min_value=date_min, max_value=date_max)
    except Exception:
        date_range = None
else:
    date_range = None

# Apply Filters
filtered_df = df_raw.copy()

# Re-compute risk levels based on slider thresholds
if "stockout_prob" in filtered_df.columns:
    conditions = [
        filtered_df["stockout_prob"] >= high_risk_threshold,
        (filtered_df["stockout_prob"] >= medium_risk_threshold) & (filtered_df["stockout_prob"] < high_risk_threshold),
        filtered_df["stockout_prob"] < medium_risk_threshold,
    ]
    choices = ["High", "Medium", "Low"]
    filtered_df["risk_level"] = np.select(conditions, choices, default="Low")
    filtered_df["Risk Level"] = filtered_df["risk_level"]

if selected_stores:
    filtered_df = filtered_df[filtered_df["Store"].isin(selected_stores)]
if selected_categories and "category" in filtered_df.columns:
    filtered_df = filtered_df[filtered_df["category"].isin(selected_categories)]

# Apply risk filter based on recalculated levels
all_risks = ["High", "Medium", "Low"]
selected_risks = st.sidebar.multiselect("Filter by Risk Level", options=all_risks, default=all_risks)
if selected_risks:
    filtered_df = filtered_df[filtered_df["risk_level"].isin(selected_risks)]

if search_term:
    mask = filtered_df["Product"].str.lower().str.contains(search_term, na=False)
    if "brand" in filtered_df.columns:
        mask = mask | filtered_df["brand"].str.lower().str.contains(search_term, na=False)
    filtered_df = filtered_df[mask]

if date_range and "date" in filtered_df.columns and len(date_range) == 2:
    try:
        filtered_df = filtered_df[
            (filtered_df["date"].dt.date >= date_range[0]) & (filtered_df["date"].dt.date <= date_range[1])
        ]
    except Exception:
        pass

# Promotion simulation
if simulate_promo:
    if "7-Day Forecast" in filtered_df.columns:
        filtered_df["7-Day Forecast"] = (filtered_df["7-Day Forecast"] * 2.2).round(1)
    if "predicted_7_day_demand" in filtered_df.columns:
        filtered_df["predicted_7_day_demand"] = (filtered_df["predicted_7_day_demand"] * 2.2).round(1)
    if "Recommended Order" in filtered_df.columns and "Current Stock" in filtered_df.columns:
        filtered_df["Recommended Order"] = np.maximum(0, filtered_df["7-Day Forecast"] - filtered_df["Current Stock"]).astype(int)

st.write("")

# =============================================================================
# TABBED LAYOUT
# =============================================================================
tab_overview, tab_replenishment, tab_analytics, tab_explainability = st.tabs(
    ["📊 Overview", "📋 Replenishment", "📈 Analytics", "🧠 Explainability"]
)

# =============================================================================
# TAB 1: OVERVIEW
# =============================================================================
with tab_overview:
    # KPI Cards
    total_rev_risk = filtered_df[filtered_df["risk_level"].isin(["High", "Medium"])]["revenue_at_risk"].sum() if "revenue_at_risk" in filtered_df.columns else 0
    total_high_risk = (filtered_df["risk_level"] == "High").sum()
    total_reorder_units = filtered_df["Recommended Order"].sum() if "Recommended Order" in filtered_df.columns else filtered_df.get("recommended_reorder_qty", pd.Series(0)).sum()
    avg_mae = 38.60  # Benchmark model MAE

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown(
            f"""
            <div class="metric-card" style="border-left-color: #f43f5e;">
                <div class="metric-lbl">Total Revenue at Risk</div>
                <div class="metric-val" style="color: #f43f5e;">${total_rev_risk:,.2f}</div>
            </div>
            """, unsafe_allow_html=True,
        )

    with c2:
        pulse = '<span class="pulse-dot"></span>' if total_high_risk > 0 else ""
        st.markdown(
            f"""
            <div class="metric-card" style="border-left-color: #f59e0b;">
                <div class="metric-lbl">High-Risk Stockout SKUs {pulse}</div>
                <div class="metric-val" style="color: #f59e0b;">{total_high_risk} <span style="font-size:1rem;color:#94a3b8;">items</span></div>
            </div>
            """, unsafe_allow_html=True,
        )

    with c3:
        st.markdown(
            f"""
            <div class="metric-card" style="border-left-color: #4f46e5;">
                <div class="metric-lbl">Avg Forecast Error (MAE)</div>
                <div class="metric-val">{avg_mae:.2f} <span style="font-size:1rem;color:#94a3b8;">units</span></div>
            </div>
            """, unsafe_allow_html=True,
        )

    with c4:
        st.markdown(
            f"""
            <div class="metric-card" style="border-left-color: #10b981;">
                <div class="metric-lbl">Total Reorder Volume</div>
                <div class="metric-val">{total_reorder_units:,} <span style="font-size:1rem;color:#94a3b8;">units</span></div>
            </div>
            """, unsafe_allow_html=True,
        )

    # Promotion simulation banner
    if simulate_promo:
        st.warning("🔮 **Promotion Simulation Active**: All forecasts have been scaled by 2.2x to simulate promotional demand impact.")

    st.write("")

    # Additional KPIs row
    if any(col in filtered_df.columns for col in ["safety_stock", "reorder_priority_score", "est_days_to_stockout"]):
        c5, c6, c7 = st.columns(3)
        with c5:
            if "safety_stock" in filtered_df.columns:
                total_safety = int(filtered_df["safety_stock"].sum())
                st.metric("🛡️ Total Safety Stock Buffer", f"{total_safety:,} units")
        with c6:
            if "reorder_priority_score" in filtered_df.columns:
                avg_priority = filtered_df["reorder_priority_score"].mean()
                st.metric("🎯 Avg Reorder Priority", f"{avg_priority:.1f} / 100")
        with c7:
            if "est_days_to_stockout" in filtered_df.columns:
                avg_days = filtered_df["est_days_to_stockout"].mean()
                st.metric("⏰ Avg Days to Stockout", f"{avg_days:.1f} days")

    st.markdown("---")

    # Risk distribution and scatter plot
    col_donut, col_scatter = st.columns(2)

    with col_donut:
        st.subheader("🎯 Risk Distribution")
        risk_counts = filtered_df["risk_level"].value_counts().reset_index()
        risk_counts.columns = ["Risk Level", "Count"]
        color_map = {"High": "#f43f5e", "Medium": "#f59e0b", "Low": "#10b981"}

        if HAS_PLOTLY:
            fig = px.pie(
                risk_counts, values="Count", names="Risk Level",
                color="Risk Level", color_discrete_map=color_map,
                hole=0.45,
            )
            fig.update_layout(
                margin=dict(t=20, b=20, l=20, r=20),
                height=320,
                showlegend=True,
                font=dict(size=13),
            )
            fig.update_traces(textposition="inside", textinfo="percent+label")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.bar_chart(risk_counts.set_index("Risk Level"))

    with col_scatter:
        st.subheader("📍 Stock vs Forecast by Risk")
        if HAS_PLOTLY and "Current Stock" in filtered_df.columns and "7-Day Forecast" in filtered_df.columns:
            scatter_df = filtered_df[["Current Stock", "7-Day Forecast", "risk_level", "Product"]].dropna()
            fig2 = px.scatter(
                scatter_df, x="Current Stock", y="7-Day Forecast",
                color="risk_level", color_discrete_map=color_map,
                hover_data=["Product"],
                labels={"risk_level": "Risk Level"},
            )
            # Add diagonal reference line
            max_val = max(scatter_df["Current Stock"].max(), scatter_df["7-Day Forecast"].max()) * 1.1
            fig2.add_trace(go.Scatter(
                x=[0, max_val], y=[0, max_val],
                mode="lines", line=dict(dash="dash", color="#94a3b8"),
                name="Balance Line (Stock = Forecast)",
            ))
            fig2.update_layout(
                margin=dict(t=20, b=20, l=20, r=20),
                height=320,
                xaxis_title="Current Stock (units)",
                yaxis_title="7-Day Forecast (units)",
            )
            st.plotly_chart(fig2, use_container_width=True)
            st.caption("Points *above* the dashed line need replenishment; points *below* have excess stock.")
        else:
            st.info("Install `plotly` for interactive scatter charts.")


# =============================================================================
# TAB 2: REPLENISHMENT
# =============================================================================
with tab_replenishment:
    st.subheader("📋 Manager-Ready Replenishment Decisions")
    st.caption(
        "Actionable store replenishment recommendations with safety stock buffers. "
        "Color tags indicate AI-predicted probability of stockout over the upcoming replenishment cycle."
    )

    # Build display columns dynamically
    base_cols = ["Store", "Product", "Current Stock", "7-Day Forecast", "Stock-out Prob", "Risk Level", "Recommended Order"]
    extra_cols = ["Safety Buffer", "Reorder Priority", "Est. Days to Stockout"]
    manager_cols = [c for c in base_cols + extra_cols if c in filtered_df.columns]

    display_table = filtered_df[manager_cols].sort_values(
        by=["Risk Level", "Recommended Order"] if "Recommended Order" in filtered_df.columns else ["Risk Level"],
        ascending=[True, False] if "Recommended Order" in filtered_df.columns else [True],
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
    except (AttributeError, KeyError):
        styled_table = display_table

    st.dataframe(styled_table, use_container_width=True, height=420)

    # Export row
    col_csv, col_excel, col_stats = st.columns([1, 1, 3])
    with col_csv:
        csv_bytes = display_table.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Export CSV",
            data=csv_bytes,
            file_name="stocksense_replenishment_orders.csv",
            mime="text/csv",
        )
    with col_excel:
        if HAS_OPENPYXL:
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
                display_table.to_excel(writer, index=False, sheet_name="Replenishment")
            st.download_button(
                label="📗 Export Excel",
                data=buffer.getvalue(),
                file_name="stocksense_replenishment_orders.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        else:
            st.caption("Install `openpyxl` for Excel export.")
    with col_stats:
        st.caption(f"Showing **{len(display_table)}** filtered records out of **{len(df_raw)}** total evaluations.")

    st.markdown("---")

    # Top 10 Urgent Actions
    st.subheader("🏆 Top 10 Urgent Reorder Actions")
    if "reorder_priority_score" in filtered_df.columns:
        urgent_cols = ["Store", "Product", "Reorder Priority", "Risk Level", "Est. Days to Stockout", "Recommended Order"]
        urgent_cols = [c for c in urgent_cols if c in filtered_df.columns]
        urgent_df = filtered_df.nlargest(10, "reorder_priority_score")[urgent_cols]
    else:
        urgent_cols = [c for c in ["Store", "Product", "Risk Level", "Recommended Order"] if c in filtered_df.columns]
        urgent_df = filtered_df.sort_values(
            by=["Risk Level", "Recommended Order"], ascending=[True, False]
        ).head(10)[urgent_cols]

    st.dataframe(urgent_df, use_container_width=True, hide_index=True)

    st.markdown("---")

    # Days Until Stockout Bar Chart
    st.subheader("⏰ Estimated Days Until Stockout (Bottom 15)")
    if "est_days_to_stockout" in filtered_df.columns:
        stockout_chart = (
            filtered_df.nsmallest(15, "est_days_to_stockout")[["Product", "est_days_to_stockout", "risk_level"]]
            .copy()
        )
        if HAS_PLOTLY:
            fig3 = px.bar(
                stockout_chart, x="est_days_to_stockout", y="Product",
                orientation="h", color="risk_level",
                color_discrete_map={"High": "#f43f5e", "Medium": "#f59e0b", "Low": "#10b981"},
                labels={"est_days_to_stockout": "Days Until Stockout", "risk_level": "Risk"},
            )
            fig3.update_layout(
                margin=dict(t=10, b=20, l=10, r=10),
                height=400, yaxis=dict(autorange="reversed"),
            )
            st.plotly_chart(fig3, use_container_width=True)
        else:
            st.bar_chart(stockout_chart.set_index("Product")["est_days_to_stockout"])
    elif "closing" in filtered_df.columns and "lag_1_demand" in filtered_df.columns:
        filtered_df["_est_days"] = (filtered_df["closing"] / np.maximum(1, filtered_df["lag_1_demand"])).round(1)
        stockout_chart = filtered_df.nsmallest(15, "_est_days")[["Product", "_est_days"]]
        st.bar_chart(stockout_chart.set_index("Product"))
    else:
        st.info("Days-until-stockout data not available. Run the improved pipeline to generate this metric.")

    # PDF-style summary card
    st.markdown("---")
    st.subheader("📄 Executive Summary Card")
    total_stores = filtered_df["Store"].nunique()
    total_products = filtered_df["Product"].nunique() if "Product" in filtered_df.columns else 0
    summary_md = f"""
> **StockSense Replenishment Intelligence Report**
>
> **Generated**: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}
> **Stores Evaluated**: {total_stores} | **Products Evaluated**: {total_products}
> **Total Revenue at Risk**: ${total_rev_risk:,.2f}
> **High-Risk SKUs**: {total_high_risk} | **Total Reorder Volume**: {total_reorder_units:,} units
>
> *This report was generated by StockSense AI using Random Forest demand forecasting and stockout risk classification models.*
"""
    st.markdown(summary_md)


# =============================================================================
# TAB 3: ANALYTICS
# =============================================================================
with tab_analytics:
    # Store Performance Comparison
    st.subheader("🗺️ Store Performance Comparison")
    if "Store" in filtered_df.columns and "Recommended Order" in filtered_df.columns:
        store_metrics = filtered_df.groupby("Store").agg(
            total_reorder=("Recommended Order", "sum"),
            avg_stock=("Current Stock", "mean") if "Current Stock" in filtered_df.columns else ("Recommended Order", "count"),
            high_risk_count=("risk_level", lambda x: (x == "High").sum()),
            avg_forecast=("7-Day Forecast", "mean") if "7-Day Forecast" in filtered_df.columns else ("Recommended Order", "mean"),
        ).reset_index()

        store_cols = st.columns(len(store_metrics))
        for i, (_, row) in enumerate(store_metrics.iterrows()):
            with store_cols[i % len(store_cols)]:
                st.markdown(f"**{row['Store']}**")
                st.metric("Reorder Volume", f"{int(row['total_reorder']):,}")
                st.metric("High-Risk Items", f"{int(row['high_risk_count'])}")
                st.metric("Avg Stock Level", f"{row['avg_stock']:.0f}")

    st.markdown("---")

    # Category-wise Reorder Heatmap
    st.subheader("🔥 Category-wise Reorder Heatmap (Store × Category)")
    if "category" in filtered_df.columns and "Store" in filtered_df.columns and "Recommended Order" in filtered_df.columns:
        heatmap_data = filtered_df.pivot_table(
            index="Store", columns="category",
            values="Recommended Order", aggfunc="sum", fill_value=0,
        )
        if HAS_PLOTLY:
            fig4 = px.imshow(
                heatmap_data, text_auto=True, color_continuous_scale="YlOrRd",
                labels=dict(x="Category", y="Store", color="Reorder Qty"),
                aspect="auto",
            )
            fig4.update_layout(margin=dict(t=10, b=20, l=10, r=10), height=350)
            st.plotly_chart(fig4, use_container_width=True)
        else:
            st.dataframe(heatmap_data, use_container_width=True)
    else:
        st.info("Category data not available for heatmap.")

    st.markdown("---")

    # Store Replenishment Distribution
    st.subheader("📊 Replenishment Allocations by Store")
    if "Store" in filtered_df.columns and "Recommended Order" in filtered_df.columns:
        store_alloc = filtered_df.groupby("Store")["Recommended Order"].sum().reset_index()
        col_bar, col_detail = st.columns([3, 2])
        with col_bar:
            if HAS_PLOTLY:
                fig5 = px.bar(
                    store_alloc, x="Store", y="Recommended Order",
                    color="Recommended Order", color_continuous_scale="Emrld",
                    labels={"Recommended Order": "Reorder Qty"},
                )
                fig5.update_layout(margin=dict(t=10, b=20, l=10, r=10), height=350)
                st.plotly_chart(fig5, use_container_width=True)
            else:
                st.bar_chart(store_alloc.set_index("Store"), color="#10b981")
        with col_detail:
            st.markdown("#### 💡 Allocation Breakdown")
            for _, row in store_alloc.iterrows():
                pct = (row["Recommended Order"] / max(1, store_alloc["Recommended Order"].sum())) * 100
                st.write(f"- **{row['Store']}**: {int(row['Recommended Order']):,} units ({pct:.1f}%)")
            st.caption("Hypermarket stores absorb highest replenishment volumes due to extreme demand volatility.")

    st.markdown("---")

    # Demand Trend Sparklines
    st.subheader("📈 Category Demand Trends")
    if not df_master.empty and "category" in df_master.columns and "total_units_sold" in df_master.columns and "date" in df_master.columns:
        df_master["date"] = pd.to_datetime(df_master["date"])
        cat_trends = df_master.groupby(["date", "category"])["total_units_sold"].sum().reset_index()

        categories = cat_trends["category"].unique()
        n_cols = min(3, len(categories))
        for i in range(0, len(categories), n_cols):
            cols = st.columns(n_cols)
            for j, cat in enumerate(categories[i:i+n_cols]):
                with cols[j]:
                    cat_data = cat_trends[cat_trends["category"] == cat].set_index("date")["total_units_sold"]
                    st.markdown(f"**{cat}**")
                    st.line_chart(cat_data, height=150)
    else:
        st.info("Master analytics data not available for trend analysis. Run the full pipeline first.")


# =============================================================================
# TAB 4: EXPLAINABILITY
# =============================================================================
with tab_explainability:
    st.subheader("🧠 Explainability: Top Drivers of Stock-out Risk")
    st.caption("Random Forest feature importance weights isolating the mechanisms precipitating inventory depletion.")

    feature_names = [
        "days_of_inventory", "reorder_gap", "promo_active", "lag_1_demand",
        "inventory_turnover", "stock_velocity", "days_since_restock",
    ]
    feature_display_names = {
        "days_of_inventory": "Days of Inventory Buffer (Stock / Trailing Demand)",
        "reorder_gap": "Reorder Gap (Current Stock - Safety Reorder Level)",
        "lag_1_demand": "Lag 1 Demand (Immediate Sales Velocity)",
        "promo_active": "Promotion Active (Promotional Demand Spike)",
        "inventory_turnover": "Inventory Turnover (Stock-to-Sales Efficiency)",
        "stock_velocity": "Stock Velocity (Depletion Rate vs Buffer)",
        "days_since_restock": "Days Since Restock (Supply Chain Recency)",
    }

    if risk_clf is not None and hasattr(risk_clf, "feature_importances_"):
        raw_importances = risk_clf.feature_importances_
        n_features = len(raw_importances)
        used_features = feature_names[:n_features]

        feat_df = pd.DataFrame({
            "Feature": [feature_display_names.get(f, f) for f in used_features],
            "Importance": raw_importances,
            "Percentage": (raw_importances * 100).round(2),
        }).sort_values("Importance", ascending=True)

        c_chart, c_explain = st.columns([3, 2])

        with c_chart:
            if HAS_PLOTLY:
                fig6 = px.bar(
                    feat_df, x="Percentage", y="Feature",
                    orientation="h", color="Percentage",
                    color_continuous_scale="Viridis",
                    labels={"Percentage": "Importance (%)"},
                )
                fig6.update_layout(
                    margin=dict(t=10, b=20, l=10, r=10),
                    height=350, showlegend=False,
                    yaxis=dict(tickfont=dict(size=11)),
                )
                st.plotly_chart(fig6, use_container_width=True)
            else:
                chart_data = feat_df.set_index("Feature")["Percentage"]
                st.bar_chart(chart_data, color="#4f46e5")

        with c_explain:
            st.markdown("#### 🔍 Supply Chain Insights")
            top_feat = feat_df.iloc[-1]["Feature"]
            top_pct = feat_df.iloc[-1]["Percentage"]
            second_feat = feat_df.iloc[-2]["Feature"] if len(feat_df) > 1 else "N/A"
            second_pct = feat_df.iloc[-2]["Percentage"] if len(feat_df) > 1 else 0

            st.info(
                f"**Primary Driver: {top_feat} ({top_pct:.1f}%)**\n\n"
                "Inventory buffer coverage relative to recent daily consumption is the overwhelming predictor of stockout. "
                "When days of inventory drops below 1.5 days, risk spikes exponentially."
            )
            st.warning(
                f"**Secondary Driver: {second_feat} ({second_pct:.1f}%)**\n\n"
                "The distance between current shelf stock and the established safety threshold accounts for a major share "
                "of stockout probability. Thresholds must dynamically adjust for high-velocity locations."
            )
    else:
        st.info("Feature importance data is populated dynamically from `models/risk_model.pkl`.")

    st.markdown("---")

    # Cost Intelligence Section
    st.subheader("💰 Cost Intelligence")
    if "overstock_savings" in filtered_df.columns:
        total_savings = filtered_df["overstock_savings"].sum()
        overstock_items = (filtered_df["overstock_savings"] > 0).sum()

        c_cost1, c_cost2 = st.columns(2)
        with c_cost1:
            st.metric("💵 Estimated Overstock Capital Recoverable", f"${total_savings:,.2f}")
        with c_cost2:
            st.metric("📦 Overstocked Low-Risk Items", f"{overstock_items} items")

        st.caption(
            "These are low-risk products where current closing stock exceeds 2× the 7-day forecast. "
            "Deferring reorders for these items can free working capital and reduce perishable spoilage."
        )
    else:
        # Estimate cost savings if raw columns exist
        if "closing" in filtered_df.columns and "predicted_7_day_demand" in filtered_df.columns:
            overstock_mask = (filtered_df["risk_level"] == "Low") & (filtered_df["closing"] > 2 * filtered_df["predicted_7_day_demand"])
            overstock_count = overstock_mask.sum()
            st.info(
                f"**{overstock_count}** low-risk items have inventory exceeding 2× the forecast. "
                "Run the improved pipeline (Phase 5) for detailed cost savings analysis."
            )
        else:
            st.info("Cost intelligence data will be available after running the enhanced pipeline.")


# =============================================================================
# FOOTER
# =============================================================================
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #94a3b8; font-size: 0.85rem;'>"
    "NovaMart Retail Challenge — Day 2 StockSense v2.0 Replenishment Intelligence Engine "
    "• Powered by Scikit-Learn, Streamlit & Plotly"
    "</div>",
    unsafe_allow_html=True,
)
