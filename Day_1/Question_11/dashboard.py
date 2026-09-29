"""Enterprise HR Analytics Dashboard.

Streamlit application for exploratory analysis, performance-experience modeling,
pay equity compensation anomaly auditing, and econometric regression metrics.
"""

from pathlib import Path
import pandas as pd
from PIL import Image
import streamlit as st

# -----------------------------------------------------------------------------
# 1. Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Enterprise HR Analytics & Pay Equity Dashboard",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# 2. Robust Path Resolution
# -----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent


def resolve_resource_path(relative_path: str) -> Path:
    """Resolve file path relative to the script location with robust fallbacks.

    Parameters
    ----------
    relative_path : str
        Relative path from the Question_11 root (e.g., 'data/cleaned_hr_data.csv').

    Returns
    -------
    Path
        Absolute Path object pointing to the existing file or best candidate.
    """
    candidates = [
        BASE_DIR / relative_path,
        Path(relative_path),
        Path("Day_1") / "Question_11" / relative_path,
        BASE_DIR.parent.parent / "Day_1" / "Question_11" / relative_path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return (BASE_DIR / relative_path).resolve()


DATA_PATH = resolve_resource_path("data/cleaned_hr_data.csv")
EDA_VIZ_PATH = resolve_resource_path("visualizations/hr_eda_visualizations.png")
PERF_VIZ_PATH = resolve_resource_path("visualizations/hr_performance_experience.png")
ANOMALY_VIZ_PATH = resolve_resource_path("visualizations/hr_compensation_anomalies.png")


# -----------------------------------------------------------------------------
# 3. Data Ingestion
# -----------------------------------------------------------------------------
@st.cache_data
def load_hr_data(filepath: Path) -> pd.DataFrame:
    """Load and cache the cleaned HR dataset.

    Parameters
    ----------
    filepath : Path
        Path to cleaned_hr_data.csv.

    Returns
    -------
    pd.DataFrame
        Pandas dataframe containing HR compensation records.
    """
    if not filepath.exists():
        st.error(f"Dataset not found at: {filepath}")
        return pd.DataFrame()
    return pd.read_csv(filepath)


df = load_hr_data(DATA_PATH)

# -----------------------------------------------------------------------------
# 4. Sidebar: Key Machine Learning Metrics & Econometric Impact
# -----------------------------------------------------------------------------
with st.sidebar:
    st.title("💼 HR Intelligence")
    st.caption("Econometric Model & Pay Equity Audit")
    st.markdown("---")

    st.subheader("📈 Key ML Metrics")
    col_r2, col_rmse = st.columns(2)
    with col_r2:
        st.metric(
            label="R Score (R²)",
            value="0.9077",
            help="Ordinary Least Squares Linear Regression explains 90.77% of salary variance.",
        )
    with col_rmse:
        st.metric(
            label="RMSE",
            value="$16,308",
            help="Root Mean Squared Error on out-of-sample evaluation.",
        )

    st.markdown("---")
    st.subheader("⚖️ Highest Impacting Coefficients")
    st.markdown(
        """
        - **New York**: `+$67k` *(+$67,876 baseline regional premium)*
        - **Experience**: `+$1.8k/yr` *(+$1,892/yr predictable tenure return)*
        - **London**: `+$59k` *(+$58,778 regional premium)*
        - **Toronto**: `+$49k` *(+$49,421 regional premium)*
        """
    )

    st.markdown("---")
    st.subheader("🏢 Dataset Summary")
    if not df.empty:
        st.write(f"**Total Records:** {len(df):,}")
        st.write(f"**Average Salary:** ${df['Salary'].mean():,.2f}")
        st.write(f"**Locations:** {df['Location'].nunique()} global hubs")
        st.write(f"**Job Titles:** {df['Designation'].nunique()} hierarchy tiers")

    st.markdown("---")
    st.caption("Enterprise HR Analytics Hackathon | Day 1 Question 11")

# -----------------------------------------------------------------------------
# 5. Main Header & KPI Cards
# -----------------------------------------------------------------------------
st.title("💼 Enterprise HR Analytics Dashboard")
st.markdown(
    "Comprehensive compensation intelligence, exploratory analytics, performance returns, "
    "and peer-conditioned pay equity anomaly detection."
)

if not df.empty:
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.metric("Total Workforce", f"{len(df):,} Employees")
    with kpi2:
        st.metric("Average Base Salary", f"${df['Salary'].mean():,.0f}")
    with kpi3:
        st.metric("Model Fidelity (R²)", "90.77%")
    with kpi4:
        st.metric("Critical Anomalies Detected", "14 Outliers", delta="-2.8% of Total", delta_color="inverse")

    st.markdown("---")

    # -------------------------------------------------------------------------
    # 6. Dataset Preview
    # -------------------------------------------------------------------------
    with st.expander("📋 Cleaned Dataset Preview & Schema Audit", expanded=True):
        st.markdown(f"**Source File:** `{DATA_PATH}` (Displaying {len(df)} validated records)")
        st.dataframe(
            df,
            use_container_width=True,
            column_config={
                "Salary": st.column_config.NumberColumn("Salary ($)", format="$%d"),
                "Salary_Increment": st.column_config.NumberColumn("Salary Increment ($)", format="$%d"),
                "Experience": st.column_config.NumberColumn("Experience (Yrs)", format="%d yrs"),
                "Performance_Score": st.column_config.NumberColumn("Performance Rating", format="%d ★"),
            },
        )

# -----------------------------------------------------------------------------
# 7. Analytical Tabs
# -----------------------------------------------------------------------------
tab_eda, tab_perf, tab_anom = st.tabs([
    "Exploratory Analysis",
    "Performance & Growth",
    "Compensation Anomalies",
])

# Tab 1: Exploratory Analysis
with tab_eda:
    st.header("📊 Exploratory Analysis: Cross-Sectional Salary Distributions")
    st.markdown(
        "Analysis of base salary dispersion across hierarchical job designations "
        "and geographical engineering hubs (New York, London, Toronto, Bangalore)."
    )

    if EDA_VIZ_PATH.exists():
        eda_img = Image.open(EDA_VIZ_PATH)
        st.image(
            eda_img,
            caption="Figure 1: Salary Distribution by Designation and Location (Median-Sorted Boxplots)",
            use_container_width=True,
        )
    else:
        st.warning(f"Visualization not found at: {EDA_VIZ_PATH}")

    st.markdown(
        """
        ### 🔍 Key Findings & Structural Takeaways
        - **Hierarchical Stratification**: Distinct base salary brackets exist across organizational levels.
          - **Director**: Median **~$222,000**
          - **Manager**: Median **~$170,000**
          - **Team Lead**: Median **~$148,000**
          - **Senior Developer**: Median **~$113,000**
          - **Junior Developer**: Median **~$85,000**
        - **Geographic Arbitrage**: New York exhibits the highest base compensation (Median **~$144k**),
          followed by London (**~$133k**), Toronto (**~$120k**), and Bangalore (**~$74k**), reflecting
          regional market rate adjustments and cost-of-living premiums.
        """
    )

# Tab 2: Performance & Growth
with tab_perf:
    st.header("📈 Performance, Career Longevity & Incentive Growth")
    st.markdown(
        "Empirical evaluation of professional tenure returns and merit-based annual salary bonus percentages."
    )

    if PERF_VIZ_PATH.exists():
        perf_img = Image.open(PERF_VIZ_PATH)
        st.image(
            perf_img,
            caption="Figure 2: Experience vs Salary Progression & Merit Increment Rate by Performance Score",
            use_container_width=True,
        )
    else:
        st.warning(f"Visualization not found at: {PERF_VIZ_PATH}")

    st.markdown(
        """
        ### 🔍 Career Progression & Incentive Takeaways
        - **Predictable Tenure Returns**: An average compensation growth rate of **+$1,892 to +$2,000 per year of experience**
          was observed via regression modeling across career tranches.
        - **Performance Increment Efficacy**: Annual performance appraisals function strictly as percentage bonus overlays:
          - **Rating 1 (Unsatisfactory)**: `0.00%` merit bonus
          - **Rating 2 (Needs Improvement)**: `~1.94%` merit bonus
          - **Rating 3 (Meets Expectations)**: `~4.89%` merit bonus
          - **Rating 4 (Exceeds Expectations)**: `~7.90%` merit bonus
          - **Rating 5 (Outstanding)**: `~11.95%` merit bonus
        - **Base Pay Decoupling**: Rating ratings drive bonus multipliers rather than structural base salary distortion.
        """
    )

# Tab 3: Compensation Anomalies
with tab_anom:
    st.header("🚨 Compensation Anomalies & Pay Equity Audit")
    st.markdown(
        "Peer-group conditioned outlier detection isolating individual compensation packages "
        "exceeding peer medians (`Designation` × `Location` × `Experience Group`) by **over 50%**."
    )

    if ANOMALY_VIZ_PATH.exists():
        anom_img = Image.open(ANOMALY_VIZ_PATH)
        st.image(
            anom_img,
            caption="Figure 3: Peer-Relative Compensation Equity Audit & Anomaly Detection (>50% Over Peer Median)",
            use_container_width=True,
        )
    else:
        st.warning(f"Visualization not found at: {ANOMALY_VIZ_PATH}")

    st.markdown(
        """
        ### 🔍 Audit Diagnostic Results
        - **14 Severe Compensation Outliers**: 14 employees were identified earning **50% to 134% above** their peer group medians.
        - **Vulnerability Concentrations**: Outliers are predominantly clustered in Junior Developer and Team Lead positions in overseas offices.
        - **Executive Recommendations**:
          1. Initiate targeted HR compensation audits for identified outlier Employee IDs (`EMP0167`, `EMP0106`, `EMP0496`, etc.).
          2. Enforce standardized Min-Mid-Max salary bands with $\\pm 20\\%$ tolerance.
          3. Implement temporary salary freezes ('red-circling') until peer benchmarks align.
        """
    )
