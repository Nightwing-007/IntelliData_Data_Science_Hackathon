"""Social Media Brand Analytics Dashboard.

Streamlit application for campus social media sentiment analytics,
exploratory data analysis, NLP topic extraction, longitudinal trend analysis,
and Naive Bayes sentiment classifier evaluation.
"""

from pathlib import Path
import pandas as pd
from PIL import Image
import streamlit as st

# -----------------------------------------------------------------------------
# 1. Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Social Media Brand Analytics",
    page_icon="📱",
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
        Relative path from the Question_42 directory.

    Returns
    -------
    Path
        Absolute Path object pointing to the target file.
    """
    candidates = [
        BASE_DIR / relative_path,
        Path(relative_path),
        Path("Day_1") / "Question_42" / relative_path,
        BASE_DIR.parent.parent / "Day_1" / "Question_42" / relative_path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return (BASE_DIR / relative_path).resolve()


DATA_PATH = resolve_resource_path("data/cleaned_social_media_data.csv")
EDA_IMG_PATH = resolve_resource_path("eda_visualizations.png")
NLP_IMG_PATH = resolve_resource_path("nlp_topics.png")
TRENDS_IMG_PATH = resolve_resource_path("topic_trends_over_time.png")
NB_IMG_PATH = resolve_resource_path("naive_bayes_evaluation.png")


# -----------------------------------------------------------------------------
# 3. Data Ingestion
# -----------------------------------------------------------------------------
@st.cache_data
def load_data(filepath: Path) -> pd.DataFrame:
    """Load and cache the cleaned social media dataset.

    Parameters
    ----------
    filepath : Path
        Path to cleaned_social_media_data.csv.

    Returns
    -------
    pd.DataFrame
        Loaded pandas DataFrame.
    """
    if not filepath.exists():
        st.error(f"Dataset not found at: {filepath}")
        return pd.DataFrame()
    return pd.read_csv(filepath)


df = load_data(DATA_PATH)

# -----------------------------------------------------------------------------
# 4. Sidebar: Key Metrics & Core Insights
# -----------------------------------------------------------------------------
with st.sidebar:
    st.title("📱 Social Media Analytics")
    st.caption("Campus Brand Intelligence & Sentiment Diagnostic")
    st.markdown("---")

    st.subheader("📊 Key Metrics")
    st.metric(
        label="Model Performance",
        value="Accuracy: 24.49%",
        help="Multinomial Naive Bayes classifier evaluated on a 20% stratified test partition.",
    )

    st.metric(
        label="Highest Engagement Topic",
        value="Protest",
        delta="Avg Eng: 152",
        help="Posts addressing protests produced the highest mean engagement score (152) across all channels.",
    )

    st.markdown("---")
    st.subheader("💡 Core Insight")
    st.info(
        "**Core Insight:**\n\n"
        "Basic NLP vectors struggle with contextual social media sentiment."
    )

    st.markdown("---")
    if not df.empty:
        st.subheader("📋 Dataset Overview")
        st.write(f"**Total Records:** {len(df):,}")
        st.write(f"**Average Engagement:** {df['Engagement'].mean():.1f}")
        st.write(f"**Platforms:** {', '.join(sorted(df['Platform'].unique()))}")
        st.write(f"**Sentiment Classes:** {', '.join(sorted(df['Sentiment_Label'].unique()))}")

    st.markdown("---")
    st.caption("Social Media Brand Analytics | Day 1 Question 42")

# -----------------------------------------------------------------------------
# 5. Main Title & Key KPI Row
# -----------------------------------------------------------------------------
st.title("📱 Social Media Brand Analytics")
st.markdown(
    "Cross-platform monitoring, linguistic topic extraction, longitudinal engagement tracking, "
    "and NLP sentiment classification benchmark."
)

if not df.empty:
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.metric("Total Social Posts", f"{len(df):,}")
    with kpi2:
        st.metric("Mean Engagement", f"{df['Engagement'].mean():.1f}")
    with kpi3:
        st.metric("Top Engagement Topic", "Protest (Avg: 152)")
    with kpi4:
        st.metric("Naive Bayes Accuracy", "24.49%", delta="-8.8% vs Random", delta_color="inverse")

    st.markdown("---")

    # -------------------------------------------------------------------------
    # 6. Data Preview
    # -------------------------------------------------------------------------
    with st.expander("📋 Cleaned Dataset Preview", expanded=True):
        st.markdown(f"**Source File:** `{DATA_PATH}` (Displaying {len(df)} validated social records)")
        st.dataframe(
            df,
            use_container_width=True,
            column_config={
                "Engagement": st.column_config.NumberColumn("Engagement Score", format="%d"),
                "Timestamp": st.column_config.DatetimeColumn("Timestamp", format="YYYY-MM-DD HH:mm"),
                "Post_ID": st.column_config.TextColumn("Post ID"),
                "Platform": st.column_config.TextColumn("Platform"),
                "Sentiment_Label": st.column_config.TextColumn("Sentiment"),
                "Text": st.column_config.TextColumn("Post Text", width="large"),
            },
        )

# -----------------------------------------------------------------------------
# 7. Four Distinct Analytical Tabs
# -----------------------------------------------------------------------------
tab_eda, tab_nlp, tab_trend, tab_nb = st.tabs([
    "EDA & Distributions",
    "NLP Topic Extraction",
    "Trend Analysis",
    "Naive Bayes Evaluation",
])

# Tab 1: EDA & Distributions
with tab_eda:
    st.header("📊 Exploratory Data Analysis & Distributions")
    st.markdown(
        "Cross-sectional assessment of sentiment frequencies, platform activity volume, "
        "and distribution of user engagement metrics."
    )

    if EDA_IMG_PATH.exists():
        eda_img = Image.open(EDA_IMG_PATH)
        st.image(
            eda_img,
            caption="Figure 1: Exploratory Data Analysis & Distribution Grid (Sentiment, Platform, and Engagement Metrics)",
            use_container_width=True,
        )
    else:
        st.warning(f"Visualization not found at: {EDA_IMG_PATH}")

    st.markdown(
        """
        ### 🔍 Key Distribution Insights
        - **Sentiment Balance**: Posts span Negative, Neutral, and Positive categories with balanced volume across platforms.
        - **Channel Representation**: Activity is distributed across major campus communication hubs including Twitter and Instagram.
        - **Engagement Skew**: Engagement metrics show wide variation, with acute viral spikes driven by sensitive campus events.
        """
    )

# Tab 2: NLP Topic Extraction
with tab_nlp:
    st.header("🧠 NLP Topic Extraction (TF-IDF)")
    st.markdown(
        "Text preprocessing (case normalization, punctuation removal, custom stopword filtering) "
        "and TF-IDF vocabulary scoring identifying dominant campus discourse themes."
    )

    if NLP_IMG_PATH.exists():
        nlp_img = Image.open(NLP_IMG_PATH)
        st.image(
            nlp_img,
            caption="Figure 2: Top Campus Discussion Topics Ranked by Cumulative TF-IDF Score",
            use_container_width=True,
        )
    else:
        st.warning(f"Visualization not found at: {NLP_IMG_PATH}")

    st.markdown(
        """
        ### 🔍 Linguistic Takeaways
        - **Dominant Vocabulary**: Key operational themes extracted include `protest`, `mental health`, `safety`, `harassment`, and `police`.
        - **Campus Focus**: Topics are heavily centered around student welfare, physical security, and institutional responsiveness.
        """
    )

# Tab 3: Trend Analysis
with tab_trend:
    st.header("📈 Longitudinal Topic Trends & Engagement")
    st.markdown(
        "Time-series aggregation tracking monthly post volumes and comparative engagement metrics across primary discourse keywords."
    )

    if TRENDS_IMG_PATH.exists():
        trends_img = Image.open(TRENDS_IMG_PATH)
        st.image(
            trends_img,
            caption="Figure 3: Monthly Volume Progression and Interaction Trajectories across Actionable Keywords",
            use_container_width=True,
        )
    else:
        st.warning(f"Visualization not found at: {TRENDS_IMG_PATH}")

    st.markdown(
        """
        ### 🔍 Trend & Interaction Takeaways
        - **Highest Engagement Topic**: **Protest** posts command the highest average engagement (**152 interactions**), followed by safety concerns and mental health issues.
        - **Temporal Waves**: Keyword volume fluctuates with key campus calendar events (mid-terms, fall orientation, administrative policy changes).
        """
    )

# Tab 4: Naive Bayes Evaluation
with tab_nb:
    st.header("🎯 Naive Bayes Classifier Evaluation")
    st.markdown(
        "Out-of-sample confusion matrix and diagnostic evaluation of Multinomial Naive Bayes trained on TF-IDF textual features."
    )

    if NB_IMG_PATH.exists():
        nb_img = Image.open(NB_IMG_PATH)
        st.image(
            nb_img,
            caption="Figure 4: Multinomial Naive Bayes Confusion Matrix Heatmap (Actual vs Predicted Sentiment)",
            use_container_width=True,
        )
    else:
        st.warning(f"Visualization not found at: {NB_IMG_PATH}")

    st.markdown(
        """
        ### 🔍 Diagnostic Performance Takeaways
        - **Evaluation Metric**: Test accuracy of **24.49%** reflects substantial misclassification across 3-class sentiment partitions.
        - **Root Cause & Core Insight**: **Basic NLP vectors struggle with contextual social media sentiment.**
          Unigram TF-IDF representations fail to capture colloquial sarcasm, negation, informal campus slang, and polysemous phrases.
        - **Next Steps**: Transition to contextual embedding architectures (e.g., RoBERTa, DistilBERT, or LLM fine-tuning) for robust social sentiment inference.
        """
    )
