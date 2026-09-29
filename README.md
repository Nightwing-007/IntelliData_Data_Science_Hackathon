# IntelliData Data Science Hackathon 🚀

[![Python Version](https://img.shields.io/badge/Python-3.12%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.40%2B-FF4B4B.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.4%2B-F7931E.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An enterprise-grade, reproducible data science repository featuring econometric compensation modeling, internal pay equity auditing, campus brand perception diagnostics, and interactive Streamlit analytics dashboards.

---

## 🏛️ Repository Architecture

```text
IntelliData_Data_Science_Hackathon/
├── .gitignore                             # Standardized Git exclusions (IDEs, caches, envs)
├── README.md                              # Enterprise Hackathon Master Documentation
├── requirements.txt                       # Unified production dependency manifest
├── main.py                                # Unified CLI master runner & dashboard launcher
│
├── Day_1/
│   ├── Question_11/                       # Challenge 11: Enterprise HR Analytics & Pay Equity
│   │   ├── README.md                      # Executive Diagnostic & Econometric Report
│   │   ├── dashboard.py                   # Interactive Streamlit Compensation Dashboard
│   │   ├── data/
│   │   │   ├── hr_salary_dataset.csv      # Raw enterprise dataset
│   │   │   └── cleaned_hr_data.csv        # Phase 1 validated dataset
│   │   ├── src/
│   │   │   ├── generate_data.py           # Synthetic generator engine
│   │   │   ├── phase1_cleaning.py         # Schema casting & validation
│   │   │   ├── phase2_eda.py              # Cross-sectional salary distributions
│   │   │   ├── phase3_analytics.py        # Tenure progression & merit increments
│   │   │   ├── phase4_anomalies.py        # Peer-conditioned anomaly engine
│   │   │   └── phase5_ml_model.py         # Multivariate OLS econometric regression
│   │   └── visualizations/
│   │       ├── hr_eda_visualizations.png  # Boxplots by designation & location
│   │       ├── hr_performance_experience.png # Regression & merit overlays
│   │       └── hr_compensation_anomalies.png # Outlier scatter audit plot
│   │
│   └── Question_42/                       # Challenge 42: Campus Social Media Brand Analytics
│       ├── README.md                      # Technical Intelligence & NLP Diagnostic Report
│       ├── dashboard.py                   # Interactive Streamlit Social Brand Dashboard
│       ├── eda_visualizations.png         # 2x2 EDA distribution grid
│       ├── nlp_topics.png                 # TF-IDF discussion topic bar chart
│       ├── topic_trends_over_time.png     # Longitudinal keyword trajectory plot
│       ├── naive_bayes_evaluation.png     # Naive Bayes confusion matrix heatmap
│       ├── data/
│       │   ├── campus_social_media_dataset.csv # Raw harvested social records
│       │   └── cleaned_social_media_data.csv   # Deduplicated, normalized dataset
│       └── src/
│           ├── eda.py                     # Engagement distribution & platform breakdown
│           ├── nlp_analysis.py            # Text preprocessing & TF-IDF scoring
│           ├── trend_analysis.py          # Keyword tagging & monthly time series
│           └── model_pipeline.py          # Multinomial Naive Bayes classifier & metrics
```

---

## 📊 Project Overviews & Key Results

### 1. Challenge 11: Enterprise HR Analytics & Pay Equity Audit
- **Domain**: Human Capital Econometrics, Compensation Intelligence & Fair Pay Compliance.
- **Core Findings**:
  - **Predictive Generalization**: Multivariate OLS regression accounts for **90.77% of total salary variance** ($R^2 = 0.9077$, $\text{RMSE} = \$16,308$).
  - **Top Wage Drivers**: Regional geographic arbitrage commands primary premiums (New York: **+$67.8k**, London: **+$58.8k**, Toronto: **+$49.4k** vs. Bangalore baseline); tenure delivers a reliable **+$1.89k/yr** return.
  - **Pay Equity Risks**: Peer-group conditioned anomaly detection isolated **14 critical outliers** earning **50% to 134% above** their peer medians, concentrated in overseas hubs.
- **Detailed Documentation**: Refer to [Day_1/Question_11/README.md](Day_1/Question_11/README.md).

### 2. Challenge 42: Campus Social Media Brand Analytics
- **Domain**: Brand Sentiment Diagnostics, Natural Language Processing & Discourse Dynamics.
- **Core Findings**:
  - **Engagement Intensity**: High-urgency student discourse regarding **protests** generated peak engagement (**Average: 152 interactions**), followed by campus safety concerns.
  - **NLP Classifier Diagnostics**: A Multinomial Naive Bayes classifier achieved **24.49% accuracy** on 3-class sentiment prediction, demonstrating that bag-of-words/unigram TF-IDF vectors struggle with the sarcasm, slang, and context of student social media.
  - **Strategic Recommendation**: Transitioning from unigram vectors to fine-tuned contextual transformers (DistilBERT / RoBERTa) and establishing real-time alert thresholds for rapid welfare triage.
- **Detailed Documentation**: Refer to [Day_1/Question_42/README.md](Day_1/Question_42/README.md).

---

## 🚀 Quickstart & Setup Guide

### 1. Clone & Set Up Environment
```bash
git clone https://github.com/Nightwing-007/IntelliData_Data_Science_Hackathon.git
cd IntelliData_Data_Science_Hackathon

# Create and activate virtual environment (optional)
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🖥️ Running Dashboards & Pipelines

### Option A: Unified Master CLI Launcher
Execute the root runner to interactively select or launch any pipeline or dashboard:
```bash
# Interactive menu:
python main.py

# Or launch directly with flags:
python main.py --q11-dashboard    # Launch HR Analytics Dashboard
python main.py --q42-dashboard    # Launch Social Media Analytics Dashboard
python main.py --q11-pipeline     # Run all Question 11 scripts (Phases 1-5)
python main.py --q42-pipeline     # Run all Question 42 scripts
```

### Option B: Direct Streamlit Execution
Launch each dashboard independently via standard Streamlit commands:
```bash
# Launch Question 11 HR Analytics Dashboard
streamlit run Day_1/Question_11/dashboard.py

# Launch Question 42 Social Media Brand Analytics Dashboard
streamlit run Day_1/Question_42/dashboard.py
```

---

## 🔬 Tech Stack & Tools

- **Language**: Python 3.12+
- **Data Engineering**: `pandas`, `numpy`
- **Machine Learning & NLP**: `scikit-learn` (`LinearRegression`, `MultinomialNB`, `TfidfVectorizer`)
- **Data Visualization**: `matplotlib`, `seaborn`, `pillow`
- **Interactive UI**: `streamlit`
- **Version Control**: Git / GitHub

---

## 👥 Contributors & Authors
- **Team**: IntelliData Hackathon Engineering Team
- **Repository**: [Nightwing-007/IntelliData_Data_Science_Hackathon](https://github.com/Nightwing-007/IntelliData_Data_Science_Hackathon)
