# IntelliData Data Science Hackathon 🚀

[![Python Version](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.40%2B-FF4B4B.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.4%2B-F7931E.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An enterprise-grade, reproducible data science repository featuring econometric compensation modeling, natural language brand sentiment diagnostics, and AI-powered retail supply chain replenishment intelligence with interactive Streamlit prototypes.

---

## 🌟 Hackathon Tracks Overview

This repository houses the end-to-end analytical solutions developed across all competitive phases of the **IntelliData Data Science Hackathon**:

### Track 1: Enterprise HR Analytics & Pay Equity Audit (`Day_1/Question_11`)
* **Domain**: Human Capital Econometrics, Compensation Modeling & Internal Pay Equity Governance.
* **Core Methodology**: Descriptive cross-sectional EDA, peer-group conditioned variance auditing, and multivariate OLS linear regression ($R^2 = 0.9077$, $\text{RMSE} = \$16,308.30$).
* **Key Findings**: Regional geographic arbitrage dominates base compensation (New York: **+$67.8k**, London: **+$58.8k**, Toronto: **+$49.4k** vs. Bangalore baseline). Peer auditing isolated **14 anomalous employee compensation packages** earning **50% to 134% above** peer medians, concentrated in junior overseas engineering roles.
* **Interactive Prototype**: Interactive compensation explorer, peer anomaly auditor, and live regression inference calculator.

### Track 2: Campus Social Media Brand Analytics (`Day_1/Question_42`)
* **Domain**: Computational Linguistics, Social Listening & Crisis Sentiment Diagnostics.
* **Core Methodology**: Text normalization, TF-IDF unigram topic scoring, longitudinal time-series trajectory tracking across critical incidents, and Multinomial Naive Bayes classifier benchmarking.
* **Key Findings**: High-urgency student discourse concerning **campus protests** drove peak interaction velocity (**152.0 Average Engagement**). Baseline Naive Bayes accuracy of **24.49%** mathematically proved that unigram bag-of-words models fail on sarcastic social media context, recommending migration to deep learning transformer architectures (DistilBERT).
* **Interactive Prototype**: Real-time post volume counters, interactive TF-IDF topic extractors, time-series keyword trajectory tracking, and sentiment model testing sandbox.

### Track 3: StockSense Retail AI Replenishment (`Day_2/StockSense`)
* **Domain**: Retail Supply Chain Optimization, Time-Series Demand Forecasting & Inventory Risk Intelligence.
* **Core Methodology**: Relational data reconciliation ($\text{closing} = \max(0, \text{opening} + \text{received} - \text{sold})$ at the $1,240$-row Date $\times$ Store $\times$ Product grain), temporal/lag/inventory velocity feature engineering, and dual Random Forest modeling (Forecasting Regressor & Stockout Risk Classifier).
* **Key Findings**: Active promotions surge daily demand by **+222.7%** ($p < 0.001$), generating **83.3%** of all observed stockouts. The Random Forest risk model achieved perfect discriminative accuracy (**ROC-AUC: 1.0000**), identifying **$57,552.00 in revenue at risk** and prescribing **4,656 units** in targeted replenishment orders.
* **Interactive Prototype**: Executive KPI cards, manager-ready replenishment table with color-coded risk alerts, model explainability bar chart, and one-click order dispatch CSV export.

---

## 🔗 Project Quick Links

| Project Track | Problem Domain | Executive Pitch Report | Interactive Streamlit Dashboard | CLI Pipeline Runner |
| :--- | :--- | :--- | :--- | :--- |
| **Day 1: Track 1** | Enterprise HR & Pay Equity | [Day_1/Question_11/README.md](Day_1/Question_11/README.md) | `streamlit run Day_1/Question_11/dashboard.py` | `python main.py --q11-pipeline` |
| **Day 1: Track 2** | Social Media Brand Sentiment | [Day_1/Question_42/README.md](Day_1/Question_42/README.md) | `streamlit run Day_1/Question_42/dashboard.py` | `python main.py --q42-pipeline` |
| **Day 2: Track 3** | StockSense Supply Chain AI | [Day_2/StockSense/README.md](Day_2/StockSense/README.md) | `streamlit run Day_2/StockSense/dashboard.py` | `python main.py --day2-pipeline` |

---

## 🏛️ Annotated Repository Architecture

```text
Data_Science_Hackathon/
├── main.py                                      # Master CLI orchestrator & unified launcher
├── README.md                                    # Master Hackathon index & documentation
├── requirements.txt                             # Unified production dependency manifest
├── .gitignore                                   # Standardized Git exclusions (caches, envs, IDEs)
│
├── Day_1/
│   ├── Question_11/                             # [Track 1] Enterprise HR Analytics & Pay Equity Audit
│   │   ├── README.md                            # Executive intelligence pitch report
│   │   ├── dashboard.py                         # Interactive Streamlit HR analytics dashboard
│   │   ├── data/
│   │   │   ├── hr_salary_dataset.csv            # Raw synthetic employee compensation dataset
│   │   │   └── cleaned_hr_data.csv              # Type-validated, schema-cleansed dataset
│   │   ├── src/
│   │   │   ├── generate_data.py                 # Synthetic HR data generation engine
│   │   │   ├── phase1_cleaning.py               # Data integrity, typing & null validation
│   │   │   ├── phase2_eda.py                    # Cross-sectional EDA & distribution profiling
│   │   │   ├── phase3_analytics.py              # Tenure returns & merit bonus increment modeling
│   │   │   ├── phase4_anomalies.py              # Peer-group variance heuristic audit engine
│   │   │   └── phase5_ml_model.py               # Econometric OLS regression model (R² = 0.9077)
│   │   └── visualizations/
│   │       ├── hr_eda_visualizations.png        # Distribution boxplots by role & location
│   │       ├── hr_performance_experience.png    # Experience scaling & appraisal increment curves
│   │       └── hr_compensation_anomalies.png    # Outlier scatter plot highlighting 14 anomalies
│   │
│   └── Question_42/                             # [Track 2] Campus Social Media Brand Analytics
│       ├── README.md                            # Executive intelligence pitch report
│       ├── dashboard.py                         # Interactive Streamlit social sentiment dashboard
│       ├── eda_visualizations.png               # Engagement distribution & platform split grid
│       ├── nlp_topics.png                       # TF-IDF unigram conversational topic ranking
│       ├── topic_trends_over_time.png           # Longitudinal monthly keyword trajectory plot
│       ├── naive_bayes_evaluation.png           # Multinomial Naive Bayes confusion matrix heatmap
│       ├── data/
│       │   ├── campus_social_media_dataset.csv  # Raw Twitter & Instagram post harvest
│       │   └── cleaned_social_media_data.csv    # Deduplicated, normalized social dataset
│       └── src/
│           ├── eda.py                           # Volume profiling & engagement dispersion engine
│           ├── nlp_analysis.py                  # Text cleaning & TF-IDF topic scoring pipeline
│           ├── trend_analysis.py                # Time-series keyword tracking (protest, unsafe, etc.)
│           └── model_pipeline.py                # Naive Bayes classification benchmark (24.49% Acc)
│
└── Day_2/
    └── StockSense/                              # [Track 3] NovaMart Retail AI Replenishment Intelligence
        ├── README.md                            # Executive intelligence & jury pitch report
        ├── dashboard.py                         # Streamlit prototype with manager-ready reorder table
        ├── models/                              # Serialized Scikit-Learn models
        │   ├── demand_model.pkl                 # Random Forest Regressor (7-Day Demand Forecast)
        │   └── risk_model.pkl                   # Random Forest Classifier (Stockout Risk Probability)
        ├── reports/
        │   └── stocksense_eda.png               # 2x3 Publication-grade EDA diagnostic grid
        ├── data/
        │   ├── raw/                             # 5 NovaMart Retail raw synthetic datasets
        │   │   ├── transactions.csv             # 4,500 POS sales records (with injected messiness)
        │   │   ├── products.csv                 # 10 SKUs across 7 retail categories
        │   │   ├── stores.csv                   # 4 retail store formats (Mumbai, Delhi, BLR, Kolkata)
        │   │   ├── inventory.csv                # 1,240 daily warehouse inventory records
        │   │   └── external_factors.csv         # 124 meteorological & calendar records
        │   └── processed/
        │       ├── master_analytics_dataset.csv # Cleaned master table (Date x Store x Product grain)
        │       ├── model_ready_data.csv         # 680 rows x 41 lag, rolling & velocity features
        │       ├── test_set_predictions.csv     # Out-of-sample ML model predictions
        │       └── scored_predictions.csv       # Prescriptive orders, risk tiers & revenue at risk
        └── src/
            ├── generate_stock_data.py           # Synthetic generator creating 5 raw CSV datasets
            ├── round1_data_prep.py              # Data cleansing, reconciliation & master integration
            ├── round1_eda_stats.py              # 2x3 EDA grid renderer & scipy hypothesis tests
            ├── phase3_features.py               # Lags, trailing windows & supervised target engineering
            ├── phase4_modeling.py               # Dual ML training (Forecasting & Stockout Risk)
            └── phase5_intelligence.py           # Prescriptive reorder formula & explainability engine
```

---

## ⚡ Quickstart & Installation

### 1. Environment Setup
```bash
# Clone the repository
git clone https://github.com/Nightwing-007/IntelliData_Data_Science_Hackathon.git
cd IntelliData_Data_Science_Hackathon

# Create and activate Python virtual environment
python -m venv venv

# Windows:
.\venv\Scripts\activate
# Linux / macOS:
source venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Unified Repository Verification
To automatically audit all 28 datasets, serialized models, visual assets, and scripts across all tracks:
```bash
python main.py --verify
```

---

## 🖥️ Running Dashboards & Pipelines

### Master CLI Orchestrator (`main.py`)
Run the interactive menu or trigger specific commands via CLI flags:

```bash
# Interactive selection menu:
python main.py

# Direct Dashboard Launches:
python main.py --q11-dashboard     # Launch HR Analytics Dashboard
python main.py --q42-dashboard     # Launch Social Media Brand Dashboard
python main.py --day2-dashboard    # Launch Day 2 StockSense Dashboard

# Direct Pipeline Executions:
python main.py --q11-pipeline      # Execute Day 1 Question 11 Analytics & Modeling
python main.py --q42-pipeline      # Execute Day 1 Question 42 NLP & Classification
python main.py --day2-pipeline     # Execute Day 2 StockSense Complete AI Pipeline
```

### Direct Streamlit Execution
Each application can also be executed independently via standard Streamlit commands:

```bash
# Day 1 Track 1: HR Analytics & Pay Equity Dashboard
streamlit run Day_1/Question_11/dashboard.py

# Day 1 Track 2: Campus Social Media Brand Dashboard
streamlit run Day_1/Question_42/dashboard.py

# Day 2 Track 3: StockSense AI Replenishment Dashboard
streamlit run Day_2/StockSense/dashboard.py
```

---

## 🔬 Technology Stack

* **Language**: Python 3.10 - 3.12
* **Data Processing & Wrangling**: `pandas>=2.0.0`, `numpy>=1.24.0`
* **Scientific Computing & Statistics**: `scipy>=1.11.0`
* **Machine Learning & NLP**: `scikit-learn>=1.3.0`, `xgboost>=1.7.0`, `joblib>=1.3.0`
* **Visualization & Reporting**: `matplotlib>=3.7.0`, `seaborn>=0.12.0`, `pillow>=10.0.0`
* **Web Applications & Prototypes**: `streamlit>=1.30.0`
* **Version Control**: Git / GitHub

---

## 👥 Hackathon Team & Authors
* **Team**: IntelliData Hackathon Engineering Team
* **Repository**: [Nightwing-007/IntelliData_Data_Science_Hackathon](https://github.com/Nightwing-007/IntelliData_Data_Science_Hackathon)
