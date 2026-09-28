# Enterprise HR Analytics: Compensation Intelligence & Pay Equity Audit

## Executive Summary

This study delivers an enterprise-grade econometric and diagnostic assessment of compensation structures across **500 employees** spanning four strategic global engineering hubs: **New York**, **London**, **Toronto**, and **Bangalore**. Utilizing rigorous exploratory data analysis, peer-demographic benchmarking, and ordinary least squares (OLS) linear modeling, this report decodes the primary drivers of base salary, evaluates performance incentive efficacy, identifies internal equity risks, and detects critical salary anomalies.

The predictive regression model accounts for **90.77% of total salary variance** ($R^2 = 0.9077$, $\text{RMSE} = \$16,308.30$), establishing that organizational hierarchy and geographic location serve as the foremost determinants of compensation. Concurrently, peer-relative audit heuristics revealed **14 anomalous employee compensation packages** exceeding peer group medians by **50% to 134%**, highlighting pressing pay equity vulnerabilities that necessitate immediate executive remediation.

---

## Dataset Schema

The underlying enterprise dataset contains 500 validated, clean employee records formatted according to the following specifications:

| Column Name | Data Type | Description | Sample Value |
| :--- | :---: | :--- | :--- |
| `Employee_ID` | `object` / `str` | Unique alphanumeric identifier for each employee | `EMP0001` |
| `Designation` | `object` / `str` | Organizational hierarchy level (5 tiers: Junior Developer to Director) | `Junior Developer` |
| `Experience` | `int64` | Total professional tenure in years (ranging from 1 to 24) | `22` |
| `Education` | `object` / `str` | Highest educational degree attained (Bachelor, Master, PhD) | `Bachelor` |
| `Location` | `object` / `str` | Primary employment city hub (New York, London, Toronto, Bangalore) | `New York` |
| `Performance_Score` | `int64` | Annual performance appraisal rating (1 = Unsatisfactory to 5 = Outstanding) | `3` |
| `Salary` | `int64` | Base annual compensation denominated in US Dollars ($) | `118999` |
| `Salary_Increment` | `int64` | Annual merit-based salary bonus amount ($) | `5949` |

---

## System Architecture

```text
Day_1/Question_11/
├── README.md                                  # Executive Intelligence Report
├── data/
│   ├── hr_salary_dataset.csv                  # Raw synthetic generation output
│   └── cleaned_hr_data.csv                    # Phase 1 validated and typed dataset
├── src/
│   ├── generate_data.py                       # Synthetic data generation engine
│   ├── phase1_cleaning.py                     # Schema validation & type casting
│   ├── phase2_eda.py                          # Cross-sectional EDA & distribution analysis
│   ├── phase3_analytics.py                    # Experience tenure & merit increment analytics
│   ├── phase4_anomalies.py                    # Peer-group anomaly & equity audit engine
│   └── phase5_ml_model.py                     # Multivariate econometric regression model
└── visualizations/
    ├── hr_eda_visualizations.png              # 1x2 boxplot distribution charts
    ├── hr_performance_experience.png          # 1x2 regression & merit increment charts
    └── hr_compensation_anomalies.png          # Outlier scatter plot audit visualization
```

---

## Analytical Methodology

### Phase 1: Ingestion & Schema Integrity Validation
Raw records were subjected to strict schema checks. Verified zero missing or null entries across all 8 attributes. Enforced strict numeric integer casting on monetary fields (`Salary`, `Salary_Increment`), preventing categorical bleed and floating-point rounding inaccuracies.

### Phase 2: Exploratory Data Analysis & Cross-Sectional Benchmarking
Parametric summary statistics revealed an organizational mean salary of **$117,709.32** ($\sigma = \$50,963.23$, $\text{IQR} = \$81,736 - \$143,500$). Grouped multi-level analyses isolated wide structural spreads across job titles (Director: **$222,019** vs. Junior Developer: **$84,863**) and office locations (New York: **$144,245** vs. Bangalore: **$74,040**). Side-by-side distribution boxplots were ordered by median benchmarks to evaluate dispersion and identify outlier skewness.

### Phase 3: Career Progression & Incentive Structuring
Career longevity was segmented into five discrete tranches: `0-5`, `6-10`, `11-15`, `16-20`, and `21+` years. Linear regression trendlines isolated an average annual return of approximately **$1,892 to $2,000 per year of experience**. Incentive analysis mapped performance ratings (1 to 5) directly to percentage increment rates, confirming that performance bonuses function as an independent variable overlay rather than a structural determinant of base pay.

### Phase 4: Peer-Relative Anomaly Detection
Rather than applying naive global thresholds (e.g., global z-scores), the audit implemented granular **peer-group conditioning**:
$$\text{Peer\_Median} = \text{median}(\text{Salary} \mid \text{Designation}, \text{Location}, \text{Experience\_Group})$$
$$\text{Variance \%} = \frac{\text{Salary} - \text{Peer\_Median}}{\text{Peer\_Median}} \times 100$$
Employees exhibiting a **Variance % > 50%** were isolated, audited, and mapped onto a high-contrast diagnostic scatter plot.

### Phase 5: Multivariate Econometric Regression
An Ordinary Least Squares (OLS) regression model was trained on an 80/20 train-test partition using one-hot encoded covariates (`drop_first=True`) to quantify the marginal dollar impact of each demographic and organizational factor.

---

## The 7 Key Findings

```text
========================================================================================
                          EXECUTIVE DASHBOARD OF CORE FINDINGS
========================================================================================
1. GEOGRAPHIC ARBITRAGE DOMINATES BASE PAY
   New York carries a +$67,875.58 premium, London +$58,777.76, and Toronto +$49,420.80
   relative to the Bangalore baseline, reflecting strict regional cost-of-living adjustments.

2. TENURE ACCRUES PREDICTABLE LINEAR RETURNS
   Each additional year of professional experience reliably yields +$1,891.90 in base
   annual compensation across all technical functions.

3. HIERARCHICAL STRATIFICATION CREATES WIDE SALARY TIERS
   Holding other variables constant, executive leadership commands substantial premiums:
   Director baseline vs. Manager (-$52,349), Team Lead (-$71,208), Senior Dev (-$106,128),
   and Junior Dev (-$131,172).

4. PERFORMANCE RATINGS DRIVE BONUS INCREMENTS, NOT BASE SALARY
   Annual appraisals are rigorously tied to incentive percentages (Rating 1: 0.00%,
   Rating 2: 1.94%, Rating 3: 4.89%, Rating 4: 7.90%, Rating 5: 11.95%), with zero
   statistically significant distortion on base salary scales (-$266 coefficient).

5. FORMAL EDUCATION EXHIBITS MINIMAL DIRECT WAGE IMPACT
   Holding role and experience constant, advanced degrees (Master's: -$1,871; PhD: -$6,934)
   do not guarantee higher compensation over Bachelor-degree holders in software engineering.

6. FOURTEEN SEVERE COMPENSATION OUTLIERS IDENTIFIED
   Fourteen individuals earn between 62% and 134% above their peer group median, primarily
   concentrated in Junior Developer and Team Lead positions in overseas hubs.

7. HIGH PREDICTIVE FIDELITY CONFIRMS SYSTEMIC STRUCTURE
   The econometric model explains 90.77% of compensation variance (R2 = 0.9077, RMSE = $16,308),
   demonstrating that the organization adheres to a consistent global compensation formula.
========================================================================================
```

---

## Strategic HR Action Plan

```mermaid
flowchart LR
    A["Immediate Anomaly Audit"] --> B["Banding Modernization"]
    B["Banding Modernization"] --> C["Performance Incentive Alignment"]
    C["Performance Incentive Alignment"] --> D["Annual Equity Governance"]
```

### 1. Immediate Pay Equity Audit & Outlier Investigation (Q1)
* **Targeted Review**: Commission an immediate compensation audit into the **14 identified anomalous employees** (e.g., `EMP0167`, `EMP0106`, `EMP0496`).
* **Root Cause Diagnostics**: Determine whether variances stem from legacy hiring sign-on packages, out-of-cycle retention counter-offers, data entry errors, or unrecorded specialized competencies.
* **Red-Circling Policy**: For employees verified as over-compensated, apply a temporary "red-circling" policy (freeze base salary while allowing performance bonus participation) until market band convergence.

### 2. Standardization of Global Compensation Bands (Q2)
* **Tiered Range Formulation**: Establish formal `Min-Mid-Max` compensation bands per designation per location with a maximum spread of $\pm 20\%$ around the target market median.
* **Elimination of Arbitrary Off-Cycle Adjustments**: Implement strict compensation committee sign-offs for any candidate offer or promotion exceeding $10\%$ of the midpoint.

### 3. Formalized Experience-to-Leveling Matrix (Q3)
* Align career tenure bands with promotional expectations: Junior Developer (`0-4` yrs), Senior Developer (`5-9` yrs), Team Lead (`10-14` yrs), Manager/Director (`15+` yrs).
* Eliminate tenure compression by ensuring promotions from Junior to Senior incorporate a baseline step-increase matching the modeled **$25,000 to $30,000 promotion differential**.

### 4. Continuous Anomaly Monitoring Engine (Ongoing)
* Embed the `phase4_anomalies.py` algorithm into the HR Information System (HRIS) compensation cycle.
* Automatically flag potential anomalies during annual review cycles before executive sign-off and payroll transmission.
