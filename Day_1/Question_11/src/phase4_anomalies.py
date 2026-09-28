"""HR Analytics Compensation Anomaly & Pay Equity Detection Module (Phase 4).

Audits enterprise compensation data for pay equity discrepancies and salary
outliers by computing peer-group benchmark medians conditioned on job
designation, office location, and career experience tier.
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def run_anomaly_detection(
    input_path="../data/cleaned_hr_data.csv",
    output_plot="../visualizations/hr_compensation_anomalies.png"
):
    """Detect peer-relative compensation anomalies and produce an audit scatter plot.

    Parameters
    ----------
    input_path : str
        Relative or absolute path to the cleaned dataset.
    output_plot : str
        Target image path for the compensation anomalies scatter plot.
    """
    # Dynamic fallback when script is executed from different working directories
    if not os.path.exists(input_path):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        input_path = os.path.join(script_dir, "..", "data", "cleaned_hr_data.csv")
        output_plot = os.path.join(
            script_dir, "..", "visualizations", "hr_compensation_anomalies.png"
        )

    os.makedirs(os.path.dirname(output_plot), exist_ok=True)

    print(f"Loading cleaned data from: '{input_path}'...")
    df = pd.read_csv(input_path)

    # 1. Experience Bucketing
    bins = [-1, 5, 10, 15, 20, np.inf]
    labels = ['0-5', '6-10', '11-15', '16-20', '21+']
    df['Experience_Group'] = pd.cut(df['Experience'], bins=bins, labels=labels)

    # 2. Peer Group Benchmark Calculation
    # Calculate median salary for peers sharing identical role, city, and experience level
    df['Peer_Median_Salary'] = df.groupby(
        ['Designation', 'Location', 'Experience_Group'],
        observed=False
    )['Salary'].transform('median')

    # 3. Discrepancy / Variance Quantification
    # Percentage spread relative to peer median baseline
    df['Variance_Percent'] = (
        (df['Salary'] - df['Peer_Median_Salary']) / df['Peer_Median_Salary']
    ) * 100

    # 4. Outlier Filtering (>50% above benchmark median)
    anomalies = df[df['Variance_Percent'] > 50].sort_values(
        by='Variance_Percent',
        ascending=False
    )

    print("\n" + "=" * 80)
    print(f"COMPENSATION AUDIT: {len(anomalies)} Severe Anomalies Detected (>50% Above Peer Median)")
    print("=" * 80)
    print(f"{'Employee_ID':<15}{'Designation':<20}{'Location':<15}{'Salary':<15}{'Variance %':<15}")
    print("-" * 80)
    for _, row in anomalies.iterrows():
        print(
            f"{row['Employee_ID']:<15}"
            f"{row['Designation']:<20}"
            f"{row['Location']:<15}"
            f"${row['Salary']:<14,}"
            f"{row['Variance_Percent']:>8.2f}%"
        )

    # 5. Scatter Plot Rendering: Experience vs. Salary with Highlighted Anomalies
    print("\n" + "=" * 80)
    print(f"GENERATING ANOMALY AUDIT VISUALIZATION -> '{output_plot}'")
    print("=" * 80)
    plt.figure(figsize=(12, 7))
    sns.set_theme(style="whitegrid", font_scale=1.0)

    # Regular employee baseline population
    plt.scatter(
        df['Experience'],
        df['Salary'],
        color='#7f8c8d',
        alpha=0.6,
        s=50,
        label='Standard Employees'
    )

    # High-compensation anomaly overlay
    plt.scatter(
        anomalies['Experience'],
        anomalies['Salary'],
        color='#e74c3c',
        edgecolors='#922b21',
        linewidth=1.5,
        s=90,
        label=f'Compensation Anomalies (>50% Peer Median, n={len(anomalies)})',
        zorder=5
    )

    plt.title(
        "HR Compensation Analysis - Outlier & Anomaly Detection",
        fontsize=14,
        fontweight='bold',
        pad=15
    )
    plt.xlabel("Experience (Years)", fontsize=11, fontweight='semibold')
    plt.ylabel("Salary ($)", fontsize=11, fontweight='semibold')
    plt.legend(frameon=True, facecolor='white', framealpha=0.9, fontsize=10, loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()

    plt.savefig(output_plot, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Anomaly audit visualization saved to: '{output_plot}'")


if __name__ == '__main__':
    run_anomaly_detection()
