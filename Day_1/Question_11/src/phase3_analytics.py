"""HR Analytics Career Progression & Incentive Analytics Module (Phase 3).

Investigates empirical compensation dynamics across career longevity buckets
and quantifies the relationship between annual performance ratings and
salary increment percentages via regression trendlines and barplots.
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def run_analytics(
    input_path="../data/cleaned_hr_data.csv",
    output_plot="../visualizations/hr_performance_experience.png"
):
    """Execute experience progression and performance incentive analytics.

    Parameters
    ----------
    input_path : str
        Relative or absolute path to the cleaned dataset.
    output_plot : str
        Target image path for generated visualizations.
    """
    # Dynamic fallback when script is executed from different working directories
    if not os.path.exists(input_path):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        input_path = os.path.join(script_dir, "..", "data", "cleaned_hr_data.csv")
        output_plot = os.path.join(
            script_dir, "..", "visualizations", "hr_performance_experience.png"
        )

    os.makedirs(os.path.dirname(output_plot), exist_ok=True)

    print(f"Loading cleaned data from: '{input_path}'...")
    df = pd.read_csv(input_path)

    # 1. Experience Bucketing: [0-5, 6-10, 11-15, 16-20, 21+]
    bins = [-1, 5, 10, 15, 20, np.inf]
    labels = ['0-5', '6-10', '11-15', '16-20', '21+']
    df['Experience_Group'] = pd.cut(df['Experience'], bins=bins, labels=labels)

    print("\n" + "=" * 65)
    print("1. EXPERIENCE BUCKETING & CAREER TENURE SALARY GROWTH")
    print("=" * 65)
    print(f"{'Experience Group':<20}{'Average Salary':<20}{'Employee Count':<15}")
    print("-" * 65)
    exp_summary = df.groupby('Experience_Group', observed=False).agg(
        Avg_Salary=('Salary', 'mean'),
        Count=('Salary', 'count')
    )
    for group, row in exp_summary.iterrows():
        print(f"{group:<20}${row['Avg_Salary']:>12,.2f}        {int(row['Count']):<10}")

    # 2. Percentage Salary Increment Metric Derivation
    # Calculate percentage increase relative to base salary
    df['Increment_Percent'] = (df['Salary_Increment'] / df['Salary']) * 100

    print("\n" + "=" * 65)
    print("2. MERIT INCREMENT RATE BY PERFORMANCE EVALUATION SCORE")
    print("=" * 65)
    print(f"{'Performance Score':<20}{'Average Increment %':<25}{'Employee Count':<15}")
    print("-" * 65)
    perf_summary = df.groupby('Performance_Score').agg(
        Avg_Increment=('Increment_Percent', 'mean'),
        Count=('Increment_Percent', 'count')
    )
    for score, row in perf_summary.iterrows():
        print(f"{score:<20}{row['Avg_Increment']:>12.2f}%        {int(row['Count']):<10}")

    # 3. Visualizations (1x2 Side-by-Side: Regression & Incentive Barplot)
    print("\n" + "=" * 65)
    print(f"3. GENERATING ANALYTICS VISUALIZATIONS -> '{output_plot}'")
    print("=" * 65)
    sns.set_theme(style="whitegrid", font_scale=1.0)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Left Chart: Regression trendline showing experience tenure return
    sns.regplot(
        data=df,
        x='Experience',
        y='Salary',
        ax=axes[0],
        scatter_kws={'alpha': 0.5, 'color': '#2980b9'},
        line_kws={'color': '#c0392b', 'linewidth': 2.5}
    )
    axes[0].set_title("Salary Growth Trendline by Experience Tenure", fontsize=13, fontweight='bold')
    axes[0].set_xlabel("Experience (Years)", fontsize=11, fontweight='semibold')
    axes[0].set_ylabel("Salary ($)", fontsize=11, fontweight='semibold')

    # Right Chart: Incentive barplot mapping performance score to increment rate
    sns.barplot(
        data=df,
        x='Performance_Score',
        y='Increment_Percent',
        palette='Purples_r',
        ax=axes[1],
        errorbar=None,
        hue='Performance_Score',
        legend=False
    )
    axes[1].set_title("Average Increment Percentage by Performance Rating", fontsize=13, fontweight='bold')
    axes[1].set_xlabel("Performance Rating (1 - 5)", fontsize=11, fontweight='semibold')
    axes[1].set_ylabel("Average Increment (%)", fontsize=11, fontweight='semibold')
    axes[1].set_ylim(0, 14)

    # Annotate percentage labels atop each bar
    for patch in axes[1].patches:
        height = patch.get_height()
        if height > 0:
            axes[1].annotate(
                f'{height:.2f}%',
                (patch.get_x() + patch.get_width() / 2., height),
                ha='center', va='bottom',
                fontsize=11,
                fontweight='bold',
                xytext=(0, 4),
                textcoords='offset points'
            )

    plt.suptitle("HR Analytics - Experience & Performance Impact", fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_plot, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Progression and incentive charts saved to: '{output_plot}'")


if __name__ == '__main__':
    run_analytics()
