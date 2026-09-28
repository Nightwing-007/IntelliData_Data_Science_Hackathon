"""HR Analytics Exploratory Data Analysis (EDA) Module (Phase 2).

Loads verified clean employee data, calculates parametric summary statistics,
conducts cross-sectional salary aggregations across job designations and
geographic offices, and renders high-resolution boxplot distributions.
"""

import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def run_eda(
    input_path="../data/cleaned_hr_data.csv",
    output_plot="../visualizations/hr_eda_visualizations.png"
):
    """Execute exploratory statistical analysis and generate salary boxplots.

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
        output_plot = os.path.join(script_dir, "..", "visualizations", "hr_eda_visualizations.png")

    os.makedirs(os.path.dirname(output_plot), exist_ok=True)

    print(f"Loading cleaned data from: '{input_path}'...")
    df = pd.read_csv(input_path)

    # 1. Descriptive Parametric and Non-Parametric Statistics
    print("\n" + "=" * 65)
    print("1. DESCRIPTIVE STATISTICS: 'Salary' Column")
    print("=" * 65)
    salary_stats = df['Salary'].describe()
    print(salary_stats.to_string())

    # 2. Grouped Aggregations: Mean Salary by Hierarchy and Geography
    print("\n" + "=" * 65)
    print("2. GROUPED ANALYSIS: Average Salary Benchmarks")
    print("=" * 65)
    print("--- Average Salary by Designation (Sorted Descending) ---")
    mean_by_designation = df.groupby('Designation')['Salary'].mean().sort_values(ascending=False)
    for designation, mean_salary in mean_by_designation.items():
        print(f"  {designation:<22}: ${mean_salary:>12,.2f}")

    print("\n--- Average Salary by Location (Sorted Descending) ---")
    mean_by_location = df.groupby('Location')['Salary'].mean().sort_values(ascending=False)
    for location, mean_salary in mean_by_location.items():
        print(f"  {location:<22}: ${mean_salary:>12,.2f}")

    # 3. High-Resolution Visualizations (1x2 Side-by-Side Distribution Boxplots)
    print("\n" + "=" * 65)
    print(f"3. GENERATING DISTRIBUTION VISUALIZATIONS -> '{output_plot}'")
    print("=" * 65)
    sns.set_theme(style="whitegrid", font_scale=1.0)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Left Chart: Salary distribution across designations ordered by median salary
    desig_order = df.groupby('Designation')['Salary'].median().sort_values(ascending=False).index
    sns.boxplot(
        data=df,
        x='Designation',
        y='Salary',
        order=desig_order,
        palette='Blues_r',
        ax=axes[0],
        hue='Designation',
        legend=False
    )
    axes[0].set_title("Salary Distribution by Designation (Median Ranked)", fontsize=13, fontweight='bold')
    axes[0].set_xlabel("Designation", fontsize=11, fontweight='semibold')
    axes[0].set_ylabel("Salary ($)", fontsize=11, fontweight='semibold')
    axes[0].tick_params(axis='x', rotation=45)

    # Right Chart: Salary distribution across locations ordered by median salary
    loc_order = df.groupby('Location')['Salary'].median().sort_values(ascending=False).index
    sns.boxplot(
        data=df,
        x='Location',
        y='Salary',
        order=loc_order,
        palette='Greens_r',
        ax=axes[1],
        hue='Location',
        legend=False
    )
    axes[1].set_title("Salary Distribution by Office Location (Median Ranked)", fontsize=13, fontweight='bold')
    axes[1].set_xlabel("Location", fontsize=11, fontweight='semibold')
    axes[1].set_ylabel("Salary ($)", fontsize=11, fontweight='semibold')

    plt.suptitle("HR Analytics - Salary EDA Visualizations", fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_plot, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"EDA visualizations saved to: '{output_plot}'")


if __name__ == '__main__':
    run_eda()
