import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def perform_eda(input_csv="../data/cleaned_social_media_data.csv", output_plot="../eda_visualizations.png"):
    # Ensure path works whether executed from src/ or Question_42/
    if not os.path.exists(input_csv) and os.path.exists("data/cleaned_social_media_data.csv"):
        input_csv = "data/cleaned_social_media_data.csv"
        output_plot = "eda_visualizations.png"

    # Load dataset
    print(f"Loading '{input_csv}'...")
    df = pd.read_csv(input_csv)

    # 1. Descriptive Statistics: Print summary statistics for 'Engagement'
    print("\n" + "=" * 60)
    print("1. DESCRIPTIVE STATISTICS: 'Engagement' column")
    print("=" * 60)
    engagement_stats = df['Engagement'].describe()[['count', 'mean', 'std', 'min', 'max']]
    print(engagement_stats.to_string())

    # 2. Distributions: Print percentage breakdown of 'Sentiment_Label' and 'Platform'
    print("\n" + "=" * 60)
    print("2. DISTRIBUTIONS: Percentage Breakdown")
    print("=" * 60)
    print("--- 'Sentiment_Label' Percentage Breakdown ---")
    sentiment_dist = (df['Sentiment_Label'].value_counts(normalize=True) * 100).round(2)
    for label, pct in sentiment_dist.items():
        print(f"  {label:<12}: {pct:>6.2f}%")

    print("\n--- 'Platform' Percentage Breakdown ---")
    platform_dist = (df['Platform'].value_counts(normalize=True) * 100).round(2)
    for platform, pct in platform_dist.items():
        print(f"  {platform:<12}: {pct:>6.2f}%")

    # 3. Generate Visualizations: 2x2 grid of charts
    print("\n" + "=" * 60)
    print(f"3. GENERATING VISUALIZATIONS -> '{output_plot}'")
    print("=" * 60)
    sns.set_theme(style="whitegrid", font_scale=1.0)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Top-Left: Count plot showing distribution of 'Sentiment_Label'
    sns.countplot(
        data=df,
        x='Sentiment_Label',
        order=['Positive', 'Neutral', 'Negative'],
        palette='viridis',
        ax=axes[0, 0],
        hue='Sentiment_Label',
        legend=False
    )
    axes[0, 0].set_title("Distribution of Sentiment Labels", fontsize=13, fontweight='bold')
    axes[0, 0].set_xlabel("Sentiment Label", fontsize=11)
    axes[0, 0].set_ylabel("Count", fontsize=11)
    for p in axes[0, 0].patches:
        height = p.get_height()
        if height > 0:
            axes[0, 0].annotate(f'{int(height)}',
                                (p.get_x() + p.get_width() / 2., height),
                                ha='center', va='bottom', fontsize=10, xytext=(0, 3),
                                textcoords='offset points')

    # Top-Right: Box plot showing 'Engagement' distribution across each 'Platform'
    sns.boxplot(
        data=df,
        x='Platform',
        y='Engagement',
        palette='Set2',
        ax=axes[0, 1],
        hue='Platform',
        legend=False
    )
    axes[0, 1].set_title("Engagement Distribution across Platforms", fontsize=13, fontweight='bold')
    axes[0, 1].set_xlabel("Platform", fontsize=11)
    axes[0, 1].set_ylabel("Engagement", fontsize=11)

    # Bottom-Left: Grouped count plot showing 'Sentiment_Label' breakdown by 'Platform'
    sns.countplot(
        data=df,
        x='Platform',
        hue='Sentiment_Label',
        hue_order=['Positive', 'Neutral', 'Negative'],
        palette='tab10',
        ax=axes[1, 0]
    )
    axes[1, 0].set_title("Sentiment Breakdown by Platform", fontsize=13, fontweight='bold')
    axes[1, 0].set_xlabel("Platform", fontsize=11)
    axes[1, 0].set_ylabel("Count", fontsize=11)
    axes[1, 0].legend(title="Sentiment")
    for p in axes[1, 0].patches:
        height = p.get_height()
        if height > 0:
            axes[1, 0].annotate(f'{int(height)}',
                                (p.get_x() + p.get_width() / 2., height),
                                ha='center', va='bottom', fontsize=9, xytext=(0, 2),
                                textcoords='offset points')

    # Bottom-Right: Bar chart showing average 'Engagement' score for each 'Sentiment_Label'
    avg_engagement = df.groupby('Sentiment_Label')['Engagement'].mean().reindex(['Positive', 'Neutral', 'Negative']).reset_index()
    sns.barplot(
        data=avg_engagement,
        x='Sentiment_Label',
        y='Engagement',
        palette='muted',
        ax=axes[1, 1],
        hue='Sentiment_Label',
        legend=False
    )
    axes[1, 1].set_title("Average Engagement by Sentiment Label", fontsize=13, fontweight='bold')
    axes[1, 1].set_xlabel("Sentiment Label", fontsize=11)
    axes[1, 1].set_ylabel("Average Engagement Score", fontsize=11)
    for p in axes[1, 1].patches:
        height = p.get_height()
        if height > 0:
            axes[1, 1].annotate(f'{height:.2f}',
                                (p.get_x() + p.get_width() / 2., height),
                                ha='center', va='bottom', fontsize=10, xytext=(0, 3),
                                textcoords='offset points')

    plt.suptitle("Campus Social Media Dataset - Exploratory Data Analysis", fontsize=16, fontweight='bold', y=0.99)
    plt.tight_layout()
    plt.savefig(output_plot, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Chart successfully saved to '{output_plot}'.")

if __name__ == '__main__':
    perform_eda()
