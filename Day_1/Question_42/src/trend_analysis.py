import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def run_trend_analysis(input_csv="../data/cleaned_social_media_data.csv", output_plot="../topic_trends_over_time.png"):
    # Ensure path works whether executed from src/ or Question_42/
    if not os.path.exists(input_csv) and os.path.exists("data/cleaned_social_media_data.csv"):
        input_csv = "data/cleaned_social_media_data.csv"
        output_plot = "topic_trends_over_time.png"

    print(f"Loading '{input_csv}'...")
    df = pd.read_csv(input_csv)

    # 1. Keyword Tagging: Define actionable keywords and create binary columns
    keywords = ['protest', 'unsafe', 'mental', 'harassment', 'police']
    for kw in keywords:
        df[kw] = df['Text'].str.lower().str.contains(kw, case=False, regex=False).astype(int)

    print("\n" + "=" * 65)
    print("1. KEYWORD TAGGING (Binary Flags Created)")
    print("=" * 65)
    print(f"Keywords: {keywords}")
    print(df[['Text'] + keywords].head(3).to_string())

    # 2. Engagement Calculation: Average Engagement & total count per keyword
    print("\n" + "=" * 65)
    print("2. ENGAGEMENT CALCULATION (Only Posts Containing Keyword)")
    print("=" * 65)
    print(f"{'Keyword':<15}{'Post Count':<15}{'Average Engagement':<20}")
    print("-" * 65)
    
    engagement_summary = []
    for kw in keywords:
        kw_posts = df[df[kw] == 1]
        count = len(kw_posts)
        avg_eng = kw_posts['Engagement'].mean()
        engagement_summary.append({'Keyword': kw, 'Count': count, 'Avg_Engagement': avg_eng})
        print(f"{kw:<15}{count:<15}{avg_eng:<20.2f}")

    # 3. Time Series Aggregation: Convert Timestamp, set as index, resample by month
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df_indexed = df.set_index('Timestamp')
    
    # In pandas 2.2+, 'M' offset alias is replaced by 'ME' (month-end)
    try:
        monthly_trends = df_indexed[keywords].resample('M').sum()
    except ValueError:
        monthly_trends = df_indexed[keywords].resample('ME').sum()

    print("\n" + "=" * 65)
    print("3. TIME SERIES AGGREGATION (Monthly Sum of Posts per Keyword)")
    print("=" * 65)
    print(monthly_trends.to_string())

    # 4. Generate Visualization: Line chart with markers ('o'), legend, grid
    plt.figure(figsize=(12, 6))
    sns.set_theme(style="whitegrid", font_scale=1.0)

    # Format x-axis dates nicely for presentation
    month_labels = monthly_trends.index.strftime('%b %Y')

    palette = sns.color_palette("tab10", len(keywords))
    for idx, kw in enumerate(keywords):
        plt.plot(
            month_labels,
            monthly_trends[kw],
            marker='o',
            linewidth=2.2,
            markersize=7,
            label=kw.capitalize(),
            color=palette[idx]
        )

    plt.title("Topic Trends Over Time (Monthly Post Counts)", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Month", fontsize=11, fontweight='semibold')
    plt.ylabel("Monthly Post Count", fontsize=11, fontweight='semibold')
    plt.xticks(rotation=45)
    plt.legend(title="Keyword", title_fontsize='11', fontsize=10, loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()

    plt.savefig(output_plot, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nSaved visualization to '{output_plot}'.")

if __name__ == '__main__':
    run_trend_analysis()
