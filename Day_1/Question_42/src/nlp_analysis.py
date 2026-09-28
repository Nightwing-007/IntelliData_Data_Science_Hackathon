import os
import string
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS

def run_nlp_pipeline(input_csv="../data/cleaned_social_media_data.csv", output_plot="../nlp_topics.png"):
    # Ensure path works whether executed from src/ or Question_42/
    if not os.path.exists(input_csv) and os.path.exists("data/cleaned_social_media_data.csv"):
        input_csv = "data/cleaned_social_media_data.csv"
        output_plot = "nlp_topics.png"

    print(f"Loading '{input_csv}'...")
    df = pd.read_csv(input_csv)

    # 1. Text Preprocessing: Create a 'Cleaned_Text' column by lowercasing the 'Text' column,
    # removing punctuation, and filtering out common stop words (e.g., 'i', 'the', 'a', 'and', 'is', 'near', 'today').
    stop_words = set(ENGLISH_STOP_WORDS).union({'i', 'the', 'a', 'and', 'is', 'near', 'today'})

    def preprocess_text(text):
        text = str(text).lower()
        text = text.translate(str.maketrans('', '', string.punctuation))
        tokens = [word for word in text.split() if word not in stop_words]
        return " ".join(tokens)

    df['Cleaned_Text'] = df['Text'].apply(preprocess_text)
    print("\nText preprocessing completed. Sample cleaned text:")
    for text in df['Cleaned_Text'].head(5):
        print(f"  - {text}")

    # 2. Topic Extraction: Initialize a TfidfVectorizer to extract the top 15 most important feature words
    tfidf = TfidfVectorizer(max_features=15)
    tfidf_matrix = tfidf.fit_transform(df['Cleaned_Text'])
    feature_names = tfidf.get_feature_names_out()
    print(f"\nExtracted top {len(feature_names)} features: {list(feature_names)}")

    # 3. TF-IDF Scoring: Sum the TF-IDF scores for each of the top 15 words to determine their overall prevalence
    cumulative_scores = np.asarray(tfidf_matrix.sum(axis=0)).flatten()
    topics_df = pd.DataFrame({
        'Topic': feature_names,
        'Cumulative_TFIDF': cumulative_scores
    }).sort_values(by='Cumulative_TFIDF', ascending=False).reset_index(drop=True)

    # 4. Terminal Output: Print the top 10 most common discussion topics and their cumulative TF-IDF scores
    print("\n" + "=" * 60)
    print("TOP 10 MOST COMMON DISCUSSION TOPICS (by Cumulative TF-IDF)")
    print("=" * 60)
    print(f"{'Rank':<6}{'Topic':<16}{'Cumulative TF-IDF Score':<25}")
    print("-" * 60)
    for idx, row in topics_df.head(10).iterrows():
        print(f"{idx + 1:<6}{row['Topic']:<16}{row['Cumulative_TFIDF']:<25.4f}")

    # 5. Generate Visualization: Create a horizontal bar chart mapping top 15 topics (y-axis) against cumulative TF-IDF (x-axis)
    plt.figure(figsize=(10, 7))
    sns.set_theme(style="whitegrid")

    plot_df = topics_df.sort_values(by='Cumulative_TFIDF', ascending=True)

    barplot = sns.barplot(
        data=plot_df,
        x='Cumulative_TFIDF',
        y='Topic',
        palette='viridis',
        hue='Topic',
        legend=False
    )
    plt.title("Top 15 Discussion Topics by Cumulative TF-IDF Score", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Cumulative TF-IDF Score", fontsize=12)
    plt.ylabel("Topic (Feature Word)", fontsize=12)

    for p in barplot.patches:
        width = p.get_width()
        if width > 0:
            barplot.annotate(
                f'{width:.2f}',
                (width, p.get_y() + p.get_height() / 2.),
                ha='left', va='center',
                fontsize=10,
                xytext=(5, 0),
                textcoords='offset points'
            )

    plt.tight_layout()
    plt.savefig(output_plot, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nSaved visualization to '{output_plot}'.")

if __name__ == '__main__':
    run_nlp_pipeline()
