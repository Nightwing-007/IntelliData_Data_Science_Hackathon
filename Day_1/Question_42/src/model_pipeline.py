import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def run_ml_pipeline(input_csv="../data/cleaned_social_media_data.csv", output_plot="../naive_bayes_evaluation.png"):
    # Ensure path works whether executed from src/ or Question_42/
    if not os.path.exists(input_csv) and os.path.exists("data/cleaned_social_media_data.csv"):
        input_csv = "data/cleaned_social_media_data.csv"
        output_plot = "naive_bayes_evaluation.png"

    print(f"Loading '{input_csv}'...")
    df = pd.read_csv(input_csv)

    # 1. Data Splitting: X = 'Text', y = 'Sentiment_Label', 80% train / 20% test (stratify=y, random_state=42)
    X = df['Text']
    y = df['Sentiment_Label']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )
    print(f"Data split successfully: Train set = {len(X_train)} samples, Test set = {len(X_test)} samples.")

    # 2. Text Vectorization: TfidfVectorizer with English stop words
    vectorizer = TfidfVectorizer(stop_words='english')
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    print(f"TF-IDF Vectorization complete: Vocabulary size = {len(vectorizer.get_feature_names_out())} features.")

    # 3. Model Training: MultinomialNB classifier
    nb_classifier = MultinomialNB()
    nb_classifier.fit(X_train_vec, y_train)
    print("Multinomial Naive Bayes model trained successfully.")

    # 4. Model Evaluation: Predict on test set, print Accuracy and Classification Report
    y_pred = nb_classifier.predict(X_test_vec)
    acc = accuracy_score(y_test, y_pred)
    class_labels = ['Negative', 'Neutral', 'Positive']
    report = classification_report(y_test, y_pred, labels=class_labels, digits=4)

    print("\n" + "=" * 60)
    print("MODEL EVALUATION RESULTS")
    print("=" * 60)
    print(f"Accuracy Score: {acc:.4f} ({acc * 100:.2f}%)\n")
    print("Classification Report:")
    print(report)

    # 5. Confusion Matrix Visualization: Heatmap with labeled axes
    cm = confusion_matrix(y_test, y_pred, labels=class_labels)
    cm_df = pd.DataFrame(cm, index=class_labels, columns=class_labels)

    plt.figure(figsize=(8, 6))
    sns.set_theme(style="white")

    heatmap = sns.heatmap(
        cm_df,
        annot=True,
        fmt='d',
        cmap='Blues',
        cbar=True,
        annot_kws={'size': 14, 'weight': 'bold'},
        linewidths=1.0,
        linecolor='white'
    )

    plt.title("Naive Bayes Classifier - Confusion Matrix", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Predicted Sentiment", fontsize=12, fontweight='semibold', labelpad=10)
    plt.ylabel("Actual Sentiment", fontsize=12, fontweight='semibold', labelpad=10)
    plt.tight_layout()

    plt.savefig(output_plot, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix heatmap saved to '{output_plot}'.")

if __name__ == '__main__':
    run_ml_pipeline()
