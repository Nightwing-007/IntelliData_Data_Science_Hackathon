"""HR Analytics Compensation Econometric Modeling Module (Phase 5).

Trains a multivariate Linear Regression model to identify and quantify the
dollar impact of organizational hierarchy, geographic location, educational
attainment, and career experience on total employee compensation.
"""

import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


def run_ml_modeling(input_path="../data/cleaned_hr_data.csv"):
    """Train multivariate Linear Regression model and extract dollar coefficients.

    Parameters
    ----------
    input_path : str
        Relative or absolute path to the cleaned dataset.
    """
    # Dynamic fallback when script is executed from different working directories
    if not os.path.exists(input_path):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        input_path = os.path.join(script_dir, "..", "data", "cleaned_hr_data.csv")

    print(f"Loading cleaned data from: '{input_path}'...")
    df = pd.read_csv(input_path)

    # 1. Feature Encoding: One-Hot Encode Categorical Covariates (drop_first=True)
    categorical_cols = ['Designation', 'Location', 'Education']
    features_df = df.drop(columns=['Employee_ID', 'Salary_Increment', 'Salary'])
    y = df['Salary']

    # Convert binary flags to integer representation
    X = pd.get_dummies(features_df, columns=categorical_cols, drop_first=True, dtype=int)
    print(f"Features encoded successfully: {X.shape[1]} explanatory variables.")

    # 2. Train/Test Partitioning: 80% Training / 20% Evaluation
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        random_state=42
    )
    print(f"Train/Test split complete: Train = {len(X_train)} samples, Test = {len(X_test)} samples.")

    # 3. Model Estimation: Ordinary Least Squares (OLS) Linear Regression
    model = LinearRegression()
    model.fit(X_train, y_train)
    print("Linear Regression model successfully fitted.")

    # 4. Out-of-Sample Model Evaluation
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print("\n" + "=" * 65)
    print("MODEL EVALUATION & GENERALIZATION METRICS")
    print("=" * 65)
    print(f"R-squared (R2) Score : {r2:.4f} ({r2 * 100:.2f}% variance explained)")
    print(f"Root Mean Squared Err: ${rmse:,.2f}")

    # 5. Econometric Coefficient Extraction: Marginal Dollar Impact per Variable
    coef_df = pd.DataFrame({
        'Feature': X.columns,
        'Impact_in_Dollars': model.coef_
    }).sort_values(by='Impact_in_Dollars', ascending=False).reset_index(drop=True)

    print("\n" + "=" * 65)
    print("COMPENSATION FACTOR COEFFICIENTS (Marginal Dollar Contribution)")
    print("=" * 65)
    print(f"Baseline Intercept: ${model.intercept_:,.2f}\n")
    print(f"{'Rank':<6}{'Feature Name':<32}{'Impact in Dollars ($)':<25}")
    print("-" * 65)
    for idx, row in coef_df.iterrows():
        sign = "+" if row['Impact_in_Dollars'] >= 0 else ""
        print(f"{idx + 1:<6}{row['Feature']:<32}{sign}${row['Impact_in_Dollars']:>12,.2f}")


if __name__ == '__main__':
    run_ml_modeling()
