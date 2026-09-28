"""HR Analytics Data Cleaning and Schema Validation Module (Phase 1).

Loads raw employee compensation data, performs missing-value diagnostics,
enforces strict numeric schema casting for financial variables, and exports
a verified clean dataset ready for analytical modeling.
"""

import os
import pandas as pd


def clean_hr_dataset(
    input_path="../data/hr_salary_dataset.csv",
    output_path="../data/cleaned_hr_data.csv"
):
    """Load, validate, and clean the enterprise HR salary dataset.

    Parameters
    ----------
    input_path : str
        Relative or absolute path to the raw CSV dataset.
    output_path : str
        Relative or absolute target path for the cleaned CSV dataset.
    """
    # Dynamic path fallback if executed from Question_11 root instead of src/
    if not os.path.exists(input_path):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        input_path = os.path.join(script_dir, "..", "data", "hr_salary_dataset.csv")
        output_path = os.path.join(script_dir, "..", "data", "cleaned_hr_data.csv")

    print(f"Loading raw dataset from: '{input_path}'...")
    df = pd.read_csv(input_path)

    # 1. Missing Values Diagnostics
    print("\n" + "=" * 60)
    print("MISSING VALUES AUDIT: df.isnull().sum()")
    print("=" * 60)
    null_audit = df.isnull().sum()
    print(null_audit.to_string())

    # 2. Strict Type Validation for Monetary Fields
    print("\n" + "=" * 60)
    print("SCHEMA & NUMERIC TYPE VALIDATION")
    print("=" * 60)
    df['Salary'] = pd.to_numeric(df['Salary'], errors='raise')
    df['Salary_Increment'] = pd.to_numeric(df['Salary_Increment'], errors='raise')

    is_salary_numeric = pd.api.types.is_numeric_dtype(df['Salary'])
    is_inc_numeric = pd.api.types.is_numeric_dtype(df['Salary_Increment'])

    print(f"  - 'Salary' DataType: {df['Salary'].dtype} | Numeric Validated: {is_salary_numeric}")
    print(f"  - 'Salary_Increment' DataType: {df['Salary_Increment'].dtype} | Numeric Validated: {is_inc_numeric}")

    # 3. Export Cleaned Dataset
    df.to_csv(output_path, index=False)
    print(f"\nCleaned dataset persisted to: '{output_path}'")

    # 4. Final Verification and Console Reporting
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)

    print("\n" + "=" * 60)
    print("VERIFICATION: df.info()")
    print("=" * 60)
    df.info()

    print("\n" + "=" * 60)
    print("VERIFICATION: df.head()")
    print("=" * 60)
    print(df.head())


if __name__ == '__main__':
    clean_hr_dataset()
