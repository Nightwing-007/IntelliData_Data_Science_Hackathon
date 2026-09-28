"""HR Analytics Data Generation Module.

Generates a synthetic enterprise HR dataset containing 500 employee records
with realistic salary scales, geographic cost-of-living adjustments,
experience premiums, performance increments, and intentional compensation
anomalies for auditing and analytics pipelines.
"""

import os
import numpy as np
import pandas as pd


def generate_hr_dataset(output_path="../data/hr_salary_dataset.csv"):
    """Generate synthetic HR compensation records and save to CSV.

    Parameters
    ----------
    output_path : str
        Relative or absolute target CSV path.
    """
    np.random.seed(42)
    n_samples = 500

    designations = [
        'Junior Developer',
        'Senior Developer',
        'Team Lead',
        'Manager',
        'Director'
    ]
    education_levels = ['Bachelor', 'Master', 'PhD']
    locations = ['New York', 'London', 'Bangalore', 'Toronto']

    # Generate demographic and organizational attributes
    data = {
        'Employee_ID': [f'EMP{i:04d}' for i in range(1, n_samples + 1)],
        'Designation': np.random.choice(
            designations,
            n_samples,
            p=[0.4, 0.3, 0.15, 0.1, 0.05]
        ),
        'Experience': np.random.randint(1, 25, n_samples),
        'Education': np.random.choice(
            education_levels,
            n_samples,
            p=[0.6, 0.3, 0.1]
        ),
        'Location': np.random.choice(locations, n_samples),
        'Performance_Score': np.random.randint(1, 6, n_samples),
    }
    df = pd.DataFrame(data)

    # Base salary tiers by organizational hierarchy
    base_salaries = {
        'Junior Developer': 60000,
        'Senior Developer': 90000,
        'Team Lead': 120000,
        'Manager': 150000,
        'Director': 200000
    }

    # Geographic location cost-of-living multipliers
    loc_multipliers = {
        'New York': 1.2,
        'London': 1.1,
        'Bangalore': 0.5,
        'Toronto': 1.0
    }

    # Compute base compensation with experience tenure adjustment and noise
    df['Salary'] = df.apply(
        lambda row: int(
            base_salaries[row['Designation']] * loc_multipliers[row['Location']]
            + (row['Experience'] * 2000)
            + np.random.randint(-5000, 5000)
        ),
        axis=1
    )

    # Merit-based annual salary increment percentages (Score 1 to 5)
    increment_map = {1: 0.0, 2: 0.02, 3: 0.05, 4: 0.08, 5: 0.12}
    df['Salary_Increment'] = df.apply(
        lambda row: int(row['Salary'] * increment_map[row['Performance_Score']]),
        axis=1
    )

    # Inject intentional compensation anomalies (15 records) for outlier auditing
    anomaly_indices = np.random.choice(n_samples, 15, replace=False)
    df.loc[anomaly_indices, 'Salary'] = (
        df.loc[anomaly_indices, 'Salary'] * np.random.uniform(1.6, 2.2, 15)
    ).astype(int)

    # Resolve output directory hierarchy
    if not os.path.exists(os.path.dirname(output_path)):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(script_dir, "..", "data")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "hr_salary_dataset.csv")

    df.to_csv(output_path, index=False)
    print(f"Data successfully generated at: {output_path}")


if __name__ == '__main__':
    generate_hr_dataset()
