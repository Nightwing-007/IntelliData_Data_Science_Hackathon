"""
IntelliData Data Science Hackathon Master Orchestrator.

Provides a unified command-line entry point to launch interactive Streamlit
dashboards, execute analytical data pipelines, or verify repository assets across
all three completed hackathon tracks:
1. Day 1 (Question 11): Enterprise HR Analytics & Pay Equity Audit
2. Day 1 (Question 42): Campus Social Media Brand Analytics
3. Day 2 (StockSense): AI-Powered Replenishment Intelligence & Forecasting
"""

import argparse
from pathlib import Path
import subprocess
import sys

BASE_DIR = Path(__file__).resolve().parent


def launch_process(command: list, cwd: Path = BASE_DIR) -> None:
    """Execute a subprocess command with clean error handling."""
    print(f"\n[EXEC] Running: {' '.join(command)} (Cwd: {cwd})")
    try:
        subprocess.run(command, cwd=str(cwd), check=True)
    except KeyboardInterrupt:
        print("\n[INFO] Process interrupted by user.")
    except subprocess.CalledProcessError as exc:
        print(f"\n[ERROR] Command failed with exit code {exc.returncode}: {exc}", file=sys.stderr)


# -----------------------------------------------------------------------------
# Pipeline Runners
# -----------------------------------------------------------------------------
def run_q11_pipeline() -> None:
    """Run all analytical modules for Day 1 Question 11 (HR Analytics)."""
    print("\n" + "=" * 75)
    print("EXECUTING DAY 1 QUESTION 11 PIPELINE (HR Analytics & Pay Equity)")
    print("=" * 75)

    q11_src = BASE_DIR / "Day_1" / "Question_11" / "src"
    scripts = [
        "phase1_cleaning.py",
        "phase2_eda.py",
        "phase3_analytics.py",
        "phase4_anomalies.py",
        "phase5_ml_model.py",
    ]

    for script in scripts:
        script_path = q11_src / script
        if script_path.exists():
            print(f"\n--- Running {script} ---")
            launch_process([sys.executable, str(script_path)], cwd=q11_src)
        else:
            print(f"[WARN] Script not found: {script_path}")


def run_q42_pipeline() -> None:
    """Run all analytical modules for Day 1 Question 42 (Social Media Brand Analytics)."""
    print("\n" + "=" * 75)
    print("EXECUTING DAY 1 QUESTION 42 PIPELINE (Social Media Brand Analytics)")
    print("=" * 75)

    q42_src = BASE_DIR / "Day_1" / "Question_42" / "src"
    scripts = [
        "eda.py",
        "nlp_analysis.py",
        "trend_analysis.py",
        "model_pipeline.py",
    ]

    for script in scripts:
        script_path = q42_src / script
        if script_path.exists():
            print(f"\n--- Running {script} ---")
            launch_process([sys.executable, str(script_path)], cwd=q42_src)
        else:
            print(f"[WARN] Script not found: {script_path}")


def run_day2_pipeline() -> None:
    """Run all analytical modules for Day 2 StockSense (Retail AI Replenishment)."""
    print("\n" + "=" * 75)
    print("EXECUTING DAY 2 STOCKSENSE PIPELINE (AI Replenishment & Demand Forecasting)")
    print("=" * 75)

    day2_src = BASE_DIR / "Day_2" / "StockSense" / "src"
    scripts = [
        "generate_stock_data.py",
        "round1_data_prep.py",
        "round1_eda_stats.py",
        "phase3_features.py",
        "phase4_modeling.py",
        "phase5_intelligence.py",
    ]

    for script in scripts:
        script_path = day2_src / script
        if script_path.exists():
            print(f"\n--- Running {script} ---")
            launch_process([sys.executable, str(script_path)], cwd=day2_src)
        else:
            print(f"[WARN] Script not found: {script_path}")


# -----------------------------------------------------------------------------
# Dashboard Launchers
# -----------------------------------------------------------------------------
def launch_q11_dashboard() -> None:
    """Launch the Question 11 Streamlit dashboard."""
    dashboard_path = BASE_DIR / "Day_1" / "Question_11" / "dashboard.py"
    print(f"\n[STREAMLIT] Launching Question 11 Dashboard: {dashboard_path}")
    launch_process([sys.executable, "-m", "streamlit", "run", str(dashboard_path)])


def launch_q42_dashboard() -> None:
    """Launch the Question 42 Streamlit dashboard."""
    dashboard_path = BASE_DIR / "Day_1" / "Question_42" / "dashboard.py"
    print(f"\n[STREAMLIT] Launching Question 42 Dashboard: {dashboard_path}")
    launch_process([sys.executable, "-m", "streamlit", "run", str(dashboard_path)])


def launch_day2_dashboard() -> None:
    """Launch the Day 2 StockSense Streamlit dashboard."""
    dashboard_path = BASE_DIR / "Day_2" / "StockSense" / "dashboard.py"
    print(f"\n[STREAMLIT] Launching Day 2 StockSense Dashboard: {dashboard_path}")
    launch_process([sys.executable, "-m", "streamlit", "run", str(dashboard_path)])


# -----------------------------------------------------------------------------
# Repository Integrity Audit
# -----------------------------------------------------------------------------
def verify_environment() -> None:
    """Verify presence and integrity of all datasets, models, and dashboards."""
    print("\n" + "=" * 75)
    print("INTELLIDATA HACKATHON - COMPLETE REPOSITORY ASSET AUDIT")
    print("=" * 75)

    checks = [
        # Question 11
        ("Q11 Cleaned Data", BASE_DIR / "Day_1" / "Question_11" / "data" / "cleaned_hr_data.csv"),
        ("Q11 Raw Data", BASE_DIR / "Day_1" / "Question_11" / "data" / "hr_salary_dataset.csv"),
        ("Q11 Dashboard", BASE_DIR / "Day_1" / "Question_11" / "dashboard.py"),
        ("Q11 EDA Viz", BASE_DIR / "Day_1" / "Question_11" / "visualizations" / "hr_eda_visualizations.png"),
        ("Q11 Perf Viz", BASE_DIR / "Day_1" / "Question_11" / "visualizations" / "hr_performance_experience.png"),
        ("Q11 Anomaly Viz", BASE_DIR / "Day_1" / "Question_11" / "visualizations" / "hr_compensation_anomalies.png"),
        ("Q11 README", BASE_DIR / "Day_1" / "Question_11" / "README.md"),
        # Question 42
        ("Q42 Cleaned Data", BASE_DIR / "Day_1" / "Question_42" / "data" / "cleaned_social_media_data.csv"),
        ("Q42 Raw Data", BASE_DIR / "Day_1" / "Question_42" / "data" / "campus_social_media_dataset.csv"),
        ("Q42 Dashboard", BASE_DIR / "Day_1" / "Question_42" / "dashboard.py"),
        ("Q42 EDA Viz", BASE_DIR / "Day_1" / "Question_42" / "eda_visualizations.png"),
        ("Q42 NLP Viz", BASE_DIR / "Day_1" / "Question_42" / "nlp_topics.png"),
        ("Q42 Trends Viz", BASE_DIR / "Day_1" / "Question_42" / "topic_trends_over_time.png"),
        ("Q42 Naive Bayes Viz", BASE_DIR / "Day_1" / "Question_42" / "naive_bayes_evaluation.png"),
        ("Q42 README", BASE_DIR / "Day_1" / "Question_42" / "README.md"),
        # Day 2 StockSense
        ("StockSense Raw Txn", BASE_DIR / "Day_2" / "StockSense" / "data" / "raw" / "transactions.csv"),
        ("StockSense Raw Inv", BASE_DIR / "Day_2" / "StockSense" / "data" / "raw" / "inventory.csv"),
        ("StockSense Master", BASE_DIR / "Day_2" / "StockSense" / "data" / "processed" / "master_analytics_dataset.csv"),
        ("StockSense Features", BASE_DIR / "Day_2" / "StockSense" / "data" / "processed" / "model_ready_data.csv"),
        ("StockSense Scored", BASE_DIR / "Day_2" / "StockSense" / "data" / "processed" / "scored_predictions.csv"),
        ("StockSense Demand Model", BASE_DIR / "Day_2" / "StockSense" / "models" / "demand_model.pkl"),
        ("StockSense Risk Model", BASE_DIR / "Day_2" / "StockSense" / "models" / "risk_model.pkl"),
        ("StockSense EDA Grid", BASE_DIR / "Day_2" / "StockSense" / "reports" / "stocksense_eda.png"),
        ("StockSense Dashboard", BASE_DIR / "Day_2" / "StockSense" / "dashboard.py"),
        ("StockSense README", BASE_DIR / "Day_2" / "StockSense" / "README.md"),
        # Root Files
        ("Root README", BASE_DIR / "README.md"),
        ("Requirements", BASE_DIR / "requirements.txt"),
        ("Gitignore", BASE_DIR / ".gitignore"),
    ]

    all_passed = True
    for label, path in checks:
        status = "PASSED" if path.exists() else "FAILED"
        if not path.exists():
            all_passed = False
        print(f"[{status}] {label:<24} -> {path.relative_to(BASE_DIR)}")

    print("-" * 75)
    if all_passed:
        print(f"[SUCCESS] All {len(checks)} core repository assets verified successfully.")
    else:
        print("[WARNING] One or more assets failed verification.", file=sys.stderr)


def interactive_menu() -> None:
    """Display interactive CLI selection menu."""
    while True:
        print("\n" + "=" * 75)
        print("      IntelliData Data Science Hackathon - Master Execution Hub       ")
        print("=" * 75)
        print("1. Launch Day 1 Question 11 Dashboard (HR Analytics & Pay Equity)")
        print("2. Launch Day 1 Question 42 Dashboard (Social Media Brand Analytics)")
        print("3. Launch Day 2 StockSense Dashboard (AI Replenishment Intelligence)")
        print("4. Run Day 1 Question 11 Analytics & ML Pipeline (Phases 1-5)")
        print("5. Run Day 1 Question 42 NLP & Classification Pipeline")
        print("6. Run Day 2 StockSense End-to-End Supply Chain AI Pipeline")
        print("7. Run Complete Repository Asset & Integrity Audit")
        print("8. Exit")
        print("-" * 75)

        choice = input("Enter selection [1-8]: ").strip()
        if choice == "1":
            launch_q11_dashboard()
        elif choice == "2":
            launch_q42_dashboard()
        elif choice == "3":
            launch_day2_dashboard()
        elif choice == "4":
            run_q11_pipeline()
        elif choice == "5":
            run_q42_pipeline()
        elif choice == "6":
            run_day2_pipeline()
        elif choice == "7":
            verify_environment()
        elif choice in {"8", "q", "exit", "quit"}:
            print("Exiting IntelliData Master Hub.")
            break
        else:
            print("Invalid selection. Please choose an option from 1 to 8.")


def main() -> None:
    """Parse command line arguments and execute requested action."""
    parser = argparse.ArgumentParser(
        description="IntelliData Data Science Hackathon Master Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--q11-dashboard", action="store_true", help="Launch Question 11 Streamlit Dashboard")
    parser.add_argument("--q42-dashboard", action="store_true", help="Launch Question 42 Streamlit Dashboard")
    parser.add_argument("--day2-dashboard", action="store_true", help="Launch Day 2 StockSense Dashboard")
    parser.add_argument("--q11-pipeline", action="store_true", help="Execute Question 11 analytical pipelines")
    parser.add_argument("--q42-pipeline", action="store_true", help="Execute Question 42 analytical pipelines")
    parser.add_argument("--day2-pipeline", action="store_true", help="Execute Day 2 StockSense analytical pipelines")
    parser.add_argument("--verify", action="store_true", help="Verify integrity of all datasets, models, and assets")

    args = parser.parse_args()

    if args.q11_dashboard:
        launch_q11_dashboard()
    elif args.q42_dashboard:
        launch_q42_dashboard()
    elif args.day2_dashboard:
        launch_day2_dashboard()
    elif args.q11_pipeline:
        run_q11_pipeline()
    elif args.q42_pipeline:
        run_q42_pipeline()
    elif args.day2_pipeline:
        run_day2_pipeline()
    elif args.verify:
        verify_environment()
    else:
        if sys.stdin.isatty():
            interactive_menu()
        else:
            verify_environment()


if __name__ == "__main__":
    main()
