"""
Week 2 Assignment - Tesla EV Deliveries and Production Data Analysis
Master Script: Run Complete ML Pipeline
Author: Aman
Date: June 17, 2026

This script executes all pipeline steps sequentially.
"""

import subprocess
import sys
import time
from datetime import datetime

class PipelineRunner:
    def __init__(self):
        """Initialize the Pipeline Runner"""
        self.scripts = [
            ('1_data_preprocessing.py', 'Data Preprocessing'),
            ('2_eda.py', 'Exploratory Data Analysis'),
            ('3_feature_engineering.py', 'Feature Engineering'),
            ('4_regression_modeling.py', 'Regression Modeling'),
            ('5_hyperparameter_tuning.py', 'Hyperparameter Tuning'),
            ('6_time_series_forecasting.py', 'Time Series Forecasting'),
            ('7_evaluation_report.py', 'Evaluation Report')
        ]
        self.start_time = None
        self.execution_times = []
        
    def print_header(self):
        """Print pipeline header"""
        print("\n" + "=" * 100)
        print(" " * 20 + "TESLA EV DELIVERIES - COMPLETE ML PIPELINE")
        print(" " * 30 + "Celebal Excellence Internship")
        print(" " * 35 + "Week 2 Assignment")
        print("=" * 100)
        print(f"\nStarting Pipeline Execution: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Total Steps: {len(self.scripts)}")
        print("=" * 100)
        
    def run_script(self, script_name, step_name, step_number):
        """Run a single script"""
        print(f"\n{'=' * 100}")
        print(f"STEP {step_number}/{len(self.scripts)}: {step_name}")
        print(f"Script: {script_name}")
        print(f"{'=' * 100}\n")
        
        start = time.time()
        
        try:
            # Run the script
            result = subprocess.run(
                [sys.executable, script_name],
                capture_output=False,
                text=True,
                check=True
            )
            
            end = time.time()
            duration = end - start
            self.execution_times.append((step_name, duration))
            
            print(f"\n✓ {step_name} completed successfully in {duration:.2f} seconds")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"\n✗ Error in {step_name}")
            print(f"Error: {str(e)}")
            return False
        except Exception as e:
            print(f"\n✗ Unexpected error in {step_name}")
            print(f"Error: {str(e)}")
            return False
    
    def print_summary(self):
        """Print execution summary"""
        total_time = sum([t[1] for t in self.execution_times])
        
        print("\n" + "=" * 100)
        print(" " * 35 + "PIPELINE EXECUTION SUMMARY")
        print("=" * 100)
        
        print("\nExecution Times:")
        print("-" * 100)
        print(f"{'Step':<50} {'Duration (seconds)':<20} {'Duration (minutes)'}")
        print("-" * 100)
        
        for step_name, duration in self.execution_times:
            minutes = duration / 60
            print(f"{step_name:<50} {duration:<20.2f} {minutes:.2f}")
        
        print("-" * 100)
        print(f"{'TOTAL':<50} {total_time:<20.2f} {total_time/60:.2f}")
        print("-" * 100)
        
        print("\n" + "=" * 100)
        print(" " * 30 + "✓ PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 100)
        
        print("\nGenerated Files:")
        print("  📊 Data Files:")
        print("     - cleaned_data.csv")
        print("     - engineered_data.csv")
        
        print("\n  📈 Model Files:")
        print("     - best_regression_model.pkl")
        print("     - best_tuned_model.pkl")
        print("     - feature_scaler.pkl")
        
        print("\n  📋 Results Files:")
        print("     - model_comparison_results.csv")
        print("     - tuned_models_comparison.csv")
        print("     - forecast_models_comparison.csv")
        print("     - metrics_summary_table.csv")
        print("     - EVALUATION_REPORT.txt")
        
        print("\n  📊 Visualizations:")
        print("     - 1_univariate_analysis.png")
        print("     - 2_correlation_matrix.png")
        print("     - 3_bivariate_scatter_plots.png")
        print("     - 4_categorical_analysis.png")
        print("     - 5_temporal_analysis.png")
        print("     - 6_model_comparison.png")
        print("     - 7_best_model_predictions.png")
        print("     - 8_tuned_models_comparison.png")
        print("     - 9_time_series_forecasts.png")
        print("     - 10_combined_forecasts.png")
        print("     - 11_comprehensive_evaluation_report.png")
        
        print("\n" + "=" * 100)
        print(f"Pipeline completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Total execution time: {total_time/60:.2f} minutes")
        print("=" * 100)
        
        print("\n🎉 Thank you for using the Tesla EV ML Pipeline!")
        print("For detailed results, check EVALUATION_REPORT.txt")
        print("=" * 100 + "\n")
    
    def run_pipeline(self):
        """Run the complete pipeline"""
        self.print_header()
        self.start_time = time.time()
        
        # Run all scripts
        for i, (script, name) in enumerate(self.scripts, 1):
            success = self.run_script(script, name, i)
            
            if not success:
                print("\n" + "=" * 100)
                print(f"⚠ Pipeline stopped at Step {i}: {name}")
                print("=" * 100)
                return False
            
            # Small delay between steps
            time.sleep(1)
        
        # Print summary
        self.print_summary()
        return True

def check_requirements():
    """Check if required packages are installed"""
    print("\nChecking requirements...")
    
    required_packages = [
        'pandas', 'numpy', 'matplotlib', 'seaborn', 'sklearn',
        'xgboost', 'lightgbm', 'statsmodels', 'prophet', 'tensorflow'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("\n⚠ Missing required packages:")
        for package in missing_packages:
            print(f"  - {package}")
        print("\nPlease install missing packages:")
        print("  pip install -r requirements.txt")
        return False
    
    print("✓ All required packages are installed\n")
    return True

if __name__ == "__main__":
    print("\n" + "=" * 100)
    print(" " * 25 + "TESLA EV ML PIPELINE - MASTER EXECUTION SCRIPT")
    print("=" * 100)
    
    # Check requirements
    if not check_requirements():
        print("\n⚠ Please install required packages before running the pipeline.")
        sys.exit(1)
    
    # Confirm execution
    print("\nThis will execute the complete ML pipeline (7 steps).")
    print("Estimated time: 15-30 minutes (depending on system)")
    
    response = input("\nDo you want to proceed? (yes/no): ").strip().lower()
    
    if response in ['yes', 'y']:
        # Run pipeline
        runner = PipelineRunner()
        success = runner.run_pipeline()
        
        if success:
            sys.exit(0)
        else:
            sys.exit(1)
    else:
        print("\n⚠ Pipeline execution cancelled by user.")
        print("\nTo run individual scripts:")
        print("  python 1_data_preprocessing.py")
        print("  python 2_eda.py")
        print("  ... etc")
        sys.exit(0)
