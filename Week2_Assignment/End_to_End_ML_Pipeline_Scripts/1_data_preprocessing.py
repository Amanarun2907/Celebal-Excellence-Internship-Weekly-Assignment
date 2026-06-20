"""
Week 2 Assignment - Tesla EV Deliveries and Production Data Analysis
Script 1: Data Preprocessing
Author: Aman
Date: June 17, 2026
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

class DataPreprocessor:
    def __init__(self, filepath):
        """Initialize the DataPreprocessor with file path"""
        self.filepath = filepath
        self.df = None
        self.df_cleaned = None
        
    def load_data(self):
        """Load the dataset from CSV file"""
        print("=" * 80)
        print("STEP 1: LOADING DATA")
        print("=" * 80)
        
        self.df = pd.read_csv(self.filepath)
        print(f"\n✓ Data loaded successfully!")
        print(f"  - Shape: {self.df.shape}")
        print(f"  - Columns: {list(self.df.columns)}")
        print(f"\nFirst 5 rows:")
        print(self.df.head())
        
        return self.df
    
    def explore_data(self):
        """Perform initial data exploration"""
        print("\n" + "=" * 80)
        print("STEP 2: DATA EXPLORATION")
        print("=" * 80)
        
        print("\n1. Data Info:")
        print(self.df.info())
        
        print("\n2. Statistical Summary:")
        print(self.df.describe())
        
        print("\n3. Missing Values:")
        missing = self.df.isnull().sum()
        missing_pct = (missing / len(self.df)) * 100
        missing_df = pd.DataFrame({
            'Missing_Count': missing,
            'Percentage': missing_pct
        })
        print(missing_df[missing_df['Missing_Count'] > 0])
        
        print("\n4. Data Types:")
        print(self.df.dtypes)
        
        print("\n5. Unique Values per Column:")
        for col in self.df.columns:
            print(f"  - {col}: {self.df[col].nunique()} unique values")
            
        print("\n6. Categorical Column Values:")
        categorical_cols = ['Region', 'Model', 'Source_Type']
        for col in categorical_cols:
            if col in self.df.columns:
                print(f"\n  {col}: {self.df[col].unique()}")
    
    def handle_missing_values(self):
        """Handle missing values in the dataset"""
        print("\n" + "=" * 80)
        print("STEP 3: HANDLING MISSING VALUES")
        print("=" * 80)
        
        self.df_cleaned = self.df.copy()
        
        # Check for missing values
        missing_before = self.df_cleaned.isnull().sum().sum()
        print(f"\nTotal missing values before: {missing_before}")
        
        if missing_before > 0:
            # For numerical columns: fill with median grouped by Model and Region
            numerical_cols = ['Estimated_Deliveries', 'Production_Units', 'Avg_Price_USD', 
                            'Battery_Capacity_kWh', 'Range_km', 'CO2_Saved_tons', 'Charging_Stations']
            
            for col in numerical_cols:
                if col in self.df_cleaned.columns and self.df_cleaned[col].isnull().sum() > 0:
                    self.df_cleaned[col] = self.df_cleaned.groupby(['Model', 'Region'])[col].transform(
                        lambda x: x.fillna(x.median())
                    )
                    # If still missing, fill with overall median
                    self.df_cleaned[col].fillna(self.df_cleaned[col].median(), inplace=True)
            
            # For categorical columns: fill with mode
            categorical_cols = ['Region', 'Model', 'Source_Type']
            for col in categorical_cols:
                if col in self.df_cleaned.columns and self.df_cleaned[col].isnull().sum() > 0:
                    self.df_cleaned[col].fillna(self.df_cleaned[col].mode()[0], inplace=True)
        
        missing_after = self.df_cleaned.isnull().sum().sum()
        print(f"Total missing values after: {missing_after}")
        print("✓ Missing values handled successfully!")
        
        return self.df_cleaned
    
    def handle_outliers(self):
        """Detect and handle outliers using IQR method"""
        print("\n" + "=" * 80)
        print("STEP 4: DETECTING AND HANDLING OUTLIERS")
        print("=" * 80)
        
        numerical_cols = ['Estimated_Deliveries', 'Production_Units', 'Avg_Price_USD', 
                         'Battery_Capacity_kWh', 'Range_km', 'CO2_Saved_tons', 'Charging_Stations']
        
        outlier_summary = []
        
        for col in numerical_cols:
            if col in self.df_cleaned.columns:
                Q1 = self.df_cleaned[col].quantile(0.25)
                Q3 = self.df_cleaned[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers_count = ((self.df_cleaned[col] < lower_bound) | 
                                 (self.df_cleaned[col] > upper_bound)).sum()
                
                if outliers_count > 0:
                    outlier_summary.append({
                        'Column': col,
                        'Outliers_Count': outliers_count,
                        'Percentage': f"{(outliers_count/len(self.df_cleaned))*100:.2f}%",
                        'Lower_Bound': f"{lower_bound:.2f}",
                        'Upper_Bound': f"{upper_bound:.2f}"
                    })
                    
                    # Cap outliers instead of removing
                    self.df_cleaned[col] = np.where(
                        self.df_cleaned[col] < lower_bound, lower_bound, 
                        np.where(self.df_cleaned[col] > upper_bound, upper_bound, self.df_cleaned[col])
                    )
        
        if outlier_summary:
            outlier_df = pd.DataFrame(outlier_summary)
            print("\nOutliers Detected and Capped:")
            print(outlier_df.to_string(index=False))
        else:
            print("\n✓ No outliers detected!")
        
        return self.df_cleaned
    
    def create_datetime_features(self):
        """Create datetime column and extract temporal features"""
        print("\n" + "=" * 80)
        print("STEP 5: CREATING DATETIME FEATURES")
        print("=" * 80)
        
        # Create date column
        self.df_cleaned['Date'] = pd.to_datetime(
            self.df_cleaned['Year'].astype(str) + '-' + 
            self.df_cleaned['Month'].astype(str).str.zfill(2) + '-01'
        )
        
        # Extract temporal features
        self.df_cleaned['Quarter'] = self.df_cleaned['Date'].dt.quarter
        self.df_cleaned['DayOfYear'] = self.df_cleaned['Date'].dt.dayofyear
        self.df_cleaned['WeekOfYear'] = self.df_cleaned['Date'].dt.isocalendar().week
        self.df_cleaned['MonthName'] = self.df_cleaned['Date'].dt.month_name()
        
        print("\n✓ Datetime features created:")
        print("  - Date (YYYY-MM-DD)")
        print("  - Quarter (1-4)")
        print("  - DayOfYear (1-366)")
        print("  - WeekOfYear (1-53)")
        print("  - MonthName (January-December)")
        
        print(f"\nDate Range: {self.df_cleaned['Date'].min()} to {self.df_cleaned['Date'].max()}")
        
        return self.df_cleaned
    
    def save_cleaned_data(self, output_path='cleaned_data.csv'):
        """Save cleaned data to CSV"""
        print("\n" + "=" * 80)
        print("STEP 6: SAVING CLEANED DATA")
        print("=" * 80)
        
        self.df_cleaned.to_csv(output_path, index=False)
        print(f"\n✓ Cleaned data saved to: {output_path}")
        print(f"  - Shape: {self.df_cleaned.shape}")
        print(f"  - Columns: {len(self.df_cleaned.columns)}")
        
        return output_path
    
    def run_preprocessing_pipeline(self):
        """Run the complete preprocessing pipeline"""
        print("\n" + "=" * 80)
        print("TESLA EV DATA PREPROCESSING PIPELINE")
        print("=" * 80)
        
        # Execute all steps
        self.load_data()
        self.explore_data()
        self.handle_missing_values()
        self.handle_outliers()
        self.create_datetime_features()
        output_path = self.save_cleaned_data()
        
        print("\n" + "=" * 80)
        print("PREPROCESSING COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print(f"\nFinal Dataset Shape: {self.df_cleaned.shape}")
        print(f"Output File: {output_path}")
        
        return self.df_cleaned

if __name__ == "__main__":
    # Initialize preprocessor
    preprocessor = DataPreprocessor('tesla_deliveries_dataset_2015_2025.csv')
    
    # Run the preprocessing pipeline
    cleaned_data = preprocessor.run_preprocessing_pipeline()
    
    print("\n" + "=" * 80)
    print("Next Step: Run '2_eda.py' for Exploratory Data Analysis")
    print("=" * 80)
