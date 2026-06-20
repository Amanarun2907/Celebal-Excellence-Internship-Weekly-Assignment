"""
Week 2 Assignment - Tesla EV Deliveries and Production Data Analysis
Script 3: Feature Engineering
Author: Aman
Date: June 17, 2026
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

class FeatureEngineer:
    def __init__(self, filepath):
        """Initialize the Feature Engineer"""
        self.filepath = filepath
        self.df = None
        self.label_encoders = {}
        self.scaler = None
        
    def load_data(self):
        """Load the cleaned dataset"""
        print("=" * 80)
        print("LOADING DATA FOR FEATURE ENGINEERING")
        print("=" * 80)
        
        self.df = pd.read_csv(self.filepath)
        self.df['Date'] = pd.to_datetime(self.df['Date'])
        
        print(f"\n✓ Data loaded successfully!")
        print(f"  - Shape: {self.df.shape}")
        print(f"  - Columns: {len(self.df.columns)}")
        
        return self.df
    
    def create_lag_features(self):
        """Create lag features for time series"""
        print("\n" + "=" * 80)
        print("STEP 1: CREATING LAG FEATURES")
        print("=" * 80)
        
        # Sort by date
        self.df = self.df.sort_values('Date')
        
        # Create lag features for deliveries and production
        target_cols = ['Estimated_Deliveries', 'Production_Units', 'Avg_Price_USD']
        lag_periods = [1, 2, 3, 6, 12]
        
        created_features = []
        
        for col in target_cols:
            for lag in lag_periods:
                feature_name = f'{col}_Lag_{lag}'
                self.df[feature_name] = self.df.groupby(['Model', 'Region'])[col].shift(lag)
                created_features.append(feature_name)
        
        print(f"\n✓ Created {len(created_features)} lag features:")
        for feat in created_features[:10]:  # Show first 10
            print(f"  - {feat}")
        if len(created_features) > 10:
            print(f"  ... and {len(created_features) - 10} more")
        
        return self.df
    
    def create_rolling_features(self):
        """Create rolling window statistics"""
        print("\n" + "=" * 80)
        print("STEP 2: CREATING ROLLING WINDOW FEATURES")
        print("=" * 80)
        
        target_cols = ['Estimated_Deliveries', 'Production_Units', 'Avg_Price_USD']
        windows = [3, 6, 12]
        
        created_features = []
        
        for col in target_cols:
            for window in windows:
                # Rolling mean
                feature_name_mean = f'{col}_RollingMean_{window}'
                self.df[feature_name_mean] = self.df.groupby(['Model', 'Region'])[col].transform(
                    lambda x: x.rolling(window=window, min_periods=1).mean()
                )
                created_features.append(feature_name_mean)
                
                # Rolling std
                feature_name_std = f'{col}_RollingStd_{window}'
                self.df[feature_name_std] = self.df.groupby(['Model', 'Region'])[col].transform(
                    lambda x: x.rolling(window=window, min_periods=1).std()
                )
                created_features.append(feature_name_std)
                
                # Rolling max
                feature_name_max = f'{col}_RollingMax_{window}'
                self.df[feature_name_max] = self.df.groupby(['Model', 'Region'])[col].transform(
                    lambda x: x.rolling(window=window, min_periods=1).max()
                )
                created_features.append(feature_name_max)
        
        print(f"\n✓ Created {len(created_features)} rolling window features")
        print("  Examples:")
        for feat in created_features[:6]:
            print(f"  - {feat}")
        
        return self.df
    
    def create_temporal_features(self):
        """Create advanced temporal features"""
        print("\n" + "=" * 80)
        print("STEP 3: CREATING TEMPORAL FEATURES")
        print("=" * 80)
        
        created_features = []
        
        # Cyclical encoding for month
        self.df['Month_Sin'] = np.sin(2 * np.pi * self.df['Month'] / 12)
        self.df['Month_Cos'] = np.cos(2 * np.pi * self.df['Month'] / 12)
        created_features.extend(['Month_Sin', 'Month_Cos'])
        
        # Cyclical encoding for quarter
        self.df['Quarter_Sin'] = np.sin(2 * np.pi * self.df['Quarter'] / 4)
        self.df['Quarter_Cos'] = np.cos(2 * np.pi * self.df['Quarter'] / 4)
        created_features.extend(['Quarter_Sin', 'Quarter_Cos'])
        
        # Is peak season (Q4)
        self.df['Is_Peak_Season'] = (self.df['Quarter'] == 4).astype(int)
        created_features.append('Is_Peak_Season')
        
        # Years since launch (2015)
        self.df['Years_Since_Launch'] = self.df['Year'] - 2015
        created_features.append('Years_Since_Launch')
        
        # Month progress (0 to 1 within year)
        self.df['Month_Progress'] = (self.df['Month'] - 1) / 11
        created_features.append('Month_Progress')
        
        print(f"\n✓ Created {len(created_features)} temporal features:")
        for feat in created_features:
            print(f"  - {feat}")
        
        return self.df
    
    def create_interaction_features(self):
        """Create interaction features"""
        print("\n" + "=" * 80)
        print("STEP 4: CREATING INTERACTION FEATURES")
        print("=" * 80)
        
        created_features = []
        
        # Price per kWh
        self.df['Price_per_kWh'] = self.df['Avg_Price_USD'] / self.df['Battery_Capacity_kWh']
        created_features.append('Price_per_kWh')
        
        # Range per kWh (efficiency)
        self.df['Range_per_kWh'] = self.df['Range_km'] / self.df['Battery_Capacity_kWh']
        created_features.append('Range_per_kWh')
        
        # Delivery-Production ratio
        self.df['Delivery_Production_Ratio'] = self.df['Estimated_Deliveries'] / (self.df['Production_Units'] + 1)
        created_features.append('Delivery_Production_Ratio')
        
        # CO2 saved per delivery
        self.df['CO2_per_Delivery'] = self.df['CO2_Saved_tons'] / (self.df['Estimated_Deliveries'] + 1)
        created_features.append('CO2_per_Delivery')
        
        # Charging stations per 1000 deliveries
        self.df['Stations_per_1k_Deliveries'] = (self.df['Charging_Stations'] / 
                                                  (self.df['Estimated_Deliveries'] + 1)) * 1000
        created_features.append('Stations_per_1k_Deliveries')
        
        # Price-Range ratio
        self.df['Price_Range_Ratio'] = self.df['Avg_Price_USD'] / self.df['Range_km']
        created_features.append('Price_Range_Ratio')
        
        # Battery capacity category
        self.df['Battery_Category'] = pd.cut(self.df['Battery_Capacity_kWh'], 
                                             bins=[0, 75, 100, 150], 
                                             labels=['Small', 'Medium', 'Large'])
        created_features.append('Battery_Category')
        
        # Price category
        self.df['Price_Category'] = pd.cut(self.df['Avg_Price_USD'], 
                                           bins=[0, 70000, 90000, 150000], 
                                           labels=['Economy', 'Mid-Range', 'Premium'])
        created_features.append('Price_Category')
        
        print(f"\n✓ Created {len(created_features)} interaction features:")
        for feat in created_features:
            print(f"  - {feat}")
        
        return self.df
    
    def create_aggregate_features(self):
        """Create aggregate features by groups"""
        print("\n" + "=" * 80)
        print("STEP 5: CREATING AGGREGATE FEATURES")
        print("=" * 80)
        
        created_features = []
        
        # By Model
        model_agg = self.df.groupby('Model').agg({
            'Estimated_Deliveries': ['mean', 'std', 'max'],
            'Avg_Price_USD': ['mean', 'std']
        })
        model_agg.columns = ['_'.join(col).strip() for col in model_agg.columns.values]
        model_agg = model_agg.add_prefix('Model_')
        model_agg = model_agg.reset_index()
        
        self.df = self.df.merge(model_agg, on='Model', how='left')
        created_features.extend(model_agg.columns.tolist()[1:])
        
        # By Region
        region_agg = self.df.groupby('Region').agg({
            'Estimated_Deliveries': ['mean', 'std'],
            'Charging_Stations': 'mean'
        })
        region_agg.columns = ['_'.join(col).strip() for col in region_agg.columns.values]
        region_agg = region_agg.add_prefix('Region_')
        region_agg = region_agg.reset_index()
        
        self.df = self.df.merge(region_agg, on='Region', how='left')
        created_features.extend(region_agg.columns.tolist()[1:])
        
        # By Year
        year_agg = self.df.groupby('Year').agg({
            'Estimated_Deliveries': 'mean',
            'Avg_Price_USD': 'mean'
        })
        year_agg.columns = ['Year_Avg_Deliveries', 'Year_Avg_Price']
        year_agg = year_agg.reset_index()
        
        self.df = self.df.merge(year_agg, on='Year', how='left')
        created_features.extend(['Year_Avg_Deliveries', 'Year_Avg_Price'])
        
        print(f"\n✓ Created {len(created_features)} aggregate features")
        print("  Examples:")
        for feat in created_features[:8]:
            print(f"  - {feat}")
        
        return self.df
    
    def encode_categorical_features(self):
        """Encode categorical variables"""
        print("\n" + "=" * 80)
        print("STEP 6: ENCODING CATEGORICAL FEATURES")
        print("=" * 80)
        
        categorical_cols = ['Region', 'Model', 'Source_Type', 'MonthName', 
                           'Battery_Category', 'Price_Category']
        
        encoded_features = []
        
        for col in categorical_cols:
            if col in self.df.columns:
                # Label Encoding
                le = LabelEncoder()
                encoded_col = f'{col}_Encoded'
                self.df[encoded_col] = le.fit_transform(self.df[col].astype(str))
                self.label_encoders[col] = le
                encoded_features.append(encoded_col)
        
        # One-hot encoding for important categorical features
        onehot_cols = ['Region', 'Model']
        for col in onehot_cols:
            if col in self.df.columns:
                dummies = pd.get_dummies(self.df[col], prefix=col, drop_first=True)
                self.df = pd.concat([self.df, dummies], axis=1)
                encoded_features.extend(dummies.columns.tolist())
        
        print(f"\n✓ Encoded {len(categorical_cols)} categorical features")
        print(f"✓ Created {len(encoded_features)} encoded features")
        print("  Examples:")
        for feat in encoded_features[:10]:
            print(f"  - {feat}")
        
        return self.df
    
    def handle_missing_values_in_features(self):
        """Handle missing values created during feature engineering"""
        print("\n" + "=" * 80)
        print("STEP 7: HANDLING MISSING VALUES IN NEW FEATURES")
        print("=" * 80)
        
        missing_before = self.df.isnull().sum().sum()
        print(f"\nTotal missing values before: {missing_before}")
        
        # Fill missing values in lag and rolling features with forward fill then backward fill
        lag_rolling_cols = [col for col in self.df.columns if 'Lag' in col or 'Rolling' in col]
        for col in lag_rolling_cols:
            self.df[col] = self.df.groupby(['Model', 'Region'])[col].transform(
                lambda x: x.fillna(method='ffill').fillna(method='bfill').fillna(x.mean())
            )
        
        # Fill remaining missing values with median/mode
        for col in self.df.columns:
            if self.df[col].isnull().sum() > 0:
                if self.df[col].dtype in ['float64', 'int64']:
                    self.df[col].fillna(self.df[col].median(), inplace=True)
                else:
                    self.df[col].fillna(self.df[col].mode()[0], inplace=True)
        
        missing_after = self.df.isnull().sum().sum()
        print(f"Total missing values after: {missing_after}")
        print("✓ Missing values handled successfully!")
        
        return self.df
    
    def save_engineered_data(self, output_path='engineered_data.csv'):
        """Save feature-engineered dataset"""
        print("\n" + "=" * 80)
        print("STEP 8: SAVING ENGINEERED DATA")
        print("=" * 80)
        
        self.df.to_csv(output_path, index=False)
        
        print(f"\n✓ Feature-engineered data saved to: {output_path}")
        print(f"  - Shape: {self.df.shape}")
        print(f"  - Total Features: {len(self.df.columns)}")
        
        # Feature summary
        print("\n✓ Feature Categories:")
        original_features = ['Year', 'Month', 'Region', 'Model', 'Estimated_Deliveries', 
                           'Production_Units', 'Avg_Price_USD', 'Battery_Capacity_kWh', 
                           'Range_km', 'CO2_Saved_tons', 'Source_Type', 'Charging_Stations']
        new_features = [col for col in self.df.columns if col not in original_features]
        
        print(f"  - Original Features: {len(original_features)}")
        print(f"  - Engineered Features: {len(new_features)}")
        print(f"  - Total Features: {len(self.df.columns)}")
        
        return output_path
    
    def run_feature_engineering_pipeline(self):
        """Run the complete feature engineering pipeline"""
        print("\n" + "=" * 80)
        print("TESLA EV DATA - FEATURE ENGINEERING PIPELINE")
        print("=" * 80)
        
        self.load_data()
        self.create_lag_features()
        self.create_rolling_features()
        self.create_temporal_features()
        self.create_interaction_features()
        self.create_aggregate_features()
        self.encode_categorical_features()
        self.handle_missing_values_in_features()
        output_path = self.save_engineered_data()
        
        print("\n" + "=" * 80)
        print("FEATURE ENGINEERING COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print(f"\nFinal Dataset Shape: {self.df.shape}")
        print(f"Output File: {output_path}")
        
        return self.df

if __name__ == "__main__":
    # Initialize feature engineer
    feature_engineer = FeatureEngineer('cleaned_data.csv')
    
    # Run feature engineering pipeline
    engineered_data = feature_engineer.run_feature_engineering_pipeline()
    
    print("\n" + "=" * 80)
    print("Next Step: Run '4_regression_modeling.py' for Model Training")
    print("=" * 80)
