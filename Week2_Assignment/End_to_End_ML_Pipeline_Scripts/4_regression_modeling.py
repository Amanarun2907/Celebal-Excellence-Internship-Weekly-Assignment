"""
Week 2 Assignment - Tesla EV Deliveries and Production Data Analysis
Script 4: Regression Modeling
Author: Aman
Date: June 17, 2026
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
import joblib
import warnings
warnings.filterwarnings('ignore')

class RegressionModeler:
    def __init__(self, filepath, target_column='Estimated_Deliveries'):
        """Initialize the Regression Modeler"""
        self.filepath = filepath
        self.target_column = target_column
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.models = {}
        self.results = {}
        self.scaler = StandardScaler()
        
    def load_and_prepare_data(self):
        """Load and prepare data for modeling"""
        print("=" * 80)
        print("LOADING AND PREPARING DATA FOR MODELING")
        print("=" * 80)
        
        self.df = pd.read_csv(self.filepath)
        
        print(f"\n✓ Data loaded successfully!")
        print(f"  - Shape: {self.df.shape}")
        print(f"  - Target Variable: {self.target_column}")
        
        # Select features for modeling
        exclude_cols = ['Date', 'Region', 'Model', 'Source_Type', 'MonthName', 
                       'Battery_Category', 'Price_Category',
                       'Estimated_Deliveries', 'Production_Units', 'Avg_Price_USD']
        
        feature_cols = [col for col in self.df.columns if col not in exclude_cols]
        
        # Remove any remaining non-numeric columns
        feature_cols = [col for col in feature_cols if self.df[col].dtype in ['int64', 'float64']]
        
        X = self.df[feature_cols]
        y = self.df[self.target_column]
        
        print(f"\n✓ Features selected: {len(feature_cols)}")
        print(f"✓ Target variable: {self.target_column}")
        print(f"✓ Total samples: {len(y)}")
        
        # Handle any infinite values
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.median())
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=False
        )
        
        print(f"\n✓ Data split completed:")
        print(f"  - Training set: {self.X_train.shape}")
        print(f"  - Test set: {self.X_test.shape}")
        
        # Scale features
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print(f"✓ Feature scaling completed")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def define_models(self):
        """Define all regression models to compare"""
        print("\n" + "=" * 80)
        print("DEFINING REGRESSION MODELS")
        print("=" * 80)
        
        self.models = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(random_state=42),
            'Lasso Regression': Lasso(random_state=42),
            'ElasticNet': ElasticNet(random_state=42),
            'Decision Tree': DecisionTreeRegressor(random_state=42, max_depth=10),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'XGBoost': XGBRegressor(n_estimators=100, random_state=42, verbosity=0),
            'LightGBM': LGBMRegressor(n_estimators=100, random_state=42, verbosity=-1),
            'SVR': SVR(kernel='rbf')
        }
        
        print(f"\n✓ Defined {len(self.models)} regression models:")
        for i, model_name in enumerate(self.models.keys(), 1):
            print(f"  {i}. {model_name}")
        
        return self.models
    
    def train_and_evaluate_models(self):
        """Train and evaluate all models"""
        print("\n" + "=" * 80)
        print("TRAINING AND EVALUATING MODELS")
        print("=" * 80)
        
        results_list = []
        
        for model_name, model in self.models.items():
            print(f"\n{'='*60}")
            print(f"Training: {model_name}")
            print(f"{'='*60}")
            
            try:
                # Use scaled data for models that benefit from scaling
                if model_name in ['Ridge Regression', 'Lasso Regression', 'ElasticNet', 'SVR']:
                    X_train_use = self.X_train_scaled
                    X_test_use = self.X_test_scaled
                else:
                    X_train_use = self.X_train
                    X_test_use = self.X_test
                
                # Train model
                model.fit(X_train_use, self.y_train)
                
                # Make predictions
                y_train_pred = model.predict(X_train_use)
                y_test_pred = model.predict(X_test_use)
                
                # Calculate metrics
                train_metrics = self._calculate_metrics(self.y_train, y_train_pred, "Training")
                test_metrics = self._calculate_metrics(self.y_test, y_test_pred, "Testing")
                
                # Cross-validation score
                cv_scores = cross_val_score(model, X_train_use, self.y_train, 
                                           cv=5, scoring='r2', n_jobs=-1)
                cv_mean = cv_scores.mean()
                cv_std = cv_scores.std()
                
                print(f"\nCross-Validation R² Score: {cv_mean:.4f} (+/- {cv_std:.4f})")
                
                # Store results
                results_list.append({
                    'Model': model_name,
                    'Train_R2': train_metrics['R2'],
                    'Test_R2': test_metrics['R2'],
                    'Train_RMSE': train_metrics['RMSE'],
                    'Test_RMSE': test_metrics['RMSE'],
                    'Train_MAE': train_metrics['MAE'],
                    'Test_MAE': test_metrics['MAE'],
                    'Train_MAPE': train_metrics['MAPE'],
                    'Test_MAPE': test_metrics['MAPE'],
                    'CV_R2_Mean': cv_mean,
                    'CV_R2_Std': cv_std
                })
                
                # Store model and predictions
                self.results[model_name] = {
                    'model': model,
                    'train_pred': y_train_pred,
                    'test_pred': y_test_pred,
                    'metrics': {
                        'train': train_metrics,
                        'test': test_metrics,
                        'cv_r2': cv_mean
                    }
                }
                
                print(f"✓ {model_name} completed successfully!")
                
            except Exception as e:
                print(f"✗ Error training {model_name}: {str(e)}")
                continue
        
        # Create results dataframe
        self.results_df = pd.DataFrame(results_list)
        self.results_df = self.results_df.sort_values('Test_R2', ascending=False)
        
        print("\n" + "=" * 80)
        print("MODEL COMPARISON RESULTS")
        print("=" * 80)
        print("\n" + self.results_df.to_string(index=False))
        
        # Save results
        self.results_df.to_csv('model_comparison_results.csv', index=False)
        print("\n✓ Results saved to: model_comparison_results.csv")
        
        return self.results_df
    
    def _calculate_metrics(self, y_true, y_pred, dataset_name):
        """Calculate regression metrics"""
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        mape = mean_absolute_percentage_error(y_true, y_pred) * 100
        
        print(f"\n{dataset_name} Metrics:")
        print(f"  - RMSE: {rmse:.2f}")
        print(f"  - MAE:  {mae:.2f}")
        print(f"  - R²:   {r2:.4f}")
        print(f"  - MAPE: {mape:.2f}%")
        
        return {
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2,
            'MAPE': mape
        }
    
    def visualize_model_comparison(self):
        """Create visualization comparing all models"""
        print("\n" + "=" * 80)
        print("CREATING MODEL COMPARISON VISUALIZATIONS")
        print("=" * 80)
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Model Comparison - Tesla EV Deliveries Prediction', 
                    fontsize=16, fontweight='bold')
        
        # 1. R² Score Comparison
        sorted_df = self.results_df.sort_values('Test_R2')
        axes[0, 0].barh(sorted_df['Model'], sorted_df['Test_R2'], color='steelblue', alpha=0.8)
        axes[0, 0].set_xlabel('R² Score', fontweight='bold')
        axes[0, 0].set_title('Test R² Score by Model', fontweight='bold')
        axes[0, 0].axvline(x=0.8, color='r', linestyle='--', label='0.80 threshold')
        axes[0, 0].legend()
        axes[0, 0].grid(axis='x', alpha=0.3)
        
        # 2. RMSE Comparison
        sorted_df_rmse = self.results_df.sort_values('Test_RMSE', ascending=False)
        axes[0, 1].barh(sorted_df_rmse['Model'], sorted_df_rmse['Test_RMSE'], 
                       color='coral', alpha=0.8)
        axes[0, 1].set_xlabel('RMSE', fontweight='bold')
        axes[0, 1].set_title('Test RMSE by Model (Lower is Better)', fontweight='bold')
        axes[0, 1].grid(axis='x', alpha=0.3)
        
        # 3. MAE Comparison
        sorted_df_mae = self.results_df.sort_values('Test_MAE', ascending=False)
        axes[1, 0].barh(sorted_df_mae['Model'], sorted_df_mae['Test_MAE'], 
                       color='lightgreen', alpha=0.8)
        axes[1, 0].set_xlabel('MAE', fontweight='bold')
        axes[1, 0].set_title('Test MAE by Model (Lower is Better)', fontweight='bold')
        axes[1, 0].grid(axis='x', alpha=0.3)
        
        # 4. Train vs Test R² (Overfitting check)
        x_pos = np.arange(len(self.results_df))
        width = 0.35
        axes[1, 1].bar(x_pos - width/2, self.results_df['Train_R2'], width, 
                      label='Train R²', color='skyblue', alpha=0.8)
        axes[1, 1].bar(x_pos + width/2, self.results_df['Test_R2'], width, 
                      label='Test R²', color='orange', alpha=0.8)
        axes[1, 1].set_xlabel('Model', fontweight='bold')
        axes[1, 1].set_ylabel('R² Score', fontweight='bold')
        axes[1, 1].set_title('Train vs Test R² Score (Overfitting Check)', fontweight='bold')
        axes[1, 1].set_xticks(x_pos)
        axes[1, 1].set_xticklabels(self.results_df['Model'], rotation=45, ha='right')
        axes[1, 1].legend()
        axes[1, 1].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('6_model_comparison.png', dpi=300, bbox_inches='tight')
        print("\n✓ Model comparison visualization saved: 6_model_comparison.png")
        plt.close()
    
    def visualize_best_model_predictions(self):
        """Visualize predictions from the best model"""
        print("\n" + "=" * 80)
        print("VISUALIZING BEST MODEL PREDICTIONS")
        print("=" * 80)
        
        # Get best model
        best_model_name = self.results_df.iloc[0]['Model']
        best_model_results = self.results[best_model_name]
        
        print(f"\n✓ Best Model: {best_model_name}")
        print(f"  - Test R²: {best_model_results['metrics']['test']['R2']:.4f}")
        print(f"  - Test RMSE: {best_model_results['metrics']['test']['RMSE']:.2f}")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Best Model Predictions: {best_model_name}', 
                    fontsize=16, fontweight='bold')
        
        # 1. Actual vs Predicted (Test Set)
        axes[0, 0].scatter(self.y_test, best_model_results['test_pred'], alpha=0.6, s=50)
        axes[0, 0].plot([self.y_test.min(), self.y_test.max()], 
                       [self.y_test.min(), self.y_test.max()], 
                       'r--', lw=2, label='Perfect Prediction')
        axes[0, 0].set_xlabel('Actual Deliveries', fontweight='bold')
        axes[0, 0].set_ylabel('Predicted Deliveries', fontweight='bold')
        axes[0, 0].set_title('Actual vs Predicted (Test Set)', fontweight='bold')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Residuals Plot
        residuals = self.y_test - best_model_results['test_pred']
        axes[0, 1].scatter(best_model_results['test_pred'], residuals, alpha=0.6, s=50)
        axes[0, 1].axhline(y=0, color='r', linestyle='--', lw=2)
        axes[0, 1].set_xlabel('Predicted Deliveries', fontweight='bold')
        axes[0, 1].set_ylabel('Residuals', fontweight='bold')
        axes[0, 1].set_title('Residual Plot (Test Set)', fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Residuals Distribution
        axes[1, 0].hist(residuals, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        axes[1, 0].axvline(x=0, color='r', linestyle='--', lw=2)
        axes[1, 0].set_xlabel('Residuals', fontweight='bold')
        axes[1, 0].set_ylabel('Frequency', fontweight='bold')
        axes[1, 0].set_title('Distribution of Residuals', fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Predictions over time
        test_indices = self.y_test.index
        axes[1, 1].plot(test_indices, self.y_test.values, 
                       label='Actual', linewidth=2, marker='o', markersize=4)
        axes[1, 1].plot(test_indices, best_model_results['test_pred'], 
                       label='Predicted', linewidth=2, marker='s', markersize=4)
        axes[1, 1].set_xlabel('Sample Index', fontweight='bold')
        axes[1, 1].set_ylabel('Deliveries', fontweight='bold')
        axes[1, 1].set_title('Actual vs Predicted Over Time', fontweight='bold')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('7_best_model_predictions.png', dpi=300, bbox_inches='tight')
        print("\n✓ Best model predictions visualization saved: 7_best_model_predictions.png")
        plt.close()
    
    def save_best_model(self):
        """Save the best performing model"""
        print("\n" + "=" * 80)
        print("SAVING BEST MODEL")
        print("=" * 80)
        
        best_model_name = self.results_df.iloc[0]['Model']
        best_model = self.results[best_model_name]['model']
        
        # Save model and scaler
        joblib.dump(best_model, 'best_regression_model.pkl')
        joblib.dump(self.scaler, 'feature_scaler.pkl')
        
        print(f"\n✓ Best model saved: {best_model_name}")
        print(f"  - Model file: best_regression_model.pkl")
        print(f"  - Scaler file: feature_scaler.pkl")
        print(f"  - Test R²: {self.results_df.iloc[0]['Test_R2']:.4f}")
        print(f"  - Test RMSE: {self.results_df.iloc[0]['Test_RMSE']:.2f}")
        
        return best_model_name
    
    def run_modeling_pipeline(self):
        """Run the complete modeling pipeline"""
        print("\n" + "=" * 80)
        print("TESLA EV DATA - REGRESSION MODELING PIPELINE")
        print("=" * 80)
        
        self.load_and_prepare_data()
        self.define_models()
        self.train_and_evaluate_models()
        self.visualize_model_comparison()
        self.visualize_best_model_predictions()
        best_model = self.save_best_model()
        
        print("\n" + "=" * 80)
        print("REGRESSION MODELING COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print(f"\nBest Model: {best_model}")
        print("Output Files:")
        print("  - model_comparison_results.csv")
        print("  - 6_model_comparison.png")
        print("  - 7_best_model_predictions.png")
        print("  - best_regression_model.pkl")
        print("  - feature_scaler.pkl")

if __name__ == "__main__":
    # Initialize modeler
    modeler = RegressionModeler('engineered_data.csv', target_column='Estimated_Deliveries')
    
    # Run modeling pipeline
    modeler.run_modeling_pipeline()
    
    print("\n" + "=" * 80)
    print("Next Step: Run '5_hyperparameter_tuning.py' for Hyperparameter Optimization")
    print("=" * 80)
