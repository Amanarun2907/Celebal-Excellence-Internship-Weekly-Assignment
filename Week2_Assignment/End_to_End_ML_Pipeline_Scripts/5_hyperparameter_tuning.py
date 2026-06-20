"""
Week 2 Assignment - Tesla EV Deliveries and Production Data Analysis
Script 5: Hyperparameter Tuning
Author: Aman
Date: June 17, 2026
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import warnings
warnings.filterwarnings('ignore')

class HyperparameterTuner:
    def __init__(self, X_train, X_test, y_train, y_test):
        """Initialize the Hyperparameter Tuner"""
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.tuned_models = {}
        self.best_params = {}
        
    def tune_random_forest(self):
        """Tune Random Forest hyperparameters"""
        print("=" * 80)
        print("TUNING RANDOM FOREST")
        print("=" * 80)
        
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, 30, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2']
        }
        
        rf = RandomForestRegressor(random_state=42, n_jobs=-1)
        
        print("\nPerforming RandomizedSearchCV...")
        print(f"Parameter space size: {np.prod([len(v) for v in param_grid.values()])}")
        
        random_search = RandomizedSearchCV(
            rf, param_distributions=param_grid, n_iter=20,
            cv=5, scoring='r2', n_jobs=-1, random_state=42, verbose=1
        )
        
        random_search.fit(self.X_train, self.y_train)
        
        print(f"\n✓ Best parameters found:")
        for param, value in random_search.best_params_.items():
            print(f"  - {param}: {value}")
        
        print(f"\n✓ Best cross-validation R² score: {random_search.best_score_:.4f}")
        
        # Evaluate on test set
        y_pred = random_search.predict(self.X_test)
        test_r2 = r2_score(self.y_test, y_pred)
        test_rmse = np.sqrt(mean_squared_error(self.y_test, y_pred))
        
        print(f"✓ Test R² score: {test_r2:.4f}")
        print(f"✓ Test RMSE: {test_rmse:.2f}")
        
        self.tuned_models['Random Forest'] = random_search.best_estimator_
        self.best_params['Random Forest'] = random_search.best_params_
        
        return random_search.best_estimator_
    
    def tune_gradient_boosting(self):
        """Tune Gradient Boosting hyperparameters"""
        print("\n" + "=" * 80)
        print("TUNING GRADIENT BOOSTING")
        print("=" * 80)
        
        param_grid = {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [3, 5, 7],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'subsample': [0.8, 0.9, 1.0]
        }
        
        gb = GradientBoostingRegressor(random_state=42)
        
        print("\nPerforming RandomizedSearchCV...")
        print(f"Parameter space size: {np.prod([len(v) for v in param_grid.values()])}")
        
        random_search = RandomizedSearchCV(
            gb, param_distributions=param_grid, n_iter=20,
            cv=5, scoring='r2', n_jobs=-1, random_state=42, verbose=1
        )
        
        random_search.fit(self.X_train, self.y_train)
        
        print(f"\n✓ Best parameters found:")
        for param, value in random_search.best_params_.items():
            print(f"  - {param}: {value}")
        
        print(f"\n✓ Best cross-validation R² score: {random_search.best_score_:.4f}")
        
        # Evaluate on test set
        y_pred = random_search.predict(self.X_test)
        test_r2 = r2_score(self.y_test, y_pred)
        test_rmse = np.sqrt(mean_squared_error(self.y_test, y_pred))
        
        print(f"✓ Test R² score: {test_r2:.4f}")
        print(f"✓ Test RMSE: {test_rmse:.2f}")
        
        self.tuned_models['Gradient Boosting'] = random_search.best_estimator_
        self.best_params['Gradient Boosting'] = random_search.best_params_
        
        return random_search.best_estimator_
    
    def tune_xgboost(self):
        """Tune XGBoost hyperparameters"""
        print("\n" + "=" * 80)
        print("TUNING XGBOOST")
        print("=" * 80)
        
        param_grid = {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [3, 5, 7, 9],
            'min_child_weight': [1, 3, 5],
            'subsample': [0.7, 0.8, 0.9],
            'colsample_bytree': [0.7, 0.8, 0.9],
            'gamma': [0, 0.1, 0.2]
        }
        
        xgb = XGBRegressor(random_state=42, verbosity=0)
        
        print("\nPerforming RandomizedSearchCV...")
        print(f"Parameter space size: {np.prod([len(v) for v in param_grid.values()])}")
        
        random_search = RandomizedSearchCV(
            xgb, param_distributions=param_grid, n_iter=25,
            cv=5, scoring='r2', n_jobs=-1, random_state=42, verbose=1
        )
        
        random_search.fit(self.X_train, self.y_train)
        
        print(f"\n✓ Best parameters found:")
        for param, value in random_search.best_params_.items():
            print(f"  - {param}: {value}")
        
        print(f"\n✓ Best cross-validation R² score: {random_search.best_score_:.4f}")
        
        # Evaluate on test set
        y_pred = random_search.predict(self.X_test)
        test_r2 = r2_score(self.y_test, y_pred)
        test_rmse = np.sqrt(mean_squared_error(self.y_test, y_pred))
        
        print(f"✓ Test R² score: {test_r2:.4f}")
        print(f"✓ Test RMSE: {test_rmse:.2f}")
        
        self.tuned_models['XGBoost'] = random_search.best_estimator_
        self.best_params['XGBoost'] = random_search.best_params_
        
        return random_search.best_estimator_
    
    def tune_lightgbm(self):
        """Tune LightGBM hyperparameters"""
        print("\n" + "=" * 80)
        print("TUNING LIGHTGBM")
        print("=" * 80)
        
        param_grid = {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [3, 5, 7, -1],
            'num_leaves': [31, 50, 70],
            'min_child_samples': [20, 30, 50],
            'subsample': [0.7, 0.8, 0.9],
            'colsample_bytree': [0.7, 0.8, 0.9]
        }
        
        lgbm = LGBMRegressor(random_state=42, verbosity=-1)
        
        print("\nPerforming RandomizedSearchCV...")
        print(f"Parameter space size: {np.prod([len(v) for v in param_grid.values()])}")
        
        random_search = RandomizedSearchCV(
            lgbm, param_distributions=param_grid, n_iter=25,
            cv=5, scoring='r2', n_jobs=-1, random_state=42, verbose=1
        )
        
        random_search.fit(self.X_train, self.y_train)
        
        print(f"\n✓ Best parameters found:")
        for param, value in random_search.best_params_.items():
            print(f"  - {param}: {value}")
        
        print(f"\n✓ Best cross-validation R² score: {random_search.best_score_:.4f}")
        
        # Evaluate on test set
        y_pred = random_search.predict(self.X_test)
        test_r2 = r2_score(self.y_test, y_pred)
        test_rmse = np.sqrt(mean_squared_error(self.y_test, y_pred))
        
        print(f"✓ Test R² score: {test_r2:.4f}")
        print(f"✓ Test RMSE: {test_rmse:.2f}")
        
        self.tuned_models['LightGBM'] = random_search.best_estimator_
        self.best_params['LightGBM'] = random_search.best_params_
        
        return random_search.best_estimator_
    
    def compare_tuned_models(self):
        """Compare all tuned models"""
        print("\n" + "=" * 80)
        print("COMPARING TUNED MODELS")
        print("=" * 80)
        
        results = []
        
        for model_name, model in self.tuned_models.items():
            y_pred = model.predict(self.X_test)
            
            r2 = r2_score(self.y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(self.y_test, y_pred))
            mae = mean_absolute_error(self.y_test, y_pred)
            
            results.append({
                'Model': model_name,
                'R2_Score': r2,
                'RMSE': rmse,
                'MAE': mae
            })
        
        results_df = pd.DataFrame(results).sort_values('R2_Score', ascending=False)
        
        print("\n" + results_df.to_string(index=False))
        
        # Save results
        results_df.to_csv('tuned_models_comparison.csv', index=False)
        print("\n✓ Comparison results saved: tuned_models_comparison.csv")
        
        # Visualization
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle('Tuned Models Performance Comparison', fontsize=16, fontweight='bold')
        
        # R² Score
        axes[0].barh(results_df['Model'], results_df['R2_Score'], color='steelblue', alpha=0.8)
        axes[0].set_xlabel('R² Score', fontweight='bold')
        axes[0].set_title('R² Score Comparison', fontweight='bold')
        axes[0].grid(axis='x', alpha=0.3)
        
        # RMSE
        axes[1].barh(results_df['Model'], results_df['RMSE'], color='coral', alpha=0.8)
        axes[1].set_xlabel('RMSE', fontweight='bold')
        axes[1].set_title('RMSE Comparison', fontweight='bold')
        axes[1].grid(axis='x', alpha=0.3)
        
        # MAE
        axes[2].barh(results_df['Model'], results_df['MAE'], color='lightgreen', alpha=0.8)
        axes[2].set_xlabel('MAE', fontweight='bold')
        axes[2].set_title('MAE Comparison', fontweight='bold')
        axes[2].grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('8_tuned_models_comparison.png', dpi=300, bbox_inches='tight')
        print("✓ Visualization saved: 8_tuned_models_comparison.png")
        plt.close()
        
        return results_df
    
    def save_best_tuned_model(self):
        """Save the best tuned model"""
        print("\n" + "=" * 80)
        print("SAVING BEST TUNED MODEL")
        print("=" * 80)
        
        # Find best model
        best_model_name = None
        best_r2 = -np.inf
        
        for model_name, model in self.tuned_models.items():
            y_pred = model.predict(self.X_test)
            r2 = r2_score(self.y_test, y_pred)
            
            if r2 > best_r2:
                best_r2 = r2
                best_model_name = model_name
        
        best_model = self.tuned_models[best_model_name]
        
        # Save model
        joblib.dump(best_model, 'best_tuned_model.pkl')
        
        # Save best parameters
        with open('best_hyperparameters.txt', 'w') as f:
            f.write(f"Best Tuned Model: {best_model_name}\n")
            f.write(f"Test R² Score: {best_r2:.4f}\n\n")
            f.write("Best Hyperparameters:\n")
            for param, value in self.best_params[best_model_name].items():
                f.write(f"  {param}: {value}\n")
        
        print(f"\n✓ Best tuned model: {best_model_name}")
        print(f"  - Test R² Score: {best_r2:.4f}")
        print(f"  - Model saved: best_tuned_model.pkl")
        print(f"  - Parameters saved: best_hyperparameters.txt")
        
        return best_model_name, best_model
    
    def run_tuning_pipeline(self):
        """Run the complete hyperparameter tuning pipeline"""
        print("\n" + "=" * 80)
        print("HYPERPARAMETER TUNING PIPELINE")
        print("=" * 80)
        
        self.tune_random_forest()
        self.tune_gradient_boosting()
        self.tune_xgboost()
        self.tune_lightgbm()
        self.compare_tuned_models()
        best_name, best_model = self.save_best_tuned_model()
        
        print("\n" + "=" * 80)
        print("HYPERPARAMETER TUNING COMPLETED!")
        print("=" * 80)
        print(f"\nBest Model: {best_name}")
        print("\nOutput Files:")
        print("  - tuned_models_comparison.csv")
        print("  - 8_tuned_models_comparison.png")
        print("  - best_tuned_model.pkl")
        print("  - best_hyperparameters.txt")

def load_data_for_tuning():
    """Load preprocessed data for tuning"""
    df = pd.read_csv('engineered_data.csv')
    
    exclude_cols = ['Date', 'Region', 'Model', 'Source_Type', 'MonthName', 
                   'Battery_Category', 'Price_Category',
                   'Estimated_Deliveries', 'Production_Units', 'Avg_Price_USD']
    
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    feature_cols = [col for col in feature_cols if df[col].dtype in ['int64', 'float64']]
    
    X = df[feature_cols]
    y = df['Estimated_Deliveries']
    
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median())
    
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=False
    )
    
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    print("Loading data...")
    X_train, X_test, y_train, y_test = load_data_for_tuning()
    
    print(f"\nData loaded:")
    print(f"  - Training samples: {len(X_train)}")
    print(f"  - Test samples: {len(X_test)}")
    print(f"  - Features: {X_train.shape[1]}")
    
    # Initialize tuner
    tuner = HyperparameterTuner(X_train, X_test, y_train, y_test)
    
    # Run tuning pipeline
    tuner.run_tuning_pipeline()
    
    print("\n" + "=" * 80)
    print("Next Step: Run '6_time_series_forecasting.py' for Time Series Models")
    print("=" * 80)
