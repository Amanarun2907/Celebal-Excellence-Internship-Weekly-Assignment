"""
Week 2 Assignment - Tesla EV Deliveries and Production Data Analysis
Script 6: Time Series Forecasting
Author: Aman
Date: June 17, 2026
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import warnings
warnings.filterwarnings('ignore')

class TimeSeriesForecaster:
    def __init__(self, filepath):
        """Initialize the Time Series Forecaster"""
        self.filepath = filepath
        self.df = None
        self.ts_data = None
        self.train_size = None
        self.forecast_periods = 12  # Forecast next 12 months
        self.results = {}
        
    def load_and_prepare_data(self):
        """Load and prepare time series data"""
        print("=" * 80)
        print("LOADING AND PREPARING TIME SERIES DATA")
        print("=" * 80)
        
        self.df = pd.read_csv(self.filepath)
        self.df['Date'] = pd.to_datetime(self.df['Date'])
        
        # Aggregate by date for time series
        self.ts_data = self.df.groupby('Date')['Estimated_Deliveries'].sum().sort_index()
        
        print(f"\n✓ Time series data prepared")
        print(f"  - Date range: {self.ts_data.index.min()} to {self.ts_data.index.max()}")
        print(f"  - Total periods: {len(self.ts_data)}")
        print(f"  - Forecast horizon: {self.forecast_periods} months")
        
        # Split into train and test
        self.train_size = len(self.ts_data) - self.forecast_periods
        self.train_data = self.ts_data[:self.train_size]
        self.test_data = self.ts_data[self.train_size:]
        
        print(f"\n✓ Train-Test split:")
        print(f"  - Training samples: {len(self.train_data)}")
        print(f"  - Test samples: {len(self.test_data)}")
        
        return self.ts_data
    
    def arima_forecast(self):
        """ARIMA forecasting"""
        print("\n" + "=" * 80)
        print("ARIMA FORECASTING")
        print("=" * 80)
        
        print("\nFitting ARIMA model...")
        
        # Try different order combinations and select best based on AIC
        best_aic = np.inf
        best_order = None
        best_model = None
        
        p_values = range(0, 3)
        d_values = range(0, 2)
        q_values = range(0, 3)
        
        print("Finding optimal parameters...")
        
        for p in p_values:
            for d in d_values:
                for q in q_values:
                    try:
                        model = ARIMA(self.train_data, order=(p, d, q))
                        fitted = model.fit()
                        if fitted.aic < best_aic:
                            best_aic = fitted.aic
                            best_order = (p, d, q)
                            best_model = fitted
                    except:
                        continue
        
        print(f"\n✓ Best ARIMA order: {best_order}")
        print(f"✓ AIC: {best_aic:.2f}")
        
        # Forecast
        forecast = best_model.forecast(steps=self.forecast_periods)
        
        # Calculate metrics
        metrics = self._calculate_metrics(self.test_data.values, forecast.values, "ARIMA")
        
        self.results['ARIMA'] = {
            'model': best_model,
            'forecast': forecast,
            'metrics': metrics,
            'order': best_order
        }
        
        return forecast, metrics
    
    def sarima_forecast(self):
        """SARIMA forecasting"""
        print("\n" + "=" * 80)
        print("SARIMA FORECASTING")
        print("=" * 80)
        
        print("\nFitting SARIMA model...")
        
        # Using seasonal order (1,1,1,12) for monthly seasonality
        order = (1, 1, 1)
        seasonal_order = (1, 1, 1, 12)
        
        try:
            model = SARIMAX(self.train_data, 
                          order=order,
                          seasonal_order=seasonal_order,
                          enforce_stationarity=False,
                          enforce_invertibility=False)
            
            fitted = model.fit(disp=False)
            
            print(f"\n✓ SARIMA order: {order}")
            print(f"✓ Seasonal order: {seasonal_order}")
            print(f"✓ AIC: {fitted.aic:.2f}")
            
            # Forecast
            forecast = fitted.forecast(steps=self.forecast_periods)
            
            # Calculate metrics
            metrics = self._calculate_metrics(self.test_data.values, forecast.values, "SARIMA")
            
            self.results['SARIMA'] = {
                'model': fitted,
                'forecast': forecast,
                'metrics': metrics,
                'order': order,
                'seasonal_order': seasonal_order
            }
            
            return forecast, metrics
            
        except Exception as e:
            print(f"✗ Error fitting SARIMA: {str(e)}")
            print("  Using simpler configuration...")
            
            # Fallback to simpler model
            order = (1, 0, 1)
            seasonal_order = (1, 0, 1, 12)
            
            model = SARIMAX(self.train_data, order=order, seasonal_order=seasonal_order)
            fitted = model.fit(disp=False)
            forecast = fitted.forecast(steps=self.forecast_periods)
            metrics = self._calculate_metrics(self.test_data.values, forecast.values, "SARIMA")
            
            self.results['SARIMA'] = {
                'model': fitted,
                'forecast': forecast,
                'metrics': metrics,
                'order': order,
                'seasonal_order': seasonal_order
            }
            
            return forecast, metrics
    
    def prophet_forecast(self):
        """Prophet forecasting"""
        print("\n" + "=" * 80)
        print("PROPHET FORECASTING")
        print("=" * 80)
        
        print("\nPreparing data for Prophet...")
        
        # Prepare data for Prophet
        prophet_df = pd.DataFrame({
            'ds': self.train_data.index,
            'y': self.train_data.values
        })
        
        # Initialize and fit Prophet model
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
            seasonality_mode='multiplicative',
            changepoint_prior_scale=0.05
        )
        
        print("Fitting Prophet model...")
        model.fit(prophet_df)
        
        # Create future dataframe
        future = model.make_future_dataframe(periods=self.forecast_periods, freq='MS')
        
        # Make predictions
        forecast_full = model.predict(future)
        forecast = forecast_full.tail(self.forecast_periods)['yhat'].values
        
        print("✓ Prophet model fitted successfully")
        
        # Calculate metrics
        metrics = self._calculate_metrics(self.test_data.values, forecast, "Prophet")
        
        self.results['Prophet'] = {
            'model': model,
            'forecast': forecast,
            'forecast_full': forecast_full,
            'metrics': metrics
        }
        
        return forecast, metrics
    
    def lstm_forecast(self):
        """LSTM forecasting"""
        print("\n" + "=" * 80)
        print("LSTM FORECASTING")
        print("=" * 80)
        
        print("\nPreparing data for LSTM...")
        
        # Scale the data
        scaler = MinMaxScaler()
        scaled_train = scaler.fit_transform(self.train_data.values.reshape(-1, 1))
        scaled_test = scaler.transform(self.test_data.values.reshape(-1, 1))
        
        # Create sequences
        lookback = 12
        X_train, y_train = self._create_sequences(scaled_train, lookback)
        
        print(f"✓ Created sequences with lookback: {lookback}")
        print(f"  - Training sequences: {X_train.shape}")
        
        # Build LSTM model
        model = Sequential([
            LSTM(50, activation='relu', return_sequences=True, input_shape=(lookback, 1)),
            Dropout(0.2),
            LSTM(50, activation='relu'),
            Dropout(0.2),
            Dense(25, activation='relu'),
            Dense(1)
        ])
        
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        
        print("\n✓ LSTM Architecture:")
        model.summary()
        
        # Train model
        print("\nTraining LSTM model...")
        early_stop = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
        
        history = model.fit(
            X_train, y_train,
            epochs=50,
            batch_size=16,
            callbacks=[early_stop],
            verbose=0
        )
        
        print(f"✓ Training completed in {len(history.history['loss'])} epochs")
        
        # Make predictions
        last_sequence = scaled_train[-lookback:]
        forecast_scaled = []
        
        for _ in range(self.forecast_periods):
            pred = model.predict(last_sequence.reshape(1, lookback, 1), verbose=0)
            forecast_scaled.append(pred[0, 0])
            last_sequence = np.append(last_sequence[1:], pred)
        
        # Inverse transform
        forecast = scaler.inverse_transform(np.array(forecast_scaled).reshape(-1, 1)).flatten()
        
        # Calculate metrics
        metrics = self._calculate_metrics(self.test_data.values, forecast, "LSTM")
        
        self.results['LSTM'] = {
            'model': model,
            'scaler': scaler,
            'forecast': forecast,
            'metrics': metrics,
            'lookback': lookback,
            'history': history.history
        }
        
        return forecast, metrics
    
    def _create_sequences(self, data, lookback):
        """Create sequences for LSTM"""
        X, y = [], []
        for i in range(len(data) - lookback):
            X.append(data[i:i+lookback])
            y.append(data[i+lookback])
        return np.array(X), np.array(y)
    
    def _calculate_metrics(self, actual, predicted, model_name):
        """Calculate forecasting metrics"""
        rmse = np.sqrt(mean_squared_error(actual, predicted))
        mae = mean_absolute_error(actual, predicted)
        mape = mean_absolute_percentage_error(actual, predicted) * 100
        r2 = r2_score(actual, predicted)
        
        print(f"\n{model_name} Forecast Metrics:")
        print(f"  - RMSE: {rmse:.2f}")
        print(f"  - MAE:  {mae:.2f}")
        print(f"  - MAPE: {mape:.2f}%")
        print(f"  - R²:   {r2:.4f}")
        
        return {
            'RMSE': rmse,
            'MAE': mae,
            'MAPE': mape,
            'R2': r2
        }
    
    def compare_forecasts(self):
        """Compare all forecasting methods"""
        print("\n" + "=" * 80)
        print("COMPARING FORECAST MODELS")
        print("=" * 80)
        
        comparison = []
        for model_name, result in self.results.items():
            metrics = result['metrics']
            comparison.append({
                'Model': model_name,
                'RMSE': metrics['RMSE'],
                'MAE': metrics['MAE'],
                'MAPE': metrics['MAPE'],
                'R2': metrics['R2']
            })
        
        comparison_df = pd.DataFrame(comparison).sort_values('RMSE')
        
        print("\n" + comparison_df.to_string(index=False))
        
        # Save results
        comparison_df.to_csv('forecast_models_comparison.csv', index=False)
        print("\n✓ Comparison saved: forecast_models_comparison.csv")
        
        return comparison_df
    
    def visualize_forecasts(self):
        """Visualize all forecasts"""
        print("\n" + "=" * 80)
        print("CREATING FORECAST VISUALIZATIONS")
        print("=" * 80)
        
        fig, axes = plt.subplots(2, 2, figsize=(18, 12))
        fig.suptitle('Time Series Forecasting Comparison', fontsize=16, fontweight='bold')
        
        models = ['ARIMA', 'SARIMA', 'Prophet', 'LSTM']
        
        for idx, model_name in enumerate(models):
            row = idx // 2
            col = idx % 2
            
            if model_name in self.results:
                # Plot historical data
                axes[row, col].plot(self.train_data.index, self.train_data.values, 
                                   label='Training', linewidth=2, color='blue')
                axes[row, col].plot(self.test_data.index, self.test_data.values, 
                                   label='Actual', linewidth=2, color='green', marker='o')
                
                # Plot forecast
                forecast = self.results[model_name]['forecast']
                axes[row, col].plot(self.test_data.index, forecast, 
                                   label='Forecast', linewidth=2, color='red', 
                                   marker='s', linestyle='--')
                
                axes[row, col].set_title(f'{model_name} Forecast', fontweight='bold')
                axes[row, col].set_xlabel('Date', fontweight='bold')
                axes[row, col].set_ylabel('Deliveries', fontweight='bold')
                axes[row, col].legend()
                axes[row, col].grid(True, alpha=0.3)
                axes[row, col].tick_params(axis='x', rotation=45)
                
                # Add metrics text
                metrics = self.results[model_name]['metrics']
                metrics_text = f"RMSE: {metrics['RMSE']:.0f}\nR²: {metrics['R2']:.3f}"
                axes[row, col].text(0.02, 0.98, metrics_text,
                                   transform=axes[row, col].transAxes,
                                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                                   verticalalignment='top', fontsize=9)
        
        plt.tight_layout()
        plt.savefig('9_time_series_forecasts.png', dpi=300, bbox_inches='tight')
        print("\n✓ Forecast visualization saved: 9_time_series_forecasts.png")
        plt.close()
        
        # Create combined forecast plot
        self._plot_combined_forecasts()
    
    def _plot_combined_forecasts(self):
        """Create a single plot with all forecasts"""
        plt.figure(figsize=(16, 8))
        
        # Plot historical data
        plt.plot(self.train_data.index, self.train_data.values, 
                label='Training Data', linewidth=2, color='blue', alpha=0.7)
        plt.plot(self.test_data.index, self.test_data.values, 
                label='Actual Test Data', linewidth=3, color='green', marker='o', markersize=6)
        
        # Plot all forecasts
        colors = ['red', 'orange', 'purple', 'brown']
        markers = ['s', '^', 'D', 'v']
        
        for idx, (model_name, color, marker) in enumerate(zip(self.results.keys(), colors, markers)):
            forecast = self.results[model_name]['forecast']
            plt.plot(self.test_data.index, forecast, 
                    label=f'{model_name} Forecast', linewidth=2, color=color,
                    marker=marker, markersize=5, linestyle='--', alpha=0.7)
        
        plt.title('All Time Series Forecasts Comparison', fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Date', fontsize=12, fontweight='bold')
        plt.ylabel('Estimated Deliveries', fontsize=12, fontweight='bold')
        plt.legend(loc='best', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('10_combined_forecasts.png', dpi=300, bbox_inches='tight')
        print("✓ Combined forecast plot saved: 10_combined_forecasts.png")
        plt.close()
    
    def run_forecasting_pipeline(self):
        """Run the complete time series forecasting pipeline"""
        print("\n" + "=" * 80)
        print("TIME SERIES FORECASTING PIPELINE")
        print("=" * 80)
        
        self.load_and_prepare_data()
        self.arima_forecast()
        self.sarima_forecast()
        self.prophet_forecast()
        self.lstm_forecast()
        comparison = self.compare_forecasts()
        self.visualize_forecasts()
        
        # Print best model
        best_model = comparison.iloc[0]['Model']
        best_rmse = comparison.iloc[0]['RMSE']
        
        print("\n" + "=" * 80)
        print("TIME SERIES FORECASTING COMPLETED!")
        print("=" * 80)
        print(f"\nBest Model: {best_model}")
        print(f"  - RMSE: {best_rmse:.2f}")
        print("\nOutput Files:")
        print("  - forecast_models_comparison.csv")
        print("  - 9_time_series_forecasts.png")
        print("  - 10_combined_forecasts.png")

if __name__ == "__main__":
    # Initialize forecaster
    forecaster = TimeSeriesForecaster('cleaned_data.csv')
    
    # Run forecasting pipeline
    forecaster.run_forecasting_pipeline()
    
    print("\n" + "=" * 80)
    print("Next Step: Run '7_evaluation_report.py' for Final Evaluation")
    print("=" * 80)
