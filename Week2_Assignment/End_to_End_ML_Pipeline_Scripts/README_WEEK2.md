# Week 2 Assignment: Tesla EV Deliveries and Production Data Analysis

## 📊 Project Overview

This project implements a **comprehensive end-to-end Machine Learning pipeline** for analyzing and predicting Tesla EV deliveries and production data from 2015 to 2025. The pipeline covers data preprocessing, exploratory data analysis, feature engineering, regression modeling, hyperparameter tuning, and time series forecasting.

**Author:** Aman  
**Program:** Celebal Excellence Internship - Data Science  
**Week:** 2 - Classical Machine Learning  
**Date:** June 17, 2026

---

## 🎯 Objectives

1. **Data Preprocessing:** Clean and prepare Tesla EV dataset
2. **Exploratory Data Analysis (EDA):** Understand patterns, trends, and relationships
3. **Feature Engineering:** Create meaningful features for modeling
4. **Regression Modeling:** Compare multiple ML algorithms
5. **Hyperparameter Tuning:** Optimize best-performing models
6. **Time Series Forecasting:** Forecast future deliveries using ARIMA, SARIMA, Prophet, and LSTM
7. **Model Evaluation:** Comprehensive performance analysis

---

## 📁 Project Structure

```
Week2_Assignment_Tesla EA Deliveries and Production Data(2015–2025)/
│
├── tesla_deliveries_dataset_2015_2025.csv    # Original dataset
│
├── 1_data_preprocessing.py                    # Data cleaning and preprocessing
├── 2_eda.py                                   # Exploratory Data Analysis
├── 3_feature_engineering.py                   # Feature creation and transformation
├── 4_regression_modeling.py                   # Regression model training and comparison
├── 5_hyperparameter_tuning.py                 # Hyperparameter optimization
├── 6_time_series_forecasting.py               # Time series models (ARIMA, SARIMA, Prophet, LSTM)
├── 7_evaluation_report.py                     # Final evaluation and reporting
│
├── run_all.py                                 # Master script to run entire pipeline
├── requirements.txt                           # Python dependencies
├── README_WEEK2.md                            # This file
│
└── Output Files (Generated after execution):
    ├── cleaned_data.csv
    ├── engineered_data.csv
    ├── model_comparison_results.csv
    ├── tuned_models_comparison.csv
    ├── forecast_models_comparison.csv
    ├── metrics_summary_table.csv
    ├── EVALUATION_REPORT.txt
    ├── best_regression_model.pkl
    ├── best_tuned_model.pkl
    ├── feature_scaler.pkl
    └── Visualizations (*.png)
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. **Navigate to the project directory:**
```bash
cd "Week2_Assignment_Tesla EA Deliveries and Production Data(2015–2025)"
```

2. **Install required packages:**
```bash
pip install -r requirements.txt
```

### Required Libraries

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- xgboost
- lightgbm
- statsmodels
- prophet
- tensorflow
- keras

---

## 📖 Usage

### Option 1: Run Complete Pipeline (Recommended)

Execute all scripts sequentially:
```bash
python run_all.py
```

### Option 2: Run Individual Scripts

Execute scripts in order:

```bash
# Step 1: Data Preprocessing
python 1_data_preprocessing.py

# Step 2: Exploratory Data Analysis
python 2_eda.py

# Step 3: Feature Engineering
python 3_feature_engineering.py

# Step 4: Regression Modeling
python 4_regression_modeling.py

# Step 5: Hyperparameter Tuning
python 5_hyperparameter_tuning.py

# Step 6: Time Series Forecasting
python 6_time_series_forecasting.py

# Step 7: Evaluation Report
python 7_evaluation_report.py
```

---

## 🔍 Pipeline Details

### 1. Data Preprocessing (`1_data_preprocessing.py`)

**Operations:**
- Load Tesla EV dataset
- Handle missing values using grouped median/mode imputation
- Detect and cap outliers using IQR method
- Create datetime features (Date, Quarter, DayOfYear, WeekOfYear)
- Save cleaned dataset

**Output:** `cleaned_data.csv`

---

### 2. Exploratory Data Analysis (`2_eda.py`)

**Analyses:**
- **Univariate Analysis:** Distribution of key numerical variables
- **Bivariate Analysis:** Correlation matrix and scatter plots
- **Categorical Analysis:** Regional and model-wise comparisons
- **Temporal Analysis:** Time series trends and seasonality

**Outputs:**
- `1_univariate_analysis.png`
- `2_correlation_matrix.png`
- `3_bivariate_scatter_plots.png`
- `4_categorical_analysis.png`
- `5_temporal_analysis.png`

---

### 3. Feature Engineering (`3_feature_engineering.py`)

**Features Created:**
- **Lag Features:** 1, 2, 3, 6, 12-month lags for deliveries, production, and price
- **Rolling Statistics:** Mean, std, max for 3, 6, 12-month windows
- **Temporal Features:** Cyclical encoding (sine/cosine) for month and quarter
- **Interaction Features:** Price per kWh, range efficiency, delivery-production ratio
- **Aggregate Features:** Model-wise, region-wise, and yearly aggregations
- **Categorical Encoding:** Label encoding and one-hot encoding

**Output:** `engineered_data.csv` (100+ features)

---

### 4. Regression Modeling (`4_regression_modeling.py`)

**Models Evaluated:**
1. Linear Regression
2. Ridge Regression
3. Lasso Regression
4. ElasticNet
5. Decision Tree
6. Random Forest
7. Gradient Boosting
8. XGBoost
9. LightGBM
10. Support Vector Regression (SVR)

**Metrics:**
- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)
- R² Score
- MAPE (Mean Absolute Percentage Error)
- Cross-Validation R² Score

**Outputs:**
- `model_comparison_results.csv`
- `6_model_comparison.png`
- `7_best_model_predictions.png`
- `best_regression_model.pkl`

---

### 5. Hyperparameter Tuning (`5_hyperparameter_tuning.py`)

**Models Tuned:**
- Random Forest
- Gradient Boosting
- XGBoost
- LightGBM

**Method:** RandomizedSearchCV with 5-fold cross-validation

**Outputs:**
- `tuned_models_comparison.csv`
- `8_tuned_models_comparison.png`
- `best_tuned_model.pkl`
- `best_hyperparameters.txt`

---

### 6. Time Series Forecasting (`6_time_series_forecasting.py`)

**Models Implemented:**
1. **ARIMA:** Auto-regressive Integrated Moving Average
2. **SARIMA:** Seasonal ARIMA with 12-month seasonality
3. **Prophet:** Facebook's forecasting tool
4. **LSTM:** Deep learning recurrent neural network

**Forecast Horizon:** 12 months

**Outputs:**
- `forecast_models_comparison.csv`
- `9_time_series_forecasts.png`
- `10_combined_forecasts.png`

---

### 7. Evaluation Report (`7_evaluation_report.py`)

**Generates:**
- Comprehensive performance comparison across all models
- Detailed text report with findings and recommendations
- Metrics summary table
- Visual comparison of all model categories

**Outputs:**
- `EVALUATION_REPORT.txt`
- `metrics_summary_table.csv`
- `11_comprehensive_evaluation_report.png`

---

## 📊 Expected Results

### Model Performance Benchmarks

**Regression Models:**
- Best Expected R² Score: > 0.85
- Best Expected RMSE: < 2000

**Tuned Models:**
- Expected R² improvement: 2-5%
- Expected RMSE improvement: 5-10%

**Time Series Forecasting:**
- Best Expected MAPE: < 15%
- Best Expected R²: > 0.75

---

## 📈 Visualizations

The pipeline generates **11 comprehensive visualizations:**

1. Univariate analysis (distributions)
2. Correlation matrix heatmap
3. Bivariate scatter plots
4. Categorical analysis
5. Temporal trends
6. Regression model comparison
7. Best model predictions
8. Tuned models comparison
9. Time series forecasts (4 models)
10. Combined forecast plot
11. Comprehensive evaluation report

---

## 🎓 Key Learnings

1. **Data Quality:** Proper preprocessing significantly impacts model performance
2. **Feature Engineering:** Domain knowledge-based features improve predictions
3. **Model Selection:** Ensemble methods outperform linear models for complex patterns
4. **Hyperparameter Tuning:** Fine-tuning provides consistent performance gains
5. **Time Series:** Multiple approaches capture different temporal patterns

---

## 🛠️ Troubleshooting

### Common Issues

**Issue 1: Import Errors**
```bash
# Solution: Reinstall requirements
pip install --upgrade -r requirements.txt
```

**Issue 2: Memory Issues with Large Models**
```python
# Reduce n_estimators or use sampling
# Already optimized in scripts
```

**Issue 3: Prophet Installation Issues**
```bash
# Windows users may need:
pip install prophet --no-cache-dir
```

---

## 📝 Notes

- All scripts include comprehensive logging and progress tracking
- Models are saved as `.pkl` files for future use
- Visualizations are saved at 300 DPI for high quality
- Scripts handle missing data gracefully
- Random seeds are set for reproducibility (random_state=42)

---

## 🔮 Future Improvements

1. **Real-time Prediction API:** Deploy model as REST API
2. **Automated Retraining:** Schedule periodic model updates
3. **Feature Importance:** Detailed SHAP analysis
4. **Model Explainability:** LIME for prediction interpretation
5. **External Data Integration:** Market trends, competitor data
6. **Dashboard:** Interactive visualization with Plotly/Streamlit

---

## 📧 Contact

**Aman**  
Data Science Intern  
Celebal Excellence Internship Program

For questions or issues, please refer to the project documentation or contact the instructor.

---

## 📄 License

This project is part of the Celebal Excellence Internship program and is intended for educational purposes only.

---

## ✅ Checklist

- [x] Data Preprocessing
- [x] Exploratory Data Analysis
- [x] Feature Engineering
- [x] Regression Modeling (10 models)
- [x] Hyperparameter Tuning (4 models)
- [x] Time Series Forecasting (ARIMA, SARIMA, Prophet, LSTM)
- [x] Model Evaluation and Reporting
- [x] Comprehensive Documentation

---

**Last Updated:** June 17, 2026  
**Version:** 1.0  
**Status:** ✅ Complete
