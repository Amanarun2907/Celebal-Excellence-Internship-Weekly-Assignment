# 🚀 Quick Start Guide - Week 2 Assignment

## ⚡ Get Started in 3 Steps

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Run the Pipeline
```bash
python run_all.py
```

### Step 3: View Results
Check these files:
- `EVALUATION_REPORT.txt` - Detailed performance report
- `*.png` - 11 visualization files
- `*.csv` - Model comparison results

---

## 📊 What Gets Generated?

### 🔢 Data Files
- `cleaned_data.csv` - Preprocessed dataset
- `engineered_data.csv` - Dataset with 100+ features

### 🤖 Model Files
- `best_regression_model.pkl` - Best regression model
- `best_tuned_model.pkl` - Best tuned model
- `feature_scaler.pkl` - Feature scaler

### 📈 Visualizations (11 total)
1. Univariate analysis
2. Correlation matrix
3. Bivariate scatter plots
4. Categorical analysis
5. Temporal trends
6. Model comparison
7. Best model predictions
8. Tuned models comparison
9. Time series forecasts
10. Combined forecasts
11. Comprehensive evaluation

---

## ⏱️ Execution Time

**Total:** ~15-30 minutes (depending on your system)

Individual steps:
1. Data Preprocessing: ~1 min
2. EDA: ~2-3 min
3. Feature Engineering: ~2 min
4. Regression Modeling: ~3-5 min
5. Hyperparameter Tuning: ~5-10 min ⏰
6. Time Series Forecasting: ~3-5 min
7. Evaluation Report: ~1 min

---

## 🎯 Key Metrics to Look For

### Regression Models
- **R² Score:** > 0.85 (target)
- **RMSE:** < 2000 (target)
- **Best Models:** XGBoost, LightGBM, Random Forest

### Time Series Forecasts
- **MAPE:** < 15% (target)
- **Best Models:** SARIMA, Prophet, LSTM

---

## 🔧 Run Individual Scripts

If you want to run steps separately:

```bash
# Step by step execution
python 1_data_preprocessing.py
python 2_eda.py
python 3_feature_engineering.py
python 4_regression_modeling.py
python 5_hyperparameter_tuning.py
python 6_time_series_forecasting.py
python 7_evaluation_report.py
```

---

## 📝 Important Notes

- ✅ Scripts run sequentially (each depends on previous)
- ✅ Progress is logged to console
- ✅ All outputs saved automatically
- ✅ Safe to re-run (overwrites previous results)
- ✅ Models saved for future use

---

## 🐛 Troubleshooting

### Prophet installation issues (Windows)?
```bash
pip install pystan
pip install prophet
```

### TensorFlow issues?
```bash
pip install tensorflow==2.15.0
```

### Out of memory?
- Close other applications
- Reduce dataset size in scripts (optional)

---

## 📧 Need Help?

Check these files:
1. `README_WEEK2.md` - Full documentation
2. `EVALUATION_REPORT.txt` - Results and insights
3. Individual script docstrings

---

## ✅ Success Indicators

You'll know it worked when you see:
- ✓ All 7 scripts complete without errors
- ✓ 11 PNG visualization files generated
- ✓ `EVALUATION_REPORT.txt` created
- ✓ Console shows "PIPELINE COMPLETED SUCCESSFULLY!"

---

**Ready?** Run `python run_all.py` and let the magic happen! 🎉
