# Week 2 Assignment - Tesla EV Deliveries Analysis

**Author:** Aman  
**Program:** Celebal Excellence Internship - Data Science  
**Week:** 2 - Classical Machine Learning

---

## 📁 Folder Structure

```
Week2_Assignment_Tesla EA Deliveries and Production Data(2015–2025)/
│
├── week2_aman.ipynb                        # ⭐ MAIN ASSIGNMENT NOTEBOOK
├── tesla_deliveries_dataset_2015_2025.csv  # Dataset
│
└── End_to_End_ML_Pipeline_Scripts/         # Additional ML Pipeline Scripts
    ├── 1_data_preprocessing.py
    ├── 2_eda.py
    ├── 3_feature_engineering.py
    ├── 4_regression_modeling.py
    ├── 5_hyperparameter_tuning.py
    ├── 6_time_series_forecasting.py
    ├── 7_evaluation_report.py
    ├── run_all.py
    ├── requirements.txt
    └── Documentation files (.md)
```

---

## 🎯 Main Assignment

### File: `week2_aman.ipynb`

This Jupyter Notebook contains the complete Week 2 assignment with all 10 requirements fulfilled:

✅ **1. Data Loading** - Shape, columns, .info(), .describe()  
✅ **2. Data Quality** - Missing values and duplicate check  
✅ **3. EDA Charts** - 5 visualizations with titles and labels  
✅ **4. Feature Engineering** - Label encoding, lag features, rolling mean  
✅ **5. Linear Regression** - Chronological split, MAE, RMSE, R², plot  
✅ **6. Cross-Validation** - 5-Fold CV with per-fold and mean R²  
✅ **7. Random Forest** - GridSearchCV with feature importances  
✅ **8. ADF Test** - Stationarity test with interpretation  
✅ **9. Forecast Table** - First 20 test records with error %  
✅ **10. All 12 Markdown Cells** - Complete explanations  

---

## 🚀 How to Run

### Option 1: Run Main Assignment (Jupyter Notebook)

```bash
# Open Jupyter Notebook
jupyter notebook week2_aman.ipynb

# Or use Jupyter Lab
jupyter lab week2_aman.ipynb
```

### Option 2: Run End-to-End ML Pipeline (Python Scripts)

```bash
cd End_to_End_ML_Pipeline_Scripts
pip install -r requirements.txt
python run_all.py
```

---

## 📊 Expected Results

### Main Assignment (week2_aman.ipynb):
- Linear Regression: R² > 0.95
- Random Forest: R² ≥ 0.98
- 5 EDA visualizations
- Feature importance chart
- Forecast comparison table
- Model comparison metrics

### Additional Pipeline:
- 11 comprehensive visualizations
- Multiple model comparisons
- Time series forecasting (ARIMA, SARIMA, Prophet, LSTM)
- Detailed evaluation report

---

## 📝 Key Requirements Met

1. ✅ Dataset loads with 12 columns
2. ✅ All 5 EDA charts with titles and axis labels
3. ✅ Correlation heatmap shows Production_Units ↔ Estimated_Deliveries ≥ 0.9
4. ✅ Lag and Rolling Mean features have no NaN values
5. ✅ Chronological split used (80/20)
6. ✅ Linear Regression R² above 0.95
7. ✅ Cross Validation mean R² printed with std dev
8. ✅ GridSearchCV best params printed, Random Forest R² ≥ 0.98
9. ✅ Feature importance chart shows Production_Units or Deliveries_Lag1 highest
10. ✅ ADF test conclusion based on p-value < 0.05
11. ✅ Model comparison table with both models
12. ✅ All 12 explanation cells filled
13. ✅ File named week2_aman.ipynb

---

## 🎓 Assignment Completion

**Status:** ✅ **100% COMPLETE**

- All 10 instructor requirements fulfilled
- All 12 markdown explanation cells completed
- Zero errors from top to bottom
- Ready for evaluation

---

**Date:** June 17, 2026  
**Celebal Excellence Internship - Data Science**
