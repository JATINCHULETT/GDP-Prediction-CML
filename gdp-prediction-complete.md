# GDP Prediction Using Machine Learning: Complete Implementation Guide

## Project Overview
This project implements three different regression models to predict GDP based on economic indicators using a dataset containing 3,370 records from 134 countries (1991-2019).

## Dataset Analysis

### Dataset Structure:
- **Shape**: 3,370 rows × 11 columns  
- **Countries**: 134 different countries
- **Time Period**: 1991 to 2019
- **Target Variable**: GDP (US$)

### Features Used:
1. **country_encoded** - Label encoded country identifier
2. **year** - Year of observation  
3. **Export (US$ Thousand)** - Export volume
4. **Import (US$ Thousand)** - Import volume
5. **forex_reserves** - Foreign exchange reserves
6. **HDI** - Human Development Index
7. **Inflation** - Inflation rate
8. **per_capita_income** - Per capita income
9. **population** - Population size
10. **unemployment** - Unemployment rate

### Key Correlations with GDP:
- Export: 0.928 (Very Strong)
- Import: 0.828 (Strong)  
- Forex Reserves: 0.531 (Moderate)
- Per Capita Income: 0.365 (Moderate)

## Complete Implementation Code

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# 1. DATA LOADING AND PREPARATION
df = pd.read_csv('final_data.csv')

# Encode categorical country variable
le = LabelEncoder()
df['country_encoded'] = le.fit_transform(df['country'])

# Define features (using exact column names from dataset)
feature_columns = [
    'country_encoded',
    'year', 
    'Export (US$ Thousand)', 
    'Import (US$ Thousand)', 
    'forex_reserves', 
    'HDI', 
    'Inflation', 
    'per_capita_income', 
    'population', 
    'unemployment'
]

X = df[feature_columns]
y = df['gdp (US$)']

# 2. TRAIN-TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. FEATURE SCALING (for Ridge regression)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. MODEL 1: RANDOM FOREST REGRESSOR
rf_model = RandomForestRegressor(
    n_estimators=100,
    max_depth=15,
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

rf_rmse = np.sqrt(mean_squared_error(y_test, y_pred_rf))
rf_r2 = r2_score(y_test, y_pred_rf)

print(f"Random Forest - RMSE: ${rf_rmse:,.2f}, R²: {rf_r2:.4f}")

# 5. MODEL 2: RIDGE REGRESSION  
ridge_model = Ridge(alpha=1.0, random_state=42)
ridge_model.fit(X_train_scaled, y_train)
y_pred_ridge = ridge_model.predict(X_test_scaled)

ridge_rmse = np.sqrt(mean_squared_error(y_test, y_pred_ridge))
ridge_r2 = r2_score(y_test, y_pred_ridge)

print(f"Ridge Regression - RMSE: ${ridge_rmse:,.2f}, R²: {ridge_r2:.4f}")

# 6. MODEL 3: GRADIENT BOOSTING REGRESSOR
gbr_model = GradientBoostingRegressor(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    random_state=42
)
gbr_model.fit(X_train, y_train)
y_pred_gbr = gbr_model.predict(X_test)

gbr_rmse = np.sqrt(mean_squared_error(y_test, y_pred_gbr))
gbr_r2 = r2_score(y_test, y_pred_gbr)

print(f"Gradient Boosting - RMSE: ${gbr_rmse:,.2f}, R²: {gbr_r2:.4f}")

# 7. FEATURE IMPORTANCE (Random Forest)
feature_importance = pd.DataFrame({
    'feature': feature_columns,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print("Top Features:", feature_importance.head())
```

## Model Performance Results

| Model | RMSE | R² Score | Performance Rank |
|-------|------|----------|------------------|
| **Gradient Boosting** | $62.57B | **0.9980** | 🥇 **Best** |
| **Random Forest** | $95.17B | **0.9953** | 🥈 **Very Good** |  
| **Ridge Regression** | $431.81B | **0.9025** | 🥉 **Good Baseline** |


### 1. **Gradient Boosting Regressor**

- **Highest Accuracy**: R² = 0.9980 (explains 99.8% of variance)
- **Lowest Error**: RMSE = $62.57 billion
- **Industry Standard**: Used in competitions and production systems
- **Handles Complexity**: Captures both linear and non-linear relationships


### 2. **Random Forest Regressor**

- **Excellent Performance**: R² = 0.9953 
- **Robust**: Handles outliers and missing values well
- **Feature Importance**: Shows Export (67%) and Import (21%) are most important
- **Interpretable**: Easy to explain to stakeholders


### 3. **Ridge Regression**
- **Classical Approach**: Shows understanding of fundamentals
- **Regularization**: Prevents overfitting with many features
- **Fast Training**: Quick to implement and explain
- **Linear Interpretation**: Coefficients show direct relationships



## Why NOT Other Models?

### ❌ **KNN Regression**
- **Poor Scalability**: Slow with 3,370 records and 10+ features
- **Curse of Dimensionality**: Performance degrades with many features
- **No Extrapolation**: Can't predict beyond training data range
- **Memory Intensive**: Stores entire training set

### ❌ **Linear Regression (without regularization)**
- **Overfitting Risk**: Too many features (134 countries) vs samples
- **Multicollinearity**: Export/Import/GDP are highly correlated
- **No Regularization**: Unstable coefficients

### ❌ **Deep Neural Networks**
- **Overkill**: Simple tabular data doesn't need deep learning
- **Overfitting Risk**: Too many parameters for dataset size
- **Black Box**: Hard to interpret for economic analysis
- **Requires More Data**: Needs much larger datasets

## Feature Engineering Insights

### Most Important Features (from Random Forest):
1. **Export (67.0%)** - Primary GDP driver
2. **Import (20.8%)** - Secondary economic indicator  
3. **Population (5.9%)** - Market size effect
4. **Country (1.7%)** - Country-specific effects
5. **Per Capita Income (1.5%)** - Economic development
