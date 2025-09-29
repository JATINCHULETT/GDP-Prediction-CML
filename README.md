# GDP Prediction Using Machine Learning (`GDP-Prediction-CML`)

A machine learning project that predicts national **GDP (US$)** and classifies economies into GDP tiers (**Low**, **Medium**, **High**) using macroeconomic and development indicators across **134 countries** from **1991 to 2019**.

---

## Project Structure

- [`final_data (1).csv`](final_data%20(1).csv) — Macroeconomic dataset (3,370 records × 11 columns, 0 missing values)
- [`Regression & Classification.ipynb`](Regression%20&%20Classification.ipynb) — Linear Regression (with One-Hot Encoding & Standard Scaling) and GDP tier classification using Logistic Regression & KNN
- [`RandomForest.ipynb`](RandomForest.ipynb) — Random Forest Regressor training and full regression metric evaluation
- [`GradientBoosting and RandomizedSearchCV.ipynb`](GradientBoosting%20and%20RandomizedSearchCV.ipynb) — Random Forest & Gradient Boosting hyperparameter tuning with `RandomizedSearchCV`
- [`gdp-prediction-complete.md`](gdp-prediction-complete.md) — Complete implementation guide, Ridge baseline comparison, and feature engineering notes

---

## Dataset Overview

- **Records:** 3,370 rows × 11 columns (no missing values)
- **Countries:** 134 countries
- **Time Period:** 1991 – 2019
- **Target Variable:** `gdp (US$)`

### Input Features

| Feature | Description |
|---|---|
| `country` | Country name (`LabelEncoder` for tree models; `OneHotEncoder` in linear/KNN pipelines) |
| `year` | Observation year (1991–2019) |
| `Export (US$ Thousand)` | Total export volume |
| `Import (US$ Thousand)` | Total import volume |
| `forex_reserves` | Foreign exchange reserves |
| `HDI` | Human Development Index |
| `Inflation` | Annual inflation rate (%) |
| `per_capita_income` | Per capita income |
| `population` | Total population |
| `unemployment` | Unemployment rate (%) |

### Key Correlations with GDP

- **Export:** `0.928` (Very Strong)
- **Import:** `0.828` (Strong)
- **Forex Reserves:** `0.531` (Moderate)
- **Per Capita Income:** `0.365` (Moderate)

---

## Model Results

All regression and classification experiments use an **80/20 train-test split** (`random_state=42`, test set = 674 samples).

### 1. Regression Models (Predicting `gdp (US$)`)

| Model | Source | R² Score | Adjusted R² | MAE (US$) | RMSE (US$) | MAPE |
|---|---|---|---|---|---|---|
| **Gradient Boosting Regressor** (`n_estimators=200, max_depth=7, max_leaf_nodes=40`) | `GradientBoosting and RandomizedSearchCV.ipynb` | **0.9980** (99.80%) | **0.9980** | **$21.42B** (`$21,424,951,927.55`) | **$61.23B** (`$61,225,987,822.45`) | 17.01% |
| **Gradient Boosting Regressor** (`n_estimators=100, max_depth=6, lr=0.1`) | `gdp-prediction-complete.md` | 0.9980 | — | — | $62.57B | — |
| **Random Forest Regressor** (`n_estimators=100, max_depth=15`) | `RandomForest.ipynb` & `GradientBoosting and RandomizedSearchCV.ipynb` | **0.9953** (99.53%) | **0.9952** | **$27.04B** (`$27,035,314,513.74`) | **$95.17B** (`$95,171,960,500.79`) | **16.92%** |
| **Random Forest Regressor** (`n_estimators=25, max_depth=15, max_leaf_nodes=80`) | `GradientBoosting and RandomizedSearchCV.ipynb` | 0.9949 (`0.99486`) | — | — | — | — |
| **Linear Regression** (`OneHotEncoder` + `StandardScaler`) | `Regression & Classification.ipynb` | 0.9868 (`0.98682`) | — | $68.81B (`$68,808,728,638.04`) | $158.79B (`$158,786,052,423.81`) | — |
| **Ridge Regression** (`alpha=1.0`, `StandardScaler`) | `gdp-prediction-complete.md` | 0.9025 | — | — | $431.81B | — |

> **Hyperparameter Tuning (`RandomizedSearchCV`, 3-fold CV, `scoring='r2'`):**
> - Random Forest best CV estimator: `RandomForestRegressor(max_depth=20, max_leaf_nodes=80, n_estimators=25)`
> - Gradient Boosting best CV estimator: `GradientBoostingRegressor(max_depth=4, max_leaf_nodes=20, n_estimators=200)`

### 2. Classification Models (Predicting GDP Tiers: Low / Medium / High)

GDP was binned into 3 equal-frequency classes using `pd.qcut(df['gdp (US$)'], 3, labels=['Low', 'Medium', 'High'])` (test support = 674: High = 239, Low = 225, Medium = 210).

#### K-Nearest Neighbors (`n_neighbors=5`) — **96% Accuracy**

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| **High** | 0.98 | 0.99 | 0.99 | 239 |
| **Low** | 0.98 | 0.94 | 0.96 | 225 |
| **Medium** | 0.93 | 0.96 | 0.94 | 210 |
| **Overall / Weighted Avg** | **0.96** | **0.96** | **0.96** | **674** |

#### Logistic Regression (`max_iter=500`) — **95% Accuracy**

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| **High** | 0.99 | 0.95 | 0.97 | 239 |
| **Low** | 0.97 | 0.94 | 0.96 | 225 |
| **Medium** | 0.89 | 0.96 | 0.92 | 210 |
| **Overall / Weighted Avg** | **0.95** | **0.95** | **0.95** | **674** |

---

## Feature Importance (Random Forest)

From the Random Forest model analysis in [`gdp-prediction-complete.md`](gdp-prediction-complete.md):

1. **Export (`Export (US$ Thousand)`)** — **67.0%**
2. **Import (`Import (US$ Thousand)`)** — **20.8%**
3. **Population (`population`)** — **5.9%**
4. **Country (`country_encoded`)** — **1.7%**
5. **Per Capita Income (`per_capita_income`)** — **1.5%**

---

## Key Takeaways

- **Gradient Boosting Regressor** achieved the strongest regression performance (`R² = 0.9980`, `RMSE = $61.23B`, `MAE = $21.42B`).
- **Random Forest Regressor** delivered strong accuracy (`R² = 0.9953`, lowest `MAPE = 16.92%`) along with clear feature importance rankings showing trade volume (exports + imports) accounting for nearly 88% of predictive importance.
- **KNN (`n_neighbors=5`)** outperformed Logistic Regression slightly on 3-tier GDP classification (**96%** vs **95%** accuracy).

---

## Running the Notebooks

```bash
pip install pandas numpy scikit-learn gdown
```

Note: The notebooks download `final_data.csv` via `gdown`; locally, the dataset is included in this repository as `final_data (1).csv`.
