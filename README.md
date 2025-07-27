### Sales-Forecasting

### Overview
This project focuses on **predicting future sales** based on historical data using a variety of **time series forecasting** and **machine learning** techniques. It involves crafting time-based features, handling temporal validation properly, and experimenting with advanced models like **Random Forest**, **XGBoost**, and **Exponential Smoothing**.

### Dataset
- **Source**: [Kaggle - Walmart Sales Forecasting](https://www.kaggle.com/datasets/aslanahmedov/walmart-sales-forecast)
- **Files Used**:
  - `train.csv`: Weekly sales data by store and department
  - `features.csv`: Additional data like temperature, fuel price, etc.
  - `stores.csv`: Store type and size information

### Objectives
1. Merge, clean, and engineer features from the dataset.
2. Create time-based and lag features (e.g. `Month`, `Lag_1`, `Lag_52`, etc.)
3. Apply regression models like **Random Forest** and **XGBoost**.
4. Use **TimeSeriesSplit** for time-aware cross-validation.
5. Evaluate using **Weighted Mean Absolute Error (WMAE)**.
6. Forecast using **Exponential Smoothing** for one store-department pair.
7. Visualize actual vs predicted sales.

### Tools & Libraries
- Python
- pandas, numpy
- matplotlib, seaborn
- scikit-learn
- xgboost
- statsmodels (for time series)

### Key Steps & Results
## Feature Engineering
- Extracted **time-based features** from the `Date` column:
  - `Year`, `Month`, `Week`, `Day of Week`
- Created **lag features**:
  - `Lag_1`: Sales from previous week
  - `Lag_52`: Sales from the same week last year
- Created **rolling features**:
  - `Rolling_Mean_4`: 4-week moving average
- Categorical encoding:
  - Converted `IsHoliday` to integer
  - One-hot encoded `Type` column (`Type_B`, `Type_C`)
- Handled missing values using `.fillna(0)`

## Time-Aware Cross-Validation (5 Folds)
Used `TimeSeriesSplit` from scikit-learn to ensure validation respects the time-order of data:
- **Fold 1**:
  -  WMAE: `2703.55`
- **Fold 2**:
  -  WMAE: `2156.53`
- **Fold 3**:
  -  WMAE: `2297.09`
- **Fold 4**:
  -  WMAE: `1582.88`
- **Fold 5**:
  -  WMAE: `1456.64`
- ** Average WMAE across folds**: `2039.34`

## XGBoost
Applied `XGBRegressor` as an advanced boosting model:
- Model parameters: `n_estimators=100`, `learning_rate=0.1`, `random_state=42`
- Achieved **WMAE**: `1498.23`
- Performance was better than Random Forest on the final fold

## Exponential Smoothing (Classical Time Series)
Used Statsmodels' `ExponentialSmoothing` for forecasting:
- Applied to **Store 1, Dept 1**
- Seasonal method with **additive trend** and **seasonal_periods = 52**
- Forecasted **12 future weeks**
- Plotted results: actual vs forecast sales with smoothed trend

## Actual vs Predicted (Random Forest)
Visualized the Random Forest model's prediction accuracy:
- Compared **first 100** predicted values to actual sales
- Used line plot:
  - Blue = Actual
  - Orange = Predicted
- Helped to visually inspect model accuracy and trend behavior
