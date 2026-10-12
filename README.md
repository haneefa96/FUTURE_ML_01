# Sales Forecasting Using Machine Learning

## Project Overview

This project uses historical retail sales data to analyze sales patterns and forecast future monthly sales using machine learning.

It demonstrates an end-to-end workflow involving data cleaning, exploratory data analysis, feature engineering, regression modeling, model evaluation, and visualization.

## Objectives

- Clean and aggregate historical sales data.
- Identify monthly sales trends and seasonality.
- Train a regression model for sales forecasting.
- Compare model performance with a simple baseline.
- Generate future sales predictions and visualizations.
- Explain the results and their business applications.

## Dataset

**Dataset:** Superstore Sales Dataset  
**Source:** [Kaggle — Superstore Sales Dataset](https://www.kaggle.com/datasets/vivek468/superstore-dataset-final)

The project uses transaction dates and sales amounts to calculate monthly sales totals.

- Data period: January 2014 – December 2017
- Monthly records: 48
- Invalid or missing date/sales rows removed: 0

## Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Joblib
- VS Code

## Project Structure

```text
sales-forecasting/
├── data/
│   ├── raw/
│   │   └── superstore.csv
│   └── processed/
│       └── monthly_sales.csv
├── models/
│   └── sales_model.joblib
├── outputs/
│   ├── future_forecast.csv
│   ├── future_forecast.png
│   ├── historical_sales.png
│   ├── metrics.json
│   ├── model_evaluation.png
│   ├── monthly_seasonality.png
│   └── test_predictions.csv
├── src/
│   ├── prepare_data.py
│   ├── explore_data.py
│   ├── train_model.py
│   └── forecast.py
├── requirements.txt
├── README.md
└── .gitignore
```

The virtual environment is excluded from the documented project structure because it should be recreated on another computer rather than committed to Git.

## Methodology

### 1. Data Preparation

The preparation script loads the raw dataset, validates dates and sales values, aggregates transactions by month, and saves the processed dataset to `data/processed/monthly_sales.csv`.

### 2. Exploratory Data Analysis

The analysis examines historical sales and monthly patterns.

The observed highest average sales month was **November**, while the lowest average sales month was **February**.

### 3. Model Training

A regression model is trained using historical monthly data and time-based features. The final 10 months are held out for evaluation to preserve chronological order.

### 4. Model Evaluation

The trained model is compared against a baseline that predicts the next month's sales using the previous month's actual sales.

### 5. Forecasting

The saved model generates forecasts for January, February, and March 2018. These predictions extend beyond the historical dataset, which ends in December 2017.

## Evaluation Results

| Metric | Result |
|---|---:|
| Test period | 10 months |
| Model MAE | 13,311.66 |
| Model RMSE | 18,763.90 |
| Baseline MAE | 21,308.09 |
| MAE reduction versus baseline | Approximately 37.5% |

**Mean Absolute Error (MAE)** measures the average absolute difference between actual and predicted sales. Lower values indicate smaller errors.

**Root Mean Squared Error (RMSE)** penalizes larger errors more heavily.

The model achieved approximately 37.5% lower MAE than the baseline on the held-out test period. This is an encouraging initial result, but further validation is needed before relying on the model for real business decisions.

## Future Forecasts

| Month | Forecast sales |
|---|---:|
| January 2018 | 59,265.60 |
| February 2018 | 52,086.26 |
| March 2018 | 46,591.09 |

These are model-generated estimates, not observed sales. The currency follows the original dataset.

## Generated Outputs

The `outputs/` directory contains:

- `historical_sales.png` — historical sales visualization.
- `monthly_seasonality.png` — average sales by month.
- `model_evaluation.png` — model evaluation visualization.
- `future_forecast.png` — future forecast visualization.
- `metrics.json` — evaluation metrics.
- `test_predictions.csv` — predictions for the test period.
- `future_forecast.csv` — forecast values for three future months.

## Setup and Usage

### Prerequisites

- Python installed on your computer.
- Git, if cloning the project.
- VS Code recommended for development.

### 1. Create and activate a virtual environment

Run these commands from the project root in PowerShell:

```powershell
python -m venv .venv64
.\.venv64\Scripts\Activate.ps1
```

### 2. Install dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Prepare the data

Ensure the raw dataset is available at `data/raw/superstore.csv`.

```powershell
python src/prepare_data.py
```

### 4. Explore the data

```powershell
python src/explore_data.py
```

### 5. Train and evaluate the model

```powershell
python src/train_model.py
```

### 6. Generate future forecasts

```powershell
python src/forecast.py
```

Run commands from the project root so the documented relative paths work correctly. If the scripts are run from another directory, their path handling may need adjustment.

## Business Applications

Sales forecasting can support:

- Inventory and replenishment planning.
- Demand estimation.
- Resource allocation.
- Sales and budget planning.
- Identification of seasonal sales patterns.

Forecasts should be considered alongside promotions, holidays, pricing changes, and other business factors.

## Limitations

- The dataset contains only 48 monthly observations.
- The model uses a limited set of features.
- External influences such as promotions and holidays are not explicitly modeled.
- Evaluation covers only 10 test months.
- The forecasts refer to 2018, not the present day.
- Better test results do not guarantee future performance.

## Future Improvements

- Compare additional regression and time-series models.
- Add lagged sales and seasonal features.
- Use rolling time-series validation.
- Forecast sales by product or category.
- Add prediction intervals.
- Build an interactive forecasting dashboard.

## Conclusion

This project demonstrates a complete introductory sales forecasting workflow, from data preparation to prediction and evaluation. The initial model outperformed a previous-month baseline on MAE, providing a foundation for further experimentation and improvement.

## Author

Name: Haneefa Riham
GitHub profile: haneefa96
