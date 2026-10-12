from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

PROJECT_DIR = Path(__file__).resolve().parent.parent

DATA_FILE = PROJECT_DIR / "data" / "processed" / "monthly_sales.csv"
MODEL_DIR = PROJECT_DIR / "models"
OUTPUT_DIR = PROJECT_DIR / "outputs"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 1. Load monthly data
df = pd.read_csv(DATA_FILE, parse_dates=["date"])
df = df.sort_values("date").reset_index(drop=True)

if len(df) < 18:
    raise ValueError(
        "Not enough monthly records for this example. "
        "Use more historical data or choose a simpler demonstration."
    )

if df["sales"].isna().any():
    raise ValueError("Sales still contains missing values.")

# 2. Create time-based features
def create_features(data):
    result = data.copy()

    result["time_index"] = (
        (result["date"].dt.year - df["date"].dt.year.min()) * 12
        + result["date"].dt.month
        - df["date"].dt.month.min()
    )

    month = result["date"].dt.month

    result["month_sin"] = np.sin(2 * np.pi * month / 12)
    result["month_cos"] = np.cos(2 * np.pi * month / 12)

    return result[
        ["time_index", "month_sin", "month_cos"]
    ]

X = create_features(df)
y = df["sales"]

# 3. Split chronologically: first 80% train, last 20% test
split_index = int(len(df) * 0.8)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

# 4. Train the model only on the training period
model = LinearRegression()
model.fit(X_train, y_train)

# 5. Predict sales for the test period
predictions = model.predict(X_test)

# 6. Evaluate errors
mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))

# Baseline: predict each test month using the previous month's actual sales.
# This is a simple benchmark, not another trained ML model.
baseline_predictions = y.iloc[split_index - 1:-1].to_numpy()

baseline_mae = mean_absolute_error(
    y_test, baseline_predictions
)

metrics = {
    "test_months": int(len(y_test)),
    "mae": float(mae),
    "rmse": float(rmse),
    "baseline_mae": float(baseline_mae),
}

with open(OUTPUT_DIR / "metrics.json", "w") as file:
    json.dump(metrics, file, indent=4)

# 7. Save actual versus predicted test results
results = pd.DataFrame({
    "date": df["date"].iloc[split_index:].to_numpy(),
    "actual_sales": y_test.to_numpy(),
    "predicted_sales": predictions,
})

results.to_csv(
    OUTPUT_DIR / "test_predictions.csv", index=False
)

# 8. Plot test-period predictions
plt.figure(figsize=(12, 6))
plt.plot(
    results["date"], results["actual_sales"],
    marker="o", label="Actual sales"
)
plt.plot(
    results["date"], results["predicted_sales"],
    marker="o", label="Predicted sales"
)

plt.title("Actual vs Predicted Monthly Sales")
plt.xlabel("Month")
plt.ylabel("Sales")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "model_evaluation.png", dpi=150)
plt.show()

# 9. Save the trained model
joblib.dump(model, MODEL_DIR / "sales_model.joblib")

print("\nEvaluation results:")
print(json.dumps(metrics, indent=4))

if mae < baseline_mae:
    print("The model outperformed the baseline on MAE.")
else:
    print("The model did not outperform the baseline on MAE.")

print("\nSaved model and test results.")