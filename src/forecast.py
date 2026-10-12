from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_DIR = Path(__file__).resolve().parent.parent

DATA_FILE = PROJECT_DIR / "data" / "processed" / "monthly_sales.csv"
MODEL_FILE = PROJECT_DIR / "models" / "sales_model.joblib"
OUTPUT_DIR = PROJECT_DIR / "outputs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 1. Load historical data and trained model
df = pd.read_csv(DATA_FILE, parse_dates=["date"])
model = joblib.load(MODEL_FILE)

df = df.sort_values("date").reset_index(drop=True)

# 2. Generate the next three calendar months
last_date = df["date"].max()

future_dates = pd.date_range(
    start=last_date + pd.offsets.MonthBegin(1),
    periods=3,
    freq="MS",
)

future = pd.DataFrame({"date": future_dates})

# 3. Create the same features used during training
start_year = df["date"].dt.year.min()
start_month = df["date"].dt.month.min()

future["time_index"] = (
    (future["date"].dt.year - start_year) * 12
    + future["date"].dt.month
    - start_month
)

month = future["date"].dt.month

future["month_sin"] = np.sin(2 * np.pi * month / 12)
future["month_cos"] = np.cos(2 * np.pi * month / 12)

features = [
    "time_index",
    "month_sin",
    "month_cos",
]

# 4. Predict future sales
future["forecast_sales"] = model.predict(future[features])

# 5. Save predictions
output_file = OUTPUT_DIR / "future_forecast.csv"

future[["date", "forecast_sales"]].to_csv(
    output_file, index=False
)

# 6. Plot historical sales and future forecasts
plt.figure(figsize=(12, 6))

plt.plot(
    df["date"],
    df["sales"],
    marker="o",
    label="Historical sales",
)

plt.plot(
    future["date"],
    future["forecast_sales"],
    marker="o",
    linestyle="--",
    label="Forecast sales",
)

plt.axvline(
    df["date"].max(),
    linestyle=":",
    label="Forecast begins",
)

plt.title("Historical Sales and Three-Month Forecast")
plt.xlabel("Month")
plt.ylabel("Sales")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(OUTPUT_DIR / "future_forecast.png", dpi=150)
plt.show()

print("\nFuture sales forecast:")
print(future[["date", "forecast_sales"]].to_string(index=False))

print("\nForecast saved to:", output_file)