from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_DIR = Path(__file__).resolve().parent.parent

DATA_FILE = (
    PROJECT_DIR / "data" / "processed" / "monthly_sales.csv"
)

OUTPUT_DIR = PROJECT_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_FILE, parse_dates=["date"])

# Historical sales trend
plt.figure(figsize=(12, 6))
plt.plot(df["date"], df["sales"], marker="o")

plt.title("Historical Monthly Sales")
plt.xlabel("Month")
plt.ylabel("Sales")
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(OUTPUT_DIR / "historical_sales.png", dpi=150)
plt.show()

# Average sales by calendar month
df["month"] = df["date"].dt.month

average_monthly_sales = (
    df.groupby("month")["sales"].mean()
)

plt.figure(figsize=(10, 5))
average_monthly_sales.plot(kind="bar")

plt.title("Average Sales by Calendar Month")
plt.xlabel("Calendar Month")
plt.ylabel("Average Sales")
plt.tight_layout()

plt.savefig(OUTPUT_DIR / "monthly_seasonality.png", dpi=150)
plt.show()

print("Highest average sales month:",
      average_monthly_sales.idxmax())

print("Lowest average sales month:",
      average_monthly_sales.idxmin())