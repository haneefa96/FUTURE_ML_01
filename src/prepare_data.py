from pathlib import Path
import pandas as pd

PROJECT_DIR = Path.cwd()
if not (PROJECT_DIR / "data" / "raw" / "superstore.csv").exists():
    PROJECT_DIR = PROJECT_DIR.parent
RAW_FILE = PROJECT_DIR / "data" / "raw" / "superstore.csv"
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# 1. Load the original data
df = pd.read_csv(RAW_FILE, encoding="cp1250")

# Normalize column names for easier matching
df.columns = df.columns.str.strip()

# Find the required columns without assuming capitalization
column_lookup = {
    column.lower(): column for column in df.columns
}

date_column = column_lookup.get("order date")
sales_column = column_lookup.get("sales")

if date_column is None or sales_column is None:
    raise ValueError(
        f"Required columns not found. Available: {df.columns.tolist()}"
    )

# 2. Convert columns into appropriate data types
df[date_column] = pd.to_datetime(
    df[date_column], errors="coerce"
)

df[sales_column] = pd.to_numeric(
    df[sales_column], errors="coerce"
)

# 3. Remove rows with unusable dates or sales
before = len(df)

df = df.dropna(subset=[date_column, sales_column]).copy()

print("Rows removed due to missing/invalid date or sales:",
      before - len(df))

# 4. Sort transactions chronologically
df = df.sort_values(date_column)

# 5. Aggregate transactions into monthly sales
monthly_sales = (
    df.set_index(date_column)[sales_column]
    .resample("MS")
    .sum()
    .rename("sales")
    .reset_index()
)

monthly_sales = monthly_sales.rename(
    columns={date_column: "date"}
)

# 6. Save the processed data
output_file = PROCESSED_DIR / "monthly_sales.csv"

monthly_sales.to_csv(output_file, index=False)

print("\nFirst monthly records:")
print(monthly_sales.head())

print("\nTotal months:", len(monthly_sales))
print("Date range:",
      monthly_sales["date"].min(),
      "to",
      monthly_sales["date"].max())

print("\nSaved to:", output_file)