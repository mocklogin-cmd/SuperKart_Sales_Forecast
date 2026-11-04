import pandas as pd

RAW_PATH = "superkart_project/data/SuperKart.csv"

# load the raw dataset from the data folder of the repo
df = pd.read_csv(RAW_PATH)

# columns we expect to find before we treat the dataset as registered
expected_columns = [
    "Product_Id", "Product_Weight", "Product_Sugar_Content",
    "Product_Allocated_Area", "Product_Type", "Product_MRP",
    "Store_Id", "Store_Establishment_Year", "Store_Size",
    "Store_Location_City_Type", "Store_Type", "Product_Store_Sales_Total",
]

missing = [c for c in expected_columns if c not in df.columns]
if missing:
    raise ValueError(f"Dataset is missing expected columns: {missing}")

print("Dataset registered successfully.")
print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
print("Columns:", list(df.columns))
print("\nMissing values per column:")
print(df.isnull().sum())
print("\nSummary of the target (Product_Store_Sales_Total):")
print(df["Product_Store_Sales_Total"].describe())
