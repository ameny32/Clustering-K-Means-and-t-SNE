import pandas as pd

# Load the Parquet file
df = pd.read_parquet("yellow_tripdata_2025-09.parquet")

# Convert and save as CSV
df.to_csv("yellow_tripdata_2025-09.csv", index=False)
