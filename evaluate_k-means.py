"""
Evaluate K-Means with Elbow method on NYC Yellow Taxi data.
Outputs:
  - elbow_k_vs_inertia.png
  - clustered_taxi_data.csv  (original rows that had complete features + 'cluster')
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# -----------------------------
# Config
# -----------------------------
DATA_PATH = r"datasets/yellow_tripdata_2025-01.csv"
RANDOM_STATE = 0
K_MIN, K_MAX = 2, 12           # search range for k
USE_COLUMNS = [
    "passenger_count", "trip_distance",
    "fare_amount", "tip_amount", "total_amount",
    "PULocationID", "DOLocationID",
    "pickup_hour", "pickup_weekday"
]

# -----------------------------
# Load and prepare data
# -----------------------------
print("Loading data...")
df = pd.read_csv(DATA_PATH, low_memory=False)

# time features
df["tpep_pickup_datetime"] = pd.to_datetime(df["tpep_pickup_datetime"], errors="coerce")
df = df.dropna(subset=["tpep_pickup_datetime"])
df["pickup_hour"] = df["tpep_pickup_datetime"].dt.hour
df["pickup_weekday"] = df["tpep_pickup_datetime"].dt.weekday

# filter obvious invalids
df = df[(df["trip_distance"] > 0) & (df["fare_amount"] > 0)]

# feature matrix
X = df[USE_COLUMNS].dropna().copy()

# keep only rows that survive dropna for later merge-back
df_model = df.loc[X.index].copy()

# standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"Modeling rows: {X_scaled.shape[0]:,}")

# -----------------------------
# Elbow method
# -----------------------------
print("Computing Elbow curve (inertia vs k)...")
k_values = list(range(K_MIN, K_MAX + 1))
inertias = []

for k in k_values:
    km = KMeans(n_clusters=k, n_init="auto", random_state=RANDOM_STATE)
    km.fit(X_scaled)
    inertias.append(km.inertia_)
    print(f"k={k:2d}  inertia={km.inertia_:,.0f}")

plt.figure()
plt.plot(k_values, inertias, marker="o")
plt.title("Elbow Method: k vs Inertia")
plt.xlabel("k")
plt.ylabel("Inertia (WCSS)")
plt.grid(True, linestyle="--", alpha=0.5)
elbow_png = os.path.join("img/elbow_k_vs_inertia.png")
plt.savefig(elbow_png, bbox_inches="tight", dpi=150)
plt.close()
print(f"Saved: {elbow_png}")

# -----------------------------
# Choose k and fit final model
# -----------------------------
# Heuristic: manually choose based on elbow plot
best_k = 8  # replace with chosen k from Elbow visualization
print(f"Selected k (from elbow): {best_k}")

final_kmeans = KMeans(n_clusters=best_k, n_init="auto", random_state=RANDOM_STATE)
final_labels = final_kmeans.fit_predict(X_scaled)

# attach cluster labels back to original subset
df_model = df_model.copy()
df_model["cluster"] = final_labels

# tip_percent for dashboard summary
df_model["tip_percent"] = df_model["tip_amount"] / df_model["fare_amount"] * 100.0

out_csv = os.path.join("datasets/clustered_taxi_data.csv")
df_model.to_csv(out_csv, index=False)
print(f"Wrote: {out_csv}")

print("Done.")
