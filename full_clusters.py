"""
Runs K-Means and PCA(2D) on all rows for a full-dataset visualization.
Saves plot to full_pca_clusters.png and prints per-cluster means.
Use this when t-SNE on full data is infeasible.
"""

import argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="datasets/yellow_tripdata_2025-01.csv")
    parser.add_argument("--clusters", type=int, default=8)
    parser.add_argument("--out", default="img/full_clusters.png")
    args = parser.parse_args()

    df = pd.read_csv(args.csv, low_memory=False)

    df["tpep_pickup_datetime"] = pd.to_datetime(df["tpep_pickup_datetime"], errors="coerce")
    df = df.dropna(subset=["tpep_pickup_datetime"])
    df["pickup_hour"] = df["tpep_pickup_datetime"].dt.hour
    df["pickup_weekday"] = df["tpep_pickup_datetime"].dt.weekday

    cols = [
        "passenger_count","trip_distance","fare_amount","tip_amount",
        "total_amount","PULocationID","DOLocationID","pickup_hour","pickup_weekday"
    ]
    X = df[cols].dropna()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=args.clusters, n_init="auto", random_state=0)
    labels = kmeans.fit_predict(X_scaled)

    df = df.loc[X.index].copy()
    df["cluster"] = labels

    emb = PCA(n_components=2, random_state=0).fit_transform(X_scaled)
    y = df["cluster"].to_numpy()

    cmap = plt.cm.get_cmap("tab20", args.clusters)
    colors = [cmap(i) for i in range(args.clusters)]

    plt.figure(figsize=(9, 7))
    for i in range(args.clusters):
        mask = (y == i)
        if np.any(mask):
            plt.scatter(emb[mask, 0], emb[mask, 1], s=5, color=colors[i], label=f"Cluster {i}")
    plt.title("NYC Taxi: PCA colored by K-Means cluster (full dataset)")
    plt.xlabel("PC 1")
    plt.ylabel("PC 2")
    plt.legend(markerscale=3, fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    plt.close()

    print(df.groupby("cluster")[cols].mean().round(2))

if __name__ == "__main__":
    main()
