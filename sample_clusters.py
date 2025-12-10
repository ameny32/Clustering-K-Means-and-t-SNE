"""
Runs K-Means on all rows, then t-SNE on a sampled subset for visualization.
Saves plot to sample_tsne_clusters.png and prints per-cluster means.
"""

import argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="datasets/yellow_tripdata_2025-01.csv")
    parser.add_argument("--clusters", type=int, default=8)
    parser.add_argument("--sample_n", type=int, default=100_000)        # t-SNE is slow, so we sample a subset      EDIT THIS TO CHANGE THE NUMBER OF SAMPLES
    parser.add_argument("--perplexity", type=float, default=50.0)
    parser.add_argument("--out", default="img/sample_clusters.png")
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

    n = min(args.sample_n, X_scaled.shape[0])
    # t-SNE requirement: perplexity < n/3
    perp = min(args.perplexity, max(5.0, (n - 1) / 3.0))

    emb = TSNE(n_components=2, perplexity=perp, random_state=0).fit_transform(X_scaled[:n])
    y_sample = df["cluster"].to_numpy()[:n]

    cmap = plt.cm.get_cmap("tab20", args.clusters)
    colors = [cmap(i) for i in range(args.clusters)]

    plt.figure(figsize=(9, 7))
    for i in range(args.clusters):
        mask = (y_sample == i)
        if np.any(mask):
            plt.scatter(emb[mask, 0], emb[mask, 1], s=10, color=colors[i], label=f"Cluster {i}")
    plt.title("NYC Taxi: t-SNE colored by K-Means cluster (sample)")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.legend(markerscale=2, fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    plt.close()

    print(df.groupby("cluster")[cols].mean().round(2))

if __name__ == "__main__":
    main()
