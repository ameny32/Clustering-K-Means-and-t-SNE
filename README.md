Python 3.11.9

NYC Taxi and Limousine Commission (TLC)
TLC Trip Record Data: https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page

Run code files in this order:

    parquet_to_csv.py - converts .parquet source file to .csv

    evaluate_k-means.py - runs the Elbow method, determines best k, and saves datasets/clustered_taxi_data.csv

    sample_clusters.py (t-SNE) OR full_clusters.py (PCA) - visualizes the clusters and prints summary stats

    analyze_clusters.ipynb - deeper cluster analysis or testing inside Jupyter

    charts.ipynb - produces visual summary plots for presentation

    dashboard.py - launches the interactive Dash web dashboard for exploration