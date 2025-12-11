Rutgers CS439 - Intro to Data Science
Final Project - Clustering: K-Means & t-SNE
Andrew Menyhert (amm926), Rida Mohammad (rm1724), Kaushik Murali (km1526)

Python 3.11.9

NYC Taxi and Limousine Commission (TLC)
TLC Trip Record Data: https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page

This project analyzes clustering patterns within the New York City Yellow Taxi Trip Records for January 2025. 
We combine K-Means clustering with t-SNE dimensionality reduction to uncover natural structure in ride behavior using key features such as trip distance, fare amount, tip amount, pickup and drop-off zones, and temporal attributes. 
After cleaning the dataset and generating engineered features, we applied the elbow method to select an optimal cluster count of k = 8. 
The clusters reveal interpretable groups including airport rides, Manhattan locals, micro-trips, and high-passenger trips. 
We validate structure using t-SNE, compare feature impacts, and build an interactive Plotly Dash dashboard for exploratory analysis. 
Our results show that NYC taxi trips form meaningful clusters influenced by spatial density, passenger behavior, and pricing patterns.


Run code files in this order:

    parquet_to_csv.py - converts .parquet source file to .csv

    evaluate_k-means.py - runs the Elbow method, determines best k, and saves datasets/clustered_taxi_data.csv

    sample_clusters.py (t-SNE) OR full_clusters.py (PCA) - visualizes the clusters and prints summary stats

    analyze_clusters.ipynb - deeper cluster analysis or testing inside Jupyter

    charts.ipynb - produces visual summary plots for presentation

    dashboard.py - launches the interactive Dash web dashboard for exploration