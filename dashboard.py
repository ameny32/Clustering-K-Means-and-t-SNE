import pandas as pd
import numpy as np
import plotly.express as px
import webbrowser, threading
from dash import Dash, dcc, html, Input, Output

# -------------------------------------------------
# Load data
# Expect CSVs in the folder "/datasets":
#   clustered_taxi_data.csv
#   taxi_zone_lookup.csv
# -------------------------------------------------
df = pd.read_csv("datasets/clustered_taxi_data.csv", low_memory=False)

# Basic feature prep
df["tip_percent"] = np.where(df["fare_amount"] > 0, df["tip_amount"] / df["fare_amount"] * 100.0, np.nan)
if "cluster" not in df.columns:
    raise RuntimeError("clustered_taxi_data.csv must include a 'cluster' column with labels.")

# Load TLC zone lookup to label PULocationID
lookup = pd.read_csv("datasets/taxi_zone_lookup.csv")
lookup = lookup.rename(columns={"LocationID": "PULocationID"})
lookup["PULocationID"] = lookup["PULocationID"].astype(int)

# Ensure merge key type alignment
if "PULocationID" in df.columns:
    df["PULocationID"] = df["PULocationID"].astype("Int64")  # allows NaN but compares as int
else:
    # If missing, create a placeholder so the bar chart still renders
    df["PULocationID"] = pd.Series([pd.NA] * len(df), dtype="Int64")

clusters = sorted(pd.unique(df["cluster"]))
default_cluster = clusters[0] if len(clusters) else 0

# -------------------------------------------------
# App
# -------------------------------------------------
app = Dash(__name__)
app.title = "NYC Taxi Clustering Dashboard"

app.layout = html.Div(
    style={"maxWidth": "1100px", "margin": "0 auto", "fontFamily": "Arial, sans-serif"},
    children=[
        html.H2("NYC Taxi Clustering Dashboard"),

        dcc.Dropdown(
            id="cluster-dropdown",
            options=[{"label": f"Cluster {int(k)}", "value": int(k)} for k in clusters],
            value=int(default_cluster),
            clearable=False,
            style={"width": "300px", "marginBottom": "16px"}
        ),

        dcc.Graph(id="pickup-map"),

        dcc.Graph(id="distance-hist"),

        html.Div(id="summary-box", style={"fontSize": 18, "marginTop": 12, "marginBottom": 12}),

        dcc.Graph(id="fare-tip-scatter"),
    ],
)

# -------------------------------------------------
# Callback
# -------------------------------------------------
@app.callback(
    Output("pickup-map", "figure"),
    Output("distance-hist", "figure"),
    Output("summary-box", "children"),
    Output("fare-tip-scatter", "figure"),
    Input("cluster-dropdown", "value"),
)
def update_dashboard(selected_cluster: int):
    subset = df[df["cluster"] == selected_cluster].copy()

    # ---------- Top pickup zones bar (acts as "map panel" without lat/lon) ----------
    # Count trips by PULocationID
    if subset["PULocationID"].notna().any():
        subset["PULocationID"] = subset["PULocationID"].astype(int)
        top = (
            subset.groupby("PULocationID")
            .size()
            .reset_index(name="count")
            .sort_values("count", ascending=False)
            .head(20)
        )
        top = top.merge(lookup[["PULocationID", "Borough", "Zone"]], on="PULocationID", how="left")
        top[["Borough", "Zone"]] = top[["Borough", "Zone"]].fillna("Unknown")
        top["label"] = top["PULocationID"].astype(str) + " • " + top["Borough"] + " • " + top["Zone"]

        map_fig = px.bar(
            top,
            x="label",
            y="count",
            title=f"Top Pickup Zones for Cluster {selected_cluster}",
        )
        map_fig.update_layout(
            xaxis_title="Pickup Zone",
            yaxis_title="Trips",
            xaxis_tickangle=45,
            margin=dict(l=20, r=20, t=60, b=80),
        )
    else:
        # Fallback empty figure
        map_fig = px.bar(title=f"Top Pickup Zones for Cluster {selected_cluster}")
        map_fig.update_layout(
            xaxis_title="Pickup Zone",
            yaxis_title="Trips",
            annotations=[dict(text="No PULocationID in data", x=0.5, y=0.5, showarrow=False)],
        )

    # ---------- Trip distance histogram ----------
    if subset.shape[0] > 0 and "trip_distance" in subset.columns:
        hist_fig = px.histogram(
            subset,
            x="trip_distance",
            nbins=40,
            title="Trip Distance Distribution",
        )
        hist_fig.update_layout(xaxis_title="Trip Distance (miles)", yaxis_title="Count")
    else:
        hist_fig = px.histogram(title="Trip Distance Distribution")
        hist_fig.update_layout(
            annotations=[dict(text="No data for selected cluster", x=0.5, y=0.5, showarrow=False)]
        )

    # ---------- Summary box ----------
    if subset.shape[0] > 0:
        avg_fare = subset["fare_amount"].mean()
        avg_tip_pct = subset["tip_percent"].mean()
        avg_pass = subset["passenger_count"].mean()
        summary_text = f"Avg Fare: ${avg_fare:.2f} | Avg Tip %: {avg_tip_pct:.1f}% | Avg Passengers: {avg_pass:.2f}"
    else:
        summary_text = "No trips in this cluster."

    # ---------- Fare vs. tip scatter (color-coded by cluster) ----------
    if df.shape[0] > 0:
        scatter_fig = px.scatter(
            df,
            x="fare_amount",
            y="tip_amount",
            color=df["cluster"].astype(str),
            opacity=0.6,
            title="Fare vs Tip by Cluster",
        )
        scatter_fig.update_layout(xaxis_title="Fare ($)", yaxis_title="Tip ($)", legend_title="Cluster")
    else:
        scatter_fig = px.scatter(title="Fare vs Tip by Cluster")
        scatter_fig.update_layout(
            annotations=[dict(text="No data available", x=0.5, y=0.5, showarrow=False)]
        )

    return map_fig, hist_fig, summary_text, scatter_fig


if __name__ == "__main__":
    threading.Timer(1.0, lambda: webbrowser.open("http://127.0.0.1:8050/")).start()
    app.run(debug=False, port=8050)