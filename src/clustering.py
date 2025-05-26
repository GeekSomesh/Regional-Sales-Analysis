import pandas as pd
from sklearn.cluster import KMeans
import joblib

def run_clustering(data):
    df = data.groupby('Region').agg({
        'Units Sold': 'mean',
        'Holiday/Promotion': 'mean',
        'Discount': 'mean',
        'Competitor Pricing': 'mean',
        'month': 'nunique'
    }).reset_index()

    kmeans = KMeans(n_clusters=3, random_state=42)
    df['Cluster'] = kmeans.fit_predict(df.iloc[:, 1:])

    # ✅ Save output and model
    df.to_csv("outputs/cluster_output.csv", index=False)
    joblib.dump(kmeans, "models/kmeans_cluster_model.pkl")  # Save KMeans model

    return df[['Region', 'Cluster']]

def get_cluster_for_region(kmeans, df, region_name):
    region_data = df[df["Region"] == region_name].iloc[:, 1:]
    return kmeans.predict(region_data)[0]

def describe_clusters(df_with_cluster):
    return df_with_cluster.groupby("Cluster").mean()
