import pandas as pd
from src.forecasting import run_forecasting
from src.classification import run_classification
from src.clustering import run_clustering

def main():
    df = pd.read_csv("data/retail_warehouse_inventory_dataset.csv")
    print("Running Forecasting...")
    rmse = run_forecasting(df)
    print(f"Forecasting RMSE: {rmse:.2f}")

    print("\nRunning Classification...")
    classification_output = run_classification(df)
    print("Classification complete. Report saved.")

    print("\nRunning Clustering...")
    clustering_output = run_clustering(df)
    print("Clustering complete. Results saved.")

if __name__ == "__main__":
    main()
