import pandas as pd
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt
import joblib

def run_forecasting(data):
    features = ['Region', 'Product ID', 'Category', 'Seasonality', 'Price', 'Discount',
                'Holiday/Promotion', 'Competitor Pricing', 'month', 'quarter']
    X = data[features].copy()
    y = data['Units Sold']

    X[["Region", "Product ID", "Category", "Seasonality"]] = X[["Region", "Product ID", "Category", "Seasonality"]].astype('category')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = CatBoostRegressor(verbose=0)
    model.fit(X_train, y_train, cat_features=[0, 1, 2, 3])
    y_pred = model.predict(X_test)

    rmse = sqrt(mean_squared_error(y_test, y_pred))
    pd.DataFrame({'Actual': y_test.values, 'Predicted': y_pred}).to_csv("outputs/forecast_results.csv", index=False)
    joblib.dump(model, "models/regressor_model.cbm")

    return rmse


def predict_sales(model, input_df):
    input_df = input_df.copy()
    input_df[["Region", "Product ID", "Category", "Seasonality"]] = input_df[["Region", "Product ID", "Category", "Seasonality"]].astype('category')
    return model.predict(input_df)


def forecast_all(df, model, date_features):
    combinations = df[["Region", "Product ID", "Category", "Seasonality"]].drop_duplicates()
    forecast_input = []

    for _, row in combinations.iterrows():
        row_data = {
            "Region": row["Region"],
            "Product ID": row["Product ID"],
            "Category": row["Category"],
            "Seasonality": row["Seasonality"],
            "Price": df[(df["Region"] == row["Region"]) & (df["Product ID"] == row["Product ID"])]["Price"].mean(),
            "Discount": df[(df["Region"] == row["Region"]) & (df["Product ID"] == row["Product ID"])]["Discount"].mean(),
            "Holiday/Promotion": df[(df["Region"] == row["Region"]) & (df["Product ID"] == row["Product ID"])]["Holiday/Promotion"].mean(),
            "Competitor Pricing": df[(df["Region"] == row["Region"]) & (df["Product ID"] == row["Product ID"])]["Competitor Pricing"].mean(),
            "month": date_features["month"],
            "quarter": date_features["quarter"]
        }
        forecast_input.append(row_data)

    forecast_df = pd.DataFrame(forecast_input)
    forecast_df[["Region", "Product ID", "Category", "Seasonality"]] = forecast_df[["Region", "Product ID", "Category", "Seasonality"]].astype('category')
    forecast_df["Prediction"] = model.predict(forecast_df)

    return forecast_df
