import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import numpy as np
from sklearn.metrics import mean_squared_error

def run_classification(data):
    # 🔧 Aggregate features per region and month
    df = data.groupby(['Region', 'month']).agg({
        'Units Sold': 'sum',
        'Discount': 'mean',
        'Holiday/Promotion': 'mean'
    }).reset_index()

    # 🔧 Label encoding based on quantiles
    df['Performance'] = pd.qcut(df['Units Sold'], 3, labels=['Low', 'Medium', 'High'])

    # 🔧 Feature selection
    X = df[['Region', 'month', 'Discount', 'Holiday/Promotion']]
    y = df['Performance']
    X['Region'] = X['Region'].astype('category')

    # 🔧 Split and train
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = CatBoostClassifier(verbose=0)
    model.fit(X_train, y_train, cat_features=[0])  # Only 'Region' is categorical

    # 🔧 Predict and save results
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred)

    with open("outputs/classification_report.txt", "w") as f:
        f.write(report)

    joblib.dump(model, "models/classifier_model.cbm")
    return report


def predict_performance(model, input_df):
    input_df = input_df.copy()
    input_df["Region"] = input_df["Region"].astype("category")
    return model.predict(input_df)


# Calculate RMSE
