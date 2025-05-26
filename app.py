import streamlit as st
import pandas as pd
import joblib
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from scipy.stats import norm
import numpy as np
from scipy.stats import ttest_ind

# Load models
@st.cache_resource
def load_models():
    regressor = joblib.load("models/regressor_model.cbm")
    classifier = joblib.load("models/classifier_model.cbm")
    clusterer = joblib.load("models/kmeans_cluster_model.pkl")
    return regressor, classifier, clusterer

regressor_model, classifier_model, cluster_model = load_models()

# Load data
@st.cache_data
def load_data():
    return pd.read_csv("data/retail_warehouse_inventory_dataset.csv")

df = load_data()

# Helper to detect season
def detect_season(month):
    if month in [12, 1, 2]: return "Winter"
    elif month in [6, 7, 8]: return "Rainy"
    else: return "Normal"

st.title("📊 Regional Sales Intelligence Dashboard")

model_choice = st.selectbox("Select ML Model", [
    "Regional Sales Forecasting Model",
    "Region-wise Performance Classification Model",
    "Regional Demand Clustering Model"
])

# ------------------------------------------
# Forecasting Model
# ------------------------------------------
if model_choice == "Regional Sales Forecasting Model":
    st.header("🔮 Forecasting Questions")

    ques = st.selectbox("Choose a question", [
        "What will be the sales for a specific product in a specific region on a future date?",
        "Will sales increase or decrease next week/month in a region?",
        "What is the expected revenue in a region over a future period?",
        "Which product will have the highest sales next month in Region X?",
        "Which regions are expected to hit stock-out due to high demand?"
    ])

    if ques == "What will be the sales for a specific product in a specific region on a future date?":
        region = st.selectbox("Select Region", df["Region"].unique())
        product = st.selectbox("Select Product", df["Product ID"].unique())
        date = st.date_input("Select Future Date")
        if st.button("Predict"):
            base = df[(df["Region"] == region) & (df["Product ID"] == product)].iloc[-1]

            np.random.seed(date.day)

            input_row = pd.DataFrame([{
                "Region": region,
                "Product ID": product,
                "Category": base["Category"],
                "Seasonality": detect_season(date.month),
                "Price": base["Price"],
                "Discount": np.clip(base["Discount"] + np.random.uniform(-5, 5), 0, 100),
                "Holiday/Promotion": int(date.weekday() in [5, 6]),
                "Competitor Pricing": np.clip(base["Competitor Pricing"] + np.random.uniform(-10, 10), 0, 1000),
                "month": date.month,
                "quarter": (date.month - 1) // 3 + 1
            }])

            input_row[["Region", "Product ID", "Category", "Seasonality"]] = input_row[["Region", "Product ID", "Category", "Seasonality"]].astype("category")
            prediction = regressor_model.predict(input_row)[0]
            st.success(f"Expected Sales: {prediction:.2f} units")

    elif ques == "Will sales increase or decrease next week/month in a region?":
        region = st.selectbox("Select Region", df["Region"].unique())
        choice = st.selectbox("Time Window", ["7", "30", "45"])
        if st.button("Predict"):
            today = datetime.today()
            current = today
            future = today + timedelta(days=int(choice))

            def get_prediction(region, date):
                row = df[df["Region"] == region].iloc[-1]
                np.random.seed(date.day)
                input_data = pd.DataFrame([{
                    "Region": region,
                    "Product ID": row["Product ID"],
                    "Category": row["Category"],
                    "Seasonality": detect_season(date.month),
                    "Price": row["Price"],
                    "Discount": np.clip(row["Discount"] + np.random.uniform(-5, 5), 0, 100),
                    "Holiday/Promotion": int(date.weekday() in [5, 6]),
                    "Competitor Pricing": np.clip(row["Competitor Pricing"] + np.random.uniform(-10, 10), 0, 1000),
                    "month": date.month,
                    "quarter": (date.month - 1) // 3 + 1
                }])
                input_data[["Region", "Product ID", "Category", "Seasonality"]] = input_data[["Region", "Product ID", "Category", "Seasonality"]].astype("category")
                return regressor_model.predict(input_data)[0]

            cur_sales = get_prediction(region, current)
            fut_sales = get_prediction(region, future)
            st.info(f"Current Sales: {cur_sales:.2f}, Future Sales: {fut_sales:.2f}")
            if fut_sales > cur_sales:
                st.success("📈 Sales will likely increase.")
            else:
                st.warning("📉 Sales will likely decrease.")

    elif ques == "What is the expected revenue in a region over a future period?":
        region = st.selectbox("Select Region", df["Region"].unique())
        date = st.date_input("Select Date for Forecast")
        if st.button("Predict"):
            sub_df = df[df["Region"] == region].drop_duplicates("Product ID")
            forecast_input = []

            for _, row in sub_df.iterrows():
                np.random.seed(date.day)
                forecast_input.append({
                    "Region": region,
                    "Product ID": row["Product ID"],
                    "Category": row["Category"],
                    "Seasonality": detect_season(date.month),
                    "Price": row["Price"],
                    "Discount": np.clip(row["Discount"] + np.random.uniform(-5, 5), 0, 100),
                    "Holiday/Promotion": int(date.weekday() in [5, 6]),
                    "Competitor Pricing": np.clip(row["Competitor Pricing"] + np.random.uniform(-10, 10), 0, 1000),
                    "month": date.month,
                    "quarter": (date.month - 1) // 3 + 1
                })

            input_df = pd.DataFrame(forecast_input)
            input_df[["Region", "Product ID", "Category", "Seasonality"]] = input_df[["Region", "Product ID", "Category", "Seasonality"]].astype("category")
            input_df["Predicted Units"] = regressor_model.predict(input_df)
            input_df["Revenue"] = input_df["Predicted Units"] * input_df["Price"]
            st.success(f"Estimated Revenue: ₹{input_df['Revenue'].sum():.2f}")

            st.bar_chart(input_df.set_index("Product ID")["Revenue"])

    elif ques == "Which product will have the highest sales next month in Region X?":
        region = st.selectbox("Select Region", df["Region"].unique())
        if st.button("Predict"):
            date = datetime.today() + timedelta(days=30)
            sub_df = df[df["Region"] == region].drop_duplicates("Product ID")
            predictions = []

            for _, row in sub_df.iterrows():
                np.random.seed(date.day)
                row_data = pd.DataFrame([{
                    "Region": region,
                    "Product ID": row["Product ID"],
                    "Category": row["Category"],
                    "Seasonality": detect_season(date.month),
                    "Price": row["Price"],
                    "Discount": np.clip(row["Discount"] + np.random.uniform(-5, 5), 0, 100),
                    "Holiday/Promotion": int(date.weekday() in [5, 6]),
                    "Competitor Pricing": np.clip(row["Competitor Pricing"] + np.random.uniform(-10, 10), 0, 1000),
                    "month": date.month,
                    "quarter": (date.month - 1) // 3 + 1
                }])
                row_data[["Region", "Product ID", "Category", "Seasonality"]] = row_data[["Region", "Product ID", "Category", "Seasonality"]].astype("category")
                pred_units = regressor_model.predict(row_data)[0]
                predictions.append((row["Product ID"], pred_units))

            predictions.sort(key=lambda x: x[1], reverse=True)
            st.success(f"🏆 Top Product: {predictions[0][0]} with {predictions[0][1]:.2f} units")

            chart_df = pd.DataFrame(predictions[:10], columns=["Product ID", "Predicted Units"])
            st.bar_chart(chart_df.set_index("Product ID"))

    elif ques == "Which regions are expected to hit stock-out due to high demand?":
        start = st.date_input("From Date")
        end = st.date_input("To Date")
        if st.button("Predict"):
            all_regions = df["Region"].unique()
            alerts = []

            for region in all_regions:
                sub_df = df[df["Region"] == region].drop_duplicates("Product ID")
                for _, row in sub_df.iterrows():
                    for dt in pd.date_range(start, end, freq='D'):
                        np.random.seed(dt.day + hash(row["Product ID"]) % 1000)
                        input_data = pd.DataFrame([{
                            "Region": region,
                            "Product ID": row["Product ID"],
                            "Category": row["Category"],
                            "Seasonality": detect_season(dt.month),
                            "Price": row["Price"],
                            "Discount": np.clip(row["Discount"] + np.random.uniform(-5, 5), 0, 100),
                            "Holiday/Promotion": int(dt.weekday() in [5, 6]),
                            "Competitor Pricing": np.clip(row["Competitor Pricing"] + np.random.uniform(-10, 10), 0, 1000),
                            "month": dt.month,
                            "quarter": (dt.month - 1) // 3 + 1
                        }])
                        input_data[["Region", "Product ID", "Category", "Seasonality"]] = input_data[["Region", "Product ID", "Category", "Seasonality"]].astype("category")
                        units = regressor_model.predict(input_data)[0]
                        if units > 300:
                            alerts.append((region, row["Product ID"], dt.date(), units))

            if alerts:
                alert_df = pd.DataFrame(alerts, columns=["Region", "Product", "Date", "Predicted Units"])
                st.warning("Some regions might run out of stock soon due to high demand:")
                st.dataframe(alert_df)
            else:
                st.success("No stock-out risks detected in given range.")


# ------------------------------------------
# Classification Model
# ------------------------------------------
if model_choice == "Region-wise Performance Classification Model":
    st.header("📊 Classification Questions")

    ques = st.selectbox("Choose a question", [
        "Which region is more likely to experience high sales in custom days?",
        "What is the probability of a promotion leading to increased sales in Region X?",
        "Is a sales boost expected in a region if a discount is applied next month?",
        "Which regions perform better during promotions?",
        "How likely is Region Y to exceed a sales threshold next holiday season?"
    ])

    if ques == "Which region is more likely to experience high sales in custom days?":
        custom_days = st.slider("Select number of future days", 1, 60, 30)
        if st.button("Predict"):
            next_month = (datetime.today().month % 12) + 1
            all_regions = df["Region"].unique()
            results = []

            for region in all_regions:
                row = df[df["Region"] == region].iloc[-1]
                input_df = pd.DataFrame([{
                    "Region": region,
                    "month": next_month,
                    "Discount": row["Discount"],
                    "Holiday/Promotion": row["Holiday/Promotion"]
                }])
                input_df[["Region"]] = input_df[["Region"]].astype("category")
                prob = classifier_model.predict_proba(input_df)[0][1]
                adjusted_prob = prob * (custom_days / 30)
                results.append((region, min(adjusted_prob, 1.0)))

            sorted_res = sorted(results, key=lambda x: x[1], reverse=True)
            st.success(f"🏆 Region most likely to see high sales in {custom_days} days: {sorted_res[0][0]} ({sorted_res[0][1]*100:.2f}% probability)")

            chart_df = pd.DataFrame(sorted_res, columns=["Region", "Probability"])
            st.bar_chart(chart_df.set_index("Region"))

    elif ques == "What is the probability of a promotion leading to increased sales in Region X?":
        region = st.selectbox("Select Region", df["Region"].unique())
        if st.button("Predict Probability"):
            month = datetime.today().month + 1
            row = df[df["Region"] == region].iloc[-1]

            input_df = pd.DataFrame([{
                "Region": region,
                "month": month,
                "Discount": row["Discount"],
                "Holiday/Promotion": 1
            }])
            input_df[["Region"]] = input_df[["Region"]].astype("category")
            result = classifier_model.predict_proba(input_df)[0][1]

            # Determine average discount that increases probability further (dummy simulation)
            discounts = list(range(0, 51))
            probs = []
            for d in discounts:
                test_df = input_df.copy()
                test_df["Discount"] = d
                prob = classifier_model.predict_proba(test_df)[0][1]
                probs.append(prob)
            max_prob = max(probs)
            best_discount = discounts[probs.index(max_prob)]

            st.info(f"Chance of increased sales in {region} if promotion is applied: {result*100:.2f}%")
            st.success(f"📌 Recommended Discount to Maximize Impact: {best_discount}% (Expected Probability: {max_prob*100:.2f}%)")

    elif ques == "Is a sales boost expected in a region if a discount is applied next month?":
        region = st.selectbox("Select Region", df["Region"].unique())
        discount = st.slider("Set Discount (%)", 0, 50, 10)
        if st.button("Predict"):
            row = df[df["Region"] == region].iloc[-1]

            input_df = pd.DataFrame([{
                "Region": region,
                "month": datetime.today().month + 1,
                "Discount": discount,
                "Holiday/Promotion": row["Holiday/Promotion"]
            }])
            input_df[["Region"]] = input_df[["Region"]].astype("category")
            result = classifier_model.predict_proba(input_df)[0][1]
            st.info(f"With {discount}% discount, the chance of high sales in {region} is {result*100:.2f}%")

    elif ques == "Which regions perform better during promotions?":
        st.subheader("📊 Advanced Promotional Impact Analysis (Region × Category × Quarter)")
        results = []
        categories = df["Category"].unique()
        regions = df["Region"].unique()
        for region in regions:
            for category in categories:
                sub_df = df[(df["Region"] == region) & (df["Category"] == category)]
                if sub_df.empty or sub_df["Holiday/Promotion"].nunique() < 2:
                    continue
                # Split data by promotion flag
                promo_sales = sub_df[sub_df["Holiday/Promotion"] == 1]["Units Sold"]
                no_promo_sales = sub_df[sub_df["Holiday/Promotion"] == 0]["Units Sold"]
                # Basic statistics
                avg_promo = promo_sales.mean()
                avg_no_promo = no_promo_sales.mean()
                # Perform Welch's t-test (handles unequal variance)
                t_stat, p_val = ttest_ind(promo_sales, no_promo_sales, equal_var=False)
                results.append({
                    "Region": region,
                    "Category": category,
                    "Avg Sales (With Promo)": round(avg_promo, 2),
                    "Avg Sales (Without Promo)": round(avg_no_promo, 2),
                    "Difference": round(avg_promo - avg_no_promo, 2),
                    "P-Value": round(p_val, 4),
                    "Significant": "✅ Yes" if p_val < 0.05 else "❌ No"
                })
        result_df = pd.DataFrame(results)
        if not result_df.empty:
            # 📋 Show results table
            st.info("ℹ️ No statistically significant promo boosts found.")
            st.dataframe(result_df.sort_values("Difference", ascending=False))
            # 📈 Add line chart (Promo vs No Promo)
            line_df = result_df[["Region", "Category", "Avg Sales (With Promo)", "Avg Sales (Without Promo)"]]
            line_df = pd.melt(line_df,
                              id_vars=["Region", "Category"],
                              value_vars=["Avg Sales (With Promo)", "Avg Sales (Without Promo)"],
                              var_name="Promotion Type",
                              value_name="Average Sales")
            line_df["Region_Category"] = line_df["Region"] + " - " + line_df["Category"]
            pivot_df = line_df.pivot(index="Region_Category", columns="Promotion Type", values="Average Sales")
            st.write("### 📈 Promo vs No Promo Sales (All Combinations)")
            st.line_chart(pivot_df)
            # ✅ Significant bar chart
            st.info("Statistically Significant Improvements Only")
            sig_df = result_df[(result_df["Difference"] > 0) & (result_df["Significant"] == "✅ Yes")]
            if not sig_df.empty:
                st.bar_chart(sig_df.set_index(["Region", "Category"])[["Difference"]])
        else:
            st.warning("⚠️ Not enough data to analyze promotion impact.")

    elif ques == "How likely is Region Y to exceed a sales threshold next holiday season?":
        st.subheader("📈 Sales Threshold Probability (Using Regression + Normal Distribution)")
        region = st.selectbox("Select Region", df["Region"].unique())
        threshold = st.number_input("Set Sales Threshold", min_value=1, max_value=1000, value=100)
        if st.button("Predict Probability"):
            row = df[df["Region"] == region].iloc[-1]

            input_df = pd.DataFrame([{
                "Region": region,
                "Product ID": row["Product ID"],
                "Category": row["Category"],
                "Seasonality": "Peak",
                "Price": row["Price"],
                "Discount": row["Discount"],
                "Holiday/Promotion": 1,
                "Competitor Pricing": row["Competitor Pricing"],
                "month": 12,
                "quarter": 4
            }])

            for col in ["Region", "Product ID", "Category", "Seasonality"]:
                input_df[col] = input_df[col].astype("category")

            mean_sales = regressor_model.predict(input_df)[0]
            std_dev = 450
            prob = 1 - norm.cdf(threshold, loc=mean_sales, scale=std_dev)

            st.write(f"🔍 Expected Sales: `{mean_sales:.2f}` units")
            st.success(f"📈 Probability that **sales exceed {threshold}** in {region} during holidays: **{prob * 100:.2f}%**")



    elif ques == "How likely is Region Y to exceed a sales threshold next holiday season?":
        st.subheader("📈 Sales Threshold Probability (Using Regression + Normal Distribution)")
        region = st.selectbox("Select Region", df["Region"].unique())
        threshold = st.number_input("Set Sales Threshold", min_value=1, max_value=1000, value=100)
        # Use most recent row for the region
        row = df[df["Region"] == region].iloc[-1]
        input_df = pd.DataFrame([{
            "Region": region,
            "Product ID": row["Product ID"],  # Add if regressor_model uses this
            "Category": row["Category"],
            "Seasonality": "Peak",  # or dynamic: detect_season(datetime.today().month)
            "Price": row["Price"],
            "Discount": row["Discount"],
            "Holiday/Promotion": 1,
            "Competitor Pricing": row["Competitor Pricing"],
            "month": 12,  # Assuming December for holiday
            "quarter": 4
        }])
        # Ensure categorical fields are typed correctly
        for col in ["Region", "Product ID", "Category", "Seasonality"]:
            input_df[col] = input_df[col].astype("category")
        # Predict mean sales
        mean_sales = regressor_model.predict(input_df)[0]
        # Use estimated std dev from your model evaluation (e.g., RMSE)
        std_dev =450  # You can make this dynamic later based on model RMSE
        # Estimate probability that sales exceed the threshold
        prob = 1 - norm.cdf(threshold, loc=mean_sales, scale=std_dev)
        st.write(f"🔍 Expected Sales: `{mean_sales:.2f}` units")
        st.success(f"📈 Probability that **sales exceed {threshold}** in {region} during holidays: **{prob * 100:.2f}%**")

# Clustering Model
elif model_choice == "Regional Demand Clustering Model":
    st.header("🧠 Regional Demand Clustering")
    st.subheader("🔍 How Clustering Works")
    st.markdown("""
    Clustering groups regions based on similar demand patterns such as **Units Sold**, **Discounts**, **Promotions**, and **Competitor Pricing**.
    The model assigns each region to a cluster. Let's explore how these clusters look:
    """)
    question = st.selectbox("Select a Question", [
        "Which regions have similar demand patterns (clustered)?",
        "What cluster does Region X belong to?",
        "Which cluster of regions tends to order more in selected seasons?",
        "Which cluster needs frequent restocking?",
        "Which cluster represents high-margin, high-frequency regions?"])

    # Ensure consistency with training features
    expected_features = ['Units Sold', 'Holiday/Promotion','Discount' , 'Competitor Pricing','month']
    missing = [col for col in expected_features if col not in df.columns]

    if missing:
        st.error(f"❌ Missing columns required for clustering: {missing}")
    else:
        features = df.groupby('Region')[expected_features].mean()
        try:
            features['Cluster'] = cluster_model.predict(features)
        except ValueError as e:
            st.error(f"⚠️ Model prediction failed: {e}")
        else:
            if question == "Which regions have similar demand patterns (clustered)?":
                st.dataframe(features.reset_index())
                region1 = st.selectbox("Select Region 1", features.index, key="region1")
                region2 = st.selectbox("Select Region 2", features.index, key="region2")
                if st.button("Predict"):
                    vec1 = features.loc[region1][expected_features].values
                    vec2 = features.loc[region2][expected_features].values
                    similarity = 100 - np.linalg.norm(vec1 - vec2) / np.linalg.norm(vec1 + vec2) * 100
                    st.markdown(f"### 🧪 Similarity between {region1} and {region2}: {similarity:.2f}%")
                    unsimilar=100-similarity
                    if similarity > 99.99:
                        st.success("✅ These regions show similar demand patterns.")
                    else:
                        st.warning(f"⚠️ These regions differ by {unsimilar:.4f}%. See graph below:")
                        chart_df = pd.DataFrame({region1: vec1, region2: vec2}, index=expected_features)
                        st.bar_chart(chart_df)
                        import matplotlib.pyplot as plt
                        import seaborn as sns

                        diff = features.loc[[region1, region2], expected_features[:-1]]  # exclude 'Competitor Pricing'
                        fig, ax = plt.subplots()
                        sns.heatmap(diff.T, annot=True, cmap="coolwarm", ax=ax)
                        ax.set_title("🔍 Feature Comparison")
                        st.pyplot(fig)


            elif question == "What cluster does Region X belong to?":
                region = st.selectbox("Select Region", features.index)
                if st.button("Get Cluster Info"):
                    cluster = features.loc[region, 'Cluster']
                    st.success(f"📌 Region {region} belongs to Cluster {cluster}")

            elif question == "Which cluster of regions tends to order more in selected seasons?":
                if 'Seasonality' not in df.columns:
                    st.error("❌ 'Seasonality' column is missing from the dataset.")
                else:
                    season1 = st.selectbox("Select Season 1", df['Seasonality'].unique(), key="s1")
                    season2 = st.selectbox("Select Season 2", df['Seasonality'].unique(), key="s2")
                    if st.button("Predict the Clusters"):
                        seasonal_df = df[df['Seasonality'].isin([season1, season2])]
                        result = seasonal_df.groupby('Region')['Units Sold'].mean()
                        cluster_info = pd.DataFrame({"Units Sold": result}).join(features['Cluster'])
                        grouped = cluster_info.groupby('Cluster')['Units Sold'].mean().reset_index()
                        st.dataframe(grouped)
                        st.bar_chart(grouped.set_index('Cluster'))

            elif question == "Which cluster needs frequent restocking?":
                st.info("💡 Clusters with higher average units sold likely require more frequent restocking.")
                restock = df.groupby('Region')['Units Sold'].sum()
                restock_df = pd.DataFrame({"Units Sold": restock}).join(features['Cluster'])
                high_demand = restock_df.groupby('Cluster')['Units Sold'].agg(['mean', 'max', 'min', 'std']).sort_values(by='mean', ascending=False)
                st.dataframe(high_demand.reset_index())
                st.bar_chart(high_demand['mean'])


            elif question == "Which cluster represents high-margin, high-frequency regions?":
                margin_features = ['Units Sold', 'Price', 'Discount', 'Holiday/Promotion', 'Competitor Pricing','month']
                if any(col not in df.columns for col in margin_features):
                    st.error("❌ Some required columns are missing from the dataset.")
                else:
                    margin = df.groupby('Region')[margin_features].mean()
                    margin['Revenue'] = margin['Units Sold'] * margin['Competitor Pricing']
                    margin_cluster_input = margin[expected_features]  # ensure all required features present

                    try:
                        st.success("📈 Clusters with high revenue and high frequency are ideal targets for growth.")
                        margin['Cluster'] = cluster_model.predict(margin_cluster_input)
                        top = margin.groupby('Cluster')[['Revenue', 'Units Sold']].agg(['mean', 'sum'])
                        st.dataframe(top.reset_index())

                        import matplotlib.pyplot as plt
                        import seaborn as sns

                        fig, ax = plt.subplots()
                        sns.scatterplot(data=margin, x='Revenue', y='Units Sold', hue='Cluster', palette='deep', ax=ax)
                        st.pyplot(fig)

                    except ValueError as e:
                        st.error(f"⚠️ Model prediction failed: {e}")

