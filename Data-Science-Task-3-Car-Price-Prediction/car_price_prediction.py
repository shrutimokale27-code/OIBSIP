import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Load Dataset
df = pd.read_csv("car_data.csv")

print("First 5 Rows:")
print(df.head())

print("\nDataset Shape:", df.shape)

# 2. Data Cleaning
print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:", df.duplicated().sum())

df = df.drop_duplicates()
df = df.dropna()

# 3. Feature Engineering - Extract Car Brand
df["Brand"] = df["CarName"].str.split().str[0]

print("\nCar Brands:")
print(df["Brand"].unique())

# 4. Remove unnecessary columns
df = df.drop(columns=["car_ID", "CarName"])

# 5. Separate Features and Target
X = df.drop(columns=["price"])
y = df["price"]

# 6. Identify categorical and numerical columns
categorical_cols = X.select_dtypes(include=["object"]).columns
numerical_cols = X.select_dtypes(exclude=["object"]).columns

# 7. Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ("num", "passthrough", numerical_cols)
    ]
)

# 8. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

# 9. Linear Regression Model
linear_model = Pipeline([
    ("preprocessor", preprocessor),
    ("model", LinearRegression())
])

linear_model.fit(X_train, y_train)
linear_pred = linear_model.predict(X_test)

# 10. Random Forest Model
random_forest = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestRegressor(
        n_estimators=100,
        random_state=42
    ))
])

random_forest.fit(X_train, y_train)
rf_pred = random_forest.predict(X_test)

# 11. Model Evaluation
def evaluate_model(name, actual, predicted):
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    r2 = r2_score(actual, predicted)

    print(f"\n{name}")
    print("MAE :", round(mae, 2))
    print("RMSE:", round(rmse, 2))
    print("R2 Score:", round(r2, 2))

    return r2

linear_r2 = evaluate_model(
    "Linear Regression", y_test, linear_pred
)

rf_r2 = evaluate_model(
    "Random Forest Regression", y_test, rf_pred
)

# 12. Best Model
if rf_r2 > linear_r2:
    print("\nBest Model: Random Forest Regression")
else:
    print("\nBest Model: Linear Regression")

# 13. Actual vs Predicted Price
plt.figure(figsize=(8, 5))
plt.scatter(y_test, rf_pred)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Car Prices")
plt.tight_layout()
plt.show()

# 14. Correlation Heatmap
numeric_df = df.select_dtypes(include=np.number)

plt.figure(figsize=(12, 8))
sns.heatmap(numeric_df.corr(), annot=False)
plt.title("Car Price Correlation Heatmap")
plt.tight_layout()
plt.show()
# 15. Feature Importance
rf_model = random_forest.named_steps["model"]
feature_names = random_forest.named_steps["preprocessor"].get_feature_names_out()

importance = pd.Series(
    rf_model.feature_importances_,
    index=feature_names
).sort_values(ascending=False)

print("\nTop 10 Important Features:")
print(importance.head(10))

plt.figure(figsize=(10, 6))
importance.head(10).sort_values().plot(kind="barh")
plt.xlabel("Importance")
plt.title("Top 10 Features Affecting Car Price")
plt.tight_layout()
# 15. Feature Importance
rf_model = random_forest.named_steps["model"]
feature_names = random_forest.named_steps["preprocessor"].get_feature_names_out()

importance = pd.Series(
    rf_model.feature_importances_,
    index=feature_names
).sort_values(ascending=False)

print("\nTop 10 Important Features:")
print(importance.head(10))

plt.figure(figsize=(10, 6))
importance.head(10).sort_values().plot(kind="barh")
plt.xlabel("Importance")
plt.title("Top 10 Features Affecting Car Price")
plt.tight_layout()
# 15. Feature Importance
rf_model = random_forest.named_steps["model"]
feature_names = random_forest.named_steps["preprocessor"].get_feature_names_out()

importance = pd.Series(
    rf_model.feature_importances_,
    index=feature_names
).sort_values(ascending=False)

print("\nTop 10 Important Features:")
print(importance.head(10))

plt.figure(figsize=(10, 6))
importance.head(10).sort_values().plot(kind="barh")
plt.xlabel("Importance")
plt.title("Top 10 Features Affecting Car Price")
plt.tight_layout()
# 15. Feature Importance
rf_model = random_forest.named_steps["model"]
feature_names = random_forest.named_steps["preprocessor"].get_feature_names_out()

importance = pd.Series(
    rf_model.feature_importances_,
    index=feature_names
).sort_values(ascending=False)

print("\nTop 10 Important Features:")
print(importance.head(10))

plt.figure(figsize=(10, 6))
importance.head(10).sort_values().plot(kind="barh")
plt.xlabel("Importance")
plt.title("Top 10 Features Affecting Car Price")
plt.tight_layout()
rf_model = random_forest.named_steps["model"]
features = random_forest.named_steps["preprocessor"].get_feature_names_out()

importance = pd.Series(rf_model.feature_importances_, index=features)
importance = importance.sort_values(ascending=False).head(10)

print("\nTop 10 Features:")
print(importance)

importance.sort_values().plot(kind="barh", figsize=(10, 6))
plt.title("Top 10 Features Affecting Car Price")
plt.xlabel("Importance")
plt.tight_layout()
rf_model = random_forest.named_steps["model"]
features = random_forest.named_steps["preprocessor"].get_feature_names_out()
importance = pd.Series(rf_model.feature_importances_, index=features)
importance = importance.sort_values(ascending=False).head(10)
print("\nTop 10 Features:")
print(importance)
importance.sort_values().plot(kind="barh", figsize=(10, 6))
plt.title("Top 10 Features Affecting Car Price")
plt.xlabel("Importance")
plt.tight_layout()
plt.show()