import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load Dataset
df = pd.read_csv("Unemployment in India.csv")

print("First 5 Rows:")
print(df.head())

print("\nDataset Shape:", df.shape)

# 2. Clean Column Names
df.columns = df.columns.str.strip()

print("\nColumn Names:")
print(df.columns.tolist())

# 3. Data Cleaning
df = df.dropna()

print("\nMissing Values:")
print(df.isnull().sum())

# 4. Convert Date Column
df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)

# 5. Basic Statistics
print("\nStatistical Summary:")
print(df.describe())

# 6. Average Unemployment Rate by State
state_avg = df.groupby("Region")["Estimated Unemployment Rate (%)"].mean()
state_avg = state_avg.sort_values(ascending=False)

print("\nAverage Unemployment Rate by State:")
print(state_avg)

# 7. State-wise Unemployment Plot
plt.figure(figsize=(12, 7))
state_avg.head(10).plot(kind="bar")
plt.title("Top 10 States by Average Unemployment Rate")
plt.xlabel("State")
plt.ylabel("Average Unemployment Rate (%)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 8. Monthly Unemployment Trend
monthly = df.groupby("Date")["Estimated Unemployment Rate (%)"].mean()

plt.figure(figsize=(12, 6))
plt.plot(monthly.index, monthly.values)
plt.title("Monthly Unemployment Rate in India")
plt.xlabel("Date")
plt.ylabel("Average Unemployment Rate (%)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 9. Heatmap of Correlation
numeric_cols = [
    "Estimated Unemployment Rate (%)",
    "Estimated Employed",
    "Estimated Labour Participation Rate (%)"
]

plt.figure(figsize=(8, 6))
sns.heatmap(df[numeric_cols].corr(), annot=True, fmt=".2f")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()

# 10. Highest and Lowest Unemployment
highest_state = state_avg.idxmax()
highest_rate = state_avg.max()

lowest_state = state_avg.idxmin()
lowest_rate = state_avg.min()

print("\nHighest Average Unemployment:")
print(highest_state, ":", round(highest_rate, 2), "%")

print("\nLowest Average Unemployment:")
print(lowest_state, ":", round(lowest_rate, 2), "%")

# 11. Conclusion
print("\nConclusion:")
print("The analysis shows unemployment variation across different states and over time.")
print("The visualizations help identify states with higher and lower unemployment rates.")