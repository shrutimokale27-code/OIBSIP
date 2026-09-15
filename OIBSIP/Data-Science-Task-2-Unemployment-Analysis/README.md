# Car Price Prediction with Machine Learning

## Project Overview

This project predicts the price of a car using Machine Learning techniques. The dataset contains different car specifications such as engine size, horsepower, fuel type, car body, mileage, and other features.

Two regression models are used:

* Linear Regression
* Random Forest Regression

The models are evaluated using MAE, RMSE, and R² Score.

## Objectives

* Clean and preprocess the car dataset.
* Perform exploratory data analysis.
* Extract car brands from car names.
* Encode categorical features.
* Analyze correlations using a heatmap.
* Build regression models for car price prediction.
* Compare model performance.
* Identify important features affecting car prices.

## Dataset

The dataset contains **205 car records and 26 columns**.

The target variable is:

**Price** – The selling price of the car.

Important features include:

* Fuel Type
* Aspiration
* Car Body
* Drive Wheel
* Engine Type
* Engine Size
* Horsepower
* City MPG
* Highway MPG
* Car Brand

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn

## Data Preprocessing

The following preprocessing steps were performed:

1. Loaded the dataset using Pandas.
2. Checked for missing values.
3. Checked for duplicate records.
4. Extracted the car brand from `CarName`.
5. Removed unnecessary columns such as `car_ID` and `CarName`.
6. Encoded categorical variables using One-Hot Encoding.
7. Split the dataset into training and testing sets.

## Exploratory Data Analysis

A correlation heatmap was created to understand the relationship between numerical features and car price.

An Actual vs Predicted Price scatter plot was also created to evaluate the prediction performance.

## Machine Learning Models

### 1. Linear Regression

Linear Regression was used as a baseline regression model.

**R² Score: 0.90**

### 2. Random Forest Regression

Random Forest Regression was used to capture non-linear relationships between car features and price.

**R² Score: 0.96**

## Model Performance

| Model                    |     MAE |    RMSE | R² Score |
| ------------------------ | ------: | ------: | -------: |
| Linear Regression        | 1835.33 | 2811.38 |     0.90 |
| Random Forest Regression | 1296.98 | 1831.07 |     0.96 |

### Best Model

**Random Forest Regression** performed better than Linear Regression with an R² Score of **0.96**.

## Feature Importance

Random Forest Feature Importance was used to identify the most influential features affecting car prices.

The project generates a **Top 10 Features Affecting Car Price** graph.

## Project Structure

```text
Data-Science-Task-3-Car-Price-Prediction/
│
├── car_data.csv
├── car_price_prediction.py
└── README.md
```

## Conclusion

This project demonstrates how Machine Learning can be used to predict car prices based on vehicle specifications. Among the two models, Random Forest Regression provided better prediction performance than Linear Regression.

The project also demonstrates data preprocessing, feature engineering, exploratory data analysis, categorical encoding, model evaluation, and feature importance analysis.

