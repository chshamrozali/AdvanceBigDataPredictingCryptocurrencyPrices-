# Importing required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Step 1: Load the cryptocurrency dataset
file_path = 'currencies_data.csv'  # Replace with your actual file path
data = pd.read_csv(file_path)

# Step 2: Display the first few rows of the dataset
print("First few rows of the dataset:")
print(data.head())

# Step 3: Data Wrangling
# Cleaning data: Drop rows with missing or invalid data in important columns
data = data[['name', 'symbol', 'price', 'marketCap', 'volume24h', 'circulatingSupply',
             'totalSupply', 'percentChange1h', 'percentChange24h', 'percentChange7d']]

# Handling missing values (drop or fill them)
data = data.dropna()

# Normalizing numerical features for easier modeling
numerical_features = ['price', 'marketCap', 'volume24h', 'circulatingSupply', 'totalSupply', 'percentChange1h', 'percentChange24h', 'percentChange7d']
for feature in numerical_features:
    data[f'Normalized_{feature}'] = (data[feature] - data[feature].mean()) / data[feature].std()

# Step 4: Exploratory Data Analysis (EDA)
# Correlation heatmap for selected numerical features
plt.figure(figsize=(12, 8))
sns.heatmap(data[numerical_features].corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap for Cryptocurrency Features')
plt.show()

# Step 5: Data Preparation for Modeling
# Selecting relevant features for prediction
X = data[['price', 'marketCap', 'volume24h', 'circulatingSupply', 'totalSupply', 'percentChange1h', 'percentChange24h', 'percentChange7d']]
y = data['price']

# Splitting data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Model Training and Evaluation
# Model 1: Random Forest Regressor
rf = RandomForestRegressor(random_state=42, n_estimators=100)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

# Model 2: Gradient Boosting Regressor
gb = GradientBoostingRegressor(random_state=42, n_estimators=100, learning_rate=0.1)
gb.fit(X_train, y_train)
y_pred_gb = gb.predict(X_test)

# Model 3: Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train) # here we train the Model
y_pred_lr = lr.predict(X_test)

# Step 7: Model Evaluation
def evaluate_model(name, y_test, y_pred):
    print(f"--- {name} Model Evaluation ---")
    print(f"RMSE: {mean_squared_error(y_test, y_pred, squared=False):.2f}")
    print(f"MAE: {mean_absolute_error(y_test, y_pred):.2f}")
    print(f"R2 Score: {r2_score(y_test, y_pred):.2f}")
    print("\n")

# Evaluating all models
evaluate_model("Random Forest", y_test, y_pred_rf)
evaluate_model("Gradient Boosting", y_test, y_pred_gb)
evaluate_model("Linear Regression", y_test, y_pred_lr)

# Step 8: Visualizing Predictions
plt.figure(figsize=(12, 6))
plt.plot(y_test.values, label='Actual Price', linestyle='dashed', color='blue')
plt.plot(y_pred_rf, label='Random Forest Predictions', linestyle='solid', color='orange')
plt.plot(y_pred_gb, label='Gradient Boosting Predictions', linestyle='solid', color='green')
plt.plot(y_pred_lr, label='Linear Regression Predictions', linestyle='solid', color='red')
plt.legend()
plt.title('Comparison of Actual and Predicted Prices')
plt.xlabel('Sample Index')
plt.ylabel('Price (USD)')
plt.show()