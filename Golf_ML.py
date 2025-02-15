import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import mutual_info_regression
from xgboost import XGBRegressor

# --- 1. Data Loading and Cleaning ---

def clean_numeric(val):
    if pd.isnull(val):
        return None
    # Remove any character that is not a digit, period, or minus sign.
    cleaned = re.sub(r'[^\d\.-]', '', str(val))
    try:
        return float(cleaned)
    except Exception:
        return None

# File and folder path
fol = ''
fn = 'FS_Golf_DB.xlsx'
df = pd.read_excel(fol+fn)

# Normalize column names:
df.columns = df.columns.str.strip()  # remove leading/trailing whitespace
df.columns = df.columns.str.replace('\xa0', ' ')  # replace non-breaking spaces with normal spaces

# List of columns that should be numeric.
numeric_cols = [
    "Ball (mph)", "Club (mph)", "Smash Factor", "Carry (yds)",
    "Total (yds)", "Roll (yds)", "Spin (rpm)", "Height (ft)",
    "Time (s)", "AOA (°)", "Spin Loft (°)", "Swing V (°)",
    "Spin Axis (°)", "Launch H (°)", "Launch V (°)",
    "DescentV (°)", "Curve Dist (yds)", "Lateral Impact (in)", "Vertical Impact (in)"
]

# Apply cleaning function to numeric columns if they exist in the DataFrame.
for col in numeric_cols:
    if col in df.columns:
        df[col] = df[col].apply(clean_numeric)


# Write the first few rows to an Excel file so you can inspect all columns.
df.head().to_excel("df_head.xlsx", index=False)

# --- 2. Create Dummy Variables for Categorical Features ---

# List of categorical columns that you may wish to include.
categorical_cols = ["Club", "Golfer", "Shot Type", "Mode", "Location"]
for col in categorical_cols:
    if col in df.columns:
        df[col] = df[col].astype('category')

# Create dummy/indicator variables (drop first level to avoid collinearity)
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# --- 3. Define Features and Target ---

# In this example, our target variable is "Carry (yds)".
# We drop columns that are identifiers or the target itself.
# Adjust the list of dropped columns as appropriate.
drop_cols = ["Mombo ShotID", "Time", "Shot", "Video"]  # and any other columns not used for prediction
X = df.drop(columns=drop_cols + ["Carry (yds)"], errors='ignore')
y = df["Carry (yds)"]

print("Features used in the model:")
print(X.columns)

# --- 4. Train/Test Split ---

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --- 5. Model Training using XGBoost ---

# Instantiate the XGBoost regressor.
model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)

# Fit the model on training data.
model.fit(X_train, y_train)

# Predict on the test set.
y_pred = model.predict(X_test)

# Compute performance metric (RMSE).
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print("Test RMSE:", rmse)

# --- 6. Feature Importance from XGBoost ---

# Get feature importance (using gain by default)
importances = model.feature_importances_
feature_names = X_train.columns
feature_importance = pd.Series(importances, index=feature_names).sort_values(ascending=False)
print("\nFeature Importances:")
print(feature_importance)

# Plot the feature importance
plt.figure(figsize=(10, 6))
feature_importance.plot(kind='bar')
plt.ylabel("Importance")
plt.title("XGBoost Feature Importances")
plt.tight_layout()
plt.show()

# --- 7. Mutual Information for Feature Selection ---
# (This filter method helps assess how much information each feature gives about the target.)
mi = mutual_info_regression(X_train, y_train, random_state=42)
mi_series = pd.Series(mi, index=X_train.columns).sort_values(ascending=False)
print("\nMutual Information Scores:")
print(mi_series)

# --- Optional: Compare Models with and without Ball (mph) ---
# For instance, if you suspect Ball (mph) is redundant with other swing parameters,
# you might drop it and retrain the model.
if "Ball (mph)" in X_train.columns:
    X_train_reduced = X_train.drop(columns=["Ball (mph)"])
    X_test_reduced = X_test.drop(columns=["Ball (mph)"])
    
    model_reduced = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    model_reduced.fit(X_train_reduced, y_train)
    
    y_pred_reduced = model_reduced.predict(X_test_reduced)
    rmse_reduced = np.sqrt(mean_squared_error(y_test, y_pred_reduced))
    print("\nTest RMSE without 'Ball (mph)':", rmse_reduced)
