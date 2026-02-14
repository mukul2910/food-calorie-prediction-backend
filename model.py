# ===============================
# FOOD CALORIE PREDICTION MODEL
# ===============================

import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

print("🔥 Model Started Running...")

# ===============================
# 1. Load Dataset
# ===============================
data = pd.read_csv("nutrients.csv")

print("✅ Dataset Loaded Successfully")
print("Dataset Shape:", data.shape)

# Replace 't' with 0
data.replace('t', 0, inplace=True)

# Convert numeric columns
numeric_cols = ['Grams', 'Calories', 'Protein', 'Fat', 
                'Sat.Fat', 'Fiber', 'Carbs']

for col in numeric_cols:
    data[col] = pd.to_numeric(data[col], errors='coerce')

data.dropna(inplace=True)

print("✅ Data Cleaned Successfully")

# ===============================
# 2. Features & Target
# ===============================
X = data[['Protein', 'Fat', 'Sat.Fat', 'Fiber', 'Carbs']]
y = data['Calories']

# ===============================
# 3. Train-Test Split
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("✅ Data Split Done")

# ===============================
# 4. Decision Tree Model
# ===============================
dt_model = DecisionTreeRegressor(random_state=42)
dt_model.fit(X_train, y_train)

dt_pred = dt_model.predict(X_test)

print("\n--- Decision Tree Results ---")
print("MAE:", mean_absolute_error(y_test, dt_pred))
print("R2 Score:", r2_score(y_test, dt_pred))

# ===============================
# 5. Neural Network Model
# ===============================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

nn_model = MLPRegressor(hidden_layer_sizes=(50,50),
                        max_iter=1000,
                        random_state=42)

nn_model.fit(X_train_scaled, y_train)
nn_pred = nn_model.predict(X_test_scaled)

print("\n--- Neural Network Results ---")
print("MAE:", mean_absolute_error(y_test, nn_pred))
print("R2 Score:", r2_score(y_test, nn_pred))

# ===============================
# 6. Save Model
# ===============================
joblib.dump(nn_model, "food_calorie_model.pkl")

print("\n✅ Model Saved Successfully as food_calorie_model.pkl")
print("🎯 Backend Ready for Presentation")