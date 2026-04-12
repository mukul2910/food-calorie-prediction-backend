import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_absolute_error
import streamlit as st
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# ---------------- LOAD DATA ----------------
data = pd.read_csv("Indian_Food_Nutrition_Processed.csv")

# ---------------- SELECT REQUIRED COLUMNS ----------------
data = data[["Dish Name", "Calories (kcal)", "Protein (g)", "Carbohydrates (g)", "Fats (g)"]]

# ---------------- RENAME COLUMNS ----------------
data.columns = ["Description", "Calories", "Protein", "Carbs", "Fat"]

# ---------------- CLEAN DATA ----------------
data = data.dropna()
data["Description"] = data["Description"].str.lower()

data = data.groupby("Description").mean().reset_index()

# (optional) limit data for fast training


# ---------------- INPUT & OUTPUT ----------------
X = data["Description"]
y = data["Calories"]

# ---------------- TEXT TO NUMERIC ----------------
vectorizer = TfidfVectorizer(ngram_range=(1,2))
X_vec = vectorizer.fit_transform(X)

# ---------------- TRAIN TEST SPLIT ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X_vec, y, test_size=0.2, random_state=42
)

# ---------------- DECISION TREE ----------------
dt_model = DecisionTreeRegressor()
dt_model.fit(X_train, y_train)

dt_pred = dt_model.predict(X_test)
dt_error = mean_absolute_error(y_test, dt_pred)

# ---------------- NEURAL NETWORK ----------------
nn_model = Sequential()
nn_model.add(Dense(64, activation='relu', input_dim=X_train.shape[1]))
nn_model.add(Dense(32, activation='relu'))
nn_model.add(Dense(1))

nn_model.compile(optimizer='adam', loss='mse')
nn_model.fit(X_train.toarray(), y_train, epochs=100, verbose=1)

nn_pred = nn_model.predict(X_test.toarray())
nn_error = mean_absolute_error(y_test, nn_pred)

# ---------------- UI ----------------
st.title("🍔 Food Calorie Predictor (Real Dataset)")

food_input = st.text_input("Enter food name:").lower()

if st.button("Predict"):
    input_vec = vectorizer.transform([food_input])

    dt_result = dt_model.predict(input_vec)[0]
    nn_result = nn_model.predict(input_vec.toarray())[0][0]

    st.write(f"Decision Tree Calories: {dt_result:.2f}")
    st.write(f"Neural Network Calories: {nn_result:.2f}")

st.write("------")
st.write(f"Decision Tree Error (MAE): {dt_error:.2f}")
st.write(f"Neural Network Error (MAE): {nn_error:.2f}")