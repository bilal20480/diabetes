import streamlit as st
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Set page config
st.set_page_config(page_title="Diabetes Progression Predictor", layout="centered")

st.title("🔬 Diabetes Progression Prediction")
st.markdown("This app uses a Random Forest Regressor to predict disease progression based on patient data.")

# Load dataset
diabetes = load_diabetes()
X = diabetes.data
y = diabetes.target
feature_names = diabetes.feature_names

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Sidebar for user input
st.sidebar.header("🧬 Input Patient Data")
user_input = []
for i, feature in enumerate(feature_names):
    val = st.sidebar.slider(
        f"{feature}", 
        float(np.min(X[:, i])), 
        float(np.max(X[:, i])), 
        float(np.mean(X[:, i]))
    )
    user_input.append(val)

user_input = np.array(user_input).reshape(1, -1)

# Make prediction
prediction = model.predict(user_input)[0]

# Display the prediction
st.subheader("📈 Predicted Disease Progression")
st.success(f"Predicted Value: **{prediction:.2f}**")

# Show model evaluation metrics
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

st.markdown("### 📊 Model Evaluation")
st.write(f"Mean Squared Error: **{mse:.2f}**")
st.write(f"R² Score: **{r2:.2f}**")
