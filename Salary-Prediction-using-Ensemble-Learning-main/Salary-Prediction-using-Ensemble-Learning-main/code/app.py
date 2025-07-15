import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

st.title("💼 Salary Prediction App")


# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("data/salary_train.csv")
    df = df.dropna(subset=["salary"])
    return df


df = load_data()


# Train model on startup
@st.cache_resource
def train_model(df):
    X = df.drop(columns=["ID", "salary"])
    y = df["salary"]

    for col in X.select_dtypes(include=["float64", "int64"]).columns:
        X[col] = X[col].fillna(X[col].mean())

    for col in X.select_dtypes(include=["object"]).columns:
        X[col] = X[col].fillna(X[col].mode()[0])

    X_encoded = pd.get_dummies(X)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_encoded)

    X_train, X_val, y_train, y_val = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)

    rmse = np.sqrt(np.mean((model.predict(X_val) - y_val) ** 2))

    return model, scaler, X_encoded.columns.tolist(), rmse, X


model, scaler, feature_list, rmse, raw_X = train_model(df)
st.success(f"Model trained ✅ Validation RMSE: ₹{rmse:.2f}")

# Collect input features
st.header("📥 Enter Features to Predict Salary")

input_data = {}

# Numeric Inputs
for col in raw_X.select_dtypes(include=["float64", "int64"]).columns:
    val = st.number_input(f"{col}", value=float(raw_X[col].mean()))
    input_data[col] = val

# Categorical Inputs
for col in raw_X.select_dtypes(include=["object"]).columns:
    options = sorted(raw_X[col].dropna().unique())
    val = st.selectbox(f"{col}", options)
    input_data[col] = val

# Preprocess input
input_df = pd.DataFrame([input_data])
for col in input_df.select_dtypes(include=["float64", "int64"]).columns:
    input_df[col] = input_df[col].fillna(raw_X[col].mean())
for col in input_df.select_dtypes(include=["object"]).columns:
    input_df[col] = input_df[col].fillna(raw_X[col].mode()[0])

input_encoded = pd.get_dummies(input_df)
input_encoded = input_encoded.reindex(columns=feature_list, fill_value=0)
input_scaled = scaler.transform(input_encoded)

if st.button("🔍 Predict Salary"):
    pred = model.predict(input_scaled)[0]
    st.success(f"💰 Predicted Salary: ₹{pred:,.2f}")
