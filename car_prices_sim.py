import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Car Price Predictor", page_icon="🚗", layout="wide")

st.title("🚗 Car Price Predictor")
st.markdown("### Predict car prices based on features like Mark, PS, KM, and Fabrication Year.")

# --- Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("car_prices.csv")
    return df

df = load_data()
st.write("### Sample of dataset", df.head())

# --- Split features and target
X = df[["MarkCode", "PS", "KM", "Year"]].values
y = df["Price"].values

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train model
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

score = model.score(X_test, y_test)

st.success(f"✅ Model trained. R² Score on test set: {score:.3f}")

# --- User input for prediction
st.sidebar.header("🔧 Enter Car Details")

mark_code = st.sidebar.selectbox(
    "Brand Code",
    options=[10, 9, 8, 7, 6, 5, 4, 3],
    format_func=lambda x: {
        10: "Mercedes / Porsche",
        9: "BMW / Audi",
        8: "Toyota / Honda / Tesla",
        7: "DS / VW",
        6: "Ford / Nissan / KIA / Hyundai / Skoda / Seat",
        5: "Renault / Dacia / Citroen / Fiat / Peugeot",
        4: "Opel / Lada",
        3: "Other"
    }[x]
)

ps = st.sidebar.slider("Horsepower (PS)", 60, 500, 150)
km = st.sidebar.slider("Mileage (KM)", 5000, 250000, 80000, step=5000)
year = st.sidebar.slider("Fabrication Year", 1998, 2024, 2015)

# --- Make prediction
input_data = np.array([[mark_code, ps, km, year]])
input_scaled = scaler.transform(input_data)
predicted_price = model.predict(input_scaled)[0]

st.subheader("💰 Predicted Car Price")
st.success(f"Estimated Price: **${predicted_price:,.0f}**")

# --- Show performance
st.subheader("📊 Model Performance (Actual vs Predicted)")
y_pred = model.predict(X_test)
perf_df = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})
st.line_chart(perf_df.reset_index(drop=True))