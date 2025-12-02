# src/api/app.py
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI(
    title="Turbine Thermal Health API",
    description="Predict blade surface temperature & risk level from operating conditions.",
    version="1.0"
)

# Load models & scaler
reg_model = joblib.load("models/regression_model.pkl")
class_model = joblib.load("models/classification_model.pkl")
scaler_bundle = joblib.load("data/processed/scaler.pkl")
scaler = scaler_bundle["scaler"]
scale_cols = scaler_bundle["columns"]


# Expected input data format
class TurbineInput(BaseModel):
    rpm: float
    load_pct: float
    t_inlet_K: float
    p_inlet_bar: float
    t_coolant_K: float
    m_dot_coolant_kg_s: float
    fuel_h2_frac: float
    k_material_W_mK: float
    blade_thickness_m: float
    chord_length_m: float

@app.get("/")
def home():
    return {"message": "Turbine Thermal Health API is running!"}

@app.post("/predict")
def predict(data: TurbineInput):
    # Convert to dataframe
    df = pd.DataFrame([data.dict()])

    # Add engineered features (same as training)
    df["temp_ratio"] = df["t_inlet_K"] / df["t_coolant_K"]
    df["cooling_eff"] = df["m_dot_coolant_kg_s"] * df["t_coolant_K"]
    df["load_rpm_interaction"] = df["load_pct"] * df["rpm"]
    df["geometry_factor"] = df["blade_thickness_m"] * df["chord_length_m"]

    # Scale numeric columns
    numeric_cols = df.columns
    df[scale_cols] = scaler.transform(df[scale_cols])

    # Predict temperature
    temp_pred = reg_model.predict(df)[0]

    # Predict risk class
    risk_pred = class_model.predict(df)[0]
    risk_map = {0: "safe", 1: "warning", 2: "critical"}

    return {
        "predicted_temperature_C": round(float(temp_pred), 2),
        "risk_level": risk_map[risk_pred]
    }
