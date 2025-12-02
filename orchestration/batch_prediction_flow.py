# orchestration/batch_prediction_flow.py

from prefect import flow, task
import pandas as pd
import joblib
from datetime import datetime
from pathlib import Path

# Load models
reg_model = joblib.load("models/regression_model.pkl")
class_model = joblib.load("models/classification_model.pkl")

# Load scaler
scaler_bundle = joblib.load("data/processed/scaler.pkl")
scaler = scaler_bundle["scaler"]
scale_cols = scaler_bundle["columns"]

@task
def load_daily_log(path: str):
    print(f"📥 Loading daily turbine log: {path}")
    return pd.read_csv(path)

@task
def preprocess(df: pd.DataFrame):
    df = df.copy()

    # Add engineered features (same as training)
    df["temp_ratio"] = df["t_inlet_K"] / df["t_coolant_K"]
    df["cooling_eff"] = df["m_dot_coolant_kg_s"] * df["t_coolant_K"]
    df["load_rpm_interaction"] = df["load_pct"] * df["rpm"]
    df["geometry_factor"] = df["blade_thickness_m"] * df["chord_length_m"]

    # Scale
    df[scale_cols] = scaler.transform(df[scale_cols])
    return df

@task
def predict(df: pd.DataFrame):
    preds_temp = reg_model.predict(df)
    preds_class = class_model.predict(df)
    return preds_temp, preds_class

@task
def save_predictions(df, preds_temp, preds_class):
    df_out = df.copy()
    df_out["predicted_temp_C"] = preds_temp
    df_out["risk_class"] = preds_class

    Path("data/predictions").mkdir(exist_ok=True)
    fname = f"data/predictions/pred_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

    df_out.to_csv(fname, index=False)
    print(f"📤 Saved predictions to {fname}")

@flow(name="turbine_batch_prediction_flow")
def batch_prediction_flow(log_path: str):
    df = load_daily_log(log_path)
    df_proc = preprocess(df)
    preds_temp, preds_class = predict(df_proc)
    save_predictions(df, preds_temp, preds_class)
    print("✅ Batch prediction pipeline completed!")

if __name__ == "__main__":
    sample_log = "data/daily_logs/sample_log.csv"
    batch_prediction_flow(sample_log)
