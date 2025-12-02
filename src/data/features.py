# src/data/features.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
from pathlib import Path

def load_data(path="data/raw/turbine_synthetic.csv"):
    return pd.read_csv(path)

def feature_engineer(df: pd.DataFrame):
    df = df.copy()

    # Example derived features (domain-inspired)
    df["temp_ratio"] = df["t_inlet_K"] / df["t_coolant_K"]
    df["cooling_eff"] = df["m_dot_coolant_kg_s"] * (df["t_coolant_K"])
    df["load_rpm_interaction"] = df["load_pct"] * df["rpm"]
    df["geometry_factor"] = df["blade_thickness_m"] * df["chord_length_m"]

    # Drop columns that leak the target
    drop_cols = ["t_surface_K", "t_surface_C", "risk_class"]
    X = df.drop(columns=drop_cols)

    y_reg = df["t_surface_C"]      # regression target
    y_class = df["risk_class"]     # classification target

    return X, y_reg, y_class

def preprocess_and_split(test_size=0.2, random_state=42):
    df = load_data()
    X, y_reg, y_class = feature_engineer(df)

    X_train, X_test, y_reg_train, y_reg_test, y_class_train, y_class_test = train_test_split(
        X, y_reg, y_class, test_size=test_size, random_state=random_state
    )

    # scale continuous features
    numeric_cols = X_train.columns  # since all X are numeric

    scaler = StandardScaler()
    X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

    # Save processed data
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    X_train.to_csv("data/processed/X_train.csv", index=False)
    X_test.to_csv("data/processed/X_test.csv", index=False)
    y_reg_train.to_csv("data/processed/y_reg_train.csv", index=False)
    y_reg_test.to_csv("data/processed/y_reg_test.csv", index=False)
    y_class_train.to_csv("data/processed/y_class_train.csv", index=False)
    y_class_test.to_csv("data/processed/y_class_test.csv", index=False)

    # Save scaler + columns
    joblib.dump(
        {
            "scaler": scaler,
            "columns": numeric_cols.tolist()
        },
    "data/processed/scaler.pkl"
    )


    print("✔ Feature engineering completed.")
    print("✔ Train/test split saved to data/processed/")
    print("✔ Scaler saved as scaler.pkl")

    return X_train, X_test, y_reg_train, y_reg_test, y_class_train, y_class_test

if __name__ == "__main__":
    preprocess_and_split()
