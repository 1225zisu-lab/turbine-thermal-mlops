# src/data/generate_synthetic_data.py
import numpy as np
import pandas as pd
from pathlib import Path

rng = np.random.default_rng(42)
N_SAMPLES = 5000

def generate_data(n_samples: int = N_SAMPLES) -> pd.DataFrame:
    rpm = rng.uniform(3000, 12000, size=n_samples)              # rotor speed
    load_pct = rng.uniform(0.4, 1.0, size=n_samples)            # fraction of rated load
    t_inlet = rng.uniform(900, 1500, size=n_samples)            # K
    p_inlet = rng.uniform(8, 32, size=n_samples)                # bar
    t_coolant = rng.uniform(300, 650, size=n_samples)           # K
    m_dot_coolant = rng.uniform(0.05, 0.3, size=n_samples)      # kg/s (per blade, fake)
    fuel_h2_frac = rng.uniform(0.0, 1.0, size=n_samples)        # 0 = no H2, 1 = pure H2

    k_material = rng.uniform(15, 35, size=n_samples)            # W/mK
    blade_thickness = rng.uniform(2e-3, 7e-3, size=n_samples)   # m
    chord_length = rng.uniform(0.05, 0.18, size=n_samples)      # m

    t_base = t_inlet + 120 * load_pct + 0.002 * (rpm - 6000)

    cooling_effect = (
        200 * (m_dot_coolant)
        + 0.03 * (t_coolant - 300)
    )

    mat_geo_effect = (
        -0.6 * (k_material - 20)
        -150 * (blade_thickness - 0.0035)
    )

    fuel_effect = 50 * fuel_h2_frac
    noise = rng.normal(0, 25, size=n_samples)

    t_surface = t_base - cooling_effect + mat_geo_effect + fuel_effect + noise
    t_surface_c = t_surface - 273.15

    df = pd.DataFrame({
        "rpm": rpm,
        "load_pct": load_pct,
        "t_inlet_K": t_inlet,
        "p_inlet_bar": p_inlet,
        "t_coolant_K": t_coolant,
        "m_dot_coolant_kg_s": m_dot_coolant,
        "fuel_h2_frac": fuel_h2_frac,
        "k_material_W_mK": k_material,
        "blade_thickness_m": blade_thickness,
        "chord_length_m": chord_length,
        "t_surface_K": t_surface,
        "t_surface_C": t_surface_c
    })

    conditions = [
        df["t_surface_C"] < 850,
        df["t_surface_C"].between(850, 950),
        df["t_surface_C"] > 950
    ]
    choices = [0, 1, 2]
    df["risk_class"] = np.select(conditions, choices)

    return df

def main():
    out_path = Path("data/raw/turbine_synthetic.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = generate_data()
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} rows to {out_path.resolve()}")

if __name__ == "__main__":
    main()
