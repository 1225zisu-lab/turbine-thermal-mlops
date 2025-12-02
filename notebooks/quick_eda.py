# notebooks/quick_eda.py
import pandas as pd
df = pd.read_csv("data/raw/turbine_synthetic.csv")
print(df.head())
print("\nRisk class counts:\n", df["risk_class"].value_counts())
print("\nDescriptive stats (t_surface_C):\n", df["t_surface_C"].describe())
