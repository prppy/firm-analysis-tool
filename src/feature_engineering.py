import numpy as np
import pandas as pd
from config import CURRENT_YEAR

def log_transform(df, cols):
    for col in cols:
        safe_col = col.lower().replace(" ", "_").replace("(", "").replace(")", "")
        df[f"log_{safe_col}"] = np.log1p(df[col])
    return df

def create_features(df):
    # Company age
    df["company_age"] = CURRENT_YEAR - df["Year Found"]

    # Efficiency metrics
    df["revenue_per_employee"] = df["Revenue (USD)"] / (df["Employees Total"] + 1)
    df["log_revenue_per_employee"] = np.log1p(df["revenue_per_employee"])
    df["employees_per_site"] = df["Employees Total"] / (df["Employees Single Site"] + 1)
    df["it_spend_per_employee"] = df["IT spend"] / (df["Employees Total"] + 1)

    # IT infrastructure density
    infra_cols = [
        "No. of Servers",
        "No. of Routers",
        "No. of Storage Devices"
    ]
    df["infra_density_score"] = df[infra_cols].sum(axis=1) / (df["Employees Total"] + 1)

    # Device mix
    df["laptop_desktop_ratio"] = (
        df["No. of Laptops"] / (df["No. of Desktops"] + 1)
    )

    return df
