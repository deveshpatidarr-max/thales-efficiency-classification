"""Shared logic: data loading, feature engineering, prediction helpers.
Used by both train.py (training) and app.py (dashboard) so they never disagree."""
import numpy as np
import pandas as pd

DATA_PATH = "data/Thales_Group_Manufacturing.csv"
TARGET = "Efficiency_Status"
CLASS_ORDER = ["Low", "Medium", "High"]

# Raw numeric columns, renamed to short, safe names
RENAME = {
    "Temperature_C": "temperature",
    "Vibration_Hz": "vibration",
    "Power_Consumption_kW": "power",
    "Network_Latency_ms": "latency",
    "Packet_Loss_%": "packet_loss",
    "Quality_Control_Defect_Rate_%": "defect_rate",
    "Production_Speed_units_per_hr": "speed",
    "Predictive_Maintenance_Score": "maint_score",
    "Error_Rate_%": "error_rate",
}
SENSOR = ["temperature", "vibration", "power", "maint_score"]
NETWORK = ["latency", "packet_loss"]
QUALITY = ["defect_rate"]
PRODUCTION = ["speed", "error_rate"]

# Known sensor ranges (from the data) used for 0-1 normalisation of indices
RANGES = {"temperature": (30, 90), "vibration": (0, 5), "power": (1, 10),
          "latency": (1, 50), "packet_loss": (0, 5)}


def load_raw(path=DATA_PATH):
    df = pd.read_csv(path)
    df["datetime"] = pd.to_datetime(df["Date"] + " " + df["Timestamp"], format="%d-%m-%Y %H:%M:%S")
    df = df.sort_values("datetime").reset_index(drop=True)      # time-based ordering
    df = df.rename(columns=RENAME)
    return df


def add_features(df):
    """Feature engineering. Every feature here can be computed from ONE record,
    so the model works in real time without needing history."""
    d = df.copy()
    n = lambda c: (d[c] - RANGES[c][0]) / (RANGES[c][1] - RANGES[c][0])
    # Sensor stress index: how far temperature, vibration and power sit from their range floors (0 = calm, 1 = extreme)
    d["sensor_stress_index"] = (n("temperature") + n("vibration") + n("power")) / 3
    # Network reliability score: 1 = perfect link, 0 = worst latency + worst packet loss
    d["network_reliability"] = 1 - 0.5 * n("latency") - 0.5 * n("packet_loss")
    # Energy efficiency ratio: kW consumed per unit/hour produced (lower = better)
    d["energy_per_unit"] = d["power"] / (d["speed"] + 1)
    # Error-to-output ratio: % errors per unit/hour (lower = better)
    d["error_to_output"] = d["error_rate"] / (d["speed"] + 1)
    # Defects per unit/hour
    d["defect_to_output"] = d["defect_rate"] / (d["speed"] + 1)
    # Thermal-vibration load: interaction of heat and shaking
    d["thermal_vibration_load"] = d["temperature"] * d["vibration"]
    return d


def add_rolling_stability(d, window=10):
    """Sensor stability indicators over a machine's last `window` readings (history needed -> EDA / extension only)."""
    d = d.sort_values(["Machine_ID", "datetime"]).copy()
    g = d.groupby("Machine_ID")
    d["temp_roll_std"] = g["temperature"].transform(lambda s: s.rolling(window, min_periods=3).std())
    d["vib_roll_std"] = g["vibration"].transform(lambda s: s.rolling(window, min_periods=3).std())
    d[["temp_roll_std", "vib_roll_std"]] = d[["temp_roll_std", "vib_roll_std"]].fillna(
        d[["temp_roll_std", "vib_roll_std"]].median())
    return d.sort_values("datetime").reset_index(drop=True)


NUMERIC_FULL = SENSOR + NETWORK + QUALITY + PRODUCTION + [
    "sensor_stress_index", "network_reliability", "energy_per_unit",
    "error_to_output", "defect_to_output", "thermal_vibration_load"]
CATEGORICAL = ["Operation_Mode"]
MODES = ["Active", "Idle", "Maintenance"]

# Scenario B: only sensors + network + quality (NO production speed / error rate, and nothing derived from them)
NUMERIC_NO_PROD = SENSOR + NETWORK + QUALITY + [
    "sensor_stress_index", "network_reliability", "thermal_vibration_load"]


def model_matrix(d, numeric):
    """Numeric columns + one-hot Operation_Mode, always in the same column order."""
    X = d[numeric].copy()
    for m in MODES:
        X[f"mode_{m}"] = (d["Operation_Mode"] == m).astype(int)
    return X
