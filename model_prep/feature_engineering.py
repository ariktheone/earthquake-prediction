"""
Feature Engineering Module
--------------------------
Transforms raw earthquake CSV into ML-ready features.
Industry-standard feature engineering for seismic data.
"""

import pandas as pd
import numpy as np
from pathlib import Path


def load_raw_data(csv_path: str | Path) -> pd.DataFrame:
    """Load the processed earthquake CSV."""
    df = pd.read_csv(csv_path, index_col=0)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full feature engineering pipeline:
      1. Temporal features from Date/Time
      2. Spatial features (lat/lon transforms)
      3. Seismological derived features
      4. Rolling / lag statistics (temporal context)
      5. Encoding of categorical columns
    """
    df = df.copy()

    # ---- 1. Temporal Features ----
    df["datetime"] = pd.to_datetime(
        df["Date(YYYY/MM/DD)"] + " " + df["Time(UTC)"],
        format="mixed",
        errors="coerce",
    )
    df = df.sort_values("datetime").reset_index(drop=True)

    df["year"] = df["datetime"].dt.year
    df["month"] = df["datetime"].dt.month
    df["day_of_year"] = df["datetime"].dt.dayofyear
    df["hour"] = df["datetime"].dt.hour
    df["day_of_week"] = df["datetime"].dt.dayofweek

    # Cyclic encoding for month, hour, day_of_week
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

    # Time since previous event (seconds)
    df["time_since_prev"] = df["datetime"].diff().dt.total_seconds().fillna(0)
    # Log-transform (heavy right skew)
    df["log_time_since_prev"] = np.log1p(df["time_since_prev"])

    # ---- 2. Spatial Features ----
    # Convert lat/lon to radians for geospatial models
    df["lat_rad"] = np.radians(df["Latitude(deg)"])
    df["lon_rad"] = np.radians(df["Longitude(deg)"])

    # Distance from a reference point (center of data)
    ref_lat = df["Latitude(deg)"].median()
    ref_lon = df["Longitude(deg)"].median()
    df["dist_from_center"] = np.sqrt(
        (df["Latitude(deg)"] - ref_lat) ** 2
        + (df["Longitude(deg)"] - ref_lon) ** 2
    )

    # Spatial grid bin (coarse location bucketing)
    df["lat_bin"] = pd.cut(df["Latitude(deg)"], bins=20, labels=False)
    df["lon_bin"] = pd.cut(df["Longitude(deg)"], bins=20, labels=False)
    df["spatial_cell"] = df["lat_bin"] * 20 + df["lon_bin"]

    # ---- 3. Seismological Derived Features ----
    df["log_depth"] = np.log1p(df["Depth(km)"])
    df["stations_per_gap"] = df["No_of_Stations"] / (df["Gap"] + 1)
    df["close_depth_interaction"] = df["Close"] * df["Depth(km)"]
    df["rms_log"] = np.log1p(df["RMS"])
    df["gap_close_ratio"] = df["Gap"] / (df["Close"] + 1)

    # ---- 4. Rolling / Lag Features (temporal context) ----
    # Previous N events' magnitude statistics
    for window in [5, 10, 30]:
        df[f"mag_rolling_mean_{window}"] = (
            df["Magnitude(ergs)"]
            .shift(1)
            .rolling(window=window, min_periods=1)
            .mean()
        )
        df[f"mag_rolling_std_{window}"] = (
            df["Magnitude(ergs)"]
            .shift(1)
            .rolling(window=window, min_periods=1)
            .std()
            .fillna(0)
        )
        df[f"mag_rolling_max_{window}"] = (
            df["Magnitude(ergs)"]
            .shift(1)
            .rolling(window=window, min_periods=1)
            .max()
        )
        df[f"depth_rolling_mean_{window}"] = (
            df["Depth(km)"]
            .shift(1)
            .rolling(window=window, min_periods=1)
            .mean()
        )

    # Lag features (direct previous values)
    for lag in [1, 2, 3]:
        df[f"mag_lag_{lag}"] = df["Magnitude(ergs)"].shift(lag)
        df[f"depth_lag_{lag}"] = df["Depth(km)"].shift(lag)
        df[f"lat_lag_{lag}"] = df["Latitude(deg)"].shift(lag)
        df[f"lon_lag_{lag}"] = df["Longitude(deg)"].shift(lag)

    # Magnitude difference from previous event (shifted to avoid leakage)
    df["mag_diff_lag1_lag2"] = df["Magnitude(ergs)"].shift(1) - df["Magnitude(ergs)"].shift(2)
    df["mag_diff_lag2_lag3"] = df["Magnitude(ergs)"].shift(2) - df["Magnitude(ergs)"].shift(3)

    # Event count in spatial cell over rolling window (seismic density)
    df["spatial_event_count_30"] = (
        df.groupby("spatial_cell")["Magnitude(ergs)"]
        .transform(lambda x: x.shift(1).rolling(30, min_periods=1).count())
    )

    # ---- 5. Encode Categorical ----
    df["Magnitude_type_enc"] = df["Magnitude_type"].astype("category").cat.codes

    # ---- 6. Drop rows with NaN from lag features ----
    df = df.dropna().reset_index(drop=True)

    return df


def get_feature_target_split(df: pd.DataFrame):
    """
    Returns X (features) and y (target = Magnitude).
    Drops non-feature columns (identifiers, raw strings, datetime).
    """
    drop_cols = [
        "Date(YYYY/MM/DD)",
        "Time(UTC)",
        "datetime",
        "Magnitude(ergs)",  # target
        "SRC",              # single value (NCSN)
        "EventID",          # identifier
        "Magnitude_type",   # already encoded
    ]

    existing_drops = [c for c in drop_cols if c in df.columns]
    X = df.drop(columns=existing_drops)
    y = df["Magnitude(ergs)"]
    return X, y


if __name__ == "__main__":
    data_path = Path(__file__).resolve().parent.parent / "data" / "Earthquake_data_processed.csv"
    raw = load_raw_data(data_path)
    print(f"Raw data shape: {raw.shape}")
    engineered = engineer_features(raw)
    X, y = get_feature_target_split(engineered)
    print(f"Engineered features shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"\nFeature columns ({X.shape[1]}):")
    for col in X.columns:
        print(f"  - {col}")
