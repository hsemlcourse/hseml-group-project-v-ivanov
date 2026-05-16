"""Data preprocessing pipeline for Beijing PM2.5 prediction."""

import os
import glob
import numpy as np
import pandas as pd

RANDOM_SEED = 42

# Wind direction encoding
WIND_DIR_MAP = {
    "N": 0, "NNE": 22.5, "NE": 45, "ENE": 67.5,
    "E": 90, "ESE": 112.5, "SE": 135, "SSE": 157.5,
    "S": 180, "SSW": 202.5, "SW": 225, "WSW": 247.5,
    "W": 270, "WNW": 292.5, "NW": 315, "NNW": 337.5,
    "cv": np.nan,
}


def load_raw_data(raw_dir: str) -> pd.DataFrame:
    """Load all station CSV files from raw_dir into a single DataFrame."""
    pattern = os.path.join(raw_dir, "*.csv")
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"No CSV files found in {raw_dir}")
    frames = [pd.read_csv(f) for f in sorted(files)]
    df = pd.concat(frames, ignore_index=True)
    return df


def make_datetime(df: pd.DataFrame) -> pd.DataFrame:
    """Create a datetime column from year/month/day/hour columns."""
    df = df.copy()
    df["datetime"] = pd.to_datetime(
        df[["year", "month", "day", "hour"]].rename(
            columns={"hour": "hour"}
        ).assign(minute=0, second=0)
    )
    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Drop duplicate rows."""
    n_before = len(df)
    df = df.drop_duplicates()
    n_after = len(df)
    if n_before != n_after:
        print(f"Removed {n_before - n_after} duplicate rows.")
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing values:
    - Numeric columns: forward-fill per station, then median fill.
    - Categorical: fill with mode.
    """
    df = df.copy()

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_cols:
        df[col] = df.groupby("station")[col].transform(
            lambda x: x.ffill().bfill()
        )
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)

    cat_cols = df.select_dtypes(include=["object", "str"]).columns.tolist()
    for col in cat_cols:
        if col != "station":
            mode_val = df[col].mode(dropna=True)
            if len(mode_val) > 0:
                df[col] = df[col].fillna(mode_val[0])

    return df


def remove_outliers_iqr(df: pd.DataFrame, target_col: str = "PM2.5") -> pd.DataFrame:
    """Remove rows where target is extreme outlier (beyond 3*IQR)."""
    df = df.copy()
    q1 = df[target_col].quantile(0.25)
    q3 = df[target_col].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 3 * iqr
    upper = q3 + 3 * iqr
    mask = df[target_col].between(lower, upper)
    n_removed = (~mask).sum()
    if n_removed > 0:
        print(f"Removed {n_removed} outlier rows from '{target_col}'.")
    return df[mask].reset_index(drop=True)


def encode_wind_direction(df: pd.DataFrame) -> pd.DataFrame:
    """Encode wind direction as sine and cosine of angle in degrees."""
    df = df.copy()
    df["wd_deg"] = df["wd"].map(WIND_DIR_MAP)
    df["wd_sin"] = np.sin(np.deg2rad(df["wd_deg"]))
    df["wd_cos"] = np.cos(np.deg2rad(df["wd_deg"]))
    df = df.drop(columns=["wd", "wd_deg"])
    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add cyclical time features and basic temporal flags."""
    df = df.copy()
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["day_of_week"] = df["datetime"].dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    return df


def add_lag_features(df: pd.DataFrame, target_col: str = "PM2.5", lags: list = None) -> pd.DataFrame:
    """Add lag features for the target per station (sorted by datetime)."""
    if lags is None:
        lags = [1, 2, 3, 24]
    df = df.copy().sort_values(["station", "datetime"])
    for lag in lags:
        df[f"{target_col}_lag{lag}"] = df.groupby("station")[target_col].shift(lag)
    df = df.dropna(subset=[f"{target_col}_lag{lag}" for lag in lags])
    return df.reset_index(drop=True)


def add_rolling_features(df: pd.DataFrame, target_col: str = "PM2.5", windows: list = None) -> pd.DataFrame:
    """Add rolling mean features for the target per station."""
    if windows is None:
        windows = [3, 24]
    df = df.copy().sort_values(["station", "datetime"])
    for w in windows:
        df[f"{target_col}_roll{w}"] = (
            df.groupby("station")[target_col]
            .transform(lambda x: x.shift(1).rolling(w, min_periods=1).mean())
        )
    return df.reset_index(drop=True)


def encode_station(df: pd.DataFrame) -> pd.DataFrame:
    """Label-encode the station column."""
    df = df.copy()
    df["station_enc"] = df["station"].astype("category").cat.codes
    return df


def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add domain-driven interaction and ratio features."""
    df = df.copy()
    # Oxidant ratio: high O3 suppresses PM2.5 (photochemical reaction)
    df["O3_to_NO2"] = df["O3"] / (df["NO2"] + 1)
    # Wind vector components (speed × direction); cv (calm) → 0
    df["wind_u"] = df["WSPM"] * df["wd_sin"].fillna(0)
    df["wind_v"] = df["WSPM"] * df["wd_cos"].fillna(0)
    # PM10/PM2.5 ratio indicates coarse particle fraction
    df["pm10_pm25_ratio"] = df["PM10"] / (df["PM2.5"] + 1)
    # Temperature-humidity index (affects particulate behavior)
    df["temp_dewp_diff"] = df["TEMP"] - df["DEWP"]
    # Season (0=winter, 1=spring, 2=summer, 3=autumn)
    df["season"] = ((df["month"] % 12) // 3).astype(int)
    return df


def time_based_split(df: pd.DataFrame, val_year: int = 2016, test_year: int = 2017):
    """
    Split dataset by time to avoid data leakage:
      - train: years < val_year
      - val:   year == val_year
      - test:  year >= test_year
    """
    train = df[df["year"] < val_year].copy()
    val = df[df["year"] == val_year].copy()
    test = df[df["year"] >= test_year].copy()
    return train, val, test


def build_feature_matrix(
    df: pd.DataFrame,
    target_col: str = "PM2.5",
    drop_cols: list = None,
):
    """Return (X, y) arrays from a processed DataFrame."""
    if drop_cols is None:
        drop_cols = ["datetime", "station", "year", "month", "day", "hour", target_col]
    drop_existing = [c for c in drop_cols if c in df.columns]
    X = df.drop(columns=drop_existing)
    y = df[target_col]
    return X, y


def full_pipeline(
    raw_dir: str,
    processed_dir: str,
    add_lags: bool = True,
    add_rolling: bool = True,
) -> tuple:
    """
    End-to-end preprocessing pipeline.

    Returns (train_df, val_df, test_df) — DataFrames with all engineered features.
    """
    df = load_raw_data(raw_dir)
    df = make_datetime(df)
    df = remove_duplicates(df)
    df = handle_missing_values(df)
    df = remove_outliers_iqr(df, target_col="PM2.5")
    df = encode_wind_direction(df)
    df = encode_station(df)
    df = add_time_features(df)
    if add_lags:
        df = add_lag_features(df)
    if add_rolling:
        df = add_rolling_features(df)
    df = add_interaction_features(df)

    train, val, test = time_based_split(df)

    os.makedirs(processed_dir, exist_ok=True)
    train.to_parquet(os.path.join(processed_dir, "train.parquet"), index=False)
    val.to_parquet(os.path.join(processed_dir, "val.parquet"), index=False)
    test.to_parquet(os.path.join(processed_dir, "test.parquet"), index=False)

    return train, val, test
