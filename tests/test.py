"""Tests for the preprocessing pipeline."""

import sys
import os

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.preprocessing import (
    make_datetime,
    remove_duplicates,
    handle_missing_values,
    remove_outliers_iqr,
    encode_wind_direction,
    encode_station,
    add_time_features,
    add_lag_features,
    add_rolling_features,
    add_interaction_features,
    time_based_split,
    build_feature_matrix,
)


STATIONS = ["Aotizhongxin", "Changping"]


def _make_sample_df(n: int = 200, seed: int = 42) -> pd.DataFrame:
    """Create a minimal sample DataFrame mimicking the raw dataset."""
    rng = np.random.default_rng(seed)
    rows = []
    for station in STATIONS:
        for i in range(n // len(STATIONS)):
            year = 2013 + (i * len(STATIONS) * 24 // (365 * 24))
            rows.append({
                "year": min(year, 2017),
                "month": (i % 12) + 1,
                "day": (i % 28) + 1,
                "hour": i % 24,
                "PM2.5": rng.uniform(5, 300),
                "PM10": rng.uniform(10, 400),
                "SO2": rng.uniform(1, 50),
                "NO2": rng.uniform(5, 100),
                "CO": rng.uniform(200, 2000),
                "O3": rng.uniform(1, 150),
                "TEMP": rng.uniform(-10, 40),
                "PRES": rng.uniform(990, 1030),
                "DEWP": rng.uniform(-20, 25),
                "RAIN": rng.uniform(0, 5),
                "wd": rng.choice(["N", "NE", "E", "SE", "S", "SW", "W", "NW"]),
                "WSPM": rng.uniform(0, 10),
                "station": station,
            })
    return pd.DataFrame(rows)


def test_make_datetime():
    df = _make_sample_df()
    df = make_datetime(df)
    assert "datetime" in df.columns
    assert pd.api.types.is_datetime64_any_dtype(df["datetime"])


def test_remove_duplicates():
    df = _make_sample_df()
    df_dup = pd.concat([df, df.iloc[:10]], ignore_index=True)
    df_clean = remove_duplicates(df_dup)
    assert len(df_clean) == len(df)


def test_handle_missing_values_no_nan():
    df = _make_sample_df()
    df = handle_missing_values(df)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    assert df[numeric_cols].isnull().sum().sum() == 0


def test_remove_outliers_iqr_reduces_rows():
    df = _make_sample_df()
    df.loc[0, "PM2.5"] = 99999  # Экстремальный выброс
    df_clean = remove_outliers_iqr(df, target_col="PM2.5")
    assert len(df_clean) < len(df)
    assert df_clean["PM2.5"].max() < 99999


def test_encode_wind_direction():
    df = _make_sample_df()
    df = encode_wind_direction(df)
    assert "wd" not in df.columns
    assert "wd_sin" in df.columns
    assert "wd_cos" in df.columns
    assert df["wd_sin"].between(-1, 1).all()


def test_encode_station():
    df = _make_sample_df()
    df = encode_station(df)
    assert "station_enc" in df.columns
    assert df["station_enc"].nunique() == len(STATIONS)


def test_add_time_features():
    df = _make_sample_df()
    df = make_datetime(df)
    df = add_time_features(df)
    for col in ["hour_sin", "hour_cos", "month_sin", "month_cos", "day_of_week", "is_weekend"]:
        assert col in df.columns


def test_add_lag_features():
    df = _make_sample_df(n=200)
    df = make_datetime(df)
    df = add_lag_features(df, target_col="PM2.5", lags=[1, 3])
    assert "PM2.5_lag1" in df.columns
    assert "PM2.5_lag3" in df.columns
    assert df["PM2.5_lag1"].isnull().sum() == 0


def test_add_rolling_features():
    df = _make_sample_df(n=200)
    df = make_datetime(df)
    df = add_rolling_features(df, target_col="PM2.5", windows=[3])
    assert "PM2.5_roll3" in df.columns


def test_time_based_split_no_leak():
    df = _make_sample_df(n=200)
    df = make_datetime(df)
    df["year"] = df["datetime"].dt.year
    # Генерируем данные с разными годами
    frames = []
    for yr in [2014, 2015, 2016, 2017]:
        chunk = _make_sample_df(n=40)
        chunk["year"] = yr
        frames.append(chunk)
    df = pd.concat(frames, ignore_index=True)
    df = make_datetime(df)

    train, val, test = time_based_split(df, val_year=2016, test_year=2017)
    assert train["year"].max() < 2016
    assert val["year"].unique().tolist() == [2016]
    assert test["year"].min() >= 2017


def test_build_feature_matrix():
    df = _make_sample_df(n=100)
    df = make_datetime(df)
    df = encode_wind_direction(df)
    df = encode_station(df)
    X, y = build_feature_matrix(df, target_col="PM2.5")
    assert "PM2.5" not in X.columns
    assert len(X) == len(y)


def test_add_interaction_features():
    df = _make_sample_df(n=100)
    df = encode_wind_direction(df)
    df = add_interaction_features(df)
    for col in ["O3_to_NO2", "wind_u", "wind_v", "pm10_pm25_ratio", "temp_dewp_diff", "season"]:
        assert col in df.columns
    assert df["season"].between(0, 3).all()
