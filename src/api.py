"""FastAPI service for Beijing PM2.5 prediction."""

import datetime
import os

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

WIND_DIR_MAP = {
    "N": 0, "NNE": 22.5, "NE": 45, "ENE": 67.5,
    "E": 90, "ESE": 112.5, "SE": 135, "SSE": 157.5,
    "S": 180, "SSW": 202.5, "SW": 225, "WSW": 247.5,
    "W": 270, "WNW": 292.5, "NW": 315, "NNW": 337.5,
    "cv": None,
}

STATION_ENC = {
    "Aotizhongxin": 0, "Changping": 1, "Dingling": 2, "Dongsi": 3,
    "Guanyuan": 4, "Gucheng": 5, "Huairou": 6, "Nongzhanguan": 7,
    "Shunyi": 8, "Tiantan": 9, "Wanliu": 10, "Wanshouxigong": 11,
}

app = FastAPI(
    title="Beijing PM2.5 Prediction API",
    description=(
        "Predicts hourly PM2.5 concentration (µg/m³) using LightGBM trained on "
        "Beijing Multi-Site Air-Quality Data (2013–2016). "
        "Requires last-known PM2.5 lag values (1h, 2h, 3h, 24h ago) and rolling means."
    ),
    version="1.0.0",
)

_model = None
_feature_names = None


def _load():
    global _model, _feature_names
    if _model is None:
        model_path = os.getenv("MODEL_PATH", "models/lgbm_final.joblib")
        features_path = os.getenv("FEATURES_PATH", "models/feature_names.joblib")
        _model = joblib.load(model_path)
        _feature_names = joblib.load(features_path)
    return _model, _feature_names


class PredictionRequest(BaseModel):
    year: int = Field(..., example=2017, ge=2010, le=2030)
    month: int = Field(..., example=3, ge=1, le=12)
    day: int = Field(..., example=15, ge=1, le=31)
    hour: int = Field(..., example=14, ge=0, le=23)
    station: str = Field(..., example="Aotizhongxin", description=f"One of: {list(STATION_ENC)}")
    DEWP: float = Field(..., example=-5.0, description="Dew point (°C)")
    TEMP: float = Field(..., example=12.0, description="Temperature (°C)")
    PRES: float = Field(..., example=1015.0, description="Pressure (hPa)")
    WSPM: float = Field(..., example=2.5, description="Wind speed (m/s)")
    wd: str = Field(..., example="NE", description=f"Wind direction. One of: {list(WIND_DIR_MAP)}")
    SO2: float = Field(..., example=15.0, description="SO2 (µg/m³)")
    NO2: float = Field(..., example=40.0, description="NO2 (µg/m³)")
    CO: float = Field(..., example=800.0, description="CO (µg/m³)")
    O3: float = Field(..., example=60.0, description="O3 (µg/m³)")
    PM10: float = Field(..., example=80.0, description="PM10 (µg/m³)")
    RAIN: float = Field(0.0, example=0.0, description="Precipitation (mm)")
    PM25_lag1: float = Field(..., example=45.0, description="PM2.5 measured 1h ago (µg/m³)")
    PM25_lag2: float = Field(..., example=42.0, description="PM2.5 measured 2h ago (µg/m³)")
    PM25_lag3: float = Field(..., example=40.0, description="PM2.5 measured 3h ago (µg/m³)")
    PM25_lag24: float = Field(..., example=35.0, description="PM2.5 measured 24h ago (µg/m³)")
    PM25_roll3: float = Field(..., example=42.3, description="PM2.5 rolling mean over last 3h (µg/m³)")
    PM25_roll24: float = Field(..., example=38.5, description="PM2.5 rolling mean over last 24h (µg/m³)")


class PredictionResponse(BaseModel):
    predicted_pm25: float
    station: str
    unit: str = "µg/m³"


@app.get("/health", tags=["service"])
def health():
    """Liveness check."""
    return {"status": "ok"}


@app.get("/stations", tags=["service"])
def stations():
    """List valid station names and their encoded IDs."""
    return {"stations": STATION_ENC}


@app.get("/features", tags=["service"])
def features():
    """Return ordered list of feature names expected by the model."""
    _, feature_names = _load()
    return {"feature_names": feature_names, "count": len(feature_names)}


@app.post("/predict", response_model=PredictionResponse, tags=["prediction"])
def predict(req: PredictionRequest):
    """
    Predict next-hour PM2.5 concentration.

    Provide current meteorological/chemical measurements and the recent PM2.5
    history (lag values). Returns predicted PM2.5 in µg/m³.
    """
    model, feature_names = _load()

    if req.station not in STATION_ENC:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown station '{req.station}'. Valid: {list(STATION_ENC)}",
        )
    if req.wd not in WIND_DIR_MAP:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown wind direction '{req.wd}'. Valid: {list(WIND_DIR_MAP)}",
        )

    wd_deg = WIND_DIR_MAP[req.wd]
    if wd_deg is None:
        wd_sin, wd_cos = 0.0, 0.0
    else:
        rad = np.deg2rad(wd_deg)
        wd_sin, wd_cos = float(np.sin(rad)), float(np.cos(rad))

    hour_sin = float(np.sin(2 * np.pi * req.hour / 24))
    hour_cos = float(np.cos(2 * np.pi * req.hour / 24))
    month_sin = float(np.sin(2 * np.pi * req.month / 12))
    month_cos = float(np.cos(2 * np.pi * req.month / 12))
    day_of_week = datetime.date(req.year, req.month, req.day).weekday()
    is_weekend = int(day_of_week >= 5)
    season = int((req.month % 12) // 3)

    O3_to_NO2 = req.O3 / (req.NO2 + 1)
    wind_u = req.WSPM * wd_sin
    wind_v = req.WSPM * wd_cos
    # PM2.5 not yet known → use lag1 as best available proxy
    pm10_pm25_ratio = req.PM10 / (req.PM25_lag1 + 1)
    temp_dewp_diff = req.TEMP - req.DEWP

    row = {
        "No": 0,
        "PM10": req.PM10,
        "SO2": req.SO2,
        "NO2": req.NO2,
        "CO": req.CO,
        "O3": req.O3,
        "TEMP": req.TEMP,
        "PRES": req.PRES,
        "DEWP": req.DEWP,
        "RAIN": req.RAIN,
        "WSPM": req.WSPM,
        "station_enc": STATION_ENC[req.station],
        "wd_sin": wd_sin,
        "wd_cos": wd_cos,
        "hour_sin": hour_sin,
        "hour_cos": hour_cos,
        "month_sin": month_sin,
        "month_cos": month_cos,
        "day_of_week": day_of_week,
        "is_weekend": is_weekend,
        "PM2.5_lag1": req.PM25_lag1,
        "PM2.5_lag2": req.PM25_lag2,
        "PM2.5_lag3": req.PM25_lag3,
        "PM2.5_lag24": req.PM25_lag24,
        "PM2.5_roll3": req.PM25_roll3,
        "PM2.5_roll24": req.PM25_roll24,
        "O3_to_NO2": O3_to_NO2,
        "wind_u": wind_u,
        "wind_v": wind_v,
        "pm10_pm25_ratio": pm10_pm25_ratio,
        "temp_dewp_diff": temp_dewp_diff,
        "season": season,
    }

    try:
        x = np.array([[row[f] for f in feature_names]], dtype=float)
    except KeyError as exc:
        raise HTTPException(status_code=500, detail=f"Feature mismatch: {exc}") from exc

    prediction = max(0.0, float(model.predict(x)[0]))
    return PredictionResponse(predicted_pm25=round(prediction, 2), station=req.station)
