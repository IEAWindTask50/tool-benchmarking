from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _build_openmeteo_like_header(latitude: float, longitude: float, elevation: float) -> str:
    keys = [
        "latitude",
        "longitude",
        "elevation",
        "utc_offset_seconds",
        "timezone",
        "timezone_abbreviation",
    ]
    vals = [latitude, longitude, elevation, 0, "GMT", "GMT"]
    return ",".join(str(k) for k in keys) + "\n" + ",".join(str(v) for v in vals) + "\n\n"


def _to_deg_c(temp_k: pd.Series) -> pd.Series:
    return temp_k.astype(float) - 273.15


def prepare_openmeteo_like_resource_files(base_dir: Path) -> tuple[Path, Path]:
    workspace_root = base_dir.parent.parent
    input_csv = workspace_root / "input_ts_France_good_wind.csv"
    out_dir = base_dir / "resource_data"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not input_csv.exists():
        raise FileNotFoundError(f"Expected shared weather input at {input_csv}")

    df = pd.read_csv(input_csv, index_col=0, parse_dates=True)

    required_cols = ["WS_1", "WS_100", "WD_1", "WD_100", "temp_air_1", "ghi", "dni", "dhi"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required weather columns: {missing}")

    # Match HyDesign behavior: enforce complete 365-day years by trimming trailing
    # records when needed (for leap-year hourly data, this drops the last 24 hours).
    t = pd.DatetimeIndex(df.index)
    if len(t) % 365 != 0:
        n_sel = len(t) - (len(t) % 365)
        df = df.iloc[:n_sel].copy()
        t = pd.DatetimeIndex(df.index)

    if len(t) != 8760:
        raise ValueError(f"Expected 8760 hourly records after preprocessing, got {len(t)}")

    # Wind resource file compatible with OpenMeteoHistoricalWindResource.load_data().
    wind_df = pd.DataFrame(
        {
            "time": t,
            "wind_speed_10m (m/s)": df["WS_1"].astype(float).values,
            "wind_speed_100m (m/s)": df["WS_100"].astype(float).values,
            "wind_direction_10m (deg)": df["WD_1"].astype(float).values,
            "wind_direction_100m (deg)": df["WD_100"].astype(float).values,
            "temperature_2m (degC)": _to_deg_c(df["temp_air_1"]).values,
            "surface_pressure (hPa)": np.full(len(df), 1013.25),
            "precipitation (mm/h)": np.zeros(len(df)),
            "relative_humidity_2m (unitless)": np.full(len(df), 0.7),
            "is_day (percent)": np.where(df["ghi"].astype(float).values > 0.0, 100.0, 0.0),
        }
    )

    # Solar resource file compatible with OpenMeteoHistoricalSolarResource.load_data().
    solar_df = pd.DataFrame(
        {
            "time": t,
            "wind_speed_10m (m/s)": df["WS_1"].astype(float).values,
            "wind_direction_10m (deg)": df["WD_1"].astype(float).values,
            "temperature_2m (C)": _to_deg_c(df["temp_air_1"]).values,
            "surface_pressure (hPa)": np.full(len(df), 1013.25),
            "relative_humidity_2m (percent)": np.full(len(df), 70.0),
            "shortwave_radiation (W/m**2)": df["ghi"].astype(float).values,
            "direct_normal_irradiance (W/m**2)": df["dni"].astype(float).values,
            "diffuse_radiation (W/m**2)": df["dhi"].astype(float).values,
            "dew_point_2m (C)": _to_deg_c(df["temp_air_1"]).values - 2.0,
            "snow_depth (m)": np.zeros(len(df)),
            "rain (mm)": np.zeros(len(df)),
            "albedo (percent)": np.full(len(df), 20.0),
        }
    )

    wind_out = out_dir / "france_shared_wind_openmeteo_like.csv"
    solar_out = out_dir / "france_shared_solar_openmeteo_like.csv"

    header = _build_openmeteo_like_header(latitude=48.744116, longitude=-0.864258, elevation=50.0)
    wind_out.write_text(header + wind_df.to_csv(index=False), encoding="utf-8")
    solar_out.write_text(header + solar_df.to_csv(index=False), encoding="utf-8")

    return wind_out, solar_out


if __name__ == "__main__":
    bdir = Path(__file__).resolve().parent
    wind_path, solar_path = prepare_openmeteo_like_resource_files(bdir)
    print("Prepared:", wind_path)
    print("Prepared:", solar_path)
