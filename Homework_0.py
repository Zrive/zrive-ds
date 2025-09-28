# meteo_homework.py

import os
import time
from typing import Dict, List, Any
import requests
import pandas as pd
import matplotlib.pyplot as plt

API_URL = "https://archive-api.open-meteo.com/v1/archive"

COORDINATES: Dict[str, Dict[str, float]] = {
    "Madrid": {"latitude": 40.416775, "longitude": -3.703790},
    "London": {"latitude": 51.507351, "longitude": -0.127758},
    "Rio": {"latitude": -22.906847, "longitude": -43.172896},
}

VARIABLES: List[str] = [
    "temperature_2m_mean",
    "precipitation_sum",
    "wind_speed_10m_max",
]

# ------------------- API -------------------


def _request_with_retry(
    url: str, params: Dict[str, Any], retries: int = 5, backoff: float = 1.0
) -> Dict[str, Any]:
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            r = requests.get(url, params=params, timeout=30)
            if r.status_code == 429:  # rate limit
                time.sleep(backoff * attempt)
                continue
            r.raise_for_status()
            return r.json()
        except requests.RequestException as e:
            last_err = e
            time.sleep(backoff * attempt)
    raise RuntimeError(f"API request failed after {retries} retries: {last_err}")


def _validate_response(payload: Dict[str, Any], expected_vars: List[str]) -> None:
    if "daily" not in payload or "time" not in payload["daily"]:
        raise ValueError("Respuesta sin bloque 'daily' válido")
    n = len(payload["daily"]["time"])
    for v in expected_vars:
        if v not in payload["daily"]:
            raise ValueError(f"Falta variable en respuesta: {v}")
        if len(payload["daily"][v]) != n:
            raise ValueError(f"Longitud inconsistente para {v}")


def get_data_meteo_api(
    city: str,
    start_date: str,
    end_date: str,
    variables: List[str] = None,
    timezone: str = "UTC",
) -> pd.DataFrame:
    if variables is None:
        variables = VARIABLES
    if city not in COORDINATES:
        raise KeyError(f"Ciudad no soportada: {city}")
    coords = COORDINATES[city]
    params = {
        "latitude": coords["latitude"],
        "longitude": coords["longitude"],
        "start_date": start_date,
        "end_date": end_date,
        "daily": ",".join(variables),
        "timezone": timezone,
    }
    payload = _request_with_retry(API_URL, params)
    _validate_response(payload, variables)
    daily = payload["daily"]
    df = pd.DataFrame({"date": pd.to_datetime(daily["time"])})
    for v in variables:
        df[v] = daily[v]
    df.insert(1, "city", city)
    return df


# ------------------- Procesado -------------------

AGG = {
    "temperature_2m_mean": "mean",
    "precipitation_sum": "sum",
    "wind_speed_10m_max": "mean",
}


def to_monthly(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    value_cols = [c for c in df.columns if c in AGG]
    out = (
        df.set_index("date")
        .groupby("city")[value_cols]
        .resample("M")
        .agg(AGG)
        .reset_index()
    )
    return out


# ------------------- Plots -------------------


def plot_city_monthly(df_monthly: pd.DataFrame, city: str, outdir: str = "docs") -> str:
    os.makedirs(outdir, exist_ok=True)
    d = df_monthly[df_monthly["city"] == city].copy()
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    axes[0].plot(d["date"], d["temperature_2m_mean"])
    axes[0].set_ylabel("Temp °C")
    axes[0].set_title(f"{city} · temperatura media mensual")

    axes[1].plot(d["date"], d["precipitation_sum"])
    axes[1].set_ylabel("Precip mm")
    axes[1].set_title(f"{city} · precipitación mensual")

    axes[2].plot(d["date"], d["wind_speed_10m_max"])
    axes[2].set_ylabel("Viento m/s")
    axes[2].set_title(f"{city} · viento (media del máximo diario)")
    axes[2].set_xlabel("Fecha")

    fig.tight_layout()
    outpath = os.path.join(outdir, f"{city.lower()}_monthly.png")
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    return outpath


def plot_variable_comparison(
    df_monthly: pd.DataFrame, variable: str, outdir: str = "docs"
) -> str:
    os.makedirs(outdir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    for city in df_monthly["city"].unique():
        d = df_monthly[df_monthly["city"] == city]
        ax.plot(d["date"], d[variable], label=city)
    ax.set_title(f"Comparativa mensual: {variable}")
    ax.set_xlabel("Fecha")
    ax.set_ylabel(variable)
    ax.legend()
    fig.tight_layout()
    outpath = os.path.join(outdir, f"compare_{variable}.png")
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    return outpath


# ------------------- Main -------------------

START = "2010-01-01"
END = "2020-12-31"


def main():
    os.makedirs("data", exist_ok=True)
    dfs = []
    for city in COORDINATES.keys():
        df = get_data_meteo_api(city, START, END)
        df.to_csv(f"data/{city.lower()}_daily.csv", index=False)
        dfs.append(df)
    daily = pd.concat(dfs, ignore_index=True)
    monthly = to_monthly(daily)
    monthly.to_csv("data/monthly_all_cities.csv", index=False)

    for city in monthly["city"].unique():
        plot_city_monthly(monthly, city)
    for v in VARIABLES:
        plot_variable_comparison(monthly, v)


# ------------------- Tests rápidos -------------------


def _test_processing():
    data = {
        "date": pd.date_range("2020-01-01", periods=4, freq="D").tolist(),
        "city": ["X"] * 4,
        "temperature_2m_mean": [10, 12, 14, 16],
        "precipitation_sum": [1, 2, 0, 3],
        "wind_speed_10m_max": [5, 7, 6, 8],
    }
    df = pd.DataFrame(data)
    out = to_monthly(df)
    assert len(out) == 1
    row = out.iloc[0]
    assert abs(row["temperature_2m_mean"] - (10 + 12 + 14 + 16) / 4) < 1e-9
    assert row["precipitation_sum"] == 6
    assert abs(row["wind_speed_10m_max"] - (5 + 7 + 6 + 8) / 4) < 1e-9
    print("Test procesamiento OK")


if __name__ == "__main__":
    # Descomenta una u otra según necesites
    main() 
    #_test_processing()
