import requests
import time
import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt

API_URL = "https://archive-api.open-meteo.com/v1/archive"

COORDINATES = {
    "Madrid": {"latitude": 40.416775, "longitude": -3.703790},
    "London": {"latitude": 51.507351, "longitude": -0.127758},
    "Rio": {"latitude": -22.906847, "longitude": -43.172896},
}

VARIABLES = ["temperature_2m_mean", "precipitation_sum", "wind_speed_10m_max"]

def llamada_api_con_rate_limit(url, **kwargs):
    intentos = 0
    max_intentos = 5
    kwargs.setdefault("timeout", 30)

    while intentos < max_intentos:
        try:
            response = requests.get(url, **kwargs)
            status = response.status_code


            if status == 429:
                retry_after = response.headers.get("Retry-After")
                if retry_after and retry_after.isdigit():
                    sleep_time = float(retry_after)
                else:
                    sleep_time = 2 ** intentos
                print(f"Límite de tasa: esperando {sleep_time}s...")
                time.sleep(sleep_time)
                intentos += 1
                continue


            if 500 <= status < 600:
                sleep_time = 2 ** intentos
                print(f"Server {status}. Reintento en {sleep_time}s...")
                time.sleep(sleep_time)
                intentos += 1
                continue


            response.raise_for_status()

            return response.json()

        except requests.exceptions.RequestException as e:
            intentos += 1
            sleep_time = 2 ** intentos
            print(f"Error de red: {e}. Reintento en {sleep_time}s...")
            time.sleep(sleep_time)

    print("Se ha alcanzado el máximo de reintentos.")
    return None

#SCHEMA VALIDATION

def schema_validation(daily: dict, expected_vars: list[str]) -> None:
    if "time" not in daily:
        raise ValueError("Falta 'time' en daily.")
    n = len(daily["time"])
    for v in expected_vars:
        if v not in daily:
            raise ValueError(f"Falta la variable '{v}' en daily.")
        if len(daily[v]) != n:
            raise ValueError(
                f"Longitud distinta en '{v}' ({len(daily[v])}) vs 'time' ({n})."
            )
    print("Schema validation OK")


#TIME SERIES
def time_series(daily:dict, city_name: str) -> pd.DataFrame:
    df = pd.DataFrame(daily)
    df["time"] = pd.to_datetime(df["time"])
    df["city"] = city_name
    df = df.sort_values("time").reset_index(drop=True)
    return df
#PASAR TIME A ANUAL

def annual_time (df_daily: pd.DataFrame) -> pd.DataFrame:
    agg_variables = {
        "temperature_2m_mean": "mean",
        "precipitation_sum": "sum",
        "wind_speed_10m_max" : "max"
    }

    df_idx = df_daily.set_index("time").sort_index()

    anual = (
        df_idx
        .groupby("city")
        .resample("YE")
        .agg(agg_variables)
        .reset_index()
    )

    anual["year"] = anual["time"].dt.year

    cols = ["year", "city"] + list(agg_variables.keys())
    anual = anual [cols].sort_values(["year", "city"]).reset_index(drop=True)
    return anual

#GRAFICOS
def plot_variables(
        df_year: pd.DataFrame,
        variable: str,
        ylabel: str
) -> None:
    
    table = (
        df_year
        .pivot(index="year", columns= "city", values=variable)
        .sort_index()
    )

    plt.figure(figsize=(8,4))

    for city in table.columns:
        plt.plot(table.index, table[city], marker="o", label=city)

    plt.xlabel("Año")
    plt.ylabel(ylabel or variable)
    titulo_legible = variable.replace("_", " ")
    plt.title(f"Evolución anual de {titulo_legible}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.show()

def plot_all_yearly(df_year: pd.DataFrame) -> None:
    plot_variables(df_year, "temperature_2m_mean", ylabel="ºC")
    plot_variables(df_year, "precipitation_sum",  ylabel="mm")
    plot_variables(df_year, "wind_speed_10m_max", ylabel="m/s")

#GRAFICO COMBINADO

def plot_combinado (df_year:pd.DataFrame) -> None:
    for city in df_year["city"].unique():
        df_city = df_year[df_year["city"] == city ]
        fig, ax1 = plt.subplots(figsize=(10,5))

        ax1.bar(
            df_city["year"], 
            df_city["temperature_2m_mean"], 
            color="skyblue", 
            alpha=0.7, 
            label="Temp media (ºC)"
        )
        ax1.set_xlabel("Año")
        ax1.set_ylabel("Temperatura media (ºC)", color="blue")
        ax1.tick_params(axis="y", labelcolor="blue")

        ax2 = ax1.twinx()
        ax2.plot(
            df_city["year"], 
            df_city["precipitation_sum"], 
            color="green", 
            marker="o", 
            label="Precipitación (mm)"
        )
        ax2.set_ylabel("Precipitación anual (mm)", color="green")
        ax2.tick_params(axis="y", labelcolor="green")

        plt.title(f"Temperatura y Precipitación anual - {city}")
        fig.tight_layout()
        plt.show()

#MAIN

def main():
    start = dt.date(2010, 1, 1)
    end   = dt.date(2020, 1, 1)

    frames = []

    for city, coords in COORDINATES.items():
        params = {
            "latitude":  coords["latitude"],
            "longitude": coords["longitude"],
            "start_date": start.isoformat(),
            "end_date":   end.isoformat(),
            "daily": ",".join(VARIABLES),
            "timezone": "UTC",
        }

        data = llamada_api_con_rate_limit(API_URL, params=params, timeout=30)
        if not data or "daily" not in data:
            print(f"{city}: no llegaron datos válidos")
            continue

        daily = data["daily"]
        try:
            schema_validation(daily, VARIABLES)
            print(f"{city}: OK ({len(daily['time'])} días)")
        except ValueError as e:
            print(f"{city}: error de esquema {e}")
            continue
        df_city = time_series(daily, city)
        frames.append(df_city)

        if not frames:
            print("No hay datos")
            return
            
    df_daily = pd.concat(frames, ignore_index=True)
    df_year = annual_time(df_daily)
    plot_combinado(df_year)   
    plot_all_yearly(df_year)

if __name__ == "__main__":
    main()