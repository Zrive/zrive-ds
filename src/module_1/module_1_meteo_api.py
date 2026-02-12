# Script que se conecta a la API, y datos proporcionados con sus variables

API_URL = "https://archive-api.open-meteo.com/v1/archive?"
COORDINATES = {
    "Madrid": {"latitude": 40.416775, "longitude": -3.703790},
    "London": {"latitude": 51.507351, "longitude": -0.127758},
    "Rio": {"latitude": -22.906847, "longitude": -43.172896},
}
VARIABLES = ["temperature_2m_mean", "precipitation_sum", "wind_speed_10m_max"]


# =============================================================================
# IMPORTS CORREGIDOS
# =============================================================================
import requests
import time
import pandas as pd
import matplotlib.pyplot as plt

#######################################################################################
# Funciones implementadas para la tarea 1

# =============================================================================
# FUNCIÓN 1 (auxiliar): Hacer llamadas a APIs (generales) de forma segura
# =============================================================================
def call_api(url: str, params: dict, max_retries: int = 3):
    """
    Función auxiliar genérica para llamadas API con gestión de errores
    Args:
        url: La dirección de internet a la que llamar
        params: Los parámetros de la llamada (como filtros de búsqueda)
        max_retries: Número máximo de intentos si falla
    Returns:
        Los datos que nos devuelve la API, o None si falla
    """
    print(f"Llamando a la API: {url}")

    # Intentar varias veces por si falla
    for attempt in range(max_retries):
        try:
            # Hacer la llamada a internet
            response = requests.get(url, params=params, timeout=30)
            print(f"Intento {attempt + 1}: Código {response.status_code}")

            # Verificar si la llamada fue exitosa
            if response.status_code == 200: # 200 significa llamada correcta
                print("Llamada exitosa!")
                return response.json()  # Convertir la respuesta a datos usables
            elif response.status_code == 429:  # Demasiadas llamadas
                wait_time = 2 ** attempt  # Esperar 1, 2, 4, 8... segundos
                print(f"Demasiadas llamadas. Esperando {wait_time} segundos...")
                time.sleep(wait_time)
            else:
                print(f"Error HTTP {response.status_code}: {response.text}")
                return None

        except requests.exceptions.RequestException as e:
            print(f"Error de conexión (intento {attempt + 1}): {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Esperar antes de reintentar

    print("Todos los intentos fallaron")
    return None


# =============================================================================
# FUNCIÓN 2: Obtener datos meteorológicos de una ciudad
# =============================================================================
def get_data_meteo_api(city: str, start_date: str = "2010-01-01", end_date: str = "2020-12-31"):
    """
    Obtiene datos meteorológicos para una ciudad específica

    Args:
        city: Nombre de la ciudad ("Madrid", "London", "Rio")
        start_date: Fecha de inicio en formato "AAAA-MM-DD"
        end_date: Fecha de fin en formato "AAAA-MM-DD"

    Returns:
        DataFrame con los datos o None si falla
    """
    print(f"Obteniendo datos para {city}...")

    # Verificar que la ciudad existe en nuestro diccionario
    if city not in COORDINATES:
        print(f"Ciudad '{city}' no encontrada. Ciudades disponibles: {list(COORDINATES.keys())}")
        return None

    # Preparar los parámetros para la API (como los filtros de búsqueda)
    params = {
        "latitude": COORDINATES[city]["latitude"],  # Latitud de la ciudad
        "longitude": COORDINATES[city]["longitude"],  # Longitud de la ciudad
        "start_date": start_date,  # Fecha de inicio
        "end_date": end_date,  # Fecha de fin
        "daily": ",".join(VARIABLES),  # Variables que queremos
        "timezone": "auto"  # Zona horaria automática
    }

    # Llamar a la API usando nuestra función segura que creamos antes
    data = call_api(API_URL, params)

    # Verificar si obtuvimos datos correctamente
    if data and "daily" in data:
        print(f"Datos para {city} obtenidos correctamente")

        # Convertir los datos a una tabla (DataFrame) para poder trabajar con ellos
        df = pd.DataFrame(data["daily"])
        df["time"] = pd.to_datetime(df["time"])  # Convertir la columna de tiempo a formato fecha
        df["city"] = city  # Añadir una columna con el nombre de la ciudad

        print(f"Período: {df['time'].min().date()} a {df['time'].max().date()}")
        print(f"Registros obtenidos: {len(df)}")

        return df
    else:
        print(f"Error obteniendo datos para {city}")
        return None


# =============================================================================
# FUNCIÓN 3: Procesar y reducir datos meteorológicos
# =============================================================================
def process_weather_data(df, resample_freq: str = "M"):
    """
    Procesa y reduce la resolución temporal de los datos meteorológicos

    Args:
        df: DataFrame con datos meteorológicos diarios
        resample_freq: Frecuencia para reducir datos
                      "M" = mensual, "Q" = trimestral, "Y" = anual

    Returns:
        DataFrame procesado con menos filas pero más significativas
    """
    print(f"Procesando datos: reduciendo a frecuencia {resample_freq}...")

    # Hacer una copia para no modificar el original
    processed_df = df.copy()

    # Convertir la columna de tiempo a índice (para poder hacer resample)
    processed_df.set_index("time", inplace=True)

    # Reducir la resolución temporal (agrupar por meses, trimestres, etc.)
    if resample_freq == "M":  # Mensual
        result = processed_df.resample("M").agg({
            "temperature_2m_mean": "mean",  # Temperatura promedio del mes
            "precipitation_sum": "sum",  # Precipitación total del mes
            "wind_speed_10m_max": "max",  # Viento máximo del mes
            "city": "first"  # Mantener el nombre de la ciudad
        })
    elif resample_freq == "Q":  # Trimestral
        result = processed_df.resample("Q").agg({
            "temperature_2m_mean": "mean",  # Temperatura promedio del trimestre
            "precipitation_sum": "sum",  # Precipitación total del trimestre
            "wind_speed_10m_max": "max",  # Viento máximo del trimestre
            "city": "first"
        })
    else:  # Anual o por defecto mensual
        result = processed_df.resample("Y").agg({
            "temperature_2m_mean": "mean",  # Temperatura promedio del año
            "precipitation_sum": "sum",  # Precipitación total del año
            "wind_speed_10m_max": "max",  # Viento máximo del año
            "city": "first"
        })

    # Eliminar filas con datos faltantes y resetear el índice
    result = result.dropna().reset_index()

    print(f"Datos procesados: {len(df)} → {len(result)} registros")
    return result


# =============================================================================
# FUNCIÓN 4: Crear gráficos de evolución temporal
# =============================================================================
def plot_weather_evolution(all_cities_data, variable: str, resample_freq: str = "M"):
    """
    Crea gráficos de evolución temporal para una variable meteorológica

    Args:
        all_cities_data: Diccionario con DataFrames de todas las ciudades
        variable: Variable a graficar ("temperature_2m_mean", "precipitation_sum", "wind_speed_10m_max")
        resample_freq: Frecuencia de los datos ("M", "Q", "Y")
    """
    print(f"📈 Creando gráfico para {variable}...")

    # Configurar el tamaño del gráfico
    plt.figure(figsize=(12, 6))

    # Colores diferentes para cada ciudad
    colors = {"Madrid": "red", "London": "blue", "Rio": "green"}

    # Para cada ciudad, añadir su línea al gráfico
    for city_name, city_data in all_cities_data.items():
        if city_data is not None:
            # Procesar datos si no están procesados
            if len(city_data) > 365:  # Si tiene más de 1 año de datos diarios
                city_data = process_weather_data(city_data, resample_freq)

            # Crear la línea en el gráfico
            plt.plot(city_data["time"],
                     city_data[variable],
                     label=city_name,
                     color=colors.get(city_name, "black"),
                     linewidth=2,
                     marker='o',
                     markersize=4)

    # Configurar títulos y etiquetas según la variable
    variable_config = {
        "temperature_2m_mean": {
            "title": "Evolución de la Temperatura Media (2010-2020)",
            "ylabel": "Temperatura (°C)",
            "color": "darkred"
        },
        "precipitation_sum": {
            "title": "Evolución de la Precipitación Acumulada (2010-2020)",
            "ylabel": "Precipitación (mm)",
            "color": "darkblue"
        },
        "wind_speed_10m_max": {
            "title": "Evolución de la Velocidad Máxima del Viento (2010-2020)",
            "ylabel": "Velocidad del Viento (km/h)",
            "color": "darkgreen"
        }
    }

    config = variable_config.get(variable, {"title": variable, "ylabel": variable, "color": "black"})

    # Personalizar el gráfico
    plt.title(config["title"], fontsize=14, fontweight='bold', color=config["color"])
    plt.xlabel("Fecha", fontsize=12)
    plt.ylabel(config["ylabel"], fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)

    # Ajustar diseño para que no se corten las etiquetas
    plt.tight_layout()

    # Guardar el gráfico como imagen
    filename = f"grafico_{variable}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"   💾 Gráfico guardado como: {filename}")

    # Mostrar el gráfico
    plt.show()

    return filename


# =============================================================================
# FUNCIÓN AUXILIAR: Obtener datos completos para todas las ciudades
# =============================================================================
def get_complete_weather_data():
    """
    Obtiene datos completos para las 3 ciudades (2010-2020)
    """
    print("Obteniendo datos completos para todas las ciudades...")
    print("Esto puede tomar unos segundos...")

    cities_data = {}

    for city in COORDINATES.keys():
        print(f"Descargando datos de {city}...")
        city_data = get_data_meteo_api(city, "2010-01-01", "2020-12-31")

        if city_data is not None:
            # Procesar datos a resolución mensual
            processed_data = process_weather_data(city_data, "M")
            cities_data[city] = processed_data
            print(f"{city}: {len(processed_data)} registros mensuales")
        else:
            print(f"Error con {city}")
            cities_data[city] = None

    return cities_data


# =============================================================================
# FUNCIÓN MAIN
# =============================================================================
def main():
    """
    Función principal que orquesta todo el proceso
    """
    print("INICIANDO ANÁLISIS METEOROLÓGICO COMPLETO")
    print("=" * 60)

    # OBTENER DATOS COMPLETOS
    print("OBTENIENDO DATOS METEOROLÓGICOS (2010-2020)...")
    all_cities_data = get_complete_weather_data()

    # VERIFICAR QUE TENEMOS DATOS
    successful_cities = [city for city, data in all_cities_data.items() if data is not None]

    if not successful_cities:
        print("No se pudieron obtener datos de ninguna ciudad")
        return

    print(f"Datos obtenidos de: {', '.join(successful_cities)}")

    # CREAR GRÁFICOS
    print("CREANDO GRÁFICOS DE EVOLUCIÓN...")

    # Gráfico 1: Temperatura
    print("Creando gráfico de temperatura...")
    plot_weather_evolution(all_cities_data, "temperature_2m_mean", "M")

    # Gráfico 2: Precipitación
    print("Creando gráfico de precipitación...")
    plot_weather_evolution(all_cities_data, "precipitation_sum", "M")

    # Gráfico 3: Viento
    print("Creando gráfico de velocidad del viento...")
    plot_weather_evolution(all_cities_data, "wind_speed_10m_max", "M")

    # RESUMEN FINAL
    print("\n" + "=" * 60)
    print("ANÁLISIS COMPLETADO EXITOSAMENTE!")
    print("Se crearon 3 gráficos guardados como:")
    print("   - grafico_temperature_2m_mean.png")
    print("   - grafico_precipitation_sum.png")
    print("   - grafico_wind_speed_10m_max.png")
    print("PRÓXIMOS PASOS:")
    print("   1. Revisa los gráficos generados")
    print("   2. Añade los gráficos a tu Pull Request")
    print("   3. Escribe conclusiones sobre los datos")
    print("=" * 60)


if __name__ == "__main__":
    main()