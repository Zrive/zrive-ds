import requests  # Importa el módulo requests para enviar solicitudes HTTP.
import json  # Importa el módulo json para codificar y decodificar datos en formato JSON.
import matplotlib.pyplot as plt  # Importa pyplot para la creación de gráficos.
import numpy as np  # Importa numpy para realizar cálculos numéricos y manejo de arrays.

# Un diccionario con las coordenadas geográficas de tres ciudades.
COORDINATES = {
    "Madrid": {"latitude": 40.416775, "longitude": -3.703790},
    "London": {"latitude": 51.507351, "longitude": -0.127758},
    "Rio": {"latitude": -22.906847, "longitude": -43.172896},
}
# Cadena de texto que contiene las variables climáticas que queremos obtener de la API.
VARIABLES = "temperature_2m_mean,precipitation_sum,soil_moisture_0_to_10cm_mean"
# La URL base de la API de pronóstico del clima.
API_URL = "https://api.open-meteo.com/v1/forecast"

# Función auxiliar para hacer llamadas a la API.
def make_api_request(url, params=None):
    """Realiza una solicitud a una API y maneja errores comunes como el límite de solicitudes."""
    try:
        response = requests.get(url, params=params)  # Realiza una solicitud GET a la API.
        response.raise_for_status()  # Si la respuesta es un código de error, lanza una excepción.
        return response.json()  # Convierte la respuesta a JSON y la retorna.
    except requests.exceptions.HTTPError as http_err:
        print(f"Error HTTP: {http_err}")  # Imprime el error HTTP si ocurre.
    except requests.exceptions.RequestException as req_err:
        print(f"Error en la solicitud: {req_err}")  # Imprime cualquier error de solicitud.
    except json.JSONDecodeError as json_err:
        print(f"Error al decodificar JSON: {json_err}")  # Imprime un error si JSON no puede decodificarse.
    return None  # Retorna None si hubo un error.

# Función para obtener datos climáticos de la API para una ciudad dada.
def get_data_meteo_api(city):
    # Verifica si la ciudad está en el diccionario de coordenadas.
    if city not in COORDINATES:
        print(f"La ciudad '{city}' no está definida en las coordenadas.")
        return None  # Si la ciudad no está definida, imprime un mensaje y retorna None.

    coords = COORDINATES[city]  # Obtiene las coordenadas de la ciudad del diccionario.

    # Parámetros para la solicitud a la API.
    params = {
        "latitude": coords["latitude"],  # Latitud de la ciudad.
        "longitude": coords["longitude"],  # Longitud de la ciudad.
        "start": 1950,  # Año de inicio para obtener datos.
        "end": 2050,  # Año final para obtener datos.
        "variables": VARIABLES,  # Variables climáticas solicitadas.
    }
    
    # Realiza la solicitud a la API utilizando la función auxiliar.
    data = make_api_request(API_URL, params=params)
    return data  # Retorna los datos obtenidos.

# Función principal que se ejecuta cuando el script es llamado directamente.
def main():
    cities = ["Madrid", "London", "Rio"]  # Lista de ciudades para las cuales obtener datos.

    # Itera sobre la lista de ciudades.
    for city in cities:
        print(f"Obteniendo datos para {city}...")  # Imprime que está obteniendo datos para la ciudad actual.
        data = get_data_meteo_api(city)  # Obtiene los datos de la ciudad actual.
        if data:
            print(f"Datos para {city}:\n{data}")  # Si los datos están disponibles, los imprime.

# Comprueba si este script es el módulo principal ejecutado y, de ser así, llama a la función main.
if __name__ == "__main__":
    main()
