""" This is a dummy example """

def main():
    raise NotImplementedError

if __name__ == "__main__":
    main()


# Funcion para gestionar APIs


# Madrid
latitude=40.4165
longitude=-3.7026

# Londres
latitude=51.507351
longitude=-0.127758

# Rio de Janeiro
latitude=-22.906847
longitude=-43.172896


start_date = '2010-01-01'
end_date   = '2020-12-31'

variables = ['temperature_2m_mean', 'precipitation_sum', 'wind_speed_10m_max']

def get_data_meteo_api(latitude, longitude, start_date, end_date, variables, resolution='daily'):
    '''
    resolution: posibles valores 'daily' o 'jourly'.
    '''
    import requests
    latitude  = str(latitude)
    longitude = str(longitude)

    # Construir URL
    API_url = 'https://archive-api.open-meteo.com/v1/archive?'+'latitude='+latitude+'&'+'longitude='+longitude+'&'+'start_date='+start_date+'&'+'end_date='+end_date+'&'+resolution+'='+variables
    
    # Llamada a la API (construir función)
    res = requests.get(API_url)
    
    return res.json()

get_data_meteo_api

# VERSION 2: Que la función ponga el nombre de una ciudad, localización, ... 
# que con la API de Google Maps se encuentre esa localización
# que devuelva las variables de dicha localización

#Desarrollo del ejercicio del modulo 1