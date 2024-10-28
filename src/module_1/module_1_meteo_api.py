

API_URL = "<https://archive-api.open-meteo.com/v1/archive?"

COORDINATES = {
    "Madrid": {"latitude": 40.416775, "longitude": -3.703790},
    "London": {"latitude": 51.507351, "longitude": -0.127758},
    "Rio": {"latitude": -22.906847, "longitude": -43.172896},
}

VARIABLES = ["temperature_2m_mean", "precipitation_sum", "wind_speed_10m_max"]

def main():
    raise NotImplementedError

if __name__ == "__main__":
    main()


start_date = '2010-01-01'
end_date   = '2020-12-31'

RESOLUTION ='daily' # L

# Función que crea la URL de la API

# Por cada ciudad se hace una llamada
# Para recolectar datos de varias ciudades se hacen varias llamadas
# Como la entrada es el nombre de la ciudad, puedo recorrer en un bucle el diccionario de coordenadas
# que se quieren conocer.

def get_data_meteo_api(nombre_ciudad, start_date, end_date, lista_variables, resolution):
    '''
    resolution: posibles valores 'daily' o 'hourly'.
    '''
    import requests
    import json

    nombre_ciudad = nombre_ciudad.capitalize()
    
    # Tomar variables y pasarlas a "formato url"
    latitude_url   = 'latitude=' + str(COORDINATES[nombre_ciudad]["latitude"])
    longitude_url  = 'longitude='+ str(COORDINATES[nombre_ciudad]["longitude"])
    start_date_url = 'start_date=' + start_date
    end_date_url   = 'end_date=' + end_date
    resolution_url = resolution
    
    # variables_url
    variables_url = ''
    for variable in lista_variables:
        variables_url+=variable
        variables_url+=','
    variables_url = variables_url[:-1] #Elimino la última coma residual

    # Construir URL
    API_url = 'https://archive-api.open-meteo.com/v1/archive?'+latitude_url+'&'+longitude_url+'&'+start_date_url+'&'+end_date_url+'&'+resolution+'='+variables_url

    # Llamada a la API (construir función)
    res = requests.get(API_url)

    return res.json()


def print_dict_schema(data, indent=0):
    """
    Función para imprimir la estructura de un diccionario (keys y sus tipos).
    :param data: Diccionario a inspeccionar
    :param indent: Nivel de indentación para el formato (usado internamente para la recursión)
    """
    # Si el dato es un diccionario
    if isinstance(data, dict):
        for key, value in data.items():
            print(' ' * indent + f"Key: '{key}' | Tipo: {type(value).__name__}")
            # Si el valor es otro diccionario, llamamos recursivamente para descomponerlo
            if isinstance(value, dict):
                print_dict_schema(value, indent + 4)  # Incrementa la indentación para los diccionarios anidados
            # Si el valor es una lista o tupla, revisamos cada elemento
            elif isinstance(value, (list, tuple)):
                if value:  # Si la lista no está vacía, mostramos el tipo del primer elemento
                    print(' ' * (indent + 4) + f"(Elementos de tipo: {type(value[0]).__name__})")
                else:
                    print(' ' * (indent + 4) + "(Lista vacía)")
    # Si el dato no es un diccionario
    else:
        print(' ' * indent + f"Valor: {data} | Tipo: {type(data).__name__}")





# 1 - Llamada a la API
# 2 - Pongo los datos que me interesan en un dataframe de Pandas
# 3 - Agrupo los datos en un dataframe de pandas

import pandas as pd

# Bulce para hacer las llamadas a la API

data_list = []

for ciudad, coordenadas in COORDINATES.items():

    API_data = get_data_meteo_api(ciudad, start_date, end_date, lista_variables, RESOLUTION)['daily']
    pd_data = pd.DataFrame(dict_madrid, index=dict_madrid['time'])
    pd_data.index = pd.to_datetime(pd_data.index)
    pd_data.drop('time', axis=1, inplace=True)
    pd_data['city'] = ciudad

    data_list.append(pd_data)



# Bucle para hacer la transformación de los datos

for data_ciudad in data_list:
    for variable in VARIABLES:
        nombre_ciudad = data_ciudad['city'].iloc[0]
        data_ciudad_variable =data_ciudad[variable].resample('MS').agg(['max', 'mean', 'min'])

        plt.figure()
        plt.plot(data_ciudad_variable.index, data_ciudad_variable['mean'])
        plt.fill_between(data_ciudad_variable.index, data_ciudad_variable['min'], data_ciudad_variable['max'], alpha=0.2 )
        plt.grid()
        plt.xlabel('Fecha (mes)')
        plt.ylabel(variable)
        plt.title(nombre_ciudad)
        
        plt.savefig(f"graficas/{nombre_ciudad}" + "_" + str(variable))



# Quiero sacar un único gráfico por ciudad y variable con datos máximos y mínimos


    


# Representacion de las variables

# 1a version: Matplotlib (sencillito)

import matplotlib.pyplot as plt

pd_data_graph = pd_data[VARIABLES].resample('MS').mean() # poner 'M' si se prefiere que índice sea el último día del mes

plt.figure(figsize=(9, 3))

plt.plot(pd_data_graph)
plt.title('Madrid')
plt.grid()

plt.show()


meteo_Madrid['daily_units']

# 'temperature_2m_mean': '°C',
# 'precipitation_sum': 'mm',
# 'wind_speed_10m_max': 'km/h'

# GRAFICA CON 3 EJES DISTINTOS

fig, ax = plt.subplots()
fig.subplots_adjust(right=0.75)

twin1 = ax.twinx()
twin2 = ax.twinx()

# Offset the right spine of twin2.  The ticks and label have already been
# placed on the right by twinx above.
twin2.spines.right.set_position(("axes", 1.2))

p1, = ax.plot(pd_data_graph.index, pd_data_graph['temperature_2m_mean'], "b-", label="Temperatura media a 2m")
p2, = twin1.plot(pd_data_graph.index, pd_data_graph['precipitation_sum'], "r-", label="Precipitaciones")
p3, = twin2.plot(pd_data_graph.index, pd_data_graph['wind_speed_10m_max'], "g-", label="Velocidad máxima del viento a 10m")

ax.set_xlabel("Fecha")
ax.set_ylabel("Temperatura media a 2m")
twin1.set_ylabel("Precipitaciones")
twin2.set_ylabel("Velocidad máxima del viento a 10m")

ax.yaxis.label.set_color(p1.get_color())
twin1.yaxis.label.set_color(p2.get_color())
twin2.yaxis.label.set_color(p3.get_color())

tkw = dict(size=4, width=1.5)
ax.tick_params(axis='y', colors=p1.get_color(), **tkw)
twin1.tick_params(axis='y', colors=p2.get_color(), **tkw)
twin2.tick_params(axis='y', colors=p3.get_color(), **tkw)
ax.tick_params(axis='x', **tkw)

ax.legend(handles=[p1, p2, p3])

plt.show()

# Inclinar las fechas que aparecen en el eje Y

# ¿hago 1 grafico por variable? ¿Poner 3 ejes distintos (con sus respectivas uds)?
# Como agrupo los datos?
# Idealmente el grafico es interactivo y muestra el valor exacto sobre el que se tiene el cursor

# He visto en la wiki de la API que representa

# VERSION 2: Que la función ponga el nombre de una ciudad, localización, ... 
# que con la API de Google Maps se encuentre esa localización
# que devuelva las variables de dicha localización