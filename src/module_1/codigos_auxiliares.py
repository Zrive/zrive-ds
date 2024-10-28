
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






# Representacion de las variables

# 1a version: Matplotlib (sencillito)

import matplotlib.pyplot as plt

pd_data_graph = pd_data[VARIABLES].resample('MS').mean() # poner 'M' si se prefiere que índice sea el último día del mes

plt.figure(figsize=(9, 3))

plt.plot(pd_data_graph)
plt.title('Madrid')
plt.grid()

plt.show()






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