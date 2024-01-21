""" This is a dummy example to show how to import code from src/ for testing"""

from src.module_1.module_1_meteo_api import get_data_meteo_api #Error: ModuleNotFoundError: No module named 'tests'

all_city_data = get_data_meteo_api()
for city, df in all_city_data.items():
  print(city)
  print(df)
