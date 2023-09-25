""" This is a dummy example to show how to import code from src/ for testing"""
import os
import sys
import time
sys.path.append(os.path.join(os.path.dirname(__file__),'../../'))
from src.module_1.module_1_meteo_api import call_api
from src.module_1.module_1_meteo_api import get_data_meteo_api
from src.module_1.module_1_meteo_api import data_parse
from src.module_1.module_1_meteo_api import conf_interval

def mean_test(df):
    df2 = df.copy()
    df2['Avg'] = df2.mean(numeric_only=True, axis=1)
    return df2['Avg'].mean(), df2['Avg'].std()

def test_main():
    COORDINATES = {
    "Madrid": {"latitude": 40.416775, "longitude": -3.703790}
    }
    start_date='2022-01-01'
    end_date='2022-12-31'
    VARIABLES = "temperature_2m_mean,precipitation_sum,soil_moisture_0_to_10cm_mean"
    city_data = get_data_meteo_api(
        coordinates=COORDINATES['Madrid'],
        start_date=start_date,
        end_date=end_date,
        vars=VARIABLES,
    )
    viariable_data = data_parse(df=city_data, variables=VARIABLES)
    test_mean = mean_test(city_data)
    variable_intervals = conf_interval(df=viariable_data)
    print(viariable_data.head())
    print(variable_intervals.head())
    print(test_mean)

if __name__ == "__main__":
    start_time = time.time()
    test_main()
    end_time = time.time()
    print(f"Timepo de ejecutción: {end_time-start_time}s")