from tests.module_1.test_meteo_api import (
  schema_validation, VARIABLES
 )

def test_schema_validation ():
    daily = {
        "time":["2020-01-01", "2020-01-02"],
        "temperature_2m_mean": [10, 12],
        "precipitation_sum": [0,5],
        "wind_speed_10m_max": [20, 25]
    }

    schema_validation(daily, VARIABLES)
    assert True