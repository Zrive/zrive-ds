""" This is a dummy example to show how to import code from src/ for testing"""
import requests
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__),'../../'))
from src.module_1.module_1_meteo_api import call_api
from src.module_1.module_1_meteo_api import get_data_meteo_api

#Variables 
COORDINATES = {
"Madrid": {"latitude": 40.416775, "longitude": -3.703790},
"London": {"latitude": 51.507351, "longitude": -0.127758},
"Rio": {"latitude": -22.906847, "longitude": -43.172896},
}
start_date='1950-01-01'
end_date='2023-08-30'
VARIABLES = "temperature_2m_mean,precipitation_sum,soil_moisture_0_to_10cm_mean"

api_result = call_api(url= f"https://climate-api.open-meteo.com/v1/climate?latitude={COORDINATES['Madrid']['latitude']}&longitude={COORDINATES['Madrid']['longitude']}&start_date={start_date}&end_date={end_date}&models=CMCC_CM2_VHR4,FGOALS_f3_H,HiRAM_SIT_HR,MRI_AGCM3_2_S,EC_Earth3P_HR,MPI_ESM1_2_XR,NICAM16_8S&daily={VARIABLES}")
print(api_result)

### def test_main():
#     raise NotImplementedError