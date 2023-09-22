""" This is a dummy example to show how to import code from src/ for testing"""
import requests
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__),'../../'))
from src.module_1.module_1_meteo_api import call_api
from src.module_1.module_1_meteo_api import get_data_meteo_api


#api_result = call_api(url = "https://climate-api.open-meteo.com/v1/climate?latitude=52.52&longitude=13.41&start_date=1950-01-01&end_date=2050-12-31&models=CMCC_CM2_VHR4,FGOALS_f3_H,HiRAM_SIT_HR,MRI_AGCM3_2_S,EC_Earth3P_HR,MPI_ESM1_2_XR,NICAM16_8S&daily=temperature_2m_mean,precipitation_sum,soil_moisture_0_to_10cm_mean")
#print(api_result)

### def test_main():
#     raise NotImplementedError

COORDINAT = {
    "Madrid": {"latitude": 40.416775, "longitude": -3.703790},
    "London": {"latitude": 51.507351, "longitude": -0.127758},
    "Rio": {"latitude": -22.906847, "longitude": -43.172896},
    }
print(COORDINAT["Madrid"]["latitude"])