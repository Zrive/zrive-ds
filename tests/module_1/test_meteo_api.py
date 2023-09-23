""" This is a dummy example to show how to import code from src/ for testing"""


import os
import sys

dir_route = os.getcwd()
sys.path.append(dir_route)

from src.module_1.module_1_meteo_api import COORDINATES, get_data_meteo_api, api_request


def test_main():

    city = 'Rio'
    api_url = get_data_meteo_api(city, "1950-01-01", "2049-12-31")
    data_dict = api_request(api_url)
    
    assert city in COORDINATES.keys()
    assert isinstance(api_url, str)
    assert isinstance(data_dict, dict)
   

if __name__ == "__main__":
    test_main()
