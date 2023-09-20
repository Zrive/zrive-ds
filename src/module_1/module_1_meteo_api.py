"""Import libraries"""
import requests
import logging

"""Define global variables"""
API_URL = "https://climate-api.open-meteo.com/v1/climate?"

COORDINATES = {
"Madrid": {"latitude": 40.416775, "longitude": -3.703790},
"London": {"latitude": 51.507351, "longitude": -0.127758},
"Rio": {"latitude": -22.906847, "longitude": -43.172896},
}

VARIABLES = "temperature_2m_mean,precipitation_sum,soil_moisture_0_to_10cm_mean"

MODELS = "CMCC_CM2_VHR4,FGOALS_f3_H,HiRAM_SIT_HR,MRI_AGCM3_2_S,EC_Earth3P_HR,MPI_ESM1_2_XR,NICAM16_8S"

"""Define auxiliar functions"""
def call_api(url):
#Need to add the verify=False as working from my corporate laptop. Tried to authenticate SSL by changing
#lots of different things but issue still arises. Disabled SSL but only works from the office
    try:
        #to-do: add the cool off
        response = requests.get(url, verify=False)
        if response.status_code == 200:
            return response
        else:
            logging.error(f"API request failed with Code: {response.status_code}")
            return None
    except requests.exceptions.RequestException as exception:
            logging.error(f"API request failed with Exception: {exception}")
            return None


def get_data_meteo_api(city, from_year, until_year):
    coordinates = COORDINATES.get(city)
    if coordinates is None:
        print("No data for the city")
        return None
    
    #Define local variables for lattitude and longitude
    lat = coordinates['latitude']
    long = coordinates['longitude']

    #Create the final URL
    url = f"{API_URL}latitude={lat}&longitude={long}&start_date={from_year}-01-01&end_date={until_year}-12-31&models={MODELS}&daily={VARIABLES}"
    url = "https://climate-api.open-meteo.com/v1/climate?latitude=52.52&longitude=13.41&start_date=1950-01-01&end_date=2050-12-31&models=CMCC_CM2_VHR4,FGOALS_f3_H,HiRAM_SIT_HR,MRI_AGCM3_2_S,EC_Earth3P_HR,MPI_ESM1_2_XR,NICAM16_8S&daily=temperature_2m_mean,precipitation_sum,soil_moisture_0_to_10cm_mean"
    data = call_api(url)
    return data


def process_data():
    pass
def plot_data():
    pass

def main():
    data = get_data_meteo_api("Madrid", 1950, 2050)
    print(data.headers)
    #print(data.text)
    #print(data.json())
    

if __name__ == "__main__":
    main()
    
