"""Import libraries"""
import requests
import logging
import time
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import pickle 

"""Define global variables"""
API_URL = "https://climate-api.open-meteo.com/v1/climate?"

COOL_OFF_TIME = 5
MAX_RETRY_ATTEMPTS = 3

COORDINATES = {
"Madrid": {"latitude": 40.416775, "longitude": -3.703790},
"London": {"latitude": 51.507351, "longitude": -0.127758},
"Rio": {"latitude": -22.906847, "longitude": -43.172896},
}

VARIABLES = "temperature_2m_mean,precipitation_sum,soil_moisture_0_to_10cm_mean"

#All models have data for the 3 variables except soil moisture, which is only provided by MRI_AGCM3 and EC_Earth
MODELS = "CMCC_CM2_VHR4,FGOALS_f3_H,HiRAM_SIT_HR,MRI_AGCM3_2_S,EC_Earth3P_HR,MPI_ESM1_2_XR,NICAM16_8S"

"""Define auxiliar functions"""
def call_api(url):
#Need to add the verify=False as working from my corporate laptop. Tried to authenticate SSL by changing
#lots of different things but issue still arises. Disabled SSL but only works from the office - not the best practice I know
    try:
        #to-do: add the cool off
        response = requests.get(url, verify=False)
        if response.status_code == 200:
            print("\nConnected Succesfully!")
            return response
        else:
            logging.error(f"\nAPI request failed with Code: {response.status_code}")
            return None
    except requests.exceptions.RequestException as exception:
            logging.error(f"\nAPI request failed with Exception: {exception}")
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

    #Define a num of max attempts for calling again
    retry_count=0

    while retry_count < MAX_RETRY_ATTEMPTS:
        data = call_api(url)
        if data:
            return data.json()
        else:
            print(f"Retrying after {COOL_OFF_TIME} seconds again...\n")
            time.sleep(COOL_OFF_TIME)
            retry_count += 1
    print("You reached the number of maximum attempts. Stopping the execution")
    sys.exit()

def clean_data(raw_data):
    cleaned_data = {}
    for key, value in raw_data.items():
        if value is not None:
            if isinstance(value, dict):
                cleaned_value = clean_data(value)  # Recursively clean sub-dictionaries starting again
                if cleaned_value:  # Check if the sub-dictionary is not empty after cleaning
                    cleaned_data[key] = cleaned_value

            elif isinstance(value, list): #do the same but for list within the dict
                # Clean each item in a list
                cleaned_list = [item for item in value if item is not None]
                if cleaned_list:
                    cleaned_data[key] = cleaned_list
            else:
                cleaned_data[key] = value
    return cleaned_data

def process_data(data):

    #FIX!
    # Create an empty dict to store processed data
    processed_data = {}
    daily_data = data['daily']

    for variable in daily_data.keys():
        for i in range(len(daily_data.keys())):

        # Extract the 'time' and 'value' data for the variable
            variable_data = daily_data[variable]
            time = variable_data[0]
            values = variable_data[i]

        # Calculate average and deviation for this variable
        #average = np.mean(values)
        #deviation = np.std(values)

        # Store the results in the processed_data dictionary
        #processed_data[variable] = {'average': average, 'deviation': deviation}

    return processed_data

def process_data_alternative(clean_data):
    #trying another  solution ???
    # Create an empty dict to store processed data
    processed_data = {}
    daily_data = clean_data['daily']
    print(clean_data)
    # Extract the 'daily' data
    #print(daily_data)
    #return daily_data
    # Loop through each variable defined in VARIABLES
    for index in range(len(daily_data['time'])):
        lista = []
        for variable in daily_data.keys():
            #print(variable)
            lista.append(daily_data[variable][index-1])

            #variable_data = daily_data[variable]
            
            
            processed_data[lista[0]] = lista[1:]
            # Extract the 'time' and 'value' data for the variable
            #print(processed_data)

            #value_data_point = data_point[variable]
            #time = variable_data[variable]
            #values = variable_data[variable_data[0]]
            #time.append(time_data_point)
            #values.append(value_data_point)

            # Calculate average and deviation for this variable
            #average = np.mean(values)
            #deviation = np.std(values)

            # Store the results in the processed_data dictionary
            #processed_data[variable] = {'average': average, 'deviation': deviation}
    print(processed_data)
        #return processed_data'''



def plot_data():
    pass

def main():
    data = get_data_meteo_api("Madrid", 1950, 2050)
    cleaned_data = clean_data(data)
    processed_data = process_data(cleaned_data)
  
    

if __name__ == "__main__":
    main()
    

