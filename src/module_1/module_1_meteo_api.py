import pandas as pd 
import requests 
import matplotlib.pyplot as plt
import time
import logging
from requests.models import Response
from datetime import datetime

API_URL = "https://climate-api.open-meteo.com/v1/climate?"
COORDINATES = {
"Madrid": {"latitude": 40.416775, "longitude": -3.703790},
"London": {"latitude": 51.507351, "longitude": -0.127758},
"Rio": {"latitude": -22.906847, "longitude": -43.172896},
}
VARIABLES = ["temperature_2m_mean","precipitation_sum","soil_moisture_0_to_10cm_mean"]
MODELS = ["CMCC_CM2_VHR4",
    "FGOALS_f3_H",
    "HiRAM_SIT_HR",
    "MRI_AGCM3_2_S",
    "EC_Earth3P_HR",
    "MPI_ESM1_2_XR",
    "NICAM16_8S",
]
MAX_CALL_ATTEMPTS = 50

# Logging configuration
logging.basicConfig(level=logging.INFO)

def validate_response(response: Response) -> bool:
    # Check if the response is a dictionary
    if not isinstance(response, dict):
        return False

    # Check if the required keys are in the response
    required_keys = ['latitude', 'longitude', 'generationtime_ms', 'timezone', 'timezone_abbreviation', 'daily', 'daily_units']
    if not all(key in response for key in required_keys):
        return False

    # Check if 'daily' is a dictionary
    if not isinstance(response['daily'], dict):
        return False

    # Check if 'daily_units' is a dictionary
    if not isinstance(response['daily_units'], dict):
        return False

    return True

def call_api (params: dict):
    """
    Makes an API call with given parameters, handling rate limits and other errors.
    :param params: Dictionary with the parameters to be passed to the API call.
    :return: The API response if successful, None otherwise.
    """
    call_attempts= 0 
    backoff_time = 1  
    while call_attempts < MAX_CALL_ATTEMPTS:
        try:
            response = requests.get(API_URL, params=params)
            if response.status_code == 200:
                if validate_response(response):
                    return response
            elif response.status_code == 429:
                # Use the Retry-After header to determine the cool-off time
                cool_off_time = response.headers.get("Retry-After",backoff_time)
                logging.info(f"Rate limit exceeded. Waiting for {cool_off_time} seconds.")
                logging.info(response.text)
                time.sleep(cool_off_time)
            else: 
                raise Exception(f"API request failed with status code {response.status_code}")
        except Exception as e:
            print(f"API request failed with exception {e}")
            break
        call_attempts += 1
        backoff_time *= 2 
    logging.info(f"Max number of call attempts reached. Returning empty result")
    return response
 
        
def get_data_meteo_api(city: str, start_date: str, end_date: str, variable: str) -> pd.DataFrame():
    """
    Fetches daily weather data for a specific city and variable from the Meteo API.
    :param city: The name of the city to fetch the data for. Must be a key in the COORDINATES dictionary.
    :param start_date: The start date for the data in the format 'YYYY-MM-DD'.
    :param end_date: The end date for the data in the format 'YYYY-MM-DD'.
    :param variable: The weather variable to fetch the data for.
    :return: A DataFrame containing the fetched data.
    :raises ValueError: If the specified city is not in the COORDINATES dictionary.
    """
    if city not in COORDINATES.keys():
        raise ValueError("City not available")
    params = {
        "latitude": COORDINATES[city]["latitude"],
        "longitude": COORDINATES[city]["longitude"],
        "start_date": start_date,
        "end_date": end_date,  
        "models":  MODELS,
        "daily": variable
    }
    data = call_api(params)
    if data is None:
        return None
    daily_data = data.json()["daily"]
    # Convert the daily data into a DataFrame
    df = pd.DataFrame(daily_data)
    # Add the city name as a column
    df['city'] = city
    df['time'] = pd.to_datetime(df['time'])
    return df

def calculate_mean_std(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates the mean and standard deviation for each prefix in the DataFrame's columns.
    :param df: The DataFrame to calculate the mean and standard deviation for.
    :return: A new DataFrame with the mean and standard deviation for each prefix in the original DataFrame's columns.
    """
    df = df.copy()
    # Extracting prefixes (assuming they are separated by underscores)
    prefixes = set(col.split('_')[0] for col in df.columns if '_' in col)
    for prefix in prefixes:
        relevant_cols = [col for col in df.columns if col.startswith(prefix)]
        mean_values = df[relevant_cols].mean(axis=1)
        df[prefix + "_mean"] = mean_values
        std_values = df[relevant_cols].std(axis=1)
        df[prefix + '_std'] = std_values
        df = df.drop(relevant_cols, axis=1)
    return df

def plot_data (df: pd.DataFrame, variable: str) -> None:
    """
    Plots the annual mean and dispersion of a specified variable for each city in the DataFrame.

    :param df: The DataFrame containing the data to plot.
    :param variable: The variable to plot.
    :return: None
    """
    plt.figure(figsize=(10, 6))
    
    for city in COORDINATES.keys():
        city_data = df[df['city'] == city]
        plt.errorbar(city_data['year'], city_data[f'{variable}_mean'], 
                     yerr=city_data[f'{variable}_std'], label=city, fmt='-o', capsize=5)
    
    plt.title(f'Annual Mean and Dispersion of {variable.capitalize()}')
    plt.xlabel('Year')
    plt.ylabel(f'{variable.capitalize()}')
    plt.legend()
    plt.show()
    
def main():
    list_data_cities = []
    for city in COORDINATES.keys():
        logging.info(f"Getting data for {city}")
        list_data_variables= []
        for variable in VARIABLES:
            logging.info(f"Getting data for {variable} in {city}")
            data = get_data_meteo_api(city, "1950-01-01", "1960-12-31", variable)
            list_data_variables.append(data)
        combined_df = list_data_variables[0]
        for df in list_data_variables[1:]:
            combined_df = pd.merge(combined_df,df, on=['time', 'city'])
        
        data_with_calculation = calculate_mean_std(combined_df)
        list_data_cities.append(data_with_calculation)
              
    df_total = pd.concat(list_data_cities)
    
    df_total['time'] = pd.to_datetime(df_total['time'])
    df_total['year'] = df_total['time'].dt.year
    
    # Annualize the data by taking the mean of the daily values for a given year
    annual_data = df_total.groupby(['city', 'year']).mean().reset_index()
    
    # Plot the data for each variable dynamically
    for variable in annual_data.columns:
        if variable not in ['city', 'year']:
            plot_data(annual_data, variable)
    

if __name__ == "__main__":
    main()
