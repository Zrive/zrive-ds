import openmeteo_requests

import requests_cache
import pandas as pd
import matplotlib.pyplot as plt
import json
import os

from retry_requests import retry
import requests
import time

OUTPUT_DIR = 'outputs'


# Setup the Open-Meteo API client with cache and retry on error
cache_session = requests_cache.CachedSession('.cache', expire_after = 3600)
retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
openmeteo = openmeteo_requests.Client(session = retry_session)

API_URL = "https://climate-api.open-meteo.com/v1/climate"

COORDINATES = {
    "Madrid": {"latitude": 40.416775, "longitude": -3.703790},
    "London": {"latitude": 51.507351, "longitude": -0.127758},
    "Rio": {"latitude": -22.906847, "longitude": -43.172896},
}

VARIABLES = "temperature_2m_mean, precipitation_sum, soil_moisture_0_to_10cm_mean"


""" Given the response to the API call, prints the according error. """
def handle_api_error(response):
    if response.status_code == 401:
        print("Invalid API key. Check credentials.")
    
    elif response.status_code == 404:
        print("Location not found.")
    
    else:
        print("API error, status code: ", response.status_code)
        print("API error, reason: ", response.reason)



""" Given a set of parameters, calls the Meteo API. If the call is successful, it returns the corresponding json;
otherwise, it returns None, handling the corresponding error."""
def call_API(params):
    # call the Meteo API

    try:
        response = requests.get(API_URL, params)
    
        if response.status_code == 200:
            print("API request has been successful")
            return response.json()
        
        elif response.status_code == 429:
            print("Rate limit exceeded. Please try again later.")
            return None
        else:
            handle_api_error(response) 
            return None
        
    except:
        print(f"API Request failed: {requests.RequestException}")
        return None
    

""" Given a city, defines the parameters and calls the API. 
If the call is successful, it returns the corresponding daily dataframe; otherwise, it tries reading the
corresponding csv file."""
def get_data_meteo_api(city):

    latitude = COORDINATES[city]['latitude']
    longitude = COORDINATES[city]['longitude']

    params = {
	"latitude": latitude,
	"longitude": longitude,
	"start_date": "1950-01-01",   # from 1950 to 2050
	"end_date": "2050-12-31",
	"models": "EC_Earth3P_HR", # One of the models with all weather variables available
	"daily": ["temperature_2m_mean", "precipitation_sum", "soil_moisture_0_to_10cm_mean"] # Data we want to extract
    }

    # Call the API
    responses = call_API(params)

    file_path = os.path.join(OUTPUT_DIR, f'{city}_daily_data.csv')

    if responses:     
        daily_json = get_daily_data(responses, city)
        # Transform to dataframe
        daily_df = pd.read_json(json.dumps(daily_json))
        
        # Save it in a csv only if there is not already one
        if not os.path.exists(file_path):
            daily_df.to_csv(file_path, index=False)
        
        return daily_df
    
    else:
        try:
            print("Failed to get API data. Loading data from CSV.")
            return pd.read_csv(file_path)
        
        except FileNotFoundError:
            print("No csv file found.")
            return None
    

""" Given the response from the API and the city, gets the returned time series and saves them in a dataframe. 
The dataframe is then saved in a csv ({city}_daily_data.csv). """
def get_daily_data(response, city):
    daily_data = response.get('data', {})
    
    date = daily_data.get('time', [])
    daily_temperature_2m_mean = daily_data.get('temperature_2m_mean', [])
    daily_precipitation_sum = daily_data.get('precipitation_sum', [])
    daily_soil_moisture_0_to_10cm_mean = daily_data.get('soil_moisture_0_to_10cm_mean', [])
    
    daily_df = daily_df = pd.DataFrame({
        "date": date,
        "temperature_2m_mean": daily_temperature_2m_mean,
        "precipitation_sum": daily_precipitation_sum,
        "soil_moisture_0_to_10cm_mean": daily_soil_moisture_0_to_10cm_mean
    })

    # Save it in a csv
    daily_df.to_csv(f'{city}_daily_data.csv', index=True)

    return daily_df
    

""" Given a daily dataframe, transforms it into mothly data. """
def reduce_temporal_resolution(daily_df, city):
     
    daily_df['date'] = pd.to_datetime(daily_df['date'])
    daily_df.set_index('date', inplace=True)  # Set 'date' as the index

    # Group by month and calculate the mean
    monthly_data = daily_df.groupby(pd.Grouper(freq='M')).mean()

    print(monthly_data.head())
    return monthly_data
    



""" Computes the mean and deviation for the three time series present in the monthly dataframe. 
The result is saved in a csv: {city}_statistics.csv """
def get_statistics(monthly_df, city):
    
    city_statistics = {}

    # Get all the variable's mean and std
    temperature_mean = monthly_df['temperature_2m_mean'].mean()
    temperature_dev = monthly_df['temperature_2m_mean'].std()

    precipitation_mean = monthly_df['precipitation_sum'].mean()
    precipitation_dev = monthly_df['precipitation_sum'].std()

    soil_mean = monthly_df['soil_moisture_0_to_10cm_mean'].mean()
    soil_dev = monthly_df['soil_moisture_0_to_10cm_mean'].std()

    # Save data in the dictionary
    city_statistics = {
        'Statistic': ['Mean', 'Deviation'],
        'Temperature': [temperature_mean, temperature_dev],
        'Precipitation': [precipitation_mean, precipitation_dev],
        'Soil Moisture': [soil_mean, soil_dev],
    }

    statistics_df = pd.DataFrame(city_statistics)
    file_path = os.path.join(OUTPUT_DIR, f"{city}_statistics.csv")
    statistics_df.to_csv(file_path, index=False)
    return city_statistics



""" Given the monthly data and the city, plots the three time series."""
def plot_data(data, city):
    if 'date' not in data.columns:
        data = data.reset_index()

    # Create a figure with 3 subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 15), sharex=True)

    # Temperature plot
    ax1.plot(data["date"], data['temperature_2m_mean'], color='tab:blue')
    ax1.set_ylabel('Temperature (°C)')
    ax1.set_title('Monthly Temperature')
    ax1.grid(True)

    # Precipitation plot
    ax2.plot(data["date"], data['precipitation_sum'], color='tab:green')
    ax2.set_ylabel('Precipitation (mm)')
    ax2.set_title('Monthly Precipitation Sum')
    ax2.grid(True)

    # Soil Moisture plot
    ax3.plot(data["date"], data['soil_moisture_0_to_10cm_mean'], color='tab:orange')
    ax3.set_ylabel('Soil Moisture (0-10 cm) (%)')
    ax3.set_title('Monthly Average Soil Moisture')
    ax3.grid(True)

    # Set the x-axis label only on the last subplot
    ax3.set_xlabel('Month')

    # Adjust layout to prevent overlap
    plt.tight_layout()
    file_path = os.path.join(OUTPUT_DIR, f'{city}_climate_data.png')
    plt.savefig(file_path) 



def main():
    # Iterate through all the available cities
    for city in COORDINATES:
        print("city: ", city)
        
        daily_data = get_data_meteo_api(city)
        print(daily_data.head())
        
        if daily_data is not None:
            print("Getting monthly data")
            monthly_data = reduce_temporal_resolution(daily_data, city)

            print("Plotting data")
            plot_data(monthly_data, city)
            #raise NotImplementedError

            print("Get statistics")
            stats = get_statistics(monthly_data, city)
            print(stats)

        
        else:
            print("Failed to get API data and csv data")


if __name__ == "__main__":
    main()