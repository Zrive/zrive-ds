import requests
import pandas as pd
import matplotlib.pyplot as plt
import time


#Base URL for Open Meteo API
API_URL = "https://archive-api.open-meteo.com/v1/archive"
#Coordinates of the cities
COORDINATES = {
    "Madrid": {"latitude": 40.42, "longitude": -3.70},
    "London": {"latitude": 51.51, "longitude": -0.13},
     "Rio": {"latitude": -22.91, "longitude": -43.17},
}

#Varibles to extract
VARIABLES = ["temperature_2m_mean", "precipitation_sum", "wind_speed_10m_max"]

#Function to call the API
def get_data_meteo_API(city, start_year, end_year):
   #Variables latitude and longitude to be able to then recover multiple cities with a for
   latitude = COORDINATES[city]["latitude"]
   longitude = COORDINATES[city]["longitude"]

   #Array to stock all the data that will be recover from the API
   all_data = []

   #For to pass through all the years in the selected interval
   for year in range(start_year, end_year):
      params = {
         "latitude" : latitude,
         "longitude" : longitude,
         "start_date" : f"{year}-01-01",
         "end_date" : f"{year}-12-31",
         "daily" : VARIABLES,
         "timezone" : "Europe/Madrid"

      }

      #Request to the API all the parameters wished
      response = requests.get(API_URL, params=params)

      if response.status_code == 200:
         data = response.json()
         all_data.append (pd.DataFrame(data["daily"],columns = ["time"]+ VARIABLES))
        
      
      else:
         print(f"Error fetching data for {city} in {year} also {start_year} and {end_year}. Status Code: {response.status_code}")
         print(f"Response: {response.text}")  # This might give more details on the error
         break
         #time.sleep(5) #Wait before retrying

    
   time.sleep(1) #Prevent API rate limiting

   return pd.concat(all_data).reset_index(drop=True)

#This function takes the data frame with all the weather data got from the API and reformate the column time to be sure that it is on the correct pandas format. This allow to use many pandas utilities
def process_data(df):
   df["time"] = pd.to_datetime(df["time"])
   return df



   # Function to plot the data
def plot_weather_data(df, city):
    fig, ax = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

    ax[0].plot(df["time"], df["temperature_2m_mean"], label="Temperature (C)")
    ax[0].set_title(f"Temperature in {city} (2010-2020)")
    ax[0].set_ylabel("Temp (°C)")
    
    ax[1].plot(df["time"], df["precipitation_sum"], label="Precipitation (mm)", color="blue")
    ax[1].set_title(f"Precipitation in {city} (2010-2020)")
    ax[1].set_ylabel("Precipitation (mm)")
    
    ax[2].plot(df["time"], df["wind_speed_10m_max"], label="Wind Speed (m/s)", color="green")
    ax[2].set_title(f"Wind Speed in {city} (2010-2020)")
    ax[2].set_ylabel("Wind Speed (m/s)")
    ax[2].set_xlabel("Year")
    
    plt.tight_layout()
    plt.show()

#Main function with the 3 cities that we want the data and a for going through them. Using the get_data_meteo_API function and process_data
def main():
    cities = ["Madrid", "London", "Rio"]
    
    for city in cities:
        print(f"Fetching data for {city}...")
        df = get_data_meteo_API(city, 2010, 2020)
        df = process_data(df)
        print(f"Plotting data for {city}...")
        plot_weather_data(df, city)



if __name__ == "__main__":
    main()

