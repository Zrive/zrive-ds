################################################
### Semana 1: Recoger datos API Meteorologia ###
################################################
import requests
import pandas as pd
import matplotlib as plt
#import backoff
#import time


#@backoff.on_exception(backoff.expo, request.error.RateLimitError)
def call_api(url, params=None):
    try:
        response = requests.get(url, params=params)
        response.raise_for_status() 
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error al hacer la solicitud a la API: {e}")
        return None

def get_data_meteo_api(city, API_URL):
    params = {"city": city}
    data = call_api(API_URL, params)
    if data:
        return data
    else:
        return None


## def data_parse():
#     #GETTING WEATHER FORECAST USING OPEN METEO API FOR EACH CITY
#     data_list=[]
#     for i in city.keys():
#         url_data='https://api.open-meteo.com/v1/forecast?latitude='+str(city[i][0])+'&longitude='+str(city[i][1])+'&hourly='+variable+'&timezone=GMT&start_date='+start_date+'&end_date='+end_date
#         response=requests.get(url_data).json() #get data response in json
#         df=pd.DataFrame(response['hourly']) #convert data to pandas dataframe
#         n_data=len(df)
#         lat,lon=[city[i][0]]*n_data,[city[i][1]]*n_data #generate lon,lat for dataframe
#         city_name=[i]*n_data
#         df['lat']=lat
#         df['lon']=lon
#         df['city']=city_name #add city name to dataframe
#         data_list.append(df)
#         time.sleep(10) #pause 5 seconds before getting a new data

    #JOIN ALL DATAFRAMES
    data_con=pd.concat(data_list,axis=0)
    data_con.head()

def main():

    URL = 'https://climate-api.open-meteo.com/v1/climate?latitude=52.52&longitude=13.41&start_date=1950-01-01&end_date=2050-12-31&models=CMCC_CM2_VHR4,FGOALS_f3_H,HiRAM_SIT_HR,MRI_AGCM3_2_S,EC_Earth3P_HR,MPI_ESM1_2_XR,NICAM16_8S&daily=temperature_2m_mean,precipitation_sum,soil_moisture_0_to_10cm_mean'
    COORDINATES = {
    "Madrid": {"latitude": 40.416775, "longitude": -3.703790},
    "London": {"latitude": 51.507351, "longitude": -0.127758},
    "Rio": {"latitude": -22.906847, "longitude": -43.172896},
    }
    start_date='1950-01-01'
    end_date='2023-08-30'
    VARIABLES = "temperature_2m_mean,precipitation_sum,soil_moisture_0_to_10cm_mean"


if __name__ == "__main__":
    result = call_api(url = "https://open-meteo.com/en/docs/climate-api")
    main()
