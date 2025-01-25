import time
import requests
import json
import pandas as pd
import matplotlib.pyplot as plt
import logging
from typing import Any, Dict, Optional, List

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
logger.addHandler(ch)

API_URL = "https://archive-api.open-meteo.com/v1/archive?"

COORDINATES = {
 "Madrid": {"latitude": 40.416775, "longitude": -3.703790},
 "London": {"latitude": 51.507351, "longitude": -0.127758},
 "Rio": {"latitude": -22.906847, "longitude": -43.172896},
}
VARIABLES = ["temperature_2m_mean", "precipitation_sum", "wind_speed_10m_max"]


def get_data_meteo_api(
        city:str,start:str,end:str
):
    
    lat = COORDINATES[city]['latitude']
    lon = COORDINATES[city]['longitude']
    
    headers = {}
    params = {
    "latitude": lat,
    "longitude": lon,
    "start_date": start,
    "end_date": end,
    "daily": ','.join(VARIABLES),
    }
    response = request_with_cooloff(API_URL, headers,params)
    return json.loads(response.content.decode('utf-8'))

def request_with_cooloff(
        url:str,
        headers:dict[str,Any],
        params:dict[str,Any],
        num_attempts:int = 5,
        payload:Optional[Dict[str,Any]] = None
) -> Dict[Any,Any]:
    
    cooloff = 1
    for call_count in range(num_attempts):
        try:
            if payload is None:
                response = requests.get(url,headers=headers,params=params)
                logger.info('API call was successful')
            else:
                response = requests.get(url,params=params,headers=headers,json=payload)
            response.raise_for_status()

        except requests.exceptions.ConnectionError as e:
            logger.info('API refused the connection')
            logger.warning(e)

            if call_count != (num_attempts-1):
                time.sleep(cooloff)
                cooloff *=2
                continue
            else:
                raise

        except requests.exceptions.HTTPError as e:
            logger.warning(e)
            if response.status_code == 404:
                raise
            
            logger.info(f'API return status code {response.status_code} cooloff at {cooloff}')

            if call_count != (num_attempts-1):
                time.sleep(cooloff)
                cooloff *=2
                continue
            else:
                raise

        return response

def compute_monthly_statistics(data: pd.DataFrame, variables: List[str]):
    # compute max, min and mean of each variable for each city
    
    # convert 'time' to datetime
    data['time'] = pd.to_datetime(data['time'])

    # group by city and month
    grouped = data.groupby([data['city'], data['time'].dt.to_period('M')])

    results = []
    for (city,month), group in grouped:
        monthly_stats = {'city': city, 'month': month.to_timestamp()}

        for variable in variables:
            monthly_stats[f'{variable}_max'] = group[variable].max()
            monthly_stats[f'{variable}_min'] = group[variable].min()
            monthly_stats[f'{variable}_mean'] = group[variable].mean()
            
        results.append(monthly_stats)

    return pd.DataFrame(results)

def plot_timeseries(data: pd.DataFrame):
    rows = len(VARIABLES)
    cols = 1
    fig, axs = plt.subplots(rows, cols, figsize=(10, 6*rows))

    for i, variable in enumerate(VARIABLES):
        for city in data['city'].unique():
            city_data = data[data['city'] == city]
            # plot mean values
            axs[i].plot(
                city_data['month'],
                city_data[f'{variable}_mean'],
                label = f'{city}',
            )

            # labels
            axs[i].set_title(f'mean {variable}')
            axs[i].legend(loc='upper right')

    fig.savefig('src/module_1/climate_evolution.png')

def main():
    data_list = []
    start = '2010-12-31'
    end = '2020-12-31'

    for city in COORDINATES:
        data = pd.DataFrame(
            get_data_meteo_api(city, start, end)['daily']
        ).assign(city=city)
        data_list.append(data)
    
    data = pd.concat(data_list)
   
    processed_data = compute_monthly_statistics(data, VARIABLES)

    plot_timeseries(processed_data)

if __name__ == "__main__":
    main()
