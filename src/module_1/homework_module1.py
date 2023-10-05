import time
import requests
import logging
import json
import pandas as pd
from urllib.parse import urlencode
import matplotlib.pyplot as plt

from typing import Dict

logger = logging.getLogger(__name__)

logger.level = logging.INFO

API_URL = "https://climate-api.open-meteo.com/v1/climate?"

COORDINATES = {
    "Madrid": {"latitude": 40.416775, "longitude": -3.703790},
    "London": {"latitude": 51.507351, "longitude": -0.127758},
    "Rio": {"latitude": -22.906847, "longitude": -43.172896},
}

VARIABLES = "temperature_2m_mean,precipitation_sum,soil_moisture_0_to_10cm_mean"


def get_data_meteo_api(city: Dict[str, float]):
    headers = {}  # Empty because we don't need any headers for this API

    params = {
        "latitude": city["latitude"],
        "longitude": city["longitude"],
        "start_date": "1950-01-01",
        "end_date": "2050-12-31",
        # We consider all climatic models avaialble on the API
        "models": "CMCC_CM2_VHR4,FGOALS_f3_H,HiRAM_SIT_HR,MRI_AGCM3_2_S,EC_Earth3P_HR,MPI_ESM1_2_XR,NICAM16_8S",  # noqa
        "daily": VARIABLES,
    }

    return request_with_cooloff(API_URL + urlencode(params, safe=","), headers)


def _request_with_cooloff(
    url: str, headers: Dict[str, any], num_attempts: int, payload: Dict[str, any] = None
):
    """
    Call the url using requests. If the endpoint returns an error wait a cooloff
    period and try again, doubling the period each attempt up to a max num_attempts.
    """
    cooloff = 1
    for call_count in range(num_attempts):
        try:
            if payload is None:
                response = requests.get(url, headers=headers)
            else:
                response = requests.post(url, headers=headers, json=payload)
            response.raise_for_status()

        # If we're overloading the endpoint it may refuse a connection
        except requests.exceptions.ConnectionError as e:
            logger.info("API refused the connection")
            logger.warning(e)
            if call_count != (num_attempts - 1):
                time.sleep(cooloff)
                cooloff *= 2
                call_count += 1
                continue
            else:
                raise

        # Catch non 200 return codes on the HTTP header
        except requests.exceptions.HTTPError as e:
            logger.warning(e)
            if response.status_code == 404:
                raise

            logger.info(f"API return code {response.status_code} cooloff at {cooloff}")
            if call_count != (num_attempts - 1):
                time.sleep(cooloff)
                cooloff *= 2
                call_count += 1
                continue
            else:
                raise

        # We got through the loop without error so we've received a valid response
        return response



def request_with_cooloff(
    url: str,
    headers: Dict[str, any],
    payload: Dict[str, any] = None,
    num_attempts: int = 10,
):
    return json.loads(
        _request_with_cooloff(
            url,
            headers,
            num_attempts,
            payload,
        ).content.decode("utf-8")
    )


def compute_variable_mean_and_std(data: pd.DataFrame):
    """
    It expects a dataframe for each city and computes mean and std of each climate
    variable.
    """
    calculated_ts = data[["city", "time"]].copy()
    for variable in VARIABLES.split(","):
        idxs = [col for col in data.columns if col.startswith(variable)]
        calculated_ts[f"{variable}_mean"] = data[idxs].mean(axis=1)
        calculated_ts[f"{variable}_std"] = data[idxs].std(axis=1)
    return calculated_ts


def plot_timeseries(data: pd.DataFrame):
    """
    Represent in a very basic form mean +- std for each variable and city.
    We do one plot per variable so we can compare cities across.
    """
    rows = 3
    cols = 1
    fig, axs = plt.subplots(rows, cols, figsize=(15, 10))
    axs = axs.flatten()

    data["year"] = pd.to_datetime(data["time"]).dt.year
    for k, city in enumerate(data.city.unique()):
        city_data = data.loc[lambda x: x.city == city, :]
        print(city_data.head())
        for i, variable in enumerate(VARIABLES.split(",")):
            city_data["mid_"] = city_data[f"{variable}_mean"]
            city_data["upper_"] = (
                city_data[f"{variable}_mean"] + city_data[f"{variable}_std"]
            )
            city_data["lower_"] = (
                city_data[f"{variable}_mean"] - city_data[f"{variable}_std"]
            )

            # Plot yearly mean values
            city_data.groupby("year")["mid_"].apply("mean").plot(
                ax=axs[i], label=f"{city}", color=f"C{k}"
            )
            city_data.groupby("year")["upper_"].apply("mean").plot(
                ax=axs[i], ls="--", label="_nolegend_", color=f"C{k}"
            )
            city_data.groupby("year")["lower_"].apply("mean").plot(
                ax=axs[i], ls="--", label="_nolegend_", color=f"C{k}"
            )
            axs[i].set_title(variable)

    plt.tight_layout()
    plt.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.1),
        fancybox=True,
        shadow=True,
        ncol=5,
    )
    plt.savefig(
        "src/module_1/climate_evolution.png", bbox_inches="tight"
    )  # save the figure to file


def main():
    data_list = []
    for city, coordinates in COORDINATES.items():
        data_list.append(
            (pd.DataFrame(get_data_meteo_api(coordinates)["daily"]).assign(city=city))
        )

    data = pd.concat(data_list)
    print(data.head())

    calculated_ts = compute_variable_mean_and_std(data)
    print(calculated_ts.head())

    plot_timeseries(calculated_ts)


if __name__ == "__main__":
    main()
