import requests
import json

API_URL = "https://climate-api.open-meteo.com/v1/climate?"
COORDINATES = {
    "Madrid": {"latitude": 40.416775, "longitude": -3.703790},
    "London": {"latitude": 51.507351, "longitude": -0.127758},
    "Rio": {"latitude": -22.906847, "longitude": -43.172896},
}
VARIABLES = ["temperature_2m_mean", "precipitation_sum", "soil_moisture_0_to_10cm_mean"]
START_DATE = "1950-01-01"
END_DATE = "2050-12-31"


def main():
    a = get_data_meteo_api("Madrid")
    # b = a["daily"]["time"]
    # print(b[:300])
    # print(a["daily"].keys())
    # a = [1, 3, 5, 7, 8]
    # b = get_data_meteo_api("Madrid")
    # c = b["daily"]["soil_moisture_0_to_10cm_mean"]
    # print(mean_calculation(c))
    # print(variance_calculation(c, mean_calculation(c)))
    print(daily_to_yearly(a["daily"]))


def get_data_meteo_api(city):
    param_city = COORDINATES[city]
    param_city["start_date"] = START_DATE
    param_city["end_date"] = END_DATE
    param_city["daily"] = VARIABLES
    data = api_connection(param_city)
    data_json = json.loads(data.text)
    return data_json


def api_connection(query_params):
    try:
        response = requests.get(API_URL, params=query_params, timeout=1)
        response.raise_for_status()
        return response
    except requests.exceptions.HTTPError as err1:
        print(err1)
    except requests.exceptions.ConnectionError as err2:
        print(err2)
    except requests.ConnectionError as err3:
        print(err3)
    except requests.Timeout as err4:
        print(err4)


def mean_calculation(list):
    suma = 0
    for val in list:
        suma = suma + val
    mean = suma / len(list)
    return round(mean, 3)


def variance_calculation(list, mean):
    suma = 0
    for val in list:
        c = val - mean
        suma = suma + c**2
    var = suma / len(list)
    return round(var, 3)


def daily_to_yearly(daily):
    actual_year = 1950
    k = 0
    yearly = {}
    temp = list()
    prec = list()
    soil = list()
    yearly = {
        "time": [],
        "temperature_2m_mean": [],
        "precipitation_sum": [],
        "soil_moisture_0_to_10cm_mean": [],
    }

    for day in daily["time"]:
        year = day.split("-")
        if actual_year != int(year[0]):
            yearly["time"].append(actual_year)
            yearly["temperature_2m_mean"].append(mean_calculation(temp))
            yearly["precipitation_sum"].append(mean_calculation(prec))
            yearly["soil_moisture_0_to_10cm_mean"].append(mean_calculation(soil))
            actual_year = int(year[0])
            temp.clear()
            prec.clear()
            soil.clear()
        temp.append(daily["temperature_2m_mean"][k])
        prec.append(daily["precipitation_sum"][k])
        soil.append(daily["soil_moisture_0_to_10cm_mean"][k])
        k = k + 1
    return yearly


def plotting(list):
    a


if __name__ == "__main__":
    main()
