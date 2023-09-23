import requests
import json
import matplotlib.pyplot as plt

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
    api_data = get_data_meteo_api("Madrid")
    daily_data = api_data["daily"]
    # a = [1, 3, 5, 7, 8]
    b = get_data_meteo_api("Madrid")
    c = b["daily"]["temperature_2m_mean"]
    # print(c[:300])
    print(mean_calculation(c))
    print(variance_calculation(c, mean_calculation(c)))
    plotting(daily_data)
    # print(yearly_data)


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
            yearly["precipitation_sum"].append(round(sum(prec), 3))
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
    list1 = daily_to_yearly(list)
    x = range(len(list1["time"]))
    y1 = list1["temperature_2m_mean"]
    y2 = list1["precipitation_sum"]
    y3 = list1["soil_moisture_0_to_10cm_mean"]
    fig, ax1 = plt.subplots(figsize=(20, 6))
    ax2 = ax1.twinx()
    ax3 = ax1.twinx()
    ax3.spines.right.set_position(("axes", 1.05))

    l1 = create_labels("temperature_2m_mean", list)
    l2 = create_labels("precipitation_sum", list)
    l3 = create_labels("soil_moisture_0_to_10cm_mean", list)
    (p1,) = ax1.plot(x, y1, "r-", label=l1)
    p2 = ax2.bar(x, y2, color="blue", label=l2)
    (p3,) = ax3.plot(x, y3, "g-", label=l3)
    ax1.yaxis.label.set_color(p1.get_color())
    ax2.yaxis.label.set_color("blue")
    ax3.yaxis.label.set_color(p3.get_color())

    ax1.set_xlabel("Year")
    ax1.set_ylabel("°C")
    ax2.set_ylabel("mm")
    ax3.set_ylabel("m3/m3")
    plt.xticks(x, list1["time"])
    plt.xlim(-1, 101)
    plt.locator_params(axis="x", nbins=10)
    tkw = dict(size=4, width=1.5)
    ax1.tick_params(axis="y", colors=p1.get_color(), **tkw)
    ax2.tick_params(axis="y", colors="blue", **tkw)
    ax3.tick_params(axis="y", colors=p3.get_color(), **tkw)
    ax1.tick_params(axis="x", **tkw)

    fig.tight_layout()
    # plt.show()
    ax1.legend(handles=[p1, p2, p3])
    plt.savefig("mygraph.png")


def create_labels(text, list):
    label = (
        text[:4]
        + "_mean="
        + str(mean_calculation(list[text]))
        + " "
        + text[:4]
        + "_disp="
        + str(
            variance_calculation(
                list[text],
                mean_calculation(list[text]),
            )
        )
    )
    return label


if __name__ == "__main__":
    main()
