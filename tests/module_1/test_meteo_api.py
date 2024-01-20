import pandas as pd
from src.module_1.module_1_meteo_api import (
    get_data_meteo_api,
    call_api,
    model_data,
    plot_climate_data,
    main,
)


def test_get_data_meteo_api():
    city = "Madrid"
    result = get_data_meteo_api(city)
    assert result is None, "The function should return None"


def test_call_api():
    city = "Madrid"
    model = "CMCC_CM2_VHR4"
    data = call_api(city, model)
    assert isinstance(data, dict), "The output should be a dictionary"
    assert set(data.keys()) == {
        "time",
        "temperature_2m_mean",
        "precipitation_sum",
        "soil_moisture_0_to_10cm_mean",
    }, "The keys are not correct"


def test_model_data():
    city = "Madrid"
    df = model_data(city)
    assert isinstance(df, pd.DataFrame), "The output should be a DataFrame"
    assert set(df.columns) == {
        "date",
        "temperature_2m_mean",
        "precipitation_sum",
        "soil_moisture_0_to_10cm_mean",
    }, "The DataFrame columns are not correct"


def test_plot_climate_data():
    city = "Madrid"
    df = model_data(city)
    result = plot_climate_data(df, city)
    assert (
        str(type(result)) == "<class 'module'>"
    ), "The output should be a matplotlib module"


def test_main():
    assert main() is None, "The main function should return None"
