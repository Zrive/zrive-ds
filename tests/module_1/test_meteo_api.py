""" This is a dummy example to show how to import code from src/ for testing"""

from src.module_1.module_1_meteo_api import (
    join_models_mean_std,
)
import pandas as pd
import pytest


def test_join_models_mean_std():
    input_data = {
        "city": ["Madrid", "Madrid"],
        "year": [2020, 2021],
        "temperature_2m_mean_CMCC_CM2_VHR4": [20, 21],
        "temperature_2m_mean_FGOALS_f3_H": [19, 20],
    }
    df = pd.DataFrame(input_data)
    result_df = join_models_mean_std(df)

    assert all(
        column in result_df.columns
        for column in ["temperature_2m_mean_mean", "temperature_2m_mean_std"]
    )
    assert result_df["temperature_2m_mean_mean"].iloc[0] == 19.5
    assert result_df["temperature_2m_mean_std"].iloc[0] == pytest.approx(0.707, 0.001)


def test_plot_climate_data():
    city = "Madrid"
    df = model_data(city)
    result = plot_climate_data(df, city)
    assert (
        str(type(result)) == "<class 'module'>"
    ), "The output should be a matplotlib module"
