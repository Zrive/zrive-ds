""" This is a dummy example to show how to import code from src/ for testing"""

from src.module_1.module_1_meteo_api import main
from src.module_1.module_1_meteo_api import calc_stats
import pandas as pd
import pytest


def test_calc_stats_by_month():
    data = {
        "daily": [
            "2023-01-01",
            "2023-01-02",
            "2023-02-01",
            "2023-02-02",
            "2024-01-01",
            "2024-01-02",
            "2024-02-01",
            "2024-02-02",
        ],
        "temp": [10.0, 12.0, 9.0, 11.0, 8.0, 7.0, 10.0, 9.0],
        "prec": [0.5, 0.7, 0.2, 0.3, 0.6, 0.8, 0.4, 0.5],
        "moist": [30.0, 32.0, 28.0, 29.0, 31.0, 33.0, 29.0, 30.0],
    }
    sample_df = pd.DataFrame(data)
    res = calc_stats(sample_df, freq="monthly")

    assert "year" in res.columns
    assert "month" in res.columns
    assert "temperature_2m_mean" in res.columns
    assert "precipitation_sum" in res.columns
    assert "soil_moisture_0_to_10cm_mean" in res.columns


def test_calc_stats_by_year():
    data = {
        "daily": [
            "2023-01-01",
            "2023-01-02",
            "2023-02-01",
            "2023-02-02",
            "2024-01-01",
            "2024-01-02",
            "2024-02-01",
            "2024-02-02",
        ],
        "temp": [10.0, 12.0, 9.0, 11.0, 8.0, 7.0, 10.0, 9.0],
        "prec": [0.5, 0.7, 0.2, 0.3, 0.6, 0.8, 0.4, 0.5],
        "moist": [30.0, 32.0, 28.0, 29.0, 31.0, 33.0, 29.0, 30.0],
    }
    sample_df = pd.DataFrame(data)
    res = calc_stats(sample_df, freq="yearly")

    assert "month" in res.columns
    assert "temperature_2m_mean" in res.columns
    assert "precipitation_sum" in res.columns
    assert "soil_moisture_0_to_10cm_mean" in res.columns


def test_main():
    raise NotImplementedError
