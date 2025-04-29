# tests/module_1/test_meteo_api.py
import pytest
from unittest.mock import MagicMock, patch
import pandas as pd
from datetime import datetime , timedelta
import pytz
from src.module_1.module_1_meteo_api import organize_data, get_data_meteo_api

def test_get_data_meteo_api():
    mock_response = MagicMock()
    mock_response.Daily.return_value = MagicMock()
    with patch("src.module_1.module_1_meteo_api.use_api", return_value=mock_response):
        daily = get_data_meteo_api("Madrid")
        assert daily is not None
        
        
        

def test_organize_data():
    daily = MagicMock()
    daily.Variables.side_effect = [
        MagicMock(ValuesAsNumpy=lambda: [1, 2, 3]),  # precipitation_sum
        MagicMock(ValuesAsNumpy=lambda: [10, 11, 12]),  # temperature_2m_mean
        MagicMock(ValuesAsNumpy=lambda: [20, 21, 22])  # wind_speed_10m_max
    ]
   
    start_date = datetime(2010, 1, 1, tzinfo=pytz.UTC)
    end_date = datetime(2010, 1, 4, tzinfo=pytz.UTC)  # 2010-01-01 a 2010-01-03
    daily.Time.return_value = int(start_date.timestamp())   
    daily.TimeEnd.return_value = int(end_date.timestamp())  
    interval = timedelta(days=1)
    daily.Interval.return_value = int(interval.total_seconds())  

    result = organize_data(daily)
    
    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == ["date", "precipitation_sum", "temperature_2m_mean", "wind_speed_10m_max"]
    assert len(result) == 1  # 
    assert result["precipitation_sum"].iloc[0] == 6  # sum([1, 2, 3])
    assert result["temperature_2m_mean"].iloc[0] == 11  # mean([10, 11, 12])
    assert result["wind_speed_10m_max"].iloc[0] == 21  # mean([20, 21, 22])
