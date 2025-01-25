from src.module_1.module_1_meteo_api import (
    main,
    compute_monthly_statistics,
    request_with_cooloff
)
from unittest.mock import Mock
import requests
import pytest
import pandas as pd


class MockResponse:
    def __init__(self, json_data, status_code):
        self.json_data = json_data
        self.status_code = status_code

    def json(self):
        return self.json_data
    
    def raise_for_status(self):
        if self.status_code != 200:
            raise requests.exceptions.HTTPError(f'HTTPError: {self.status_code}') 
    
def test_request_with_cooloff_200(monkeypatch: pytest.MonkeyPatch):
    headers = {}
    params = {}
    mocked_response = Mock(return_value=MockResponse('json dummy',200))
    monkeypatch.setattr(requests, 'get', mocked_response)
    response = request_with_cooloff('mock url',headers, params, num_attempts=3, payload=None)
    assert response.status_code == 200
    assert response.json() == 'json dummy'

def test_request_with_cooloff_404(monkeypatch: pytest.MonkeyPatch):
    with pytest.raises(requests.exceptions.HTTPError):
        headers = {}
        params = {}
        mocked_response = Mock(return_value=MockResponse('mocked response',404))
        monkeypatch.setattr(requests, 'get', mocked_response)
        response = request_with_cooloff('mock url',headers, params, num_attempts=3, payload=None)
        assert response.status_code == 404 

def test_request_with_cooloff_429(monkeypatch: pytest.MonkeyPatch):
    with pytest.raises(requests.exceptions.HTTPError):
        headers = {}
        params = {}
        mocked_response = Mock(return_value=MockResponse('mocked response',429))
        monkeypatch.setattr(requests, 'get', mocked_response)
        response = request_with_cooloff('mock url',headers, params, num_attempts=3, payload=None)
        expected_msgs = [
            f'API return status code {response.status_code} cooloff at {1}'
            f'API return status code {response.status_code} cooloff at {2}'
            f'API return status code {response.status_code} cooloff at {4}'
        ]
        assert response.status_code == 429
        assert  [r.msg for r in pytest.caplog.records] == expected_msgs

def test_compute_monthly_statistics():
    test_variable = 'test_variable'
    data = pd.DataFrame({
        'city': ['London'] * 4,
        'time': pd.date_range(start='2010-01-01',periods=4),
        f'{test_variable}':[0,10,None,50],
    })
    expected = pd.DataFrame({
        'city':['London'],
        'month':pd.to_datetime(['2010-01-01']),
        f'{test_variable}_max':[50],
        f'{test_variable}_min':[0],
        f'{test_variable}_mean':[20],
    })
    actual = compute_monthly_statistics(data, {test_variable})
    pd.testing.assert_frame_equal(actual,expected, check_dtype=False)

