import pandas as pd
import requests
from unittest.mock import Mock
import pytest

from src.module_1.module_1_meteo_api import compute_variable_mean_and_std, VARIABLES, _request_with_cooloff


def test_compute_variable_mean_and_std():
    test_variable = VARIABLES.split(",")[0]
    data = pd.DataFrame(
        {
            "city": ["Madrid", "Madrid", "Madrid"],
            "time": ["2021-01-01", "2021-01-02", "2021-01-03"],
            f"{test_variable}_model1": [10, 20, 30],
            f"{test_variable}_model2": [1, 2, 3],
            f"{test_variable}_model3": [5, 6, 7],
        }
    )
    expected = pd.DataFrame(
        {
            "city": {0: "Madrid", 1: "Madrid", 2: "Madrid"},
            "time": {0: "2021-01-01", 1: "2021-01-02", 2: "2021-01-03"},
            f"{test_variable}_mean": {
                0: 5.333333333333333,
                1: 9.333333333333334,
                2: 13.333333333333334,
            },
            f"{test_variable}_std": {
                0: 4.509249752822894,
                1: 9.451631252505216,
                2: 14.571661996262929,
            },
        }
    )

    # We want to keep only the processed test_variable but not all the other variables
    info_cols = ["city", "time"]
    test_cols = [f"{test_variable}_mean", f"{test_variable}_std"]
    pd.testing.assert_frame_equal(
        compute_variable_mean_and_std(data)[info_cols + test_cols],
        expected,
    )


class MockResponse:
    def __init__(self, json_data, status_code):
        self.json_data = json_data
        self.status_code = status_code

    def json(self):
        return self.json_data

    def raise_for_status(self):
        if self.status_code != 200:
            raise requests.exceptions.HTTPError(f"HTTPError: {self.status_code}")


def test_request_with_cooloff_200(monkeypatch):
    headers = {}
    mocked_response = Mock(return_value=MockResponse('mocked_response', 200))
    monkeypatch.setattr(requests, 'get', mocked_response)
    response = _request_with_cooloff('mock_url', headers, num_attempts=10, payload=None)
    assert response.status_code == 200
    assert response.json() == 'mocked_response'


def test_request_with_cooloff_404(monkeypatch):
    with pytest.raises(requests.exceptions.HTTPError):
        headers = {}
        mocked_response = Mock(return_value=MockResponse('mocked_response', 404))
        monkeypatch.setattr(requests, 'get', mocked_response)
        response = _request_with_cooloff('mock_url', headers, num_attempts=10, payload=None)


def test_request_with_cooloff_429(monkeypatch, caplog):
    """ After n_max_attemps, the function should also raise an exception"""
    with pytest.raises(requests.exceptions.HTTPError):
        headers = {}
        mocked_response = Mock(return_value=MockResponse('mocked_response', 429))
        monkeypatch.setattr(requests, 'get', mocked_response)
        response = _request_with_cooloff('mock_url', headers, num_attempts=2, payload=None)
        expected_msgs = [
            f"API return code {response.status_code} cooloff at {1}",
            f"API return code {response.status_code} cooloff at {2}"
        ]
        assert [r.msg for r in caplog.records] == expected_msgs