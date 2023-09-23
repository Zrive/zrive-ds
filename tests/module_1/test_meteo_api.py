from src.module_1.module_1_meteo_api import mean_calculation, variance_calculation


list = [1, 2, 3, 4, 5, 6, 7, 8]


def test_mean():
    assert mean_calculation(list) == 4.5, "incorrect mean"


def test_variance():
    assert variance_calculation(list, 4.5) == 5.25, "incorrect variance"
