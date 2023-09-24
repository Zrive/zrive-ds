""" This is a dummy example to show how to import code from src/ for testing"""

from src.module_1.module_1_meteo_api import main
from src.module_1.module_1_meteo_api import APIConnector
import pytest



def test_main():
    # We are going to test the calc_stats function
