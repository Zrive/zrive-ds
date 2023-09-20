""" This is a dummy example """
import pandas as pd
import requests
import matplotlib.pyplot as plt

class APIConnector:

    url = "https://open-meteo.com/en/docs/climate-api"

    def __init__(self, url: str):
        self._url = url
    
    def get_city(city: str):
        match city:
            case "Madrid":
                return (40.4165, -3.7026)
            case "Londres":
                return (51.507351, -0.127758)
            case "Rio de Janeiro":
                return (-22.906847, -43.172896)
            case _:
                raise Exception("Invalid city")
    

        


def main():
    raise NotImplementedError

if __name__ == "__main__":
    main()
