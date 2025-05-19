import pandas as pd 
import os

data_dir = "./data"
files = {
    "orders": "orders.parquet",
    "regulars": "regulars.parquet",
    "abandoned_carts": "abandoned_carts.parquet",
    "inventory": "inventory.parquet",
    "users": "users.parquet"
}

#para guardar los dataframes
dfs = {}

#abrir los archivos con pandas
for name, file in files.items():
    path = os.path.join(data_dir, file)
    dfs[name] = pd.read_parquet(path)

#leemos que se hayan abierto bien
for name, df in dfs.items():
    print(df.head())
