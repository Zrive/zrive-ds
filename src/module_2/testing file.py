import pandas as pd
import numpy as np
df_orders = pd.read_parquet(
    '/Users/alvaroleal/Desktop/DS/Zrive/orders.parquet')
print(df_orders)

for column in df_orders.columns:
    non_scalar_values = df_orders[column].apply(
        lambda x: isinstance(x, (list, dict, set, np.ndarray)))
    if non_scalar_values.any():
        print(f"Column {column} contains non-scalar values.")
