import pandas as pd
df = pd.read_parquet("C:/Users/manue/OneDrive/Escritorio/Zrive_DS/Module 6/task_module_6/zrive-ds/data/orders.parquet")
print(df["user_id"].iloc[5000])
print(df.columns)