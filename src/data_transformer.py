import pandas as pd
import numpy as np

orders = pd.read_parquet("data/orders.parquet")
grouped = (
    orders.groupby(["user_id", "id"])["ordered_items"]
    .apply(list)
    .reset_index()
)
grouped["item_ids"] = grouped["ordered_items"]
merged = pd.merge(
    orders,
    grouped[["user_id", "id", "item_ids"]],
    on=["user_id", "id"],
    how="left"
)
merged.to_parquet("C:/Users/manue/OneDrive/Escritorio/Zrive_DS/Module 6/task_module_6/zrive-ds/data/orders.parquet", index=False)
print(merged["item_ids"].apply(type).head(20))

