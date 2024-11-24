import os
import pandas as pd


STORAGE = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..", "data"))


def load_orders() -> pd.DataFrame:
    orders = pd.read_parquet(os.path.join(STORAGE, "orders.parquet"))
    orders = orders.sort_values(by=["user_id", "created_at"])
    orders["item_count"] = orders.apply(lambda x: len(x.item_ids), axis=1)
    orders["user_order_seq"] = (
        orders.groupby(["user_id"])["created_at"].rank().astype(int)
    )
    return orders


def load_regulars() -> pd.DataFrame:
    return pd.read_parquet(os.path.join(STORAGE, "regulars.parquet"))


def get_mean_item_price() -> float:
    inventory = pd.read_parquet(os.path.join(STORAGE, "inventory.parquet"))
    return inventory.our_price.mean()
