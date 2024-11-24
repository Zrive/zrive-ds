import numpy as np
import pandas as pd


def count_regulars_in_order(order: pd.DataFrame, user_regulars: pd.DataFrame) -> int:
    return len(set(order.item_ids).intersection(set(user_regulars.variant_id.values)))


def count_regulars_in_orders(
    orders: pd.DataFrame, regulars: pd.DataFrame
) -> np.ndarray:
    counts = []
    for _, order in orders.iterrows():
        user_regulars = regulars.loc[lambda x: x.user_id == order.user_id]
        counts += [count_regulars_in_order(order, user_regulars)]
    return np.array(counts)


def compute_basket_value(orders: pd.DataFrame, mean_item_price: float) -> float:
    return orders.item_count * mean_item_price


def enrich_orders(
    orders: pd.DataFrame, regulars: pd.DataFrame, mean_item_price: float
) -> pd.DataFrame:
    enriched_orders = orders.copy()
    enriched_orders["regulars_count"] = count_regulars_in_orders(
        enriched_orders, regulars
    )
    enriched_orders["basket_value"] = compute_basket_value(
        enriched_orders, mean_item_price
    )
    return enriched_orders


def build_prior_orders(enriched_orders: pd.DataFrame) -> pd.DataFrame:
    prior_orders = enriched_orders.copy()
    prior_orders["user_order_seq_plus_1"] = prior_orders.user_order_seq + 1
    prior_orders["prior_basket_value"] = prior_orders["basket_value"]
    prior_orders["prior_item_count"] = prior_orders["item_count"]
    prior_orders["prior_regulars_count"] = prior_orders["regulars_count"]
    return prior_orders.loc[
        :,
        [
            "user_id",
            "user_order_seq_plus_1",
            "prior_item_count",
            "prior_regulars_count",
            "prior_basket_value",
        ],
    ]


def build_feature_frame(
    orders: pd.DataFrame, regulars: pd.DataFrame, mean_item_price: float
) -> pd.DataFrame:
    enriched_orders = enrich_orders(orders, regulars, mean_item_price)
    prior_orders = build_prior_orders(enriched_orders)
    return pd.merge(
        enriched_orders.loc[
            :,
            [
                "user_id",
                "created_at",
                "user_order_seq",
                "basket_value",
                "regulars_count",
            ],
        ],
        prior_orders,
        how="inner",
        left_on=("user_id", "user_order_seq"),
        right_on=("user_id", "user_order_seq_plus_1"),
    ).drop(["user_order_seq", "user_order_seq_plus_1"], axis=1)
