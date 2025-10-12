import pandas as pd
import boto3
import os
import matplotlib.pyplot as plt
from typing import Any

credenciales: dict[str, str] = {}
with open("Credenciales.txt", "r") as f:
    for linea in f:
        if "=" in linea:
            clave, valor = linea.strip().split("=", 1)
            credenciales[clave] = valor

# Crear cliente S3
s3: Any = boto3.client(
    "s3",
    aws_access_key_id=credenciales.get("Acces_Key_ID"),
    aws_secret_access_key=credenciales.get("Password"),
    region_name="eu-west-1",
)

# Descargar archivo desde S3
bucket: str = "zrive-ds-data"
prefix: str = "groceries/sampled-datasets/"
output_dir: str = "Datos_Parquet"
os.makedirs(output_dir, exist_ok=True)
dataframes: dict[str, pd.DataFrame] = {}

# Listar objetos bajo ese prefijo
response: dict[str, Any] = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
for obj in response.get("Contents", []):
    key: str = obj["Key"]
    filename: str = os.path.basename(key)
    if not filename:
        continue
    local_path: str = os.path.join(output_dir, filename)
    s3.download_file(bucket, key, local_path)
    if filename.endswith(".parquet"):
        df: pd.DataFrame = pd.read_parquet(local_path, engine="pyarrow")
        dataframes[filename] = df

# Creamos cada dataframe espcífico para poder trabajar con ellos
df_abandoned_carts: pd.DataFrame = dataframes["abandoned_carts.parquet"]
df_inventory: pd.DataFrame = dataframes["inventory.parquet"]
df_orders: pd.DataFrame = dataframes["orders.parquet"]
df_regulars: pd.DataFrame = dataframes["regulars.parquet"]
df_users: pd.DataFrame = dataframes["users.parquet"]

for nombre, df in dataframes.items():
    print(f"\n--- {nombre} ---")
    print(df.info())
    print(df.isna().sum())
    for col in df.columns:
        tipos: pd.Series = df[col].map(type).value_counts()
        if len(tipos) > 1:
            print(f"Columna '{col}' tiene múltiples tipos:")
            print(tipos)
# De aquí sacamos como probablemente las columnas "count_" en df_users no tengan mucha relevancia

# Comenzamos el análisis
df_abandoned_carts["id"] = df_abandoned_carts["id"].astype(str)
df_orders["id"] = df_orders["id"].astype(str)

# Ver si lo que se compra ha sido abandoned previamente
match_abandoned_orders: pd.DataFrame = df_abandoned_carts.merge(
    df_orders, on="id", how="inner", suffixes=("_abandoned", "_orders")
)
print("Coincidencias (filas):", len(match_abandoned_orders))
print(match_abandoned_orders.head())

df_regulars["id"] = df_orders["id"].astype(str)

# Vemos si lo que está en regulars suele estar en abandoned
df_regulars["user_id"] = df_regulars["user_id"].astype(str)
df_abandoned_carts["variant_id"] = (
    df_abandoned_carts["variant_id"]
    .astype(str)
    .str.replace("[", "", regex=False)
    .str.replace("]", "", regex=False)
    .str.strip()
)
df_regulars["variant_id"] = df_regulars["variant_id"].astype(str)
df_inventory["variant_id"] = df_inventory["variant_id"].astype(str)
df_inventory["product_type"] = df_inventory["product_type"].astype(str)

df_abandoned_carts["variant_list"]: pd.Series = df_abandoned_carts["variant_id"].str.split()
df_abandoned_expanded: pd.DataFrame = df_abandoned_carts.explode("variant_list")
df_abandoned_expanded.drop(columns=["variant_id"], inplace=True)
df_abandoned_expanded.rename(columns={"variant_list": "variant_id"}, inplace=True)

match_abandoned_regulars: pd.DataFrame = df_abandoned_expanded.merge(
    df_regulars,
    on=["user_id", "variant_id"],
    how="inner",
    suffixes=("_abandoned", "_regular"),
)
print("Coincidencias encontradas:", len(match_abandoned_regulars))

product_counts_abandoned: pd.DataFrame = (
    match_abandoned_regulars.groupby(["user_id", "variant_id"])
    .size()
    .reset_index(name="count")
    .sort_values(["user_id", "count"], ascending=[False, True])
)

product_counts_abandoned = product_counts_abandoned.merge(
    df_inventory[["variant_id", "product_type"]], on="variant_id", how="left"
)
print(product_counts_abandoned.head())

product_user_counts_abandoned: pd.DataFrame = (
    match_abandoned_regulars.groupby(["user_id"])
    .size()
    .reset_index(name="count")
    .sort_values(["user_id", "count"], ascending=[False, True])
)
print(product_user_counts_abandoned.head())

# Analizamos como se comportan df_orders y df_regulars
df_orders.rename(columns={"ordered_items": "variant_id"}, inplace=True)
df_orders["variant_id"] = (
    df_orders["variant_id"]
    .astype(str)
    .str.replace("[", "", regex=False)
    .str.replace("]", "", regex=False)
    .str.strip()
)
df_orders["variant_id"] = df_orders["variant_id"].astype(str)
df_orders["variant_list"]: pd.Series = df_orders["variant_id"].str.split()
df_orders_expanded: pd.DataFrame = df_orders.explode("variant_list")
df_orders_expanded.drop(columns=["variant_id"], inplace=True)
df_orders_expanded.rename(columns={"variant_list": "variant_id"}, inplace=True)

# Join con df_regulars
match_orders_regulars: pd.DataFrame = df_orders_expanded.merge(
    df_regulars,
    on=["user_id", "variant_id"],
    how="inner",
    suffixes=("_orders", "_regular"),
)
print("Coincidencias encontradas:", len(match_orders_regulars))

product_counts_orders: pd.DataFrame = (
    match_orders_regulars.groupby(["user_id", "variant_id"])
    .size()
    .reset_index(name="count")
    .sort_values(["user_id", "count"], ascending=[False, True])
)

product_counts_orders = product_counts_orders.merge(
    df_inventory[["variant_id", "product_type"]], on="variant_id", how="left"
)

product_user_counts_orders: pd.DataFrame = (
    match_orders_regulars.groupby(["user_id"])
    .size()
    .reset_index(name="count")
    .sort_values(["user_id", "count"], ascending=[False, True])
)

# Ver si lo que se pone en abandoned se compra en el corto plazo
df_abandoned_expanded["created_at"] = pd.to_datetime(
    df_abandoned_expanded["created_at"], errors="coerce"
)
df_orders_expanded["created_at"] = pd.to_datetime(
    df_orders_expanded["created_at"], errors="coerce"
)

df_abandoned_orders: pd.DataFrame = df_abandoned_expanded.merge(
    df_orders_expanded,
    on=["user_id", "variant_id"],
    how="inner",
    suffixes=("_abandoned", "_order"),
)

mask: pd.Series[bool] = (
    df_abandoned_orders["created_at_order"] >= df_abandoned_orders["created_at_abandoned"]
) & (
    (
        df_abandoned_orders["created_at_order"]
        - df_abandoned_orders["created_at_abandoned"]
    ).dt.days
    <= 7
)

df_abandoned_orders_filtered: pd.DataFrame = df_abandoned_orders.loc[mask].copy()
print("Coincidencias con pedido en <= 7 días:", len(df_abandoned_orders_filtered))

grouped_variants: pd.DataFrame = (
    df_abandoned_orders_filtered.groupby(["user_id", "created_at_order"])
    .agg({"variant_id": lambda x: " ".join(sorted(x.astype(str)))})
    .reset_index()
)
print("Total de combinaciones user_id + hora de pedido:", len(grouped_variants))

# Poner cantidades de dinero a estas cuentas, viendo cuánto es el valor de df_abandoned_carts, y viendo el de df_abandoned_orders_filtered
df_abandoned_priced: pd.DataFrame = df_abandoned_expanded.merge(
    df_inventory[["variant_id", "price"]], on="variant_id", how="left"
)

abandoned_sums: pd.DataFrame = (
    df_abandoned_priced.groupby(["user_id", "created_at"])["price"]
    .sum()
    .reset_index(name="total_abandoned_value")
    .sort_values(["user_id", "created_at"])
)
total_abandoned_value_sum: float = abandoned_sums["total_abandoned_value"].sum()

df_abandoned_ordered_priced: pd.DataFrame = df_abandoned_orders_filtered.merge(
    df_inventory[["variant_id", "price"]], on="variant_id", how="left"
)

abandoned_ordered_sum: pd.DataFrame = (
    df_abandoned_ordered_priced.groupby(["user_id", "created_at_order"])["price"]
    .sum()
    .reset_index(name="total_abandoned_ordered_value")
    .sort_values(["user_id", "created_at_order"])
)
total_abandoned_order_value_sum: float = abandoned_ordered_sum[
    "total_abandoned_ordered_value"
].sum()

perc_not_bought: float = 1 - (total_abandoned_order_value_sum / total_abandoned_value_sum)

# Crear gráfico
plt.figure(figsize=(8, 5))
plt.hist(
    abandoned_sums["total_abandoned_value"],
    bins=30,
    alpha=0.5,
    color="blue",
    label="Abandoned",
)
plt.hist(
    abandoned_ordered_sum["total_abandoned_ordered_value"],
    bins=30,
    alpha=0.5,
    color="orange",
    label="Ordered",
)
plt.xlabel("Price")
plt.ylabel("Frequency")
plt.title("Carts distribution: Abandoned vs Ordered")
plt.legend()
plt.xlim(0, 250)
plt.grid(alpha=0.3)
plt.show()
