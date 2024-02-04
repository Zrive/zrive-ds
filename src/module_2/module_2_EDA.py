import pandas as pd
df_orders = pd.read_parquet('/Users/alvaroleal/Desktop/DS/Zrive/orders.parquet')
df_regulars = pd.read_parquet('/Users/alvaroleal/Desktop/DS/Zrive/regulars.parquet')
df_abandoned_carts = pd.read_parquet(
    '/Users/alvaroleal/Desktop/DS/Zrive/abandoned_carts.parquet')
df_inventory = pd.read_parquet('/Users/alvaroleal/Desktop/DS/Zrive/inventory.parquet')
df_users = pd.read_parquet('/Users/alvaroleal/Desktop/DS/Zrive/users.parquet')

df_feature_frame = pd.read_csv('/Users/alvaroleal/Desktop/DS/Zrive/feature_frame.csv')

# Funcion para hacer una primera visualización de los datos y quick checks


def quick_view(df):
    print("Primeras filas:")
    print(df.head())
    print("\n")
    print("Información general:")
    df.info()
    print("\n")
    print("Número de valores null:")
    print(df.isnull().sum())
    print("\n")
    print("Porcentaje de valores null:")
    print(df.isnull().mean() * 100)
    print("\n")

# Función para ver las correlaciones con una variable objetivo:


def check_corr(df, target_variable):
    correlation_matrix = df.corr()
    target_correlations = correlation_matrix[target_variable].sort_values(
        ascending=False)
    print(f"Correlaciones de '{target_variable}' con las demás variables:")
    print(target_correlations)


quick_view(df_feature_frame)
