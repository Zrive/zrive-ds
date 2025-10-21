import boto3
import pandas as pd
from io import StringIO
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import Ridge, Lasso
import numpy as np

# cargar credenciales
credenciales: dict[str, str] = {}
with open("Credenciales.txt", "r") as f:
    for linea in f:
        if "=" in linea:
            clave, valor = linea.strip().split("=", 1)
            credenciales[clave] = valor

# cliente s3
s3 = boto3.client(
    "s3",
    aws_access_key_id=credenciales.get("Acces_Key_ID"),
    aws_secret_access_key=credenciales.get("Password"),
    region_name="eu-west-1",
)

bucket = "zrive-ds-data"
key = "groceries/box_builder_dataset/feature_frame.csv"

obj = s3.get_object(Bucket=bucket, Key=key)
df = pd.read_csv(StringIO(obj["Body"].read().decode("utf-8")))

# Formamos el dataframe para poder trabajar con el
df_bought = df[df["outcome"] == 1]
counts = df_bought.groupby("order_id")["variant_id"].nunique()
valid_ids = counts[counts >= 5].index
df_filtered = df_bought[df_bought["order_id"].isin(valid_ids)]

# Vemos que productos podemos recomendar
df_filtered["user_id"].nunique()
df_filtered["order_id"].nunique()
df_filtered["variant_id"].nunique()

df_user_variant_regular = df_filtered[df_filtered["set_as_regular"] == 1][
    ["user_id", "variant_id"]
].drop_duplicates()

users_no_regular = set(df_filtered["user_id"].unique()) - set(
    df_user_variant_regular["user_id"].unique()
)
df_no_regular = df_filtered[df_filtered["user_id"].isin(users_no_regular)]
counts_2 = df_no_regular.groupby(["user_id", "variant_id"]).size()
df_users_prod_3plus = counts_2[counts_2 >= 3].reset_index(name="count")

# Modelling
# Monthly sales increase model (2%)
df_order_summary = (
    df_bought.groupby("order_id")
    .agg(
        order_cost=("normalised_price", "sum"),
        avg_price=("normalised_price", "mean"),
        avg_discount=("discount_pct", "mean"),
        avg_popularity=("global_popularity", "mean"),
        people=("people_ex_baby", "mean"),
        babies=("count_babies", "mean"),
        pets=("count_pets", "mean"),
    )
    .reset_index()
)

df_order_summary["order_at"] = pd.to_datetime(
    df_bought.groupby("order_id")["created_at"]
    .first()
    .reindex(df_order_summary["order_id"])
    .values
)
df_order_summary = df_order_summary.sort_values("order_at").reset_index(drop=True)

n = len(df_order_summary)
n_train = int(n * 0.7)
n_val = int(n * 0.2)

df_train = df_order_summary.iloc[:n_train]
df_val = df_order_summary.iloc[n_train : n_train + n_val]
df_test = df_order_summary.iloc[n_train + n_val :]

order_to_user = df_bought.groupby("order_id")["user_id"].first()

users_train = set(df_train["order_id"].map(order_to_user))
users_val = set(df_val["order_id"].map(order_to_user))
users_test = set(df_test["order_id"].map(order_to_user))


overlap_train_val = users_train & users_val
overlap_train_test = users_train & users_test
overlap_val_test = users_val & users_test

len(overlap_train_val), len(overlap_train_test), len(overlap_val_test)

X_train = df_train.drop(columns=["order_cost", "order_id", "order_at"])
y_train = df_train["order_cost"]

X_val = df_val.drop(columns=["order_cost", "order_id", "order_at"])
y_val = df_val["order_cost"]

X_test = df_test.drop(columns=["order_cost", "order_id", "order_at"])
y_test = df_test["order_cost"]

model = LinearRegression()
model.fit(X_train, y_train)

betas = model.coef_
beta0 = model.intercept_

y_val_pred = model.predict(X_val)
y_test_pred = model.predict(X_test)

val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
val_r2 = r2_score(y_val, y_val_pred)

test_rmse = np.sqrt(y_test, y_test_pred)
test_r2 = r2_score(y_test, y_test_pred)


# Standardización del modelo
def fit_best_ridge_lasso(
    X_train, y_train, X_val, y_val, alphas=[0.001, 0.01, 0.1, 1, 10]
):
    results = {}

    # --- Ridge ---
    best_ridge_rmse = np.inf
    best_ridge_model = None

    for alpha in alphas:
        model = Ridge(alpha=alpha)
        model.fit(X_train, y_train)
        y_val_pred = model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
        if rmse < best_ridge_rmse:
            best_ridge_rmse = rmse
            best_ridge_model = model

    results["ridge"] = {
        "alpha": best_ridge_model.alpha,
        "rmse_val": best_ridge_rmse,
        "coef": best_ridge_model.coef_,
        "intercept": best_ridge_model.intercept_,
    }

    # --- Lasso ---
    best_lasso_rmse = np.inf
    best_lasso_model = None

    for alpha in alphas:
        model = Lasso(alpha=alpha, max_iter=10000)
        model.fit(X_train, y_train)
        y_val_pred = model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
        if rmse < best_lasso_rmse:
            best_lasso_rmse = rmse
            best_lasso_model = model

    results["lasso"] = {
        "alpha": best_lasso_model.alpha,
        "rmse_val": best_lasso_rmse,
        "coef": best_lasso_model.coef_,
        "intercept": best_lasso_model.intercept_,
    }

    return results


res = fit_best_ridge_lasso(X_train, y_train, X_val, y_val)

print("Ridge best alpha:", res["ridge"]["alpha"])
print("Ridge val RMSE:", res["ridge"]["rmse_val"])
print("Lasso best alpha:", res["lasso"]["alpha"])
print("Lasso val RMSE:", res["lasso"]["rmse_val"])


def show_coefs(res, feature_names):
    ridge_coefs = (
        pd.DataFrame({"feature": feature_names, "coef_ridge": res["ridge"]["coef"]})
        .assign(abs_coef_ridge=lambda df: np.abs(df["coef_ridge"]))
        .sort_values(by="abs_coef_ridge", ascending=False)
    )

    lasso_coefs = (
        pd.DataFrame({"feature": feature_names, "coef_lasso": res["lasso"]["coef"]})
        .assign(abs_coef_lasso=lambda df: np.abs(df["coef_lasso"]))
        .sort_values(by="abs_coef_lasso", ascending=False)
    )

    return ridge_coefs.drop(columns="abs_coef_ridge"), lasso_coefs.drop(
        columns="abs_coef_lasso"
    )


ridge_coef_table, lasso_coef_table = show_coefs(res, X_train.columns)

print(ridge_coef_table)
print(lasso_coef_table)
