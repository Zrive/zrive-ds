import boto3
import pandas as pd
from io import StringIO
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score
from catboost import CatBoostClassifier


def load_credentials(path="Credenciales.txt"):
    out={}
    with open(path,"r") as f:
        for linea in f:
            if "=" in linea:
                k,v=linea.strip().split("=",1)
                out[k]=v
    return out


def load_data_s3(bucket,key,cred):
    s3=boto3.client(
        "s3",
        aws_access_key_id=cred.get("Acces_Key_ID"),
        aws_secret_access_key=cred.get("Password"),
        region_name="eu-west-1",
    )
    obj=s3.get_object(Bucket=bucket,Key=key)
    return pd.read_csv(StringIO(obj["Body"].read().decode("utf-8")))


def split_temporal(df):
    p70=df["order_date"].quantile(0.70)
    p90=df["order_date"].quantile(0.90)
    train=df[df["order_date"]<p70]
    val=df[(df["order_date"]>=p70)&(df["order_date"]<p90)]
    test=df[df["order_date"]>=p90]
    return train,val,test


def fit_model():

    cred=load_credentials()
    df=load_data_s3("zrive-ds-data","groceries/box_builder_dataset/feature_frame.csv",cred)

    df_bought=df[df["outcome"]==1]
    counts=df_bought.groupby("order_id")["variant_id"].nunique()
    valid_ids=counts[counts>=5].index
    df_filtered=df[df["order_id"].isin(valid_ids)]

    cols_drop=["variant_id","product_type","order_id","user_id","created_at","vendor"]
    df_model=df_filtered.drop(columns=cols_drop)

    df_model=df_model.sort_values("order_date")
    df_model["order_date"]=pd.to_datetime(df_model["order_date"])

    train,val,test=split_temporal(df_model)

    X_train=train.drop(columns=["outcome","order_date"])
    y_train=train["outcome"]
    X_val=val.drop(columns=["outcome","order_date"])
    y_val=val["outcome"]

    pipe_cb = Pipeline([
        ("scaler", StandardScaler()),
        ("cb", CatBoostClassifier(
            iterations=2000,
            learning_rate=0.03,
            depth=6,
            loss_function="Logloss",
            eval_metric="Logloss",
            random_seed=42
        ))
    ])

    pipe_cb.fit(
        X_train, y_train,
        cb__eval_set=(X_val, y_val),
        cb__early_stopping_rounds=5,
        cb__use_best_model=True,
        cb__verbose=False
    )

    return pipe_cb
