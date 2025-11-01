import boto3
import pandas as pd
from io import StringIO
from sklearn.pipeline import  Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, auc, average_precision_score
from lightgbm import LGBMClassifier
import matplotlib.pyplot as plt
from catboost import CatBoostClassifier

#Carga credenciales
def load_credentials(path="Credenciales.txt"):
    out={}
    with open(path,"r") as f:
        for linea in f:
            if "=" in linea:
                k,v=linea.strip().split("=",1)
                out[k]=v
    return out

#Carga dataset desde s3
def load_data_s3(bucket,key,cred):
    s3=boto3.client(
        "s3",
        aws_access_key_id=cred.get("Acces_Key_ID"),
        aws_secret_access_key=cred.get("Password"),
        region_name="eu-west-1",
    )
    obj=s3.get_object(Bucket=bucket,Key=key)
    return pd.read_csv(StringIO(obj["Body"].read().decode("utf-8")))

#Split temporal
def split_temporal(df):
    p70=df["order_date"].quantile(0.70)
    p90=df["order_date"].quantile(0.90)
    train=df[df["order_date"]<p70]
    val=df[(df["order_date"]>=p70)&(df["order_date"]<p90)]
    test=df[df["order_date"]>=p90]
    return train,val,test

#Entrena y evalua AUC
def fit_and_eval(pipeline, X_train, y_train, X_val, y_val, model, **fit_params):
    pipeline.fit(X_train, y_train, **fit_params)
    p = pipeline.predict_proba(X_val)[:,1]
    auc_val = roc_auc_score(y_val, p)
    ap_val  = average_precision_score(y_val, p)
    
    last_step_name = list(pipeline.named_steps.keys())[-1]
    internal_model = pipeline.named_steps[last_step_name]
    
    importances = internal_model.feature_importances_
    df_importances = pd.DataFrame({
        "feature": X_train.columns,
        "importance": importances
    }).sort_values("importance", ascending=False)

    all_results = {}
    all_results[f"result_{model}"] = {
        "model_name": model,
        "auc": auc_val,
        "ap": ap_val,
        "importances_df": df_importances
        }

    return pipeline, p, all_results


#Curvas ROC y PR
def plot_roc_pr(p,y_val,model):
    fpr,tpr,_=roc_curve(y_val,p)
    roc_auc=auc(fpr,tpr)
    plt.figure()
    plt.plot(fpr,tpr,label=f"{model} ROC={roc_auc:.3f}")
    plt.plot([0,1],[0,1])
    plt.legend()
    plt.show()

    prec,rec,_=precision_recall_curve(y_val,p)
    pr_auc=auc(rec,prec)
    plt.figure()
    plt.plot(rec,prec,label=f"{model} PR={pr_auc:.3f}")
    plt.legend()
    plt.show()

cred=load_credentials()
df=load_data_s3("zrive-ds-data","groceries/box_builder_dataset/feature_frame.csv",cred)

df_bought=df[df["outcome"]==1]
counts=df_bought.groupby("order_id")["variant_id"].nunique()
valid_ids=counts[counts>=5].index
df_filtered=df[df["order_id"].isin(valid_ids)]

#Cleanig data
cols_drop=["variant_id","product_type","order_id","user_id","created_at","vendor"]
df_model=df_filtered.drop(columns=cols_drop)

df_model=df_model.sort_values("order_date")
df_model["order_date"]=pd.to_datetime(df_model["order_date"])

train,val,test=split_temporal(df_model)

X_train=train.drop(columns=["outcome","order_date"])
y_train=train["outcome"]
X_val=val.drop(columns=["outcome","order_date"])
y_val=val["outcome"]

#Baseline
auc_baseline = roc_auc_score(y_val, X_val["global_popularity"])
ap_baseline  = average_precision_score(y_val, X_val["global_popularity"])

#Modelo RF
rf_lgbm = Pipeline([
    ("scaler", StandardScaler()),
    ("lgbm", LGBMClassifier(
        boosting_type="rf",
        n_estimators=500,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        random_state=42
    ))
])

rf_lgbm, p_rf_lgbm, res_lgbm = fit_and_eval(
    rf_lgbm,
    X_train, y_train,
    X_val, y_val,
    "RF_LGBM"
)

plot_roc_pr(p_rf_lgbm, y_val, "RF_LGBM")

#Modelo CatBoost con early stopping
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

cb_model, p_cb, res_cb = fit_and_eval(
    pipe_cb,
    X_train, y_train,
    X_val, y_val,
    "CAT",
    cb__eval_set=(X_val, y_val),
    cb__early_stopping_rounds=5,
    cb__use_best_model=True,
    cb__verbose=False
)

plot_roc_pr(p_cb,y_val,"CAT")