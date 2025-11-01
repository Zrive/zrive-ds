import os
import json
import joblib
import pandas as pd
from Module_4_push import push_model

def predict(df_new_data):

    if isinstance(df_new_data,str) and df_new_data.endswith(".json"):
        with open(df_new_data,"r") as f:
            data = json.load(f)
        df_new_data = pd.DataFrame(data)

    modelos = [f for f in os.listdir() if f.startswith("model_") and f.endswith(".pkl")]

    if len(modelos)==0:
        model_path = push_model()
    else:
        modelos.sort(reverse=True)
        model_path = modelos[0]

    model = joblib.load(model_path)
    p = model.predict_proba(df_new_data)[:,1]
    return pd.Series(p,name="Predictions")
