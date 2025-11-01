import joblib
from Module_4_fit import fit_model
from datetime import datetime

def push_model():
    model = fit_model()
    fecha = datetime.now().strftime("%Y_%m_%d")
    nombre = f"model_{fecha}.pkl"
    joblib.dump(model, nombre)
    return nombre

push_model()
