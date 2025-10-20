# DOCUMENTO DE DESCRIPCIÓN TÉCNICA - TDD
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import joblib
from datetime import datetime

# ---------------------------------------------------------------------------------------------------------
# HITO 2: CÓDIGO MVP
# ---------------------------------------------------------------------------------------------------------
# 1. Carga de datos
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "feature_frame_filtered.csv")  # dataset ya filtrado

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"No se encontró el CSV: {DATA_PATH}")

df = pd.read_csv(DATA_PATH)



# 2. Preprocesamiento
# Variable objetivo
y = df['outcome']  # 1 si compró, 0 si no
X = df.drop(columns=['outcome'])

# Columnas categóricas
cat_cols = ['product_type', 'vendor']
label_encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    label_encoders[col] = le

# Fechas
date_cols = ['created_at', 'order_date']
for col in date_cols:
    X[col] = pd.to_datetime(X[col], errors='coerce')
    X[col + '_dayofweek'] = X[col].dt.dayofweek
    X[col + '_day'] = X[col].dt.day
X = X.drop(columns=date_cols)



# 3. Reducción de dataset para pruebas rápidas (opcional)
sample_size = 50000
X_small, _, y_small, _ = train_test_split(
    X, y, train_size=sample_size, stratify=y, random_state=42
)

# Añadir columnas de ruido para verificar regularización
num_noise_cols = 10
for i in range(num_noise_cols):
     X_small[f'noise_{i}'] = np.random.rand(X_small.shape[0])



# 4. División Train/Validation/Test
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=1/3, stratify=y_temp, random_state=42
)
print("División completada:")
print(f"Train: {X_train.shape[0]}, Validation: {X_val.shape[0]}, Test: {X_test.shape[0]}")

# Escalado
scaler = MinMaxScaler(feature_range=(0, 1))
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)



# 4. Entrenamiento del modelo final (Lasso L1 elegido en H1)
model = LogisticRegression(penalty='l1', solver='saga', C=0.5, max_iter=1000, tol=1e-3)
model.fit(X_train_scaled, y_train)



# 5. Validación
y_val_pred = model.predict_proba(X_val_scaled)[:,1]
auc_val = roc_auc_score(y_val, y_val_pred)
print(f"AUC de validación: {auc_val:.4f}")

# Coeficientes de ruido
print("Coeficientes de las columnas de ruido:", model.coef_[0, -num_noise_cols:])



# 6. Guardar modelo y scaler para producción
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)
joblib.dump(model, os.path.join(MODEL_DIR, "logreg_l1_model.pkl"))
joblib.dump(scaler, os.path.join(MODEL_DIR, "minmax_scaler.pkl"))

print("Modelo y scaler guardados correctamente en:", MODEL_DIR)
