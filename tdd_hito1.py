# DOCUMENTO DE DESCRIPCIÓN TÉCNICA - TDD
import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

------------------------------------------------------------------------------------------------------------
# HITO 1: FASE DE EXPLORACIÓN
------------------------------------------------------------------------------------------------------------
# 1. Se Cargan datos
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "feature_frame_filtered.csv")
print("Cargando datos desde:", DATA_PATH)
df = pd.read_csv(DATA_PATH)
print(" Datos cargados correctamente:", df.shape)
# Devuelve --> Datos cargados correctamente: (2880549, 27)



# 2. Preprocesamiento
# 2.1 Se definen: Features (X) y target (y)
y = df['outcome'] # 1 si compró, 0 si no
feature_cols = [col for col in df.columns if col != 'outcome'] # todas las columas menos 'outcome'
X = df[feature_cols].copy()

# 2.1 Se codifican las variables categóricas para que los modelos lineales puedan procesarlas.
cat_cols = ['product_type', 'vendor']  # columnas categóricas
label_encoders = {}  # diccionario para guardar los LabelEncoders y poder usar en inferencia
for col in cat_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))  # se transforma cada categoría en un número
    label_encoders[col] = le  # se guarda el encoder para esta columna

# 2.3 Se transforman las fechas a números.
date_cols = ['created_at', 'order_date']
for col in date_cols:
    X[col] = pd.to_datetime(X[col], errors='coerce')
    X[col + '_dayofweek'] = X[col].dt.dayofweek
    X[col + '_day'] = X[col].dt.day
X = X.drop(columns=date_cols) # se eliminan las columnas originales de datetime



# 3. Se reduce el tamaño del dataset (STRATIFIED SAMPLING) porque con todos los datos las AUC son iguales
sample_size = 50000 # me quedo con 50000 filas
X_small, _, y_small, _ = train_test_split(
    X, y, train_size=sample_size, stratify=y, random_state=42
)

# 3.1 IMPORTANTE: Añadir ruido (es decir, eliminar algunas columnas para que L1 ponga coeficientes a cero)
import numpy as np
num_noise_cols = 10
for i in range(num_noise_cols):
    X_small[f'noise_{i}'] = np.random.rand(X_small.shape[0])

for i in range(num_noise_cols): # se comprueba su irrelevancia respecto a y
    col = f'noise_{i}'
    corr = np.corrcoef(X_small[col], y_small)[0,1]
    auc = roc_auc_score(y_small, X_small[col])
    print(f"{col}: correlación={corr:.3f}, AUC individual={auc:.3f}")
# Devuelve --> noise_0: correlación=0.005, AUC individual=0.513
# noise_1: correlación=-0.001, AUC individual=0.496
# noise_2: correlación=-0.001, AUC individual=0.499
# noise_3: correlación=0.000, AUC individual=0.501
# noise_4: correlación=0.000, AUC individual=0.501
# noise_5: correlación=-0.005, AUC individual=0.486
# noise_6: correlación=-0.003, AUC individual=0.492
# noise_7: correlación=-0.009, AUC individual=0.476
# noise_8: correlación=0.004, AUC individual=0.510
# noise_9: correlación=0.002, AUC individual=0.505


# 4. Se divide el dataset en: TRAIN (70%) - VALIDATION (20%) - TEST (10%)
from sklearn.model_selection import train_test_split
# 4.1 Primera división: Train (70%) y Temp (30%)
X_train, X_temp, y_train, y_temp = train_test_split(
    X_small, y_small, test_size=0.3, stratify=y_small, random_state=42
)
# 4.2 Segunda división: Validation (20%) y Test (10%) del total
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=1/3, stratify=y_temp, random_state=42
)
# 4.3 Comprobaciones:
print(" División completada:")
print(f"Train: {X_train.shape[0]} filas")
print(f"Validation: {X_val.shape[0]} filas")
print(f"Test: {X_test.shape[0]} filas")
# Devuelve --> División completada:
# Train: 35000 filas
# Validation: 10000 filas
# Test: 5000 filas


# 5. Se escalan features (mean y sd entre 0 y 1)
scaler = MinMaxScaler(feature_range=(0,1))
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)



# 6. Se entrenan varios modelos lineales para ver cuál proporciona un mín error de validación.
# 6.1 Variantes de R.Logística (Lasso (L1), Ridge (L2), estándar (sin regularización, MSE))
models = {
    "LogReg_L2": LogisticRegression(penalty='l2', solver='saga', C=0.5, max_iter=1000, tol=1e-3),
    "LogReg_L1": LogisticRegression(penalty='l1', solver='saga', C=0.5, max_iter=1000, tol=1e-3),
    "LogReg_None": LogisticRegression(penalty=None, solver='saga', max_iter=1000, tol=1e-3)
}

#6.2 Entrenamiento y Validación
best_model = None # guarda el mejor modelo
best_score = 0    # guarda la mejor métrica de validación

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_val_pred = model.predict_proba(X_val_scaled)[:, 1]
    score = roc_auc_score(y_val, y_val_pred)
    print(f"{name} AUC en validación: {score:.4f}")

    if name in ["LogReg_L1", "LogReg_L2"]: # coeficientes de las columnas de ruido
        print(f"{name} coeficientes de ruido:", model.coef_[0, -num_noise_cols:])

    if score > best_score:
        best_score = score
        best_model = model
        best_model_name = name

print(f"\nMejor modelo: {best_model_name} con AUC {best_score:.4f}")
# Devuelve: LogReg_L2 AUC en validación: 0.7177
# LogReg_L2 coeficientes de ruido: [ 0.28380544  0.11840657 -0.07367345  0.06758003 -0.06258619 -0.16851927
# LogReg_L1 AUC en validación: 0.7453
# LogReg_L1 coeficientes de ruido: [ 0.2550971   0.06224398 -0.03293451  0.00648352  0.         -0.1171099
#  -0.10486589 -0.26279983  0.          0.11304079]
# LogReg_None AUC en validación: 0.7330
# Mejor modelo: LogReg_L1 con AUC 0.7453

# CONCLUSIÓN: Me quedo con Regresión logística (Lasso)



