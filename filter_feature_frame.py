import pandas as pd
import os

# Construir la ruta relativa al archivo CSV
file_path = os.path.join(os.path.dirname(__file__), 'data', 'feature_frame.csv')

# Leer el CSV en un DataFrame
df = pd.read_csv(file_path)

# Contar cuántos productos tiene cada usuario
user_counts = df['user_id'].value_counts()

# Filtrar usuarios con al menos 5 productos
valid_users = user_counts[user_counts >= 5].index

# Filtrar el DataFrame original
filtered_df = df[df['user_id'].isin(valid_users)]

# Guardar resultado (opcional)
filtered_df.to_csv(os.path.join(os.path.dirname(__file__), 'data', 'feature_frame_filtered.csv'), index=False)

print(f"Filtrado realizado: {len(filtered_df)} filas correspondientes a {len(valid_users)} usuarios.")
# Me devuelve: Filtrado realizado: 2880549 filas correspondientes a 1937 usuarios.
