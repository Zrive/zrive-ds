import os
import boto3
from dotenv import load_dotenv
import pandas as pd

#cargo las claves
load_dotenv()

#leo las claves y las guardo
aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
region = os.getenv("AWS_DEFAULT_REGION", "us-east-1")

#entro en s3 para coger los datos
s3 = boto3.client(
    "s3",
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key,
    region_name=region
)

#listado de archivos por descagar
parquet_files = [
    "orders.parquet",
    "regulars.parquet",
    "abandoned_carts.parquet",
    "inventory.parquet",
    "users.parquet"
]

#ruta que seguir en el bucket, la carpeta data si ya esta creada no hace nada la ultima linea de este parágrafo
bucket_name = "zrive-ds-data"
prefix = "groceries/sampled-datasets/"
local_dir = "./data"
os.makedirs(local_dir, exist_ok=True)

#descargamos los archivos de uno en uno
for file_name in parquet_files:
    key = prefix + file_name
    local_path = os.path.join(local_dir, file_name)
    print("Descargando desde S3 key:", key)
    s3.download_file(bucket_name, key, local_path)

print("Fin de descarga!")

#ahora para facilitar la lectura y poder interpretar los resultados los convierto a csv
output_dir = "./data_csv"
os.makedirs(output_dir, exist_ok=True)

for file_name in parquet_files:
    parquet_path = os.path.join(local_dir, file_name)
    csv_name = file_name.replace(".parquet", ".csv")
    csv_path = os.path.join(output_dir, csv_name)

    try:
        df = pd.read_parquet(parquet_path)
        df.to_csv(csv_path, index=False)
        print(f" {csv_name} guardado en {output_dir}")
    except Exception as e:
        print(f" Error convirtiendo {file_name}: {e}")

print("\n Conversión a CSV completada.")