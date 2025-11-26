import pandas as pd
import s3fs
import os

def load_credentials(path="Credenciales.txt"):
    creds = {}
    with open(path, "r") as f:
        for line in f:
            if "=" in line:
                k, v = line.strip().split("=", 1)
                creds[k] = v
    return creds

creds = load_credentials()

fs = s3fs.S3FileSystem(
    key=creds["Acces_Key_ID"],
    secret=creds["Password"],
    client_kwargs={"region_name": creds.get("AWS_REGION", "eu-west-1")}
)

remote_dir = "s3://zrive-ds-data/groceries/sampled-datasets/"
local_dir = r"C:/Users/manue/OneDrive/Escritorio/Zrive_DS/Module 6/task_module_6/zrive-ds/data/"

files = fs.ls(remote_dir)

for f in files:
    filename = os.path.basename(f)
    local_path = os.path.join(local_dir, filename)
    df = pd.read_parquet(f, filesystem=fs)
    df.to_parquet(local_path, index=False)
    print("Descargado:", filename)
