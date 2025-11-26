import joblib
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

s3_path = "s3://zrive-ds-data/groceries/trained-models/model.joblib"

local_path = r"C:/Users/manue/OneDrive/Escritorio/Zrive_DS/Module 6/task_module_6/zrive-ds/bin/model.joblib"

os.makedirs(os.path.dirname(local_path), exist_ok=True)

with fs.open(s3_path, "rb") as f:
    with open(local_path, "wb") as out:
        out.write(f.read())

