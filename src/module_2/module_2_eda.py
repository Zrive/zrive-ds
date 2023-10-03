import boto3
import os
import pandas as pd
import matplotlib.pyplot as plt

# AWS credentials
aws_access_key_id = 'AKIAXN64CPXKVY56HGZZ'
aws_secret_access_key = '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'

# Initialize a session with boto3
session = boto3.Session(
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key,
)

# Create an S3 client
s3 = session.client('s3')

# Define the S3 bucket and objects you want to download
s3_bucket = 'zrive-ds-data'
s3_prefix = 'groceries/sampled-datasets/'

# Define a local directory to save the downloaded files
local_directory = 'Module2Data'

# Create the local directory if it doesn't exist
os.makedirs(local_directory, exist_ok=True)

# List files in the S3 bucket
response = s3.list_objects_v2(Bucket=s3_bucket, Prefix=s3_prefix)

# Download each file from S3 to the local directory
for obj in response.get('Contents', []):
    s3_object_key = obj['Key']
    local_file_path = os.path.join(local_directory, os.path.basename(s3_object_key))
    s3.download_file(s3_bucket, s3_object_key, local_file_path)

# Retrieving the data

path_data = 'Users/alvarochapela/Desktop/ZRIVE/ProyectosZrive/zrive-ds/Module2Data'

orders = pd.read_parquet(f"{path_data}/orders.parquet")
users = pd.read_parquet(f"{path_data}/users.parquet")
regulars = pd.read_parquet(f"{path_data}/regulars.parquet")
inventory = pd.read_parquet(f"{path_data}/inventory.parquet")
abandoned_carts = pd.read_parquet(f"{path_data}/abandoned_carts.parquet")

