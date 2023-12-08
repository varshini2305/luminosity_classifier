# python utils/gcs_utils.py
from google.cloud import storage

from google.oauth2 import service_account
import toml
import json
# Load secrets from secrets.toml
secrets_path = ".streamlit/secrets.toml"
secrets = toml.load(secrets_path)
gcp_service_account_json = secrets.get('gcp', {})


with open("config/gcp-service-account.json", "w") as json_file:
    json.dump(gcp_service_account_json, json_file)

service_account_json = "config/gcp-service-account.json"

import os
# Set the GOOGLE_APPLICATION_CREDENTIALS environment variable
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = service_account_json


def list_gcs_objects(bucket_name):
    # Create a GCS client using the service account credentials
    client = storage.Client()

    # Specify the GCS bucket
    bucket = client.get_bucket(bucket_name)

    # List objects in the bucket
    blobs = bucket.list_blobs()
    for blob in blobs:
        print(blob.name)

# Replace 'your-bucket-name' with your actual GCS bucket name
list_gcs_objects('de_pipelines')

from google.cloud import storage
import os

# Define your GCS bucket and paths
bucket_name = "de_pipelines"
source_folder = "light_classifier_data"
destination_folders = ["data", "models"]  # You can change this to "models" for the other folder

# Initialize the GCS client with your service account credentials
client = storage.Client.from_service_account_json(service_account_json)

# Get the bucket and list the files in the source folder
bucket = client.get_bucket(bucket_name)
blobs = bucket.list_blobs(prefix=source_folder)

# Create destination folders if they don't exist
for destination_folder in destination_folders:
    os.makedirs(destination_folder, exist_ok=True)

files_to_download = ['data/test_data.pkl',
'data/train_data.pkl',
'models/bert_model.pkl',
'models/bert_tokenizer.pkl',
'models/light_standard_scaler.pkl',
'models/light_vectorizer.pkl',
'models/lr_light_model.pkl'
]
# Get the bucket and download each specified file
for file_path in files_to_download:
    # Combine the source folder and file path
    blob_path = os.path.join(source_folder, file_path)

    # Download the file to the local destination folder
    local_destination = os.path.join("models" if "models" in file_path else "data", os.path.basename(file_path))
    os.makedirs(os.path.dirname(local_destination), exist_ok=True)

    try:
        blob = client.bucket(bucket_name).get_blob(blob_path)
        blob.download_to_filename(local_destination)
        print(f"Downloaded: {blob_path} to {local_destination}")
    except Exception as e:
        print(f"Error downloading {blob_path}: {e}")

print("Download complete.")