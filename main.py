# streamlit run main.py
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from transformers import AutoTokenizer, AutoModel
from functools import lru_cache
import numpy as np
import streamlit as st
import sys
import subprocess
from google.cloud import storage
from google.oauth2 import service_account
import toml
import json
import os
# Load secrets from secrets.toml
secrets_path = ".streamlit/secrets.toml"
secrets = toml.load(secrets_path)
gcp_service_account_json = secrets.get('gcp', {})

# Define the path for the GCP service account JSON file
service_account_json_path = "config/gcp-service-account.json"

# Ensure the existence of the config folder
config_folder = os.path.dirname(service_account_json_path)
os.makedirs(config_folder, exist_ok=True)

# Set the GOOGLE_APPLICATION_CREDENTIALS environment variable
with open(service_account_json_path, "w") as json_file:
    json.dump(gcp_service_account_json, json_file)

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = service_account_json_path
# Accessing the dictionary


# gcp_service_account_json = ast.literal_eval(secrets.get("gcp", {}).get("service_account_json", {}))

# st.write(gcp_service_account_json)
# st.write(type(gcp_service_account_json))
# st.write(gcp_service_account_json.keys())

# # Authenticate with GCP using the service account JSON key
# credentials = service_account.Credentials.from_service_account_info(gcp_service_account_json)
# client = storage.Client(credentials=credentials)


def run_dvc_command(command):
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode == 0:
        print(f"{command} - run successfully with {result.stdout}")
    else:
        print(f"{command} failed with {result.stderr}")

# # DVC commands
remote_add_command = [sys.executable, "-m", "dvc", "remote", "add", "--default", "gcs", "gs://de_pipelines/light_classifier_data"]
# config_command = [sys.executable, "-m", "dvc", "config", "core.hardlink_lock", "true"]
# dvc_tmp_lock_remove = [sys.executable, "rm", ".dvc/tmp/lock"]
# remote_list_command = [sys.executable, "-m", "dvc", "remote", "list"]
pull_command = [sys.executable, "-m", "dvc", "pull"]

# # Run DVC commands in sequence
run_dvc_command(remote_add_command)
# run_dvc_command(config_command)
# run_dvc_command(remote_list_command)
run_dvc_command(pull_command)



from pdf_parser import predict_luminosity_from_url, predict_if_lighting

# Streamlit app
def main():
    st.title("Product Classification App")
    product_name = st.text_input("Enter the product pdf url:")
    

    if product_name:
        if 'www.' in product_name or '.pdf' in product_name:
            # is_lighting, confidence_score = predict_if_lighting(product_name)
            is_lighting, product_title = predict_luminosity_from_url(product_name)
            st.write(f"The product '{product_title}' is lighting: {is_lighting}")
        else:
            is_lighting, confidence_score = predict_if_lighting(product_name)
            st.write(f"The product '{product_name}' luminosity : {is_lighting} with confidence {confidence_score}")

# Run the app
if __name__ == "__main__":
    main()
