## Luminosity Classifier

#### Classify product from product catalog pdf to check if it is a lighting product or not.


##### set up venv

```
python3.11 -m venv lclassifier_env
source lclassifier_env/bin/activate
pip install -r requirements.txt
```

##### To run the notebook - light_classifier_analysis.ipynb

```
pip install jupyterlab
pip install ipykernel
python -m ipykernel install --user --name=lclassifier_env
jupyter lab
```

- update requirements file in case of additional dependencies 

```
pipreqs . --force
```

### dvc config

- gcloud set up
```
brew install --cask google-cloud-sdk 
# or download tr.gz file online and run install.sh (will add path) 

gcloud init
gcloud --version
gcloud auth login  (complete auth using gcp google account)
```


```
dvc init
dvc remote add -d gcp gs://gcs_bucket/destination_folder_path
export GOOGLE_APPLICATION_CREDENTIALS="config/gcp-service-account.json"
```

-  the gcs_bucket, gcs_project, needs to have necessary permissions using a service account

- IAM & Admin -> Service accounts -> create service account -> copy/download service_account.json -> provide necessary permissions

- IAM & Admin -> IAM -> select the service_account created -> ADD Role - Storage Object Viewer

- Cloud Storage -> Buckets -> gcs_bucket -> Permissions -> select/add the service account created - > add roles - Storage Legacy Bucket Reader, Storage Object Viewer

- store the service account credentials.json - in config/gcp-service-account.json (to be .gitignored)

- copy the credentials json (for streamlit deployment) - .streamlit/secrets.toml (to be .gitignored)

- while deploying streamlit on cloud - Advanced Settings -> Secrets -> paste the credentials content in .toml file - for data fetch, without sharing it via git


#### To execute the streamlit app in local (after above steps are complete)
 
```
git pull
dvc pull
streamlit run main.py
```