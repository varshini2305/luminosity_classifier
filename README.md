## Luminosity Classifier

#### Classify product from product catalog pdf to check if it is a lighting product or not.


- set up env

python3.11 -m venv lclassifier_env
source lclassifier_env/bin/activate
pip install -r requirements.txt

- update requirements with 

```
pipreqs . --force
```

```
dvc remote add -d gcp gs://de_pipelines/luminosity_classifier_data
```

```
python -m ipykernel install --user --name=lclassifier_env
```

```
git pull
dvc pull
streamlit run main.py
```