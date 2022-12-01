"""API Deployment for the Census application.

Author: Ollie Tian
Date: Nov 2022
Pylint score
"""
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

import json
import requests
import warnings

warnings.filterwarnings("ignore")

# Declare predict uri and sample dictionary
PREDICT_URI = "https://deploy-ml-with-heroku-offtian.herokuapp.com/predict"

sample = {
    "age": 39,
    "workclass": "State-gov",
    "fnlgt": 77516,
    "education": "Bachelors",
    "education-num": 13,
    "marital-status": "Never-married",
    "occupation": "Adm-clerical",
    "relationship": "Not-in-family",
    "race": "White",
    "sex": "Male",
    "capital-gain": 2174,
    "capital-loss": 0,
    "hours-per-week": 40,
    "native-country": "United-States",
}

# Retrieve response using POST
response = requests.post(PREDICT_URI, data=json.dumps(sample), timeout=30, verify=False)

dictionary = {
    "Request body": sample,
    "Status code": response.status_code,
    "Response body": response.json()["fetch"],
}
print(json.dumps(dictionary, indent=4))
