"""API Deployment for the Census application.

Author: Ollie Tian
Date: Nov 2022
Pylint score
"""

import json
import requests

# Declare predict uri and sample dictionary
PREDICT_URI = "https://git.heroku.com/udacity-offtian.git/predict/"

sample = {
    "age": 39,
    "workclass": "State-gov",
    "fnlgt": 77516,
    "education": "Bachelors",
    "education_num": 13,
    "marital_status": "Never-married",
    "occupation": "Adm-clerical",
    "relationship": "Not-in-family",
    "race": "White",
    "sex": "Male",
    "capital_gain": 2174,
    "capital_loss": 0,
    "hours_per_week": 40,
    "native_country": "United-States",
}

# Retrieve response using POST
response = requests.post(PREDICT_URI, json=sample, timeout=5)

dictionary = {
    "Request body": json.dumps(sample),
    "Status code": response.status_code,
}
print(json.dumps(dictionary, indent=4))