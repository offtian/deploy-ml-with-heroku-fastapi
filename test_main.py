import json
from fastapi.testclient import TestClient
from bs4 import BeautifulSoup

from main import app

# test Fast API root
def test_api_locally_get_root():
    """ Test Fast API root route"""

    with TestClient(app) as client:
        r = client.get("/")
    assert r.status_code == 200
    assert r.json()["greeting"]== "Welcome User!"
    
def test_api_locally_get_predictions_inf1():
    """ Test Fast API predict route with a '<=50K' salary prediction result """

    expected_res = "Predicts ['<=50K']"
    test_data = {
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
    headers = {"Content-Type": "application/json"}

    with TestClient(app) as client:
        r = client.post("/predict", data=json.dumps(test_data), headers=headers)
        assert r.status_code == 200
        assert (r.json()["fetch"][: len(expected_res)]) == expected_res
        
def test_api_locally_get_predictions_inf2():
    """ Test Fast API predict route with a '>50K' salary prediction result """

    expected_res = "Predicts ['>50K']"
    test_data = {
        "age": 40,
        "workclass": "State-gov",
        "fnlgt": 77516,
        "education": "Bachelors",
        "education-num": 13,
        "marital-status": "Never-married",
        "occupation": "Adm-clerical",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Male",
        "capital-gain": 20000,
        "capital-loss": 0,
        "hours-per-week": 40,
        "native-country": "United-States",
    }
    headers = {"Content-Type": "application/json"}

    with TestClient(app) as client:
        r = client.post("/predict", data=json.dumps(test_data), headers=headers)
        assert r.status_code == 200
        assert (r.json()["fetch"][: len(expected_res)]) == expected_res