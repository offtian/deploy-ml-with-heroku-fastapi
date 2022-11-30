# Put the code for your API here.
import os
from pathlib import Path

import pickle
import pandas as pd
from starter.ml.data import process_data
from starter.ml.model import inference


from fastapi import FastAPI, HTTPException, Request
from fastapi.templating import Jinja2Templates

# BaseModel from Pydantic is used to define data objects.
from pydantic import BaseModel, Field


if "DYNO" in os.environ and os.path.isdir(".dvc"):
    os.system("dvc config core.no_scm true")
    os.system("dvc config core.hardlink_lock true")
    if os.system("dvc pull") != 0:
        exit("dvc pull failed")
    os.system("rm -r .dvc .apt/usr/lib/dvc")

# Directory Paths
cur_path = str(Path(__file__).parent.absolute())
feature_encoding_file = cur_path + '/model/encoder.pkl'
census_model_file = cur_path + '/model/rf_model.pkl'
lb_file = cur_path + '/model/lb.pkl'

# Declare the data object with its components and their type.
class census_data(BaseModel):
    age: int = Field(..., example=39)
    workclass: str = Field(..., example="State-gov")
    fnlgt: int = Field(..., example=77516)
    education: str = Field(..., example="Bachelors")
    education_num: int = Field(..., example=13, alias="education-num")
    marital_status: str = Field(..., example="Never-married", alias="marital-status")
    occupation: str = Field(..., example="Adm-clerical")
    relationship: str = Field(..., example="Not-in-family")
    race: str = Field(..., example="White")
    sex: str = Field(..., example="Male")
    capital_gain: int = Field(..., example=2174, alias="capital-gain")
    capital_loss: int = Field(..., example=0, alias="capital-loss")
    hours_per_week: int = Field(..., example=40, alias="hours-per-week")
    native_country: str = Field(..., example="United-States", alias="native-country")
    
# Declare fastapi app
app = FastAPI()

# Load model artifacts upon startup of the application
@app.on_event("startup")
async def startup_event():
    global clf, encoder, lb
    
    # load data encoder
    with open(feature_encoding_file, "rb") as file:
        encoder = pickle.load(file)
        print("census_app - loaded {}".format(feature_encoding_file))

    # load LabelBinarizer
    lb = pickle.load(open(lb_file, "rb"))
    print("census_app - loaded {}".format(lb_file))
    
    # load model
    clf = pickle.load(open(census_model_file, "rb"))
    print("census_app - loaded {}".format(census_model_file))
    
@app.get("/")
async def root(user: str = "User"):
    return {"greeting": f"Welcome {user}!"}

@app.post("/predict")
async def get_prediction(payload: census_data):
    
    # Convert input data into a dictionary and then pandas dataframe
    df = pd.DataFrame.from_dict([payload.dict(by_alias=True)])

    cat_features = [
                "workclass",
                "education",
                "marital-status",
                "occupation",
                "relationship",
                "race",
                "sex",
                "native-country",
            ]

    # Process the raw input data
    input, _, _, _ = process_data(
            df, categorical_features=cat_features, training=False, encoder=encoder, lb=lb
            )
    
    # Model inference
    y_pred = inference(clf, input)
    label = y_pred.item()

    if label==0:
        output = "<=50K"
    else:
        output = ">50K" 

    return {"fetch": f"Predicts ['{output}']"}