# Script to train machine learning model.

import pandas as pd
from sklearn.model_selection import train_test_split
from ml.data import process_data
from ml.model import (
    save_model,
    train_model,
    inference,
    compute_model_metrics,
    compute_slice_metric,
)
from pathlib import Path
import os

# Add the necessary imports for the starter code.

# Add code to load in the data.
# Load and clean raw data
data_path = os.path.join(os.path.abspath(os.getcwd()), "data")
data = pd.read_csv(data_path + "/census_clean.csv")


# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20)

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

# Proces the train data with the process_data function.
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# Proces the test data with the process_data function.
X_test, y_test, _, _ = process_data(
    test,
    categorical_features=cat_features,
    label="salary",
    training=False,
    encoder=encoder,
    lb=lb,
)

model_dir = os.path.join(
    os.path.abspath(os.getcwd()), "model"
)  # the directory where the model will be stored.
save_model(encoder, os.path.join(model_dir, "encoder.pkl"))  # save the label encoder
save_model(lb, os.path.join(model_dir, "lb.pkl"))  # save the encoder
model_path = os.path.join(model_dir, "rf_model.pkl")  # path to the model
slice_output = os.path.join(model_dir, "slice_output.txt")  # path to the slice output

# Train and save a model.
classifier = train_model(X_train, y_train)
save_model(classifier, model_path)

y_train_predict = inference(
    classifier, X_train
)  # make predictions on the training data
train_precision, train_recall, train_fbeta = compute_model_metrics(
    y_train, y_train_predict
)  # compute the metrics for the training data
print(
    f"train_precision: {train_precision}, train_recall: {train_recall}, train_fbeta: {train_fbeta}"
)  # print the metrics for the training data

y_test_predict = inference(classifier, X_test)  # make predictions on the test data
test_precision, test_recall, test_fbeta = compute_model_metrics(
    y_test, y_test_predict
)  # compute the metrics for the test data
print(
    f"test_precision: {test_precision}, test_recall: {test_recall}, test_fbeta: {test_fbeta}"
)  # print the metrics for the test data

# Output the performance of the model on slices of the data
performance = compute_slice_metric(
    classifier, "workclass", cat_features, encoder, lb, data
)
list_of_strings = [f"{key} : {performance[key]}" for key in performance]
with open(slice_output, "w") as file:
    [file.write(f"{st}\n") for st in list_of_strings]
