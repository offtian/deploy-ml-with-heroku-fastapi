from sklearn.ensemble import RandomForestClassifier
from .data import process_data
from sklearn.metrics import fbeta_score, precision_score, recall_score
import pickle

# save the model
def save_model(model, file):
    """
    save_model: saves a pickled model to a file.
    Args:
        model: The model to save
        file: The file to save the model to.
    """
    with open(file, "wb") as f:
        pickle.dump(model, f)


# load the model
def load_model(file):
    """
    load_model: loads a pickled model from a file.
    Args:
        file: The file to load the model from.
    Returns:
        model: logistic regression model
    """
    with open(file, "rb") as f:
        model = pickle.load(f)
    return model


# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """
    clf = RandomForestClassifier(
        n_estimators=150,
        n_jobs=-1,
        max_depth=25,
        min_samples_split=60,
        max_features=30,
        random_state=42,
    )
    clf.fit(X_train, y_train)
    return clf


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """Run model inferences and return the predictions.

    Inputs
    ------
    model : ???
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    return model.predict(X)


def compute_slice_metric(model, cat_feat, cat_features, encoder, lb, X):
    """Output the performance of the model on slices of the data
    Inputs
    ------
    model : ???
        Trained machine learning model.
    cat_feat: str
        A given categorical variable
    cat_features: list
        All categorical variables
    encoder : sklearn.preprocessing._encoders.OneHotEncoder
        Trained sklearn OneHotEncoder, only used if training=False.
    lb : sklearn.preprocessing._label.LabelBinarizer
        Trained sklearn LabelBinarizer, only used if training=False.
    X : np.array
        Data used for prediction.
    Returns
    -------
    performance : dict
        F1 performance of the model on slices of the data.
    """

    output = {}
    for feat in X[cat_feat].unique():
        sample_data = X[X[cat_feat] == feat]
        sample_X_test, sample_y_test, _, _ = process_data(
            sample_data,
            categorical_features=cat_features,
            label="salary",
            training=False,
            encoder=encoder,
            lb=lb,
        )
        sample_y_pred = inference(model, sample_X_test)
        _, _, f1 = compute_model_metrics(sample_y_test, sample_y_pred)
        print(f"For {feat} in {cat_feat}, the F1 of the model prediction is {f1}")
        output[feat] = f1
    return output
