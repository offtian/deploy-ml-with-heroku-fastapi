import pandas as pd
import numpy as np
import pytest

import os


@pytest.fixture
def data():
    """Retrieve Cleaned Dataset"""
    data_path = os.path.join(os.path.abspath(os.getcwd()), "data")
    train_file = os.path.join(data_path, "census_clean.csv")
    df = pd.read_csv(train_file)
    df = df.iloc[:, :-1]  # exclude label
    return df


def test_data_nullness(data: pd.DataFrame):
    """Check that data has no null value

    Args:
        data (pd.DataFrame): dataframe
    """

    assert data.shape == data.dropna().shape, "There are null values in the dataframe."


def test_data_cleaned(data):
    """Check that there are no ? characters any features"""
    columns = data.columns
    for col in columns:
        filt = data[col] == "?"
        assert filt.sum() == 0, f"Found ? character in feature {col}"


def test_data_column_name(data):
    """Check that there are no spaces in the column names"""
    col_names = data.columns
    for col in col_names:
        assert " " not in col, f"Found space character in feature {col}"
