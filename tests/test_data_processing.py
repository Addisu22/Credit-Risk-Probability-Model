import pandas as pd
from src.data_utils import load_data, split_data

def test_load_data():
    df = load_data("Data/processed/cleaned_data.csv")
    assert df is not None
    assert isinstance(df, pd.DataFrame)
    print("✅ test_load_data passed")

def test_split_data():
    df = load_data("Data/processed/cleaned_data.csv")
    X_train, X_test, y_train, y_test = split_data(df, "fraudresult")
    assert X_train.shape[0] > 0
    assert y_test.shape[0] > 0
    print("✅ test_split_data passed")
