from src.data_utils import load_and_split_data

def test_data_loading():
    X_train, X_test, y_train, y_test = load_and_split_data("Data/processed/cleaned_data.csv", "fraudresult")
    assert X_train is not None, " Data load failed"
    assert X_train.shape[0] > 0, " No rows loaded"
    print("✅ Data loading test passed")
