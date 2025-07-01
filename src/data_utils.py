import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(path):
    try:
        df = pd.read_csv(path)
        return df
    except Exception as e:
        print(f" Failed to load data: {e}")
        return None

def split_data(df, target_col, test_size=0.2, random_state=42):
    try:
        X = df.drop(columns=[target_col])
        y = df[target_col]
        return train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)
    except Exception as e:
        print(f" Failed to split data: {e}")
        return None, None, None, None
