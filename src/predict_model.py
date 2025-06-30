
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score
import mlflow
import pickle

# ------------------------------
# 1. Load Data
# ------------------------------
def load_data(file_path):
    try:
        df = pd.read_csv(file_path)
        print(f" Loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        return df
    except Exception as e:
        print(f" Failed to load data: {e}")
        return None

# ------------------------------
# 2. Feature Engineering
# ------------------------------
def preprocess_data(df):
    try:
        df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])
        df['TransactionHour'] = df['TransactionStartTime'].dt.hour
        df['TransactionDay'] = df['TransactionStartTime'].dt.day
        df['TransactionMonth'] = df['TransactionStartTime'].dt.month
        df['TransactionYear'] = df['TransactionStartTime'].dt.year
        
        df.drop(columns=['TransactionStartTime'], inplace=True)
        return df
    except Exception as e:
        print(f" Error in preprocessing: {e}")
        return None

# ------------------------------
# 3. Split Features
# ------------------------------
def split_features_target(df):
    try:
        target = 'FraudResult'
        features = [
            'AccountId', 'SubscriptionId', 'CustomerId', 'CurrencyCode',
            'CountryCode', 'ProviderId', 'ProductId', 'ProductCategory',
            'ChannelId', 'Amount', 'Value',
            'TransactionHour', 'TransactionDay', 'TransactionMonth', 'TransactionYear',
            'PricingStrategy'
        ]
        return df[features], df[target]
    except Exception as e:
        print(f" Error splitting features and target: {e}")
        return None, None

# ------------------------------
# 4. Preprocessing Pipeline
# ------------------------------
def build_pipeline(model):
    numeric = ['Amount', 'Value', 'TransactionHour', 'TransactionDay', 'TransactionMonth', 'TransactionYear']
    categorical = ['AccountId', 'SubscriptionId', 'CustomerId', 'CurrencyCode',
                   'CountryCode', 'ProviderId', 'ProductId', 'ProductCategory',
                   'ChannelId', 'PricingStrategy']
    
    numeric_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer([
        ('num', numeric_pipeline, numeric),
        ('cat', categorical_pipeline, categorical)
    ])

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])
    
    return pipeline

# ------------------------------
# 5. Train and Track Model
# ------------------------------
def train_and_track(X_train, X_test, y_train, y_test, model_name, model):
    try:
        pipeline = build_pipeline(model)
        with mlflow.start_run(run_name=model_name):
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)
            y_proba = pipeline.predict_proba(X_test)[:, 1]

            # Metrics
            roc_auc = roc_auc_score(y_test, y_proba)
            print(f"🔍 {model_name} - ROC AUC: {roc_auc:.4f}")
            print("📊 Classification Report:")
            print(classification_report(y_test, y_pred))

            # MLflow tracking
            mlflow.log_param("model", model_name)
            mlflow.log_metric("roc_auc", roc_auc)
            # mlflow.sklearn.log_model(pipeline, "model", input_example=X_test.iloc[:5], registered_model_name=model_name)
            mlflow.sklearn.log_model(sk_model=pipeline, name=model_name, input_example=X_test.iloc[:5], registered_model_name=model_name)

             # Save model locally as .pkl file
            filename = f"{model_name}_pipeline.pkl"
            with open(filename, 'wb') as f:
                pickle.dump(pipeline, f)
            print(f" {model_name} pipeline saved locally as {filename}")

            print(f" {model_name} model trained and tracked in MLflow")
    except Exception as e:
        print(f" Error in model training ({model_name}): {e}")
