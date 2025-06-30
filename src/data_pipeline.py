import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

# ---------------------
# Feature Engineering Functions
# ---------------------

def create_aggregate_features(df):
    try:
        agg_df = df.groupby("CustomerId")["Amount"].agg(
            total_transaction_amount='sum',
            avg_transaction_amount='mean',
            transaction_count='count',
            std_transaction_amount='std'
        ).reset_index()
        df = df.merge(agg_df, on="CustomerId", how="left")
        return df
    except Exception as e:
        print(f"Error in create_aggregate_features: {e}")
        raise


def extract_datetime_features(df, time_col="TransactionStartTime"):
    try:
        df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
        df["transaction_hour"] = df[time_col].dt.hour
        df["transaction_day"] = df[time_col].dt.day
        df["transaction_month"] = df[time_col].dt.month
        df["transaction_year"] = df[time_col].dt.year
        return df.drop(columns=[time_col])
    except Exception as e:
        print(f"Error in extract_datetime_features: {e}")
        raise


def clean_and_engineer(df):
    try:
        df = create_aggregate_features(df)
        df = extract_datetime_features(df)
        return df
    except Exception as e:
        print(f"Error in clean_and_engineer: {e}")
        raise

# ---------------------
# Build Preprocessing Pipeline
# ---------------------

def build_preprocessing_pipeline(numeric_features, categorical_features):
    try:
        numeric_pipeline = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler())
        ])

        categorical_pipeline = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ])

        preprocessor = ColumnTransformer(transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features)
        ])

        return preprocessor
    except Exception as e:
        print(f"Error building preprocessing pipeline: {e}")
        raise

# ---------------------
# Apply Full Preprocessing Pipeline
# ---------------------

def prepare_data(df):
    try:
        df = clean_and_engineer(df)

        numeric_features = [
            "Amount", "Value",
            "transaction_hour", "transaction_day", "transaction_month", "transaction_year",
            "total_transaction_amount", "avg_transaction_amount",
            "transaction_count", "std_transaction_amount"
        ]

        categorical_features = [
            "AccountId", "SubscriptionId", "CustomerId", "CurrencyCode",
            "CountryCode", "ProviderId", "ProductId", "ProductCategory",
            "ChannelId", "PricingStrategy"
        ]

        pipeline = build_preprocessing_pipeline(numeric_features, categorical_features)
        return df, pipeline, numeric_features + categorical_features
    except Exception as e:
        print(f"Error in prepare_data: {e}")
        raise
