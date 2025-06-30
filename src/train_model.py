import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.exceptions import NotFittedError

# ----------------------------------------
#  Load and preprocess raw credit data
# ----------------------------------------
def preprocess_raw_data(df):
    try:
        print(" Cleaning and transforming raw data...")

        # Convert to datetime
        df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'], errors='coerce')

        # Drop duplicates
        df.drop_duplicates(inplace=True)

        # Drop rows with missing TransactionStartTime
        df = df.dropna(subset=['TransactionStartTime'])

        # Extract temporal features
        df['TransactionHour'] = df['TransactionStartTime'].dt.hour
        df['TransactionDay'] = df['TransactionStartTime'].dt.day
        df['TransactionMonth'] = df['TransactionStartTime'].dt.month
        df['TransactionYear'] = df['TransactionStartTime'].dt.year

        # Drop original timestamp
        df.drop(columns=['TransactionStartTime'], inplace=True)

        # Aggregate features per customer
        agg_df = df.groupby('CustomerId').agg({
            'Amount': ['sum', 'mean', 'count', 'std'],
            'Value': 'mean',
            'TransactionHour': 'mean'
        }).reset_index()

        agg_df.columns = ['CustomerId', 'TotalTransactionAmount', 'AvgTransactionAmount', 'TransactionCount',
                          'StdTransactionAmount', 'AvgValue', 'AvgHour']

        # Merge back aggregated features
        df = df.merge(agg_df, on='CustomerId', how='left')

        print(" Preprocessing complete.")
        return df

    except Exception as e:
        print(f" Error during preprocessing: {e}")
        return None


# ----------------------------------------
#  Build preprocessing pipeline
# ----------------------------------------
def build_preprocessing_pipeline(df):
    try:
        # Define feature groups
        numeric_features = [
            'Amount', 'Value']
        # , 'TransactionHour', 'TransactionDay',         'TransactionMonth', 'TransactionYear', 'TotalTransactionAmount',          'AvgTransactionAmount', 'TransactionCount', 'StdTransactionAmount', 'AvgValue', 'AvgHour'        ]

        categorical_features = [
            # 'AccountId', 'SubscriptionId', 'CustomerId', 
            'CurrencyCode',  'CountryCode']
        # , 'ProviderId', 'ProductId', 'ProductCategory',           'ChannelId', 'PricingStrategy'        ]

        # Define numeric pipeline
        numeric_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ])

        # Define categorical pipeline
        categorical_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore'))
        ])

        # Full preprocessing pipeline
        preprocessor = ColumnTransformer([
            ('numer', numeric_pipeline, numeric_features),
            ('cat', categorical_pipeline, categorical_features)
        ])

        final_features = numeric_features + categorical_features
        return preprocessor, final_features

    except Exception as e:
        print(f" Error building pipeline: {e}")
        return None, None


# ----------------------------------------
#  Access function for processing + pipeline
# ----------------------------------------
def prepare_data(df):
    try:
        df_clean = preprocess_raw_data(df)
        if df_clean is None:
            raise ValueError("Preprocessing failed, no clean DataFrame returned.")

        pipeline, final_features = build_preprocessing_pipeline(df_clean)
        if pipeline is None:
            raise ValueError("Pipeline creation failed.")

        return df_clean, pipeline, final_features
    except Exception as e:
        print(f"Error preparing data: {e}")
        return None, None, None