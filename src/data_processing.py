import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer

def feature_engineering(df):
    try:
        df = df.copy()
        
        # --------- Aggregate Features per Customer ---------
        agg = df.groupby('CustomerId')['Amount'].agg(
            TotalTransactionAmount='sum',
            AverageTransactionAmount='mean',
            TransactionCount='count',
            StdTransactionAmount='std'
        ).reset_index()
        
        # Merge aggregated features back to main dataframe
        df = df.merge(agg, on='CustomerId', how='left')
        
        # --------- Extract Time Features ---------
        df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'], errors='coerce')
        df['TransactionHour'] = df['TransactionStartTime'].dt.hour
        df['TransactionDay'] = df['TransactionStartTime'].dt.day
        df['TransactionMonth'] = df['TransactionStartTime'].dt.month
        df['TransactionYear'] = df['TransactionStartTime'].dt.year
        
        # Drop original datetime column if not needed
        df = df.drop(columns=['TransactionStartTime'])
        
        # --------- Handle Missing Values ---------
        # For numeric columns - fill with mean
        numeric_cols = ['Amount', 'TotalTransactionAmount', 'AverageTransactionAmount', 'TransactionCount', 
                        'StdTransactionAmount', 'TransactionHour', 'TransactionDay', 'TransactionMonth', 'TransactionYear']
        for col in numeric_cols:
            if col in df.columns:
                df[col].fillna(df[col].mean(), inplace=True)
        
        # For categorical columns - fill with mode
        categorical_cols = ['CurrencyCode', 'CountryCode', 'ProviderId', 'ProductId', 'ProductCategory', 'ChannelId', 'PricingStrategy']
        for col in categorical_cols:
            if col in df.columns:
                df[col].fillna(df[col].mode()[0], inplace=True)
        
        # --------- Encode Categorical Variables ---------
        # Example: One-Hot Encoding for categorical variables
        ohe = OneHotEncoder(handle_unknown='ignore', sparse=False)
        cat_data = ohe.fit_transform(df[categorical_cols])
        cat_df = pd.DataFrame(cat_data, columns=ohe.get_feature_names_out(categorical_cols))
        
        # Drop original categorical columns and concatenate encoded ones
        df = df.drop(columns=categorical_cols).reset_index(drop=True)
        df = pd.concat([df, cat_df], axis=1)
        
        # --------- Normalize / Standardize Numerical Features ---------
        # Choose either normalization or standardization (example: standardization)
        scaler = StandardScaler()
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
        
        print(" Feature engineering completed successfully.")
        return df

    except Exception as e:
        print(f" Error during feature engineering: {e}")
        return None
