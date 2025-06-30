import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score


def feature_engineering(df):
    try:
        df = df.copy()

        required_columns = ['CustomerId', 'Amount', 'TransactionStartTime']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")

        # --------- Aggregate Features per Customer ---------
        agg = df.groupby('CustomerId')['Amount'].agg(
            TotalTransactionAmount='sum',
            AverageTransactionAmount='mean',
            TransactionCount='count',
            StdTransactionAmount='std'
        ).reset_index()
        
        df = df.merge(agg, on='CustomerId', how='left')

        # --------- Extract Time Features ---------
        df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'], errors='coerce')
        df['TransactionHour'] = df['TransactionStartTime'].dt.hour
        df['TransactionDay'] = df['TransactionStartTime'].dt.day
        df['TransactionMonth'] = df['TransactionStartTime'].dt.month
        df['TransactionYear'] = df['TransactionStartTime'].dt.year
        df = df.drop(columns=['TransactionStartTime'])

        # --------- Handle Missing Values ---------
        numeric_cols = df.select_dtypes(include='number').columns.tolist()
        for col in numeric_cols:
            df[col].fillna(df[col].mean(), inplace=True)

        categorical_cols = df.select_dtypes(include='object').columns.tolist()
        for col in categorical_cols:
            df[col].fillna(df[col].mode()[0], inplace=True)

        # --------- Encode Categorical ---------
        if categorical_cols:
            ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
            cat_data = ohe.fit_transform(df[categorical_cols])
            cat_df = pd.DataFrame(cat_data, columns=ohe.get_feature_names_out(categorical_cols))
            df = df.drop(columns=categorical_cols).reset_index(drop=True)
            df = pd.concat([df, cat_df], axis=1)

        # --------- Normalize Numerical Features ---------
        scaler = StandardScaler()
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

        print("✅ Feature engineering completed successfully.")
        return df

    except Exception as e:
        print(f"❌ Error during feature engineering: {e}")
        return None
