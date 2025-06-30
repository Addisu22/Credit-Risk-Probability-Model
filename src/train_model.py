import pandas as pd
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

def train_and_evaluate_models(data_path):
    try:
        # 1. Load data
        df = pd.read_csv(data_path)
        
        # 2. Select features and target
        features = [
            'AccountId', 'SubscriptionId', 'CustomerId',
            'CurrencyCode', 'CountryCode', 'ProviderId', 'ProductId',
            'ProductCategory', 'ChannelId', 'Amount', 'Value',
            'TransactionStartTime', 'PricingStrategy'
        ]
        target = 'FraudResult'
        
        # 3. Extract datetime features
        df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])
        df['TransactionHour'] = df['TransactionStartTime'].dt.hour
        df['TransactionDay'] = df['TransactionStartTime'].dt.day
        df['TransactionMonth'] = df['TransactionStartTime'].dt.month
        df['TransactionYear'] = df['TransactionStartTime'].dt.year
        
        features.remove('TransactionStartTime')
        features.extend(['TransactionHour', 'TransactionDay', 'TransactionMonth', 'TransactionYear'])
        
        X = df[features]
        y = df[target]
        
        # 4. Define numeric and categorical columns
        numeric_features = ['Amount', 'Value', 'TransactionHour', 'TransactionDay', 'TransactionMonth', 'TransactionYear']
        categorical_features = list(set(features) - set(numeric_features))
        
        # 5. Build preprocessing pipelines
        numeric_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ])
        categorical_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        preprocessor = ColumnTransformer([
            ('numerical', numeric_pipeline, numeric_features),
            ('categorical', categorical_pipeline, categorical_features)
        ])
        
        # 6. Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
        
        # 7. Define models
        models = {
            'LogisticRegression': LogisticRegression(max_iter=1000),
            'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
            'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
        }
        
        # 8. Train, predict and evaluate
        for name, model in models.items():
            print(f"\n Model: {name}")
            pipeline = Pipeline([
                ('preprocessing', preprocessor),
                ('classifier', model)
            ])
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)
            y_proba = pipeline.predict_proba(X_test)[:, 1]
            
            print(" Classification Report:")
            print(classification_report(y_test, y_pred))
            print("ROC-AUC Score:", round(roc_auc_score(y_test, y_proba), 4))
    
    except Exception as e:
        print(" Error during model training or evaluation:", e)
