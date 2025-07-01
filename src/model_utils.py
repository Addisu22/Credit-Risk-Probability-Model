import mlflow
import mlflow.sklearn
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

def get_pipeline(model, numeric_features, categorical_features):
    try:
        numeric_pipe = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ])

        categorical_pipe = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore'))
        ])

        preprocessor = ColumnTransformer([
            ('num', numeric_pipe, numeric_features),
            ('cat', categorical_pipe, categorical_features)
        ])

        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])
        return pipeline
    except Exception as e:
        print(f"Pipeline creation failed: {e}")
        return None

def tune_and_train(X_train, y_train, X_test, y_test, model_name, model, param_grid):
    try:
        mlflow.set_experiment("Credit_Risk_Model_Tracking")
        with mlflow.start_run(run_name=model_name):
            numeric_features = ['amount', 'value', 'TransactionHour', 'TransactionDay', 'TransactionMonth', 'TransactionYear']
            categorical_features = list(set(X_train.columns) - set(numeric_features))

            pipeline = get_pipeline(model, numeric_features, categorical_features)

            grid = GridSearchCV(pipeline, param_grid, cv=3, scoring='roc_auc', n_jobs=-1)
            grid.fit(X_train, y_train)

            best_model = grid.best_estimator_
            y_pred_proba = best_model.predict_proba(X_test)[:, 1]
            roc_auc = roc_auc_score(y_test, y_pred_proba)

            mlflow.log_param("model_name", model_name)
            mlflow.log_params(grid.best_params_)
            mlflow.log_metric("roc_auc", roc_auc)
            mlflow.sklearn.log_model(best_model, artifact_path="model", registered_model_name=model_name)

            print(f" {model_name} trained | ROC AUC: {roc_auc:.4f}")
            return best_model
    except Exception as e:
        print(f" Training failed for {model_name}: {e}")
        return None
