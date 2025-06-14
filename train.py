import mlflow
import numpy as np
# from data import X_train, X_val, y_train, y_val
from data import X_train, X_val, y_train, y_val
from sklearn.linear_model import Ridge, ElasticNet
from xgboost import XGBRegressor
from sklearn.model_selection import ParameterGrid
from params import ridge_param_grid, elasticnet_param_grid, xgb_param_grid
from utils import eval_metrics

# mlflow.set_tracking_uri('http://34.205.64.209:5000')

# Loop through the hyperparameter combinations and log results in separate runs
# for params in ParameterGrid(elasticnet_param_grid):
for params in ParameterGrid(ridge_param_grid):
    with mlflow.start_run():

        print("params passed",params)

        lr = Ridge(**params) # 1 spoon of salt, 2 spoons of sugar,  3 spoon of sauce

        lr.fit(X_train, y_train)

        y_pred = lr.predict(X_val)

        metrics = eval_metrics(y_val, y_pred) # 3 , 4 , 5

        # Logging the inputs such as dataset
        mlflow.log_input(
            mlflow.data.from_numpy(X_train.toarray()),
            context='Training dataset'
        )

        mlflow.log_input(
            mlflow.data.from_numpy(X_val.toarray()),
            context='Validation dataset'
        )

        # Logging hyperparameters
        mlflow.log_params(params)

        # Logging metrics
        mlflow.log_metrics(metrics)

        # Log the trained model
        mlflow.sklearn.log_model(
        lr,
        artifact_path="Ridge",
        input_example=X_train,
        code_paths=['train.py','data.py','params.py','utils.py']
)
