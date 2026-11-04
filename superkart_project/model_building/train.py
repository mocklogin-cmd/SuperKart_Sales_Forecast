# data handling
import pandas as pd
import numpy as np
# model building
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# saving the model and tracking experiments
import joblib
import mlflow

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("SuperKart-Sales-Forecast-Experiment")

# these four files come from the data-prep job (workflow artifact)
Xtrain = pd.read_csv("Xtrain.csv")
Xtest = pd.read_csv("Xtest.csv")
ytrain = pd.read_csv("ytrain.csv").squeeze()
ytest = pd.read_csv("ytest.csv").squeeze()

numeric_features = [
    "Product_Weight", "Product_Allocated_Area", "Product_MRP",
    "Store_Establishment_Year",
]
categorical_features = [
    "Product_Sugar_Content", "Product_Type", "Store_Size",
    "Store_Location_City_Type", "Store_Type", "Product_Category",
]

# trees do not need scaling, so numeric columns are passed as they are
preprocessor = make_column_transformer(
    (OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ("passthrough", numeric_features),
)

rf_model = RandomForestRegressor(random_state=42, n_jobs=-1)
model_pipeline = make_pipeline(preprocessor, rf_model)

# hyperparameters to try
param_grid = {
    "randomforestregressor__n_estimators": [100, 150],
    "randomforestregressor__max_depth": [8, 10, 12],
    "randomforestregressor__min_samples_leaf": [3, 5],
    "randomforestregressor__max_features": [0.5, 0.8],
}


def get_metrics(y_true, y_pred):
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


with mlflow.start_run():
    grid_search = GridSearchCV(
        model_pipeline, param_grid, cv=5,
        scoring="neg_root_mean_squared_error", n_jobs=-1,
    )
    grid_search.fit(Xtrain, ytrain)

    # log every combination that was tried as a nested run
    results = grid_search.cv_results_
    for i in range(len(results["params"])):
        with mlflow.start_run(nested=True):
            mlflow.log_params(results["params"][i])
            mlflow.log_metric("cv_rmse", -results["mean_test_score"][i])
            mlflow.log_metric("cv_rmse_std", results["std_test_score"][i])

    # best parameters go in the main run
    mlflow.log_params(grid_search.best_params_)
    best_model = grid_search.best_estimator_

    train_m = get_metrics(ytrain, best_model.predict(Xtrain))
    test_m = get_metrics(ytest, best_model.predict(Xtest))

    mlflow.log_metrics({f"train_{k}": v for k, v in train_m.items()})
    mlflow.log_metrics({f"test_{k}": v for k, v in test_m.items()})

    print("Best parameters:", grid_search.best_params_)
    print("Train metrics:", {k: round(v, 3) for k, v in train_m.items()})
    print("Test metrics :", {k: round(v, 3) for k, v in test_m.items()})

    # save the model next to app.py so Streamlit can load it from the repo
    model_path = "superkart_project/deployment/best_superkart_sales_model_v1.joblib"
    joblib.dump(best_model, model_path, compress=3)
    mlflow.log_artifact(model_path, artifact_path="model")
    print(f"Model saved to {model_path}")
