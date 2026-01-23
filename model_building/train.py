# for data manipulation
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline

# for model training, tuning, and evaluation
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, recall_score
# for model serialization
import joblib
# for creating a folder
import os
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
import mlflow

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("MLOps_experiment")

api = HfApi()

os.environ["MLOps"]= "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"
api = HfApi(token=os.getenv("MLOps"))



# Load preprocessed and split data from Hugging Face for the tourism project

Xtrain = pd.read_csv("hf://datasets/pavanipriyanka/tourism-project/Xtrain.csv")
Xtest = pd.read_csv("hf://datasets/pavanipriyanka/tourism-project/Xtest.csv")

ytrain = (pd.read_csv("hf://datasets/pavanipriyanka/tourism-project/ytrain.csv").squeeze())

ytest = (pd.read_csv("hf://datasets/pavanipriyanka/tourism-project/ytest.csv").squeeze())

# Define numeric and categorical features based on the tourism dataset
numeric_features = [
    'Age',
    'DurationOfPitch',
    'NumberOfPersonVisiting',
    'NumberOfFollowups',
    'PreferredPropertyStar',
    'NumberOfTrips',
    'PitchSatisfactionScore',
    'OwnCar',
    'NumberOfChildrenVisiting',
    'MonthlyIncome'
]
categorical_features = [
    'TypeofContact',
    'CityTier',
    'Occupation',
    'Gender',
    'ProductPitched',
    'MaritalStatus',
    'Passport',
    'Designation'
]

# Set the class weight to handle class imbalance
class_weight = ytrain.value_counts()[0] / ytrain.value_counts()[1]

# Define the preprocessing steps
preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown='ignore'), categorical_features)
)

# Define base XGBoost model
xgb_model = xgb.XGBClassifier(scale_pos_weight=class_weight, random_state=42)

# Define hyperparameter grid
param_grid = {
    'xgbclassifier__n_estimators': [50, 75, 100],
    'xgbclassifier__max_depth': [2, 3, 4],
    'xgbclassifier__colsample_bytree': [0.4, 0.5, 0.6],
    'xgbclassifier__colsample_bylevel': [0.4, 0.5, 0.6],
    'xgbclassifier__learning_rate': [0.01, 0.05, 0.1],
    'xgbclassifier__reg_lambda': [0.4, 0.5, 0.6],
}

# Model pipeline
model_pipeline = make_pipeline(preprocessor, xgb_model)

#MLFlow Training

with mlflow.start_run():

    grid_search = GridSearchCV(
        model_pipeline,
        param_grid,
        cv=5,
        n_jobs=-1,
        scoring="accuracy",
    )

    grid_search.fit(Xtrain, ytrain)

    best_model = grid_search.best_estimator_

    # Predictions
    y_pred_train = best_model.predict(Xtrain)
    y_pred_test = best_model.predict(Xtest)

    train_report = classification_report(
        ytrain, y_pred_train, output_dict=True
    )
    test_report = classification_report(
        ytest, y_pred_test, output_dict=True
    )

    # Log metrics
    mlflow.log_metrics(
        {
            "train_accuracy": train_report["accuracy"],
            "train_recall": train_report["1"]["recall"],
            "train_f1": train_report["1"]["f1-score"],
            "test_accuracy": test_report["accuracy"],
            "test_recall": test_report["1"]["recall"],
            "test_f1": test_report["1"]["f1-score"],
        }
    )

    mlflow.log_params(grid_search.best_params_)

    
    # Save Model
    
    MODEL_NAME = "best_tourism_project_v1.joblib"
    joblib.dump(best_model, MODEL_NAME)

    mlflow.log_artifact(MODEL_NAME, artifact_path="model")

    print(f"Model saved locally as {MODEL_NAME}")


# Upload Model to Hugging Face

repo_id = "pavanipriyanka/tourism-project"
repo_type = "model"

try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Repo {repo_id} exists.")
except RepositoryNotFoundError:
    create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
    print(f"Repo {repo_id} created.")

api.upload_file(
    path_or_fileobj=MODEL_NAME,
    path_in_repo=MODEL_NAME,
    repo_id=repo_id,
    repo_type=repo_type,
)

print("Model successfully uploaded to Hugging Face!")
