import pandas as pd
import numpy as np
import mlflow
import time
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
import os

# ====== CONFIGURATION ======
uri = os.getenv("MLFLOW_TRACKING_URI")
if not uri:
    raise ValueError("❌ MLFLOW_TRACKING_URI is not set in environment variables.")
print("📍 MLFLOW_TRACKING_URI:", uri)
mlflow.set_tracking_uri(uri)

print("🔧 MLflow tracking URI (applied):", mlflow.get_tracking_uri())
print("📦 Backend Store URI:", os.getenv("BACKEND_STORE_URI"))
print("🗂️ Artifact Store URI:", os.getenv("ARTIFACT_STORE_URI"))

# ====== DATA LOADING ======
def load_data(url):
    print("📥 Chargement des données...")
    return pd.read_csv(url)

# ====== PREPROCESSING ======
def preprocess_data(df):
    print("🧹 Prétraitement des données...")
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    return train_test_split(X, y, test_size=0.2)

# ====== PIPELINE CREATION ======
def create_pipeline():
    print("🛠️ Création du pipeline...")
    return Pipeline([
        ("standard_scaler", StandardScaler()),
        ("Random_Forest", RandomForestRegressor())
    ])

# ====== TRAINING ======
def train_model(pipe, X_train, y_train, param_grid, cv=2, n_jobs=-1, verbose=3):
    print("🏋️‍♂️ Entraînement du modèle avec GridSearchCV...")
    model = GridSearchCV(pipe, param_grid, n_jobs=n_jobs, verbose=verbose, cv=cv, scoring="r2")
    model.fit(X_train, y_train)
    return model

# ====== LOGGING TO MLFLOW ======
def log_metrics_and_model(model, X_train, y_train, X_test, y_test, artifact_path):
    print("📊 Log des métriques dans MLflow...")
    mlflow.log_metric("Train Score", model.score(X_train, y_train))
    mlflow.log_metric("Test Score", model.score(X_test, y_test))
    print("✅ Métriques enregistrées.")

    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path=artifact_path
    )
    print("✅ Modèle enregistré dans les artefacts.")

# ====== MAIN WORKFLOW ======
def run_experiment(experiment_name, data_url, param_grid, artifact_path):
    start_time = time.time()

    df = load_data(data_url)
    X_train, X_test, y_train, y_test = preprocess_data(df)
    pipe = create_pipeline()

    print(f"📁 Tentative de définition de l'expérience MLflow : '{experiment_name}'")
    mlflow.set_experiment(experiment_name)

    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise RuntimeError(f"❌ Expérience '{experiment_name}' introuvable ou non créée.")

    with mlflow.start_run(experiment_id=experiment.experiment_id):
        model = train_model(pipe, X_train, y_train, param_grid)
        log_metrics_and_model(model, X_train, y_train, X_test, y_test, artifact_path)

    print(f"✅ Entraînement terminé en {time.time() - start_time:.2f} secondes")

# ====== ENTRY POINT ======
if __name__ == "__main__":
    experiment_name = "hyperparameter_tuning"
    data_url = "https://julie-2-next-resources.s3.eu-west-3.amazonaws.com/full-stack-full-time/linear-regression-ft/californian-housing-market-ft/california_housing_market.csv"
    param_grid = {
        "Random_Forest__n_estimators": list(range(90, 101, 10)),
        "Random_Forest__criterion": ["squared_error"]
    }
    artifact_path = "modeling_housing_market"

    run_experiment(experiment_name, data_url, param_grid, artifact_path)

