import mlflow
from dotenv import load_dotenv
import os


def start_mlflow(params: dict, run_name: str):
    load_dotenv()
    mlflow.set_tracking_uri(os.getenv("DATABRICKS_HOST"))

    os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("DATABRICKS_USER_NAME")
    os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("DATABRICKS_TOKEN")

    databricks_user = os.getenv("DATABRICKS_USER_NAME")
    experiment_name = os.getenv("EXPERIMENT_NAME")

    # Ruta completa del experimento
    experiment_path = f"/Users/{databricks_user}/{experiment_name}"

    print("MLflow Tracking URI:", mlflow.get_tracking_uri())
    print("Usando experimento:", experiment_path)

    mlflow.set_experiment(experiment_path)
    mlflow.start_run(run_name=run_name)
    mlflow.log_params(params)

    return mlflow