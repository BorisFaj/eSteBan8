import mlflow
from dotenv import load_dotenv
import os

load_dotenv()

def start_mlflow(params: dict, run_name: str):

    mlflow.set_tracking_uri(os.getenv("DATABRICKS_HOST"))
    os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("DATABRICKS_USER_NAME")
    os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("DATABRICKS_TOKEN")

    print("MLflow Tracking URI:", mlflow.get_tracking_uri())
    client = mlflow.tracking.MlflowClient()
    print("Experimentos disponibles:", client.list_experiments())

    _params = {}


    mlflow.set_experiment(run_name)  # opcional, si quieres organizar por run
    mlflow.start_run()
    mlflow.log_params(params)

    return mlflow
