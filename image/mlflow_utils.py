import os
import torch
import mlflow
import mlflow.pytorch
from dotenv import load_dotenv
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo, nvmlDeviceGetUtilizationRates

nvmlInit()
_handle = nvmlDeviceGetHandleByIndex(0)

def start_mlflow():
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

    return mlflow

def get_gpu_stats():
    torch.cuda.synchronize()  # Espera que el trabajo termine antes de medir

    mem_info = nvmlDeviceGetMemoryInfo(_handle)
    util = nvmlDeviceGetUtilizationRates(_handle)
    return {
        "sys/gpu_memory_used_mb": mem_info.used // 1024 ** 2,
        "sys/gpu_memory_total_mb": mem_info.total // 1024 ** 2,
        "sys/gpu_utilization_percent": util.gpu
    }

def log_gpu_stats(mlflow, epoch):
    for k, v in get_gpu_stats().items():
        mlflow.log_metric(k, v, step=epoch)

def log_model_histograms(writer, model, model_name, epoch):
    model_to_log = getattr(model, "_orig_mod", model)
    for name, param in model_to_log.named_parameters():
        if param.requires_grad and param.grad is not None:
            grad = param.grad
            if torch.is_tensor(grad) and grad.numel() > 0:
                if not torch.isnan(grad).all() and not torch.isinf(grad).all() and grad.abs().sum() > 0:
                    writer.add_histogram(f"{model_name}/Weights/{name}", param.data, epoch)
                    writer.add_histogram(f"{model_name}/Grads/{name}", grad, epoch)
