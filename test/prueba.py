import mlflow
import os

mlflow.set_tracking_uri("http://192.168.18.74:5000")

with mlflow.start_run(run_name="test_run"):
    # Crear un archivo de prueba en el directorio de artefactos
    os.makedirs("/mlflow/artifacts/test_artifact", exist_ok=True)
    with open("/mlflow/artifacts/test_artifact/hello.txt", "w") as f:
        f.write("Hello, MLflow!")

    # Registrar el artefacto
    mlflow.log_artifact("/mlflow/artifacts/test_artifact/hello.txt")

