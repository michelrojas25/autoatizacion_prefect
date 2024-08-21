
import os
import pandas as pd
from sklearn.ensemble import IsolationForest
import mlflow
import mlflow.sklearn
from prefect import task, flow, get_run_logger
import seaborn as sns
import matplotlib.pyplot as plt

# Configuración inicial de MLflow
os.environ['MLFLOW_TRACKING_URI'] = 'http://192.168.18.74:5000'
mlflow.set_tracking_uri(os.environ['MLFLOW_TRACKING_URI'])
mlflow.sklearn.autolog()

# Establecer el nombre del experimento
experiment_name = "Detección_Anomalias_Reclamos"
mlflow.set_experiment(experiment_name)

# --- Tareas ---

@task
def cargar_datos(csv_path):
    df = pd.read_csv(csv_path)
    logger = get_run_logger()
    logger.info(f"Datos cargados correctamente desde {csv_path}")
    return df

@task
def preprocesar_datos(df):
    # Seleccionamos características relevantes para la detección de anomalías
    columnas = ["Claim History", "Previous Claims History", "Premium Amount", "Coverage Amount"]
    X = df[columnas]
    
    logger = get_run_logger()
    logger.info("Datos preprocesados.")
    
    return X

@task
def detectar_anomalias(X):
    # Modelo de Isolation Forest para detección de anomalías
    model = IsolationForest(contamination=0.05, random_state=42)
    
    # Entrenamos el modelo
    model.fit(X)
    
    # Predicciones (-1 para anómalos, 1 para normales)
    predicciones = model.predict(X)
    
    # Agregar predicciones al DataFrame
    X['Anomaly'] = predicciones
    
    # Calcular y loggear la proporción de anomalías detectadas
    n_anomalies = (predicciones == -1).sum()
    logger = get_run_logger()
    logger.info(f"Anomalías detectadas: {n_anomalies} de {len(X)} registros.")
    
    # Loggear el modelo y los resultados en MLflow
    mlflow.log_param("Proporción de anomalías", n_anomalies / len(X))
    mlflow.sklearn.log_model(model, "IsolationForest_Model")
    
    return X

@task
def guardar_resultados(X):
    # Guardar resultados a un archivo CSV
    resultados_path = "./data/anomalies_detected.csv"
    X.to_csv(resultados_path, index=False)
    
    # Loggear el archivo de resultados en MLflow
    mlflow.log_artifact(resultados_path)
    
    logger = get_run_logger()
    logger.info(f"Resultados guardados en {resultados_path}")

# --- Flujo Principal ---

@flow(name="Detección de Anomalías en Reclamos")
def deteccion_anomalias(csv_path):
    # Cargar los datos
    df = cargar_datos(csv_path)
    
    # Preprocesar los datos
    X = preprocesar_datos(df)
    
    # Detectar anomalías
    X_anomalies = detectar_anomalias(X)
    
    # Guardar los resultados
    guardar_resultados(X_anomalies)

# --- Ejecución del Flujo ---

if __name__ == "__main__":
    deteccion_anomalias(csv_path="./data/data_synthetic.csv")

