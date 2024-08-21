import os
from datetime import datetime
import pandas as pd
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
import mlflow
import mlflow.tensorflow
import matplotlib.pyplot as plt
from prefect import task, flow, get_run_logger
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import seaborn as sns

# Configurar la URI de seguimiento de MLflow
os.environ['MLFLOW_TRACKING_URI'] = 'http://192.168.18.74:5000'
mlflow.set_tracking_uri(os.environ['MLFLOW_TRACKING_URI'])

# Ponemos nombre al experimento 
experiment_name = "Pipeline_Seguros_TensorFlow"
mlflow.set_experiment(experiment_name)

mlflow.tensorflow.autolog()

@task(name=["Procesamiento-de-datos"], log_prints=True)
def preprocesar_datos(df):
    """Preprocesamiento de los datos"""
    df = pd.get_dummies(df.drop(columns=["Customer ID", "Policy Start Date", "Policy Renewal Date"]), drop_first=True)
    X = df.drop(columns=["Risk Profile"])
    y = df["Risk Profile"]
    
    # Normalizar características
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Codificar la variable objetivo
    encoder = LabelEncoder()
    y = encoder.fit_transform(y)
    
    return train_test_split(X, y, test_size=0.2, random_state=42)

@task(name=["construccion del modelo"], log_prints=True)
def construir_modelo(input_shape):
    """Construye y compila un modelo de TensorFlow para clasificación multiclase."""
    modelo = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_shape,)),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(4, activation='softmax')  # 4 categorías en "Risk Profile"
    ])
    modelo.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',  # Ajustado para multiclase
        metrics=['accuracy']
    )
    return modelo

@task(name=["entrenamiento-evaluacion"], log_prints=True)
def entrenar_y_evaluar_modelo(modelo, X_train, X_test, y_train, y_test):
    """Entrena y evalúa el modelo con los datos proporcionados."""
    logger = get_run_logger()
    logger.info("Entrenando el modelo...")
    
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"modelo_deteccion_fraude_{current_time}"
    
    with mlflow.start_run(run_name=run_name, tags={"version": "v2", "etapa": "entrenamiento"}):
        historia = modelo.fit(X_train, y_train, epochs=30, validation_data=(X_test, y_test))
        
        # Guardar el modelo en el formato .keras
        modelo.save('./model/model_deteccion_fraude_v2.keras')
        mlflow.log_artifact('./model/model_deteccion_fraude_v2.keras') 
        
        # Predicciones
        y_pred_prob = modelo.predict(X_test)
        y_pred = y_pred_prob.argmax(axis=1)
        
        # Calcular métricas
        report = classification_report(y_test, y_pred, target_names=[f'Clase {i}' for i in range(4)])
        logger.info(f"Classification Report:\n{report}")
        
        # Usar backend no interactivo de Matplotlib
        plt.switch_backend('Agg')

        # Matriz de confusión
        conf_matrix = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(10,7))
        sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=[f'Clase {i}' for i in range(4)], yticklabels=[f'Clase {i}' for i in range(4)])
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.title('Matriz de Confusión')
        plt.savefig("confusion_matrix.png")
        mlflow.log_artifact("confusion_matrix.png")
        
    return historia

@flow(name="Pipeline Seguros con TensorFlow")
def pipeline_seguros(csv_path, batch_size):
    # Leer y preprocesar los datos
    df = pd.read_csv(csv_path)
    X_train, X_test, y_train, y_test = preprocesar_datos(df)
    
    # Construir el modelo
    modelo = construir_modelo(X_train.shape[1])
    
    # Entrenar y evaluar el modelo
    entrenar_y_evaluar_modelo(modelo, X_train, X_test, y_train, y_test)

# Ejecutar el flujo
pipeline_seguros(csv_path="./data/data_synthetic.csv", batch_size=1000)
