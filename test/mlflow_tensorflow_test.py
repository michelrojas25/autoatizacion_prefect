import os
import mlflow
import mlflow.tensorflow
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve
import numpy as np
import matplotlib.pyplot as plt

# Configurar la URI de seguimiento de MLflow
# os.environ['MLFLOW_TRACKING_URI'] = 'http://192.168.18.74:5000'
# mlflow.set_tracking_uri(os.environ['MLFLOW_TRACKING_URI'])

# Habilitar el autologging de MLflow para TensorFlow
mlflow.tensorflow.autolog()

# Generar datos sintéticos
X, y = np.random.rand(1000, 20), np.random.randint(0, 2, 1000)

# Dividir los datos en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Construir un modelo simple de TensorFlow
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(X.shape[1],)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Iniciar una nueva ejecución en MLflow
with mlflow.start_run(run_name="prueba_tensorflow") as run:
    print(f"Run ID: {run.info.run_id}")
    artifacts = mlflow.artifacts.list_artifacts(run.info.run_id)
    for artifact in artifacts:
        print(artifact.path)
    # Entrenar el modelo
    history = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

    # Guardar el archivo YML como artifact
    config_path = 'config.yml'
    with open(config_path, 'w') as f:
        f.write("nombre_experimento: Pipeline_Seguros_TensorFlow\n")
        f.write("parametro_1: valor1\n")
        f.write("parametro_2: valor2\n")
    mlflow.log_artifact(config_path)

    # Realizar predicciones
    y_pred_prob = model.predict(X_test).squeeze()

    # Calcular y registrar métricas
    roc_auc = roc_auc_score(y_test, y_pred_prob)
    print(f"AUC: {roc_auc}")
    mlflow.log_metric("AUC", roc_auc)

    # Generar y guardar la curva ROC
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    roc_curve_path = "roc_curve.png"
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:0.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig(roc_curve_path)

    # Registrar la curva ROC en MLflow
    mlflow.log_artifact(roc_curve_path)

    # Guardar el modelo
    model_path = 'my_model.keras'
    model.save(model_path)
    mlflow.log_artifact(model_path)

    # **Registrar el modelo en el Model Registry de MLflow**
    # mlflow.tensorflow.log_model(model, artifact_path="model", registered_model_name="Modelo_Deteccion_Fraude")
    mlflow.keras.log_model(model, artifact_path="model", registered_model_name="Modelo_Deteccion_Fraude")
