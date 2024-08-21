import os
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score
import matplotlib.pyplot as plt

# Configurar la URI de seguimiento de MLflow
os.environ['MLFLOW_TRACKING_URI'] = 'http://192.168.18.74:5000'
mlflow.set_tracking_uri(os.environ['MLFLOW_TRACKING_URI'])

# Habilitar el autologging de MLflow para scikit-learn
mlflow.sklearn.autolog()

# Generar datos sintéticos
X, y = np.random.rand(1000, 5), np.random.randint(0, 2, 1000)

# Dividir los datos en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Iniciar una nueva ejecución en MLflow
with mlflow.start_run(run_name="prueba_logistic_regression"):
    
    # Entrenar un modelo de regresión logística
    model = LogisticRegression()
    model.fit(X_train, y_train)
    
    # Realizar predicciones
    y_pred = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)[:, 1]
    
    # Calcular y registrar métricas
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_prob)
    print(f"Accuracy: {accuracy}, AUC: {roc_auc}")
    
    mlflow.log_metric("Accuracy", accuracy)
    mlflow.log_metric("AUC", roc_auc)
    
    # Generar y guardar la curva ROC
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:0.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig("roc_curve.png")
    
    # Registrar la curva ROC en MLflow
    mlflow.log_artifact("roc_curve.png")

    # Registrar el modelo
    mlflow.sklearn.log_model(model, "logistic_regression_model")

