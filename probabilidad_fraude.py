import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_curve, roc_auc_score, log_loss, cohen_kappa_score,roc_curve 

import mlflow
import mlflow.sklearn
from prefect import task, flow, get_run_logger
import matplotlib.pyplot as plt
import seaborn as sns


# Configurar MLflow
mlflow.set_tracking_uri("http://192.168.18.74:5000")
mlflow.set_experiment("Fraud_Risk_Model")

# Cargar datos
@task
def cargar_datos(csv_path):
    logger = get_run_logger()
    logger.info("Cargando datos...")
    return pd.read_csv(csv_path)

# Preprocesamiento de datos
@task
def preprocesar_datos(df):
    logger = get_run_logger()
    logger.info("Preprocesando datos...")
    
    # Ingeniería de características
    df['reclamos_frecuencia'] = df['Claim History'] / df['Age']
    df['monto_total_reclamado'] = df['Coverage Amount'] * df['Claim History']
    
    # Codificación de variables categóricas
    df = pd.get_dummies(df, drop_first=True)
    
    X = df.drop(columns=["Risk Profile", "Customer ID"])
    y = df["Risk Profile"]
    
    return train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenamiento del modelo
@task
def entrenar_modelo(X_train, X_test, y_train, y_test):
    logger = get_run_logger()
    logger.info("Entrenando modelo...")

    with mlflow.start_run(run_name="Fraud_Detection_RF"):
        # Crear y entrenar el modelo
        modelo = RandomForestClassifier(n_estimators=100, random_state=42)
        modelo.fit(X_train, y_train)

        # Predicciones
        y_pred = modelo.predict(X_test)
        y_pred_prob = modelo.predict_proba(X_test)

        # Evaluación: Especifica el tipo de multiclase para ROC AUC
        roc_auc = roc_auc_score(y_test, y_pred_prob, multi_class="ovr")
        report = classification_report(y_test, y_pred)

        # Guardar en MLflow
        mlflow.log_param("n_estimators", 100)
        mlflow.log_metric("roc_auc", roc_auc)
        mlflow.sklearn.log_model(modelo, "random_forest_model")

        # Registrar resultados
        logger.info(f"ROC AUC: {roc_auc}")
        logger.info(f"Classification Report: \n{report}")

        # Usar backend no interactivo de Matplotlib
        plt.switch_backend('Agg')
        
        # Curva ROC para cada clase
        fpr = {}
        tpr = {}
        for i in range(len(modelo.classes_)):
            fpr[i], tpr[i], _ = roc_curve(y_test, y_pred_prob[:, i], pos_label=i)
            plt.plot(fpr[i], tpr[i], label=f'Class {i} ROC curve')

        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve - Multiclass')
        plt.legend(loc="lower right")
        plt.savefig("./images/roc_curve_multiclass.png")
        mlflow.log_artifact("./images/roc_curve_multiclass.png")
        plt.close()

        return modelo

@task(name="evaluar_modelo", log_prints=True)
def evaluar_modelo(y_test, y_pred, y_pred_prob):
    # Matriz de confusión
    conf_matrix = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
    plt.title("Matriz de Confusión")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig("./images/confusion_matrix.png")
    mlflow.log_artifact("./images/confusion_matrix.png")

    # Clasificación por clase
    report = classification_report(y_test, y_pred, output_dict=True)
    mlflow.log_metrics({
        "precision": report['weighted avg']['precision'],
        "recall": report['weighted avg']['recall'],
        "f1-score": report['weighted avg']['f1-score']
    })
    
    # Log-Loss
    log_loss_value = log_loss(y_test, y_pred_prob)
    mlflow.log_metric("log_loss", log_loss_value)

    # Cohen's Kappa
    kappa = cohen_kappa_score(y_test, y_pred)
    mlflow.log_metric("cohen_kappa", kappa)

    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_test, y_pred_prob[:, 1])
    plt.figure()
    plt.plot(recall, precision, marker='.', label='Precision-Recall curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.savefig("precision_recall_curve.png")
    mlflow.log_artifact("precision_recall_curve.png")

    # AUC-ROC (Ya lo tienes, pero asegúrate de que sea multi-clase)
    roc_auc = roc_auc_score(y_test, y_pred_prob, multi_class="ovr")
    mlflow.log_metric("roc_auc", roc_auc)


# Flujo principal
@flow(name="Fraud Risk Model Pipeline")
def pipeline_fraude(csv_path):
    # Cargar y preprocesar datos
    df = cargar_datos(csv_path)
    X_train, X_test, y_train, y_test = preprocesar_datos(df)

    # Entrenar el modelo
    modelo = entrenar_modelo(X_train, X_test, y_train, y_test)
    
    # Realizar predicciones y obtener probabilidades
    y_pred = modelo.predict(X_test)
    y_pred_prob = modelo.predict_proba(X_test)
    
    # Evaluar el modelo
    evaluar_modelo(y_test, y_pred, y_pred_prob)
    
    return modelo

# Ejecutar el flujo
pipeline_fraude(csv_path="./data_synthetic.csv")

