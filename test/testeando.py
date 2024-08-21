import mlflow
import pandas as pd
import os

os.environ['MLFLOW_TRACKING_URI'] = 'http://192.168.18.74:5000'
mlflow.set_tracking_uri(os.environ['MLFLOW_TRACKING_URI'])

table_dict = {
    "inputs": ["What is MLflow?", "What is Databricks?"],
    "outputs": ["MLflow is ...", "Databricks is ..."],
    "toxicity": [0.0, 0.0],
}
df = pd.DataFrame.from_dict(table_dict)
with mlflow.start_run():
    # Log the df as a table
    mlflow.log_table(data=df, artifact_file="qabot_eval_results.json")
