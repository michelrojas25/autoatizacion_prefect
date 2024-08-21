from prefect.deployments import Deployment
from prefect.infrastructure import Process
from prefect.server.schemas.schedules import IntervalSchedule
from datetime import timedelta
from main import pipeline_seguros

# Crear una implementación del flujo con cronograma
deployment = Deployment.build_from_flow(
    flow=pipeline_seguros,
    name="Pipeline Seguros con TensorFlow Programado",
    schedule=IntervalSchedule(interval=timedelta(minutes=30)),  # Ejecutar cada 5 minutos
    parameters={
        "csv_path": "./data/data_synthetic.csv",
        "batch_size": 1000
    },
    infrastructure=Process(),
    work_pool_name="pipeline-tensorflow-pool"  # Especificar el nuevo work pool
)

if __name__ == "__main__":
    deployment.apply()
    print("¡Despliegue creado y programado exitosamente en el work pool pipeline-tensorflow-pool!")
