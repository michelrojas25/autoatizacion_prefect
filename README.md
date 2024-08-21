# Detección de Fraude en Seguros

Este trabajo implementa un modelo de detección de fraude utilizando machine learning, basado en datos sintéticos de reclamaciones de seguros y pólizas. Los datos provienen de [Kaggle](https://www.kaggle.com/code/mafayed/insurance-analysis). El flujo principal del modelo y su despliegue están gestionados por `Prefect`, lo que permite una automatización y monitorización eficiente de los procesos.

## Configuración del Entorno con Docker Compose

Para facilitar la creación del entorno necesario para ejecutar este trabajo, utilizamos Docker Compose. A continuación se describe el archivo `docker-compose.yml`, que define los servicios necesarios: PostgreSQL, MLflow, y Prefect.

```yaml
version: '3.8'

services:
  postgres:
    image: postgres:16
    container_name: prefect_mlflow_postgres
    environment:
      POSTGRES_USER: prefect
      POSTGRES_PASSWORD: prefect
      POSTGRES_DB: prefectdb
    volumes:
      - E:\DockerData\Prefect\postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"

  mlflow:
    image: ubuntu:latest
    container_name: mlflow_server
    environment:
      MLFLOW_TRACKING_URI: postgresql+psycopg2://prefect:prefect@postgres:5432/mlflowdb
      MLFLOW_BACKEND_STORE_URI: postgresql+psycopg2://prefect:prefect@postgres:5432/mlflowdb
      MLFLOW_ARTIFACT_ROOT: /mlflow/artifacts
    volumes:
      - ./mlflow:/mlflow/artifacts
    ports:
      - "5000:5000"
    depends_on:
      - postgres
    command: >
      /bin/bash -c "
      apt-get update &&
      apt-get install -y python3-pip &&
      pip install --break-system-packages mlflow psycopg2-binary &&
      mlflow server --backend-store-uri $MLFLOW_BACKEND_STORE_URI --default-artifact-root $MLFLOW_ARTIFACT_ROOT --host 0.0.0.0 --port 5000"

  prefect:
    image: ubuntu:latest
    container_name: prefect_server
    environment:
      PREFECT_API_DATABASE_CONNECTION_URL: postgresql+asyncpg://prefect:prefect@postgres:5432/prefectdb
    ports:
      - "4200:4200"
    depends_on:
      - postgres
    command: >
      /bin/bash -c "
      apt-get update &&
      apt-get install -y python3-pip && 
      pip3 install --break-system-packages prefect psycopg2-binary &&
      prefect server start --host 0.0.0.0 --port 4200"

networks:
  default:
    driver: bridge
```

## Descripción de los Servicios
- **PostgreSQL (postgres)**: Este servicio proporciona una base de datos PostgreSQL utilizada tanto por MLflow como por Prefect. Se configura con un usuario y base de datos específicos para prefect.

- **MLflow (mlflow)**: Este servicio gestiona el servidor de MLflow, que se utiliza para el seguimiento de experimentos de machine learning. Está configurado para almacenar los datos en la base de datos PostgreSQL y los artefactos en un volumen local.

- **Prefect (prefect)**: Este servicio ejecuta el servidor de Prefect, que se utiliza para gestionar y monitorizar los flujos de trabajo. Prefect se conecta a la base de datos PostgreSQL para almacenar su estado y configuración.

## Instrucciones para Ejecutar
- Asegúrate de tener Docker y Docker Compose instalados en tu máquina.

- Navega hasta el directorio donde se encuentra tu archivo docker-compose.yml.

- Ejecuta el siguiente comando para levantar los servicios:

```bash
docker-compose up -d
```
- Una vez que los servicios estén en ejecución, podrás acceder a la interfaz de MLflow en http://localhost:5000 y a la interfaz de Prefect en http://localhost:4200.


## Archivos Principales

- **main.py**: Contiene la lógica principal del flujo de trabajo para la detección de fraude. Aquí se encuentra la implementación del modelo de machine learning.
- **deployment.py**: Configura y aplica la programación para el flujo, asegurando que las tareas se ejecuten de acuerdo con el cronograma definido.

## Monitorización

Puedes monitorizar la ejecución de los flujos y su correcto funcionamiento en la interfaz de Prefect UI.

## Configuración de Entorno

Para conectarse a los contenedores desde el escritorio, se deben establecer las siguientes variables de entorno:

```bash
export PREFECT_API_URL=http://192.168.18.74:4200/api
export MLFLOW_TRACKING_URI=http://192.168.18.74:5000
```

## Secuencia de Comandos

### Crear un Work-Pool en Prefect:

```bash
prefect work-pool create "pipeline-tensorflow-pool" --type process
```
### Iniciar un Worker:

```bash
prefect worker start -p "pipeline-tensorflow-pool"
```

### Ejecutar la Tarea Programada:
```bash
python deployment.py
```

## Guía de Actualizaciones
###  Cambios en el Código del Flujo (main.py)
Si realizas modificaciones en el código de tus flujos, estos cambios se reflejarán en la próxima ejecución del flujo. No es necesario reiniciar el agente; simplemente, la próxima vez que el flujo se ejecute, usará el código actualizado.

Ejemplo: Si cambias la lógica dentro de una tarea o flujo, la siguiente ejecución de ese flujo utilizará el código actualizado.

### Cambios en la Programación (deployment.py)
Si cambias la programación del flujo o implementas nuevos despliegues, simplemente re-ejecuta deployment.py para actualizar la configuración en Prefect. El agente detectará los nuevos despliegues automáticamente.

Ejemplo: Si ajustas el intervalo de ejecución de cada 10 minutos a cada 15 minutos y vuelves a ejecutar deployment.py, el flujo comenzará a ejecutarse cada 15 minutos según la nueva configuración.

### Impacto en Procesos Actuales
Los cambios que realices en el código no afectarán las ejecuciones que ya están en curso o que han sido lanzadas. Los cambios solo impactarán en las próximas ejecuciones.
