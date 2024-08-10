import mlflow
from typing import Any

import pandas as pd

from sklearn.datasets import make_classification


def create_mlflow_experiment(
    experiment_name: str, artifact_location: str, tags: dict[str, Any]
) -> str:

    try:
        experiment_id = mlflow.create_experiment(
            name=experiment_name, artifact_location=artifact_location, tags=tags
        )
    except:
        print(f"Experiment {experiment_name} already exists.")
        experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id

    mlflow.set_experiment(experiment_name=experiment_name)

    return experiment_id


def get_mlflow_experiment(
    experiment_id: str = None, experiment_name: str = None
) -> mlflow.entities.Experiment:
    
    if experiment_id is not None:
        experiment = mlflow.get_experiment(experiment_id)
    elif experiment_name is not None:
        experiment = mlflow.get_experiment_by_name(experiment_name)
    else:
        experiment = None
    return experiment


def delete_mlflow_experiment(
    experiment_id: str = None, experiment_name: str = None
) -> None:
    if experiment_id is not None:
        mlflow.delete_experiment(experiment_id)
    elif experiment_name is not None:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        experiment_id = experiment.experiment_id
        mlflow.delete_experiment(experiment_id)
    else:
        raise ValueError("Either experiment_id or experiment_name must be provided.")

def mlflow_predict(run_id, data):
    
    logged_model = f'runs:/{run_id}/decision_tree_classifier'
    loaded_model = mlflow.pyfunc.load_model(logged_model)
    return loaded_model.predict(pd.DataFrame(data))
