import mlflow
import pandas as pd # type: ignore


def mlflow_predict(run_id, data):
    logged_model = f'runs:/{run_id}/decision_tree_classifier'
    loaded_model = mlflow.pyfunc.load_model(logged_model)
    return loaded_model.predict(pd.DataFrame(data))