import mlflow
from mlflow.models import infer_signature
from mlflow_utils import get_mlflow_experiment
from mlflow_utils import create_mlflow_experiment
from mlflow_utils import mlflow_predict


from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import PrecisionRecallDisplay
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import ConfusionMatrixDisplay

import pandas as pd 
import matplotlib.pyplot as plt

csv_dir = 'data.csv'

experiment=get_mlflow_experiment(experiment_name='app_predict')

if( experiment is None):
    experiment_id = create_mlflow_experiment(experiment_name='app_predict', artifact_location='app_predict_artifacts', tags={"env": "dev", "version": "1.0.0"})
    experiment=get_mlflow_experiment(experiment_id=experiment_id)
    
print("Name: {}".format(experiment.name))

with mlflow.start_run(run_name="recommending_models", experiment_id=experiment.experiment_id) as run:
    df = pd.read_csv(csv_dir, encoding='latin-1')
    
    #adjust these to your model
    df=df.drop(columns=['Location', 'Date'])
    X = df.iloc[:,0:13]
    y= df.iloc[:,13]    
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)    
    dtc = DecisionTreeClassifier()    
    dtc.fit(X_train, y_train)    
    y_pred = dtc.predict(X_test)
    
    model_signature = infer_signature(model_input=X_train, model_output=y_pred)
    mlflow.sklearn.log_model(sk_model=dtc, artifact_path="decision_tree_classifier", signature=model_signature)
    
    fig_pr = plt.figure()
    pr_display = PrecisionRecallDisplay.from_predictions(y_test, y_pred, ax=plt.gca())
    plt.title("Precision-Recall Curve")
    plt.legend()
    mlflow.log_figure(fig_pr, "metrics/precision_recall_curve.png")

    fig_roc = plt.figure()
    roc_display = RocCurveDisplay.from_predictions(y_test, y_pred, ax=plt.gca())
    plt.title("ROC Curve")
    plt.legend()
    mlflow.log_figure(fig_roc, "metrics/roc_curve.png")

    fig_cm = plt.figure()
    cm_display = ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=plt.gca())
    plt.title("Confusion Matrix")
    plt.legend()
    mlflow.log_figure(fig_cm, "metrics/confusion_matrix.png")
    
    # print("run_id: {}".format(run.info.run_id))
    # print("experiment_id: {}".format(run.info.experiment_id))
    # print("status: {}".format(run.info.status))
    # print("start_time: {}".format(run.info.start_time))
    # print("end_time: {}".format(run.info.end_time))
    # print("lifecycle_stage: {}".format(run.info.lifecycle_stage))
    
    # data={'MinTemp': [13.1], 'MaxTemp': [30.1], 'Rainfall': [1.4], 'WindGustSpeed': [28], 'WindSpeed9am': [15], 'WindSpeed3pm': [11], 'Humidity9am': [58], 'Humidity3pm': [27], 'Pressure9am': [1007.1], 'Pressure3pm': [1005.7], 'Temp9am': [20.1],  'Temp3pm':[28.2], 'RainToday': [1]}
    # print(mlflow_predict(run_id=run.info.run_id,data=data)) 
