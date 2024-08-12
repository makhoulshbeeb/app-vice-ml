import mlflow
from mlflow.models import infer_signature
from mlflow_utils import get_mlflow_experiment
from mlflow_utils import create_mlflow_experiment
from mlflow_utils import mlflow_predict


from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split,GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler


from xgboost import XGBClassifier

from sklearn.metrics import PrecisionRecallDisplay
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import ConfusionMatrixDisplay

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

csv_dir = 'data/onehot_models/onehot_3.csv'
experiment_name='onehot_3_experiment2'
run_name="onehot_3_model"
artifact_location='onehot_3_artifacts'

experiment=get_mlflow_experiment(experiment_name=experiment_name)

if( experiment is None):
    experiment_id = create_mlflow_experiment(experiment_name=experiment_name, artifact_location=artifact_location, tags={"env": "dev", "version": "1.0.0"})
    experiment=get_mlflow_experiment(experiment_id=experiment_id)
    
print("Name: {}".format(experiment.name))

with mlflow.start_run(run_name=run_name, experiment_id=experiment.experiment_id) as run:
    df = pd.read_csv(csv_dir, encoding='latin-1')
    df_full = pd.read_csv("./data/full_label.csv")
    
    #adjust these to your model
    ss = StandardScaler()
    df[["Size", "Price"]] = ss.fit_transform(df[["Size", "Price"]])   
    
    df_full['Last Updated'] = pd.to_datetime(df_full['Last Updated'])
    df_full = df_full.sort_values(by='Last Updated')
    
    df['Last Updated'] = pd.to_datetime(df['Last Updated'])

    df = df.drop(columns=["App", "Reviews", "Rating"])

    bool_cols = df.select_dtypes(include='bool').columns
    df[bool_cols] = df[bool_cols].astype(int)

    df_full['Last Updated'] = pd.to_datetime(df_full['Last Updated'])
    df_full = df_full.sort_values(by='Last Updated')
    df_full = df_full.drop(columns="Unnamed: 0")


    df[["Size", "Price"]] = ss.fit_transform(df[["Size", "Price"]])

    train_df, test_df = train_test_split(df_full, test_size=0.2, stratify=df_full['Category'], random_state=42)

    train_indices = train_df.index
    test_indices = test_df.index

    df = df.sort_values(by='Last Updated').drop(columns="Last Updated")

    train_df = df.loc[train_indices]
    test_df = df.loc[test_indices]

    X_train = train_df.drop(columns=['Label'])
    y_train = train_df["Label"]

    X_test = test_df.drop(columns=['Label'])
    y_test = test_df["Label"]

    xgb = XGBClassifier(eval_metric='logloss', n_estimators=250, learning_rate=0.001)
    xgb.fit(X_train, y_train)

    # Get predictions
    y_pred = xgb.predict(X_test)

    
    model_signature = infer_signature(model_input=X_train, model_output=y_pred)
    mlflow.sklearn.log_model(sk_model=xgb, artifact_path="xgb_classifier", signature=model_signature)
    
    # fig_pr = plt.figure()
    # pr_display = PrecisionRecallDisplay.from_predictions(y_test, y_pred, ax=plt.gca())
    # plt.title("Precision-Recall Curve")
    # plt.legend()
    # mlflow.log_figure(fig_pr, "metrics/precision_recall_curve.png")

    # fig_roc = plt.figure()
    # roc_display = RocCurveDisplay.from_predictions(y_test, y_pred, ax=plt.gca())
    # plt.title("ROC Curve")
    # plt.legend()
    # mlflow.log_figure(fig_roc, "metrics/roc_curve.png")

    fig_cm = plt.figure()
    cm_display = ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=plt.gca())
    plt.title("Confusion Matrix")
    plt.legend()
    mlflow.log_figure(fig_cm, "metrics/confusion_matrix.png")
    
    print("run_id: {}".format(run.info.run_id))
    print("experiment_id: {}".format(run.info.experiment_id))
    print("status: {}".format(run.info.status))
    print("start_time: {}".format(run.info.start_time))
    print("end_time: {}".format(run.info.end_time))
    print("lifecycle_stage: {}".format(run.info.lifecycle_stage))
    
    # data={'MinTemp': [13.1], 'MaxTemp': [30.1], 'Rainfall': [1.4], 'WindGustSpeed': [28], 'WindSpeed9am': [15], 'WindSpeed3pm': [11], 'Humidity9am': [58], 'Humidity3pm': [27], 'Pressure9am': [1007.1], 'Pressure3pm': [1005.7], 'Temp9am': [20.1],  'Temp3pm':[28.2], 'RainToday': [1]}
    # print(mlflow_predict(run_id=run.info.run_id,data=data)) 
