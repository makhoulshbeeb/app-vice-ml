from fastapi import FastAPI
# from app.model.model import __version__ as model_version


app = FastAPI()


@app.get("/")
def home():
    return {"test": "ok"}


# @app.post("/predict")
# def predict(payload: TextIn):
#     return {"language": mlflow_predict(payload.text)}