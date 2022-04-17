from fastapi import FastAPI
from Utils import predictor, retrainer

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/retrain")
def retrain():
    return retrainer.run()

@app.get("/isrunning")
def isrunning():
    return retrainer.is_running()

@app.get("/predict")
def predict(url: str):
    predictor.predict(url)
    return {"Hello": "World"}