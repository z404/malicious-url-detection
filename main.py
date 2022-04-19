from fastapi import FastAPI
from Utils import predictor, retrainer

app = FastAPI()

@app.get("/")
def read_root():
    return {"response": "Welcome to the URL predictor API. To view documentation, visit /docs"}

@app.post("/retrain")
def retrain():
    return retrainer.run()

@app.get("/isrunning")
def isrunning():
    return retrainer.is_running()

@app.get("/predict")
def predict(url: str):
    return {"result": "benign"} if predictor.predict(url) == 0 else {"result": "malicious"}
    