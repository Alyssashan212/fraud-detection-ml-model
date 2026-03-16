from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI()

model = joblib.load("fraud_model.pkl")

@app.get("/")
def home():
    return {"message": "Fraud Detection API Running"}

@app.post("/predict")
def predict(transaction: dict):

    df = pd.DataFrame([transaction])

    prediction = model.predict(df)[0]

    if prediction == 1:
        result = "Fraud detected"
    else:
        result = "Transaction is safe"

    return {"prediction": result}
