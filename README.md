# Fraud Detection Machine Learning Model

This project demonstrates a machine learning system that detects fraudulent financial transactions.

The system trains a Random Forest classifier using transaction features such as amount, transaction frequency, and location risk.

## Features

- Machine learning fraud detection model
- Transaction risk prediction
- Model evaluation metrics
- REST API for fraud prediction

## Tech Stack

Python  
Scikit-learn  
Pandas  
FastAPI  
Random Forest Classifier

## System Architecture

Transaction Data
↓
Feature Processing
↓
Random Forest Model
↓
Fraud Prediction API

## Training the model

python train_model.py

## Run the API

uvicorn api:app --reload

Open API:

http://127.0.0.1:8000

## Example API Request

POST /predict

{
 "amount": 4500,
 "transactions_per_day": 35,
 "location_risk": 4
}

## Example Response

{
 "prediction": "Fraud detected"
}
