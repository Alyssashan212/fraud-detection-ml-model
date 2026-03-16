import joblib
import pandas as pd

model = joblib.load("fraud_model.pkl")

new_transaction = pd.DataFrame({
    "amount": [4500],
    "transactions_per_day": [35],
    "location_risk": [4]
})

prediction = model.predict(new_transaction)

if prediction[0] == 1:
    print("Fraud detected")
else:
    print("Transaction is safe")
