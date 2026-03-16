import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

# Example dataset (simulated)
data = {
    "amount": [100, 2000, 150, 5000, 120, 7000, 50, 400],
    "transactions_per_day": [2, 20, 3, 50, 1, 60, 1, 4],
    "location_risk": [1, 5, 1, 5, 1, 5, 1, 2],
    "is_fraud": [0, 1, 0, 1, 0, 1, 0, 0]
}

df = pd.DataFrame(data)

X = df.drop("is_fraud", axis=1)
y = df["is_fraud"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier()
model.fit(X_train, y_train)

predictions = model.predict(X_test)

print(classification_report(y_test, predictions))

joblib.dump(model, "fraud_model.pkl")

print("Model saved as fraud_model.pkl")
