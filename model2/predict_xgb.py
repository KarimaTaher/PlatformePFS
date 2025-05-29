# /model2/predict_xgb.py

import joblib
import pandas as pd
import os

def predict_with_xgb():
    model = joblib.load("model2/xgb_model.pkl")
    encoder = joblib.load("model2/xgb_encoder.pkl")
    X_test = pd.read_csv("model2/x_test.csv")
    y_test = pd.read_csv("model2/y_test.csv")

    y_pred = model.predict(X_test)

    results = pd.DataFrame({
        'Prix réel': y_test.values.flatten(),
        'Prix prédit': y_pred
    })

    return results
