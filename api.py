# api.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd

# Chargement modèle et scaler
model  = joblib.load("models/RandomForest.pkl")
scaler = joblib.load("models/scaler.pkl")

app = FastAPI(title="API Scoring Crédit", version="1.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ClientData(BaseModel):
    RevolvingUtilizationOfUnsecuredLines: float
    age: int
    NumberOfTime30_59DaysPastDueNotWorse: int
    DebtRatio: float
    MonthlyIncome: float
    NumberOfOpenCreditLinesAndLoans: int
    NumberOfTimes90DaysLate: int
    NumberRealEstateLoansOrLines: int
    NumberOfTime60_89DaysPastDueNotWorse: int
    NumberOfDependents: int

def build_features(data: ClientData) -> pd.DataFrame:
    d = data.dict()
    d["NumberOfTime30-59DaysPastDueNotWorse"] = d.pop("NumberOfTime30_59DaysPastDueNotWorse")
    d["NumberOfTime60-89DaysPastDueNotWorse"] = d.pop("NumberOfTime60_89DaysPastDueNotWorse")

    df = pd.DataFrame([d])

    df["TotalLatePayments"] = (
        df["NumberOfTime30-59DaysPastDueNotWorse"] +
        df["NumberOfTime60-89DaysPastDueNotWorse"] +
        df["NumberOfTimes90DaysLate"]
    )
    df["DebtToIncome"]       = df["DebtRatio"] / (df["MonthlyIncome"] + 1)
    df["IncomePerDependent"] = df["MonthlyIncome"] / (df["NumberOfDependents"] + 1)

    # Correction AgeGroup — gérer les ages hors bornes
    age = int(df["age"].iloc[0])
    if age <= 30:
        age_group = 0
    elif age <= 45:
        age_group = 1
    elif age <= 60:
        age_group = 2
    else:
        age_group = 3
    df["AgeGroup"] = age_group

    df["HighUtilization"] = (df["RevolvingUtilizationOfUnsecuredLines"] > 0.75).astype(int)

    colonnes = [
        "RevolvingUtilizationOfUnsecuredLines", "age",
        "NumberOfTime30-59DaysPastDueNotWorse", "DebtRatio",
        "MonthlyIncome", "NumberOfOpenCreditLinesAndLoans",
        "NumberOfTimes90DaysLate", "NumberRealEstateLoansOrLines",
        "NumberOfTime60-89DaysPastDueNotWorse", "NumberOfDependents",
        "TotalLatePayments", "DebtToIncome", "IncomePerDependent",
        "AgeGroup", "HighUtilization"
    ]
    return df[colonnes]

@app.get("/")
def root():
    return {"message": "API Scoring Crédit — opérationnelle ✅"}

@app.post("/predict")
def predict(data: ClientData):
    df = build_features(data)
    df_scaled = scaler.transform(df)
    prediction  = int(model.predict(df_scaled)[0])
    probabilite = float(model.predict_proba(df_scaled)[0][1])

    if probabilite < 0.3:
        risque = "FAIBLE"
    elif probabilite < 0.6:
        risque = "MOYEN"
    else:
        risque = "ÉLEVÉ"

    return {
        "prediction":      prediction,
        "probabilite":     round(probabilite, 4),
        "risque":          risque,
        "interpretation":  "Défaut probable" if prediction == 1 else "Pas de défaut prévu"
    }

@app.get("/health")
def health():
    return {"status": "ok"}