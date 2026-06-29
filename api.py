# api.py
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd

# ─────────────────────────────────────────
# 1. CHARGEMENT DU MODÈLE ET SCALER
# ─────────────────────────────────────────
model  = joblib.load("models/RandomForest.pkl")
scaler = joblib.load("models/scaler.pkl")

app = FastAPI(title="API Scoring Crédit", version="1.0")

# ─────────────────────────────────────────
# 2. SCHÉMA D'ENTRÉE
# ─────────────────────────────────────────
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

# ─────────────────────────────────────────
# 3. FEATURE ENGINEERING (même que prepare_data.py)
# ─────────────────────────────────────────
def build_features(data: ClientData) -> pd.DataFrame:
    d = data.dict()

    # Renommer pour correspondre aux colonnes originales
    d["NumberOfTime30-59DaysPastDueNotWorse"] = d.pop("NumberOfTime30_59DaysPastDueNotWorse")
    d["NumberOfTime60-89DaysPastDueNotWorse"] = d.pop("NumberOfTime60_89DaysPastDueNotWorse")

    df = pd.DataFrame([d])

    # Nouvelles features
    df["TotalLatePayments"] = (
        df["NumberOfTime30-59DaysPastDueNotWorse"] +
        df["NumberOfTime60-89DaysPastDueNotWorse"] +
        df["NumberOfTimes90DaysLate"]
    )
    df["DebtToIncome"]       = df["DebtRatio"] / (df["MonthlyIncome"] + 1)
    df["IncomePerDependent"] = df["MonthlyIncome"] / (df["NumberOfDependents"] + 1)
    df["AgeGroup"]           = pd.cut(df["age"], bins=[18, 30, 45, 60, 100],
                                       labels=[0, 1, 2, 3]).astype(int)
    df["HighUtilization"]    = (df["RevolvingUtilizationOfUnsecuredLines"] > 0.75).astype(int)

    # Forcer le bon ordre des colonnes (même ordre que pendant l'entraînement)
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

# ─────────────────────────────────────────
# 4. ENDPOINTS
# ─────────────────────────────────────────
@app.get("/")
def root():
    return {"message": "API Scoring Crédit — opérationnelle ✅"}

@app.post("/predict")
def predict(data: ClientData):
    # Construction des features
    df = build_features(data)

    # Normalisation
    df_scaled = scaler.transform(df)

    # Prédiction
    prediction    = int(model.predict(df_scaled)[0])
    probabilite   = float(model.predict_proba(df_scaled)[0][1])

    # Interprétation
    if probabilite < 0.3:
        risque = "FAIBLE"
    elif probabilite < 0.6:
        risque = "MOYEN"
    else:
        risque = "ÉLEVÉ"

    return {
        "prediction":   prediction,
        "probabilite":  round(probabilite, 4),
        "risque":       risque,
        "interpretation": "Défaut probable" if prediction == 1 else "Pas de défaut prévu"
    }

@app.get("/health")
def health():
    return {"status": "ok"}