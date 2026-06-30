# api.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
import shap

# Chargement modèle et scaler
model  = joblib.load("models/RandomForest.pkl")
scaler = joblib.load("models/scaler.pkl")

# Explainer SHAP (créé une seule fois au démarrage)
explainer = shap.TreeExplainer(model)

app = FastAPI(title="API Scoring Crédit", version="1.0")

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

COLONNES = [
    "RevolvingUtilizationOfUnsecuredLines", "age",
    "NumberOfTime30-59DaysPastDueNotWorse", "DebtRatio",
    "MonthlyIncome", "NumberOfOpenCreditLinesAndLoans",
    "NumberOfTimes90DaysLate", "NumberRealEstateLoansOrLines",
    "NumberOfTime60-89DaysPastDueNotWorse", "NumberOfDependents",
    "TotalLatePayments", "DebtToIncome", "IncomePerDependent",
    "AgeGroup", "HighUtilization"
]

LABELS_FR = {
    "RevolvingUtilizationOfUnsecuredLines": "Taux utilisation crédit",
    "age": "Âge",
    "NumberOfTime30-59DaysPastDueNotWorse": "Retards 30-59 jours",
    "DebtRatio": "Ratio dette",
    "MonthlyIncome": "Revenu mensuel",
    "NumberOfOpenCreditLinesAndLoans": "Lignes de crédit ouvertes",
    "NumberOfTimes90DaysLate": "Retards 90+ jours",
    "NumberRealEstateLoansOrLines": "Prêts immobiliers",
    "NumberOfTime60-89DaysPastDueNotWorse": "Retards 60-89 jours",
    "NumberOfDependents": "Personnes à charge",
    "TotalLatePayments": "Total retards paiement",
    "DebtToIncome": "Ratio dette/revenu",
    "IncomePerDependent": "Revenu par dépendant",
    "AgeGroup": "Tranche d'âge",
    "HighUtilization": "Forte utilisation crédit",
}

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

    age = int(df["age"].iloc[0])
    if age <= 30: age_group = 0
    elif age <= 45: age_group = 1
    elif age <= 60: age_group = 2
    else: age_group = 3
    df["AgeGroup"] = age_group

    df["HighUtilization"] = (df["RevolvingUtilizationOfUnsecuredLines"] > 0.75).astype(int)

    return df[COLONNES]

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

    # ─────────────────────────────────────
    # CALCUL SHAP RÉEL pour ce client
    # ─────────────────────────────────────
    shap_values = explainer.shap_values(df_scaled)

    # Pour un RandomForest binaire, shap_values peut être une liste [classe0, classe1]
    if isinstance(shap_values, list):
        sv = shap_values[1][0]  # classe 1 = défaut
    else:
        sv = shap_values[0]
        if sv.ndim > 1:
            sv = sv[:, 1]

    shap_contributions = []
    for i, col in enumerate(COLONNES):
        shap_contributions.append({
            "feature": LABELS_FR.get(col, col),
            "value": round(float(df[col].iloc[0]), 3),
            "shap_value": round(float(sv[i]), 4)
        })

    # Trier par impact absolu décroissant
    shap_contributions.sort(key=lambda x: abs(x["shap_value"]), reverse=True)

    return {
        "prediction":      prediction,
        "probabilite":     round(probabilite, 4),
        "risque":          risque,
        "interpretation":  "Défaut probable" if prediction == 1 else "Pas de défaut prévu",
        "shap_values":     shap_contributions[:8],
        "base_value":      round(float(explainer.expected_value[1]) if isinstance(explainer.expected_value, (list, np.ndarray)) else float(explainer.expected_value), 4)
    }

@app.get("/health")
def health():
    return {"status": "ok"}