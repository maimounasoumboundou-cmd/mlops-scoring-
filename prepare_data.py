# prepare_data.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import mlflow
import mlflow.sklearn
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────
# 1. CHARGEMENT
# ─────────────────────────────────────────
df = pd.read_csv("cs-training.csv", index_col=0)
print(f"Shape initiale : {df.shape}")
print(df.head())

# ─────────────────────────────────────────
# 2. NETTOYAGE
# ─────────────────────────────────────────

# Renommer la cible pour clarté
df.rename(columns={"SeriousDlqin2yrs": "target"}, inplace=True)

# Valeurs manquantes
print("\nValeurs manquantes avant nettoyage :")
print(df.isnull().sum())

# Imputation par médiane (robuste aux outliers)
df["MonthlyIncome"].fillna(df["MonthlyIncome"].median(), inplace=True)
df["NumberOfDependents"].fillna(df["NumberOfDependents"].median(), inplace=True)

# Suppression des outliers évidents
df = df[df["age"] > 18]
df = df[df["age"] < 100]
df = df[df["RevolvingUtilizationOfUnsecuredLines"] <= 1]
df = df[df["DebtRatio"] < 10]
df = df[df["MonthlyIncome"] < 50000]

print(f"\nShape après nettoyage : {df.shape}")

# ─────────────────────────────────────────
# 3. FEATURE ENGINEERING
# ─────────────────────────────────────────

# Nombre total de retards de paiement
df["TotalLatePayments"] = (
    df["NumberOfTime30-59DaysPastDueNotWorse"] +
    df["NumberOfTime60-89DaysPastDueNotWorse"] +
    df["NumberOfTimes90DaysLate"]
)

# Ratio dette/revenu (éviter division par zéro)
df["DebtToIncome"] = df["DebtRatio"] / (df["MonthlyIncome"] + 1)

# Ratio revenu par dépendant
df["IncomePerDependent"] = df["MonthlyIncome"] / (df["NumberOfDependents"] + 1)

# Tranche d'âge
df["AgeGroup"] = pd.cut(df["age"],
                         bins=[18, 30, 45, 60, 100],
                         labels=[0, 1, 2, 3]).astype(int)

# Flag : client très endetté
df["HighUtilization"] = (df["RevolvingUtilizationOfUnsecuredLines"] > 0.75).astype(int)

print("\nNouvelles features créées :")
print(df[["TotalLatePayments", "DebtToIncome", "IncomePerDependent",
          "AgeGroup", "HighUtilization"]].describe())

# ─────────────────────────────────────────
# 4. SÉPARATION TRAIN / TEST
# ─────────────────────────────────────────
X = df.drop(columns=["target"])
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y   # important : classes déséquilibrées
)

print(f"\nTrain : {X_train.shape} | Test : {X_test.shape}")
print(f"Distribution cible train :\n{y_train.value_counts(normalize=True).round(3)}")

# ─────────────────────────────────────────
# 5. NORMALISATION
# ─────────────────────────────────────────
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# ─────────────────────────────────────────
# 6. SAUVEGARDE
# ─────────────────────────────────────────
import joblib, os
os.makedirs("data", exist_ok=True)
os.makedirs("models", exist_ok=True)

pd.DataFrame(X_train_scaled, columns=X_train.columns).to_csv("data/X_train.csv", index=False)
pd.DataFrame(X_test_scaled,  columns=X_test.columns).to_csv("data/X_test.csv",  index=False)
y_train.to_csv("data/y_train.csv", index=False)
y_test.to_csv("data/y_test.csv",   index=False)
joblib.dump(scaler, "models/scaler.pkl")

print("\n✅ Données sauvegardées dans data/ et scaler dans models/")

# ─────────────────────────────────────────
# 7. LOG MLFLOW
# ─────────────────────────────────────────
mlflow.set_experiment("scoring-data-prep")

with mlflow.start_run(run_name="data_preparation"):
    mlflow.log_param("test_size", 0.2)
    mlflow.log_param("random_state", 42)
    mlflow.log_param("n_train_samples", X_train.shape[0])
    mlflow.log_param("n_test_samples", X_test.shape[0])
    mlflow.log_param("n_features", X_train.shape[1])
    mlflow.log_metric("target_rate_train", float(y_train.mean().round(4)))
    mlflow.log_artifact("data/X_train.csv")
    mlflow.log_artifact("data/X_test.csv")
    print("\n✅ Run MLFlow enregistré !")