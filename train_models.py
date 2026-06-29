# train_models.py
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (accuracy_score, roc_auc_score,
                             classification_report, confusion_matrix)
from imblearn.over_sampling import SMOTE
import shap

# ─────────────────────────────────────────
# 0. SCORE MÉTIER
# ─────────────────────────────────────────
COUT_FN = 500_000
COUT_FP = 50_000

def score_metier(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    FP = np.sum((y_pred == 1) & (y_true == 0))
    FN = np.sum((y_pred == 0) & (y_true == 1))
    return (FP * COUT_FP) + (FN * COUT_FN), FP, FN

# ─────────────────────────────────────────
# 1. CHARGEMENT DES DONNÉES
# ─────────────────────────────────────────
print("📂 Chargement des données...")
X_train = pd.read_csv("data/X_train.csv")
X_test  = pd.read_csv("data/X_test.csv")
y_train = pd.read_csv("data/y_train.csv").squeeze()
y_test  = pd.read_csv("data/y_test.csv").squeeze()

# ─────────────────────────────────────────
# 2. SMOTE — Gestion du déséquilibre
# ─────────────────────────────────────────
print("⚖️  Application de SMOTE...")
smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
print(f"   Après SMOTE : {pd.Series(y_train_sm).value_counts().to_dict()}")

# ─────────────────────────────────────────
# 3. MODÈLES ET HYPERPARAMÈTRES
# ─────────────────────────────────────────
modeles = {
    "LogisticRegression": {
        "model": LogisticRegression(max_iter=1000, random_state=42),
        "params": {
            "C": [0.1, 1, 10],
            "solver": ["lbfgs", "liblinear"]
        }
    },
    "RandomForest": {
        "model": RandomForestClassifier(random_state=42, n_jobs=-1),
        "params": {
            "n_estimators": [100, 200],
            "max_depth": [5, 10],
        }
    },
    "XGBoost": {
        "model": XGBClassifier(random_state=42, eval_metric="logloss"),
        "params": {
            "n_estimators": [100, 200],
            "max_depth": [3, 5],
            "learning_rate": [0.05, 0.1]
        }
    }
}

# ─────────────────────────────────────────
# 4. ENTRAÎNEMENT + MLFLOW
# ─────────────────────────────────────────
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("scoring-models")

resultats = {}

for nom, config in modeles.items():
    print(f"\n🔄 Entraînement : {nom}...")

    grid = GridSearchCV(
        config["model"],
        config["params"],
        cv=3,
        scoring="roc_auc",
        n_jobs=-1,
        verbose=0
    )
    grid.fit(X_train_sm, y_train_sm)
    best_model = grid.best_estimator_

    # Prédictions
    y_pred       = best_model.predict(X_test)
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]

    # Métriques
    auc      = roc_auc_score(y_test, y_pred_proba)
    acc      = accuracy_score(y_test, y_pred)
    cout, FP, FN = score_metier(y_test, y_pred)

    print(f"   ✅ AUC : {auc:.4f} | Accuracy : {acc:.4f}")
    print(f"   💰 Coût métier : {cout:,.0f} FCFA (FP={FP}, FN={FN})")

    resultats[nom] = {"auc": auc, "acc": acc, "cout": cout, "model": best_model}

    # Sauvegarde du modèle
    os.makedirs("models", exist_ok=True)
    joblib.dump(best_model, f"models/{nom}.pkl")

    # Log MLFlow
    with mlflow.start_run(run_name=nom):
        mlflow.log_params(grid.best_params_)
        mlflow.log_metric("auc", auc)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("cout_metier_FCFA", cout)
        mlflow.log_metric("faux_positifs", FP)
        mlflow.log_metric("faux_negatifs", FN)
        if nom == "XGBoost":
            import mlflow.xgboost
            mlflow.xgboost.log_model(best_model, name=nom)
        else:
            mlflow.sklearn.log_model(best_model, name=nom)

# ─────────────────────────────────────────
# 5. COMPARAISON DES MODÈLES
# ─────────────────────────────────────────
print("\n" + "="*55)
print("  COMPARAISON DES MODÈLES")
print("="*55)
print(f"{'Modèle':<22} {'AUC':>8} {'Accuracy':>10} {'Coût (FCFA)':>18}")
print("-"*55)
for nom, r in resultats.items():
    print(f"{nom:<22} {r['auc']:>8.4f} {r['acc']:>10.4f} {r['cout']:>18,.0f}")

meilleur = min(resultats, key=lambda x: resultats[x]['cout'])
print(f"\n🏆 Meilleur modèle (coût métier) : {meilleur}")

# ─────────────────────────────────────────
# 6. FEATURE IMPORTANCE — SHAP (XGBoost)
# ─────────────────────────────────────────
print("\n📊 Calcul SHAP pour XGBoost...")
xgb_model = resultats["XGBoost"]["model"]
explainer  = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_test[:500])

# Sauvegarde du résumé SHAP
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
shap.summary_plot(shap_values, X_test[:500], show=False)
plt.tight_layout()
plt.savefig("models/shap_summary.png", dpi=100, bbox_inches='tight')
plt.close()
print("   ✅ Graphique SHAP sauvegardé : models/shap_summary.png")

# Log SHAP dans MLFlow
with mlflow.start_run(run_name="SHAP_XGBoost"):
    mlflow.log_artifact("models/shap_summary.png")
    print("   ✅ SHAP enregistré dans MLFlow !")

print("\n✅ Étape 4 terminée !")