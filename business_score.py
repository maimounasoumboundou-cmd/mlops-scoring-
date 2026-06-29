# business_score.py
import numpy as np
import pandas as pd
import mlflow

# ─────────────────────────────────────────
# 1. DÉFINITION DU SCORE MÉTIER
# ─────────────────────────────────────────

COUT_FN = 500_000  # Faux Négatif : banque perd l'argent prêté (FCFA)
COUT_FP = 50_000   # Faux Positif : banque perd un bon client (FCFA)

def score_metier(y_true, y_pred):
    """
    Calcule le coût métier total basé sur les erreurs du modèle.
    Plus le score est bas, meilleur est le modèle.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Calcul des FP et FN
    FP = np.sum((y_pred == 1) & (y_true == 0))
    FN = np.sum((y_pred == 0) & (y_true == 1))
    TP = np.sum((y_pred == 1) & (y_true == 1))
    TN = np.sum((y_pred == 0) & (y_true == 0))

    cout_total = (FP * COUT_FP) + (FN * COUT_FN)

    print(f"\n{'='*45}")
    print(f"  SCORE MÉTIER — ANALYSE DES ERREURS")
    print(f"{'='*45}")
    print(f"  Vrais Positifs  (TP) : {TP:>8}")
    print(f"  Vrais Négatifs  (TN) : {TN:>8}")
    print(f"  Faux Positifs   (FP) : {FP:>8}  → {FP * COUT_FP:>15,.0f} FCFA")
    print(f"  Faux Négatifs   (FN) : {FN:>8}  → {FN * COUT_FN:>15,.0f} FCFA")
    print(f"{'='*45}")
    print(f"  COÛT TOTAL           : {cout_total:>15,.0f} FCFA")
    print(f"{'='*45}\n")

    return cout_total, FP, FN

# ─────────────────────────────────────────
# 2. TEST AVEC UN EXEMPLE SIMPLE
# ─────────────────────────────────────────

# Simulation : 1000 clients
np.random.seed(42)
y_true = np.random.choice([0, 1], size=1000, p=[0.937, 0.063])

# Modèle baseline : prédit toujours 0 (pas de défaut)
y_pred_baseline = np.zeros(1000, dtype=int)

# Modèle amélioré : prédit aléatoirement
y_pred_ameliore = np.random.choice([0, 1], size=1000, p=[0.85, 0.15])

print("📊 BASELINE (prédit toujours 0) :")
cout_baseline, _, _ = score_metier(y_true, y_pred_baseline)

print("📊 MODÈLE AMÉLIORÉ :")
cout_ameliore, _, _ = score_metier(y_true, y_pred_ameliore)

# ─────────────────────────────────────────
# 3. LOG MLFLOW
# ─────────────────────────────────────────
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("scoring-metier")

with mlflow.start_run(run_name="definition_score_metier"):
    mlflow.log_param("cout_faux_negatif_FCFA", COUT_FN)
    mlflow.log_param("cout_faux_positif_FCFA", COUT_FP)
    mlflow.log_param("ratio_FN_FP", COUT_FN / COUT_FP)
    mlflow.log_metric("cout_baseline_FCFA", cout_baseline)
    mlflow.log_metric("cout_modele_ameliore_FCFA", cout_ameliore)
    mlflow.log_metric("gain_FCFA", cout_baseline - cout_ameliore)
    print("✅ Score métier enregistré dans MLFlow !")