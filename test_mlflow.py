import mlflow

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("test-experiment")

with mlflow.start_run(run_name="test_run"):
    mlflow.log_param("learning_rate", 0.01)
    mlflow.log_param("n_estimators", 100)
    mlflow.log_metric("accuracy", 0.92)
    mlflow.log_metric("auc", 0.87)
    print("✅ Run enregistré avec succès !")