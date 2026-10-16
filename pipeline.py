import os
import sys
import json
import subprocess
import yaml
import joblib
import pandas as pd
import mlflow
import mlflow.sklearn

P = yaml.safe_load(open("params.yaml", encoding="utf-8"))
mlflow.set_tracking_uri(
    os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
mlflow.set_experiment("medicaments-pipeline")

algo = P["entrainement"]["algorithme"]
seuil = P["validation"]["seuil_f1"]
#Créer le run parent et les runs enfants
with mlflow.start_run(run_name=f"pipeline-{algo}"):          # run parent
    for etape in ["preparer", "entrainer", "evaluer"]:
        with mlflow.start_run(run_name=etape, nested=True):  # run enfant
            subprocess.run([sys.executable, f"src/{etape}.py"], check=True)

    mets = json.load(open("reports/metrics.json"))
# enregistre les paramètres du pipeline dans MLflow.
    mlflow.log_params({
        **{f"prep_{k}": v for k, v in P["preparation"].items()},
        "algorithme": algo,
        **{f"hp_{k}": v for k, v in P["entrainement"]["hyperparametres"][algo].items()},
        "seuil_f1": seuil,
    })
    mlflow.log_metrics(mets)
    mlflow.set_tags({"jeu_de_donnees": "referentiel-medicaments-cnops-2014",
                     "source": "data.gov.ma"})
    mlflow.log_artifact("reports/confusion.png")
    mlflow.log_artifact("reports/metrics.json")

    if mets["f1"] >= seuil:                                   # règle de validation
        exemple = (pd.read_csv("data/prepare/test.csv")
                   .drop(columns="generique").head(3).astype("float64"))
        mlflow.sklearn.log_model(
            joblib.load("models/model.pkl"),
            name="model",
            input_example=exemple,                            # -> signature
            registered_model_name=P["registre"]["nom_modele"],
            skops_trusted_types=P["registre"]["skops_trusted_types"] or None,
        )
        mlflow.set_tag("statut", "accepte")                   # après le succès
        print(f"modele accepte (F1 {mets['f1']} >= {seuil}) et enregistre")
    else:
        mlflow.set_tag("statut", "refuse")
        raise SystemExit(f"modele refuse : F1 {mets['f1']} < seuil {seuil}")