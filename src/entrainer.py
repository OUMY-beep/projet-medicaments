import os
import yaml
import joblib
import pandas as pd
from modeles import construire_modele

P = yaml.safe_load(open("params.yaml", encoding="utf-8"))["entrainement"]
nom = P["algorithme"]

tr = pd.read_csv("data/prepare/train.csv")
X, y = tr.drop(columns="generique"), tr["generique"]

modele = construire_modele(nom, P["hyperparametres"][nom]).fit(X, y)

os.makedirs("models", exist_ok=True)
joblib.dump(modele, "models/model.pkl")
print(f"entrainement : {nom} ecrit dans models/model.pkl")