import json
import mlflow
from variables import charger_brut, construire_tableau

NOM = "medicament-generique"
mlflow.set_tracking_uri("http://127.0.0.1:5000")

df = charger_brut()
X, y = construire_tableau(df)

modele = mlflow.pyfunc.load_model(f"models:/{NOM}@production")
pred = modele.predict(X)

# un générique et un princeps correctement classés par le modèle
i_g = X.index[(y == 1) & (pred == 1)][0]
i_p = X.index[(y == 0) & (pred == 0)][0]
print("Générique (attendu 1) :", df.loc[i_g, "NOM"])
print("Princeps  (attendu 0) :", df.loc[i_p, "NOM"])

sel = X.loc[[i_g, i_p]]

# requête valide : les 11 colonnes, dans l'ordre de la signature
valide = {"dataframe_split": {"columns": list(sel.columns),
                              "data": sel.values.tolist()}}

# requête invalide : la colonne "ph" est volontairement supprimée
sans_ph = sel.drop(columns="ph")
invalide = {"dataframe_split": {"columns": list(sans_ph.columns),
                                "data": sans_ph.values.tolist()}}

with open("reports/requete_valide.json", "w", encoding="utf-8") as f:
    json.dump(valide, f)
with open("reports/requete_invalide.json", "w", encoding="utf-8") as f:
    json.dump(invalide, f)

print("reports/requete_valide.json et reports/requete_invalide.json écrits")