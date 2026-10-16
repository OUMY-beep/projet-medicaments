import mlflow
from mlflow import MlflowClient
from variables import charger_brut, construire_tableau

NOM = "medicament-generique"
V_PRODUCTION = 3   # meilleur réglage (max_iter 200)  <- à adapter
V_CANDIDATE = 5    # réglage plus simple (max_iter 30) <- à adapter

mlflow.set_tracking_uri("http://127.0.0.1:5000")
c = MlflowClient()

# --- 1. lister les versions avec leur F1 ---
print("Versions au registre :")
f1_par_version = {}
run_par_version = {}
for v in sorted(c.search_model_versions(f"name='{NOM}'"), key=lambda v: int(v.version)):
    run = c.get_run(v.run_id)
    f1_par_version[int(v.version)] = run.data.metrics["f1"]
    run_par_version[v.run_id] = int(v.version)
    print(f"  version {v.version} | F1 = {run.data.metrics['f1']:.4f}"
          f" | max_iter = {run.data.params.get('hp_max_iter')}")

# --- 2. alias + description ---
c.set_registered_model_alias(NOM, "production", V_PRODUCTION)
c.set_registered_model_alias(NOM, "staging", V_CANDIDATE)
c.update_model_version(
    NOM, str(V_PRODUCTION),
    description=f"Meilleur reglage (F1 {f1_par_version[V_PRODUCTION]:.4f}) - valide, en production.")
print(f"\nproduction -> version {V_PRODUCTION} | staging -> version {V_CANDIDATE}")

# --- 3. un medicament de mon choix ---
df = charger_brut()
X, y = construire_tableau(df)
i = y[y == 1].index[0]                  # premier generique du fichier
medicament = X.loc[[i]]
print(f"\nMedicament : {df.loc[i, 'NOM']} (reel : generique)")


def qui_repond(donnees):
    """Code client : ne change JAMAIS, seul l'alias bouge cote registre."""
    modele = mlflow.pyfunc.load_model(f"models:/{NOM}@production")
    version = run_par_version.get(modele.metadata.run_id, "?")
    return version, modele.predict(donnees)


v, p = qui_repond(medicament)
print(f"[etat initial]   version {v} repond -> prediction {p}")

# --- 4. promotion de la candidate, puis retour arriere ---
c.set_registered_model_alias(NOM, "production", V_CANDIDATE)
v, p = qui_repond(medicament)
print(f"[promotion]      version {v} repond -> prediction {p}")

c.set_registered_model_alias(NOM, "production", V_PRODUCTION)
v, p = qui_repond(medicament)
print(f"[retour arriere] version {v} repond -> prediction {p}")