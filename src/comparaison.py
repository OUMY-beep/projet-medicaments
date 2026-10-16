# entraîne plusieurs algorithmes de classification, compare leurs performances et enregistre les résultats dans MLflow.
import time
import mlflow
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline #Construire un pipeline
from sklearn.preprocessing import StandardScaler #Standardiser les variables
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.svm import SVC

from variables import charger_brut, construire_tableau

TEST_SIZE = 0.25
SEED = 42

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("medicaments-exploration")

X, y = construire_tableau(charger_brut())
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=TEST_SIZE, random_state=SEED)
print(f"train : {len(Xtr)} lignes | test : {len(Xte)} lignes | variables : {X.shape[1]}")


CANDIDATS = [
    ("regression_logistique", "lineaire",
     LogisticRegression(max_iter=1000, random_state=SEED), True),
    ("knn", "plus_proches_voisins",
     KNeighborsClassifier(n_neighbors=15), True),
    ("arbre_seul", "arbre",
     DecisionTreeClassifier(max_depth=8, random_state=SEED), False),
    ("hist_gradient_boosting", "boosting",
     HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1,
                                    random_state=SEED), False),
    ("svm_rbf", "svm",
     SVC(C=1.0, kernel="rbf", random_state=SEED), True),
]
# besoin_echelle est un booléen qui indique s'il faut standardiser les variables.
resultats = []
for nom, famille, modele, besoin_echelle in CANDIDATS:
    # le scaler est DANS le pipeline : il est ajusté sur le train uniquement
    estimateur = make_pipeline(StandardScaler(), modele) if besoin_echelle else modele

    with mlflow.start_run(run_name=nom):
        debut = time.perf_counter()
        estimateur.fit(Xtr, ytr)
        duree = time.perf_counter() - debut

        pred = estimateur.predict(Xte)
        mets = {
            "f1": f1_score(yte, pred),           # classe 1 = générique
            "precision": precision_score(yte, pred),
            "recall": recall_score(yte, pred),
            "duree_entrainement_s": duree,
        }
# Enregistrer les paramètres dans MLflow
        mlflow.log_params({
            "algorithme": nom,
            "famille": famille,
            "nb_variables": X.shape[1],
            "mise_a_echelle": besoin_echelle,
            "test_size": TEST_SIZE,
            "random_state": SEED,
            # Ligne récupère automatiquement les hyperparamètres du modèle, 
            # conserve ceux qui ont une valeur simple (nombre, texte, booléen ou None), 
            # et ajoute le préfixe hp_ à leur nom et les transmet à MLflow grâce à **.
            **{f"hp_{k}": v for k, v in modele.get_params().items()
               if isinstance(v, (int, float, str, bool)) or v is None},
        })
        mlflow.log_metrics(mets)
        mlflow.set_tags({
            "jeu_de_donnees": "referentiel-medicaments-cnops-2014",
            "source": "data.gov.ma",
        })
        resultats.append((nom, famille, mets))

print("\n=== Résumé (trié par F1) ===")
print(f"{'algorithme':<24}{'famille':<22}{'F1':>7}{'préc.':>8}{'rappel':>8}{'durée(s)':>10}")
for nom, famille, m in sorted(resultats, key=lambda r: -r[2]["f1"]):
    print(f"{nom:<24}{famille:<22}{m['f1']:>7.4f}{m['precision']:>8.4f}"
          f"{m['recall']:>8.4f}{m['duree_entrainement_s']:>10.3f}")