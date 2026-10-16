from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.svm import SVC

# nom -> (classe, a besoin de la mise à l'échelle ?)
ALGOS = {
    "regression_logistique": (LogisticRegression, True),
    "knn": (KNeighborsClassifier, True),
    "arbre_seul": (DecisionTreeClassifier, False),
    "hist_gradient_boosting": (HistGradientBoostingClassifier, False),
    "svm_rbf": (SVC, True),
}


def construire_modele(nom, hyperparametres):
    classe, echelle = ALGOS[nom]
    modele = classe(**hyperparametres)
    return make_pipeline(StandardScaler(), modele) if echelle else modele