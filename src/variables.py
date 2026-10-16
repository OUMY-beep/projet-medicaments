import numpy as np
import pandas as pd

FICHIER_BRUT = "data/brut/ref-des-medicaments-cnops-2014.xlsx"
CIBLE = "PRINCEPS_GENERIQUE"


def charger_brut(chemin=FICHIER_BRUT):
    return pd.read_excel(chemin, engine="openpyxl")


def _taux_en_nombre(s):
    """'70%' -> 70.0 ; gère aussi le cas où Excel stocke 0.7."""
    t = pd.to_numeric(s.astype(str).str.replace("%", "", regex=False).str.strip(),
                      errors="coerce")
    if t.max() <= 1:          # stocké en fraction : on remet en pourcentage
        t = t * 100
    return t.fillna(0)


def construire_tableau(df): # prépare les données avant de les donner à un modèle de Machine Learning.
    """Tableau brut -> (X numérique, y). Aucune variable n'utilise la cible."""
    d = df.copy()
    X = pd.DataFrame(index=d.index)

    # --- variables simples ---
    X["ppv"] = pd.to_numeric(d["PPV"], errors="coerce").fillna(0)
    X["ph"] = pd.to_numeric(d["PH"], errors="coerce").fillna(0)
    X["prix_br"] = pd.to_numeric(d["PRIX_BR"], errors="coerce").fillna(0)
    X["taux"] = _taux_en_nombre(d["TAUX_REMBOURSEMENT"])

    # dosage : non numérique -> NaN -> médiane + indicateur "manquant"
    dosage = pd.to_numeric(d["DOSAGE1"], errors="coerce")
    X["dosage_manquant"] = dosage.isna().astype(int)
    X["dosage"] = dosage.fillna(dosage.median())

    # --- variables construites ---
    ppv_sûr = X["ppv"].replace(0, np.nan)               # évite la division par 0
    X["ratio_ph_ppv"] = (X["ph"] / ppv_sûr).fillna(0)
    X["ratio_br_ppv"] = (X["prix_br"] / ppv_sûr).fillna(0)
    X["log_ppv"] = np.log1p(X["ppv"]) # L'objectif est de réduire l'écart entre les petits et les grands prix, pour aider le modèle à mieux exploiter les données.

    # idée clé du TP : combien de présentations partagent la même molécule ?
    X["nb_par_dci"] = d.groupby("DCI1")["DCI1"].transform("count")
    # prix relatif au prix médian de la molécule
    med_dci = X.groupby(d["DCI1"])["ppv"].transform("median").replace(0, np.nan)
    X["ppv_relatif_dci"] = (X["ppv"] / med_dci).fillna(1)

    X = X.astype(float)                                  # signature MLflow : tout en double

    y = (d[CIBLE].astype(str).str.strip().str.upper() == "G").astype(int)
    return X, y