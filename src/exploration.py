from variables import charger_brut, CIBLE

df = charger_brut()

print("Dimensions :", df.shape)
print("\nColonnes :", list(df.columns))
print("\nValeurs manquantes :\n", df.isna().sum())
print("\nRépartition de la cible :\n", df[CIBLE].value_counts())
print("\nTaux de remboursement :\n", df["TAUX_REMBOURSEMENT"].value_counts())
print("\nPrix public (PPV) :\n", df["PPV"].describe())
print("\nNombre de molécules (DCI1) :", df["DCI1"].nunique())
print("\nPrix nuls : PPV =", (df["PPV"] == 0).sum(), "| PH =", (df["PH"] == 0).sum())