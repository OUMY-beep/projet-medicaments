import os
import yaml
from sklearn.model_selection import train_test_split
from variables import charger_brut, construire_tableau

P = yaml.safe_load(open("params.yaml", encoding="utf-8"))["preparation"]

X, y = construire_tableau(charger_brut())
Xtr, Xte, ytr, yte = train_test_split(
    X, y, test_size=P["test_size"], random_state=P["random_state"])

os.makedirs("data/prepare", exist_ok=True)
tr = Xtr.assign(generique=ytr)
te = Xte.assign(generique=yte)
tr.to_csv("data/prepare/train.csv", index=False)
te.to_csv("data/prepare/test.csv", index=False)
print("preparation :", len(tr), "lignes train /", len(te), "lignes test")