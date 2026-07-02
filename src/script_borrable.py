import pandas as pd
import numpy as np
from andrews import encode, normalizacion
from sublevel_persistence import persistence_diagrams, finite_diagram

df = pd.read_csv("data/winequality-red.csv", sep=";")
X = df.drop(columns=["quality"]).values
X = normalizacion(X, metodo="zscore")

for idx in [613, 486, 1102, 529]:
    curva = encode(X[idx], N=256)
    H0, _ = persistence_diagrams(curva)
    H0_finito = finite_diagram(H0)
    print(f"sample={idx}: {len(H0)} puntos totales en H0, {len(H0_finito)} finitos")