import gudhi as gd
import gudhi.wasserstein import wasserstein_distance
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from andrews import encode, normalizacion

""" Módulo sublevel_persistence.py: 
Tiene como objetivo implementar el cálculo los diagramas de persistencia de un set de 
datos y sus estadísticas topológicas
"""
# Construimos funciones auxiliares para manejar correctamente el
# lifetime infinito

 # Función finite_diagram, devuelve los intervalos con muerte finita
def finite_diagram(diagram):
    if len(diagram) == 0 :
        return diagram
    
    return diagram[np.isfinite(diagram[:, 1])]

# Función diagram_for_plot, reemplaza muertes infinias por un valor más grande
def diagram_for_plot(diagram, margin = 0.1):
    if len(diagram) == 0:
        return diagram
    
    # Hacemos una copia del diagrama
    diagram_plot = diagram.copy()
    finite = diagram_plot[np.isfinite(diagram_plot[:, 1])]

    # Si no hay intervalos con lifetimes con muertes finitas
    # se reemplaza la muerte infinita con 1.0
    if len(finite) == 0:
        replacement = 1.0
    # Si existen intervalos con muertes finitas, se reemplazan las muertes infinitas
    # con el número máximo y un margen.
    else:
        replacement = np.max(finite[:, 1] * (1 + margin))
    
    diagram_plot[np.isinf(diagram_plot[:, 1]), 1] = replacement
    return diagram_plot



# Función persistence_diagrams, calcula el diagrama de persistencia de una curva
def persistence_diagrams(curve):
    curve = np.asarray(curve) # Nos aseguramos que curve sea un arreglo
    
    # Construimos el cubical complex
    cc = gd.CubicalComplex(dimensions=[len(curve)], top_dimensional_cells = curve)
    cc.compute_persistence()

    # Extraemos el H0 y el H1
    H0 = np.array(cc.persistence_intervals_in_dimension(0))
    H1 = np.array(cc.persistence_intervals_in_dimension(1))

    return H0, H1 

# Ahora calculamos la persistencia total
def total_persistence(diagram):
    if len(diagram) == 0:
        return 0.0
    lifetimes = diagram[:,1] - diagram[:,0] # Hacemos la resta death - birth

    return np.sum(lifetimes) # Retornamos las sumas de todas las vidas.

#  Calculamos la persistencia máxima encontrada.
def max_lifetime(diagram):
    if len(diagram) == 0:
        return 0.0
    lifetimes = diagram[:, 1] - diagram[:, 0]
    return np.max(lifetimes)

# Función count_pairs, cuenta cuántos intervalos sobreviven más que un umbral tau
def count_pairs(diagram, tau = 0.1):
    if len(diagram) == 0:
        return 0  
    lifetimes = diagram[:, 1] - diagram[:, 0]
    return int(np.sum(lifetimes > tau))

# Calculamos la entropía persistente
def persistent_entropy(diagram):
    if len(diagram) == 0:
        return 0.0
    
    lifetimes = diagram[:, 1] - diagram[:, 0]
    lifetimes = lifetimes[lifetimes > 0]

    if len(lifetimes) == 0:
        return 0.0
    
    p = lifetimes / np.sum(lifetimes)

    return -np.sum(p * np.log(p))

# La función topological_isgnature utiliza reune a las funciones anteriores
# para calcular los diagramas de persistencia y estadísticas de una curva.
def topological_signature(curve, tau = 0.1):
    H0, H1 = persistence_diagrams(curve) # Calculamos el diagrama de persistencia de la curva.

    # Ahora guardamos todos los valores en un diccionario
    return {
        "H0": H0,
        "H1" : H1,

        "pairs_H0" : count_pairs(H0, tau),
        "pairs_H1" : count_pairs(H1, tau),

        "total_persistence_H0" : total_persistence(H0),
        "total_persistence_H1" : total_persistence(H1),

        "max_H0" : max_lifetime(H0),
        "max_H1" : max_lifetime(H1),

        "entropy_H0" : persistent_entropy(H0),
        "entropy_H1" : persistent_entropy(H1)
    }



if __name__ == "__main__":
    df = pd.read_csv(
        "data/winequality-red.csv",
        sep=";"
    )

    X = df.drop(columns=["quality"]).values
    X = normalizacion(X, metodo="zscore")
    curve = encode(X[0], N=256)
    resultado = topological_signature(curve)
    print(resultado)