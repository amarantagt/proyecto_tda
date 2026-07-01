import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from andrews import encode, normalizacion
from sublevel_persistence import persistence_diagrams, bottleneck, wasserstein

# Función para agregar ruido gaussiano
def gaussian_noise(x, sigma=0.05):
    ruido = np.random.normal(loc=0, scale=sigma, size=x.shape)
    return x + ruido

# Función para agregar perturbaciones aleatorias acotadas
def bounded_noise(x, epsilon= 0.05):
    ruido = np.random.uniform(low=-epsilon,high=epsilon,size=x.shape)
    return x + ruido
