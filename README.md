# Proyecto semestral CC5514-1 Análisis Topológico de Datos: TDA Aplicado a las Curvas de Andrews
## Robustez y Estabilidad de Firmas Topológicas de Curvas de Andrews
#### Integrantes:
* Amaranta Godoy
* Felipe Quilodrán
* Valeria Serrano

## Descripción del proyecto

## Estructura del proyecto

```
PROYECTO_TDA/
├── src/
|   ├── exploracion_datos.py    → implementación de una exploración inicial de los datos.
│   ├── andrews.py              → implementación de la normalización y generación de Curvas de Andrews.
|   ├──stability.py             → implementación de experimentos.
│   └── sublevel_persistence.py → implementación de cálculo de diagramas de persistencia.

├── data/
│   ├── winequality-red.csv      → dataset 1 (Vinos tintos).
│   └── winequality-white.csv    → dataset 2 (Vinos blancos).
├── figures/
|   ├── boxplot_vino_blanco.png                 → boxplot dataset 2.
|   ├── boxplot_vino_tinto.png                  → boxplot dataset 1.
|   ├── distribucion_calidad_vino_blanco.png    → gráfico de distribución de calidad del dataset 2.
|   ├── distribucion_calidad_vino_tinto.png     → gráfico de distribución de calidad del dataset 1.
|   ├── blanco_minmax.png                       → normalización de curvas dataset 2 con minmax.
|   ├── blanco_robust.png                       → normalización de curvas dataset 2 con robust.
|   ├── blanco_zscore.png                       → normalización de curvas dataset 2 con zscore.
|   ├── tinto_minmax.png                        → normalización de curvas dataset 1 con minmax.
|   ├── tinto_robust.png                        → normalización de curvas dataset 1 con robust.
|   └── tinto_zscore.png                        → normalización de curvas dataset 1 con zscore.
├── resultado_exploracion.txt                   → resultados de la ejecución de exploracion_datos.py
├── prueba_sublevel_persistence.txt             → resultados de la ejecución de sublevel_persistence.py
├── .gitignore
└── README.md
```

## Ejecución del proyecto
Para realizar la exploración de datos y guardarla en resultado_exploracion.txt

```bash
python3 src/exploracion_datos.py > resultado_exploracion.txt
```

Para revisar los resultados de las pruebas de andrews.py 
```bash
python3 src/andrews.py 
```
Para realizar las pruebas de sublevel_persistence.py
```bash
python3 src/sublevel_persistence.py > prueba_sublevel_persistence.txt
```

## Resultados generados
De la exploración de dato se crean los archivos:

* resultado_exploracion.txt
* boxplot_vino_blanco.png
* boxplot_vino_tinto.png
* distribucion_calidad_vino_blanco.png
* distribucion_calidad_vino_tinto.png

De las pruebas de andrews.py se generan las imágenes correspondientes a las normalizaciones de las curvas:

* blanco_minmax.png
* tinto_minmax.png
* blanco_robust.png
* tinto_robust.png
* blanco_zscore.png
* tinto_zscore.png

De las pruebas de sublevel_persistence.py se generan un resumen de los resultados en prueba_sublevel_persistence.txt