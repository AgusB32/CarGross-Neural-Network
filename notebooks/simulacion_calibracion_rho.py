# %% [markdown]
# # Simulación Experimental: Calibración del Parámetro de Vigilancia ($\rho$) en ART1
# **Trabajo Práctico Final Integrador — Inteligencia Artificial (2026) — Grupo 6**
#
# Este cuaderno interactivo evalúa el impacto del hiperparámetro de vigilancia $\rho \in (0, 1]$
# sobre la formación dinámica de clústeres vocacionales en la capa $F_2$ y la coherencia
# psicopedagógica de los prototipos aprendidos $T_{Ji}$ en `data/dataset_A.csv` y `data/dataset_B.csv`.

# %%
import os
import sys

sys.dont_write_bytecode = True

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(ROOT_DIR, "src")
DATA_DIR = os.path.join(ROOT_DIR, "data")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import numpy as np
import pandas as pd
from CarGross import ART1, cargar_csv_binario, evaluar_dataset, interpretar_cluster

# %% [markdown]
# ## 1. Barrido de Sensibilidad del Parámetro $\rho$ (Curva de Calibración)
# Evaluamos una grilla de valores $\rho \in \{0.20, 0.30, 0.45, 0.60, 0.75, 0.85, 0.95\}$
# para identificar el punto de equilibrio entre sub-clasificación y sobre-clasificación.

# %%
def simular_curva_calibracion(ruta_csv):
    X, cols, _, idx_nul, _ = cargar_csv_binario(ruta_csv)
    grilla_rho = [0.20, 0.30, 0.45, 0.60, 0.75, 0.85, 0.95]
    filas = []

    for rho in grilla_rho:
        red = ART1(rho=rho)
        asig, _, epocas = red.entrenar(X)

        bits_activos = []
        coherencias = []
        for j in range(red.n_categorias):
            mask = asig == j
            _, _, coh = interpretar_cluster(red.weights_td[j], cols, X[mask])
            bits_activos.append(int(np.sum(red.weights_td[j] >= 0.5)))
            coherencias.append(coh * int(np.sum(mask)))

        regimen = (
            "Sub-clasificacion (amplio)"
            if rho <= 0.40
            else ("Sobre-clasificacion (micro)" if rho >= 0.75 else "Equilibrio optimo")
        )
        filas.append({
            "Dataset": os.path.basename(ruta_csv),
            "Rho": rho,
            "Clusters_F2": red.n_categorias,
            "Alumnos_por_Cluster": round(len(X) / max(1, red.n_categorias), 2),
            "Media_Bits_T_Ji": round(float(np.mean(bits_activos)), 2),
            "Coherencia_RIASEC_%": round(float(np.sum(coherencias) / len(X)), 1),
            "Epocas": epocas,
            "Regimen": regimen,
        })

    return pd.DataFrame(filas)


if __name__ == "__main__":
    ruta_a = os.path.join(DATA_DIR, "dataset_A.csv")
    ruta_b = os.path.join(DATA_DIR, "dataset_B.csv")

    print("\n=== CURVA DE CALIBRACION DE VIGILANCIA (RHO) EN DATASET A ===")
    df_curva_a = simular_curva_calibracion(ruta_a)
    print(df_curva_a.to_string(index=False))

    print("\n=== CURVA DE CALIBRACION DE VIGILANCIA (RHO) EN DATASET B ===")
    df_curva_b = simular_curva_calibracion(ruta_b)
    print(df_curva_b.to_string(index=False))

    print("\n=== DETALLE DEL PUNTO DE EQUILIBRIO OPTIMO (RHO = 0.60) ===")
    evaluar_dataset(ruta_a, rho=0.60)

