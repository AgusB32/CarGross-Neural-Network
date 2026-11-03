"""
src/preparar_datos.py - Preprocesamiento de datos vocacionales para la red ART1.
Transforma data/career_data.csv a codificacion binaria dispersa de 110 bits (CSV puro).
"""

from itertools import combinations
import os
import sys

sys.dont_write_bytecode = True

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

# 11 variables del modelo: (columna_original, prefijo_binario, dividir_por_10, descripcion)
VARIABLES = [
    ("Math_Score", "Math", True, "Rendimiento en matematica (escala 1-100 -> 1-10)"),
    ("Science_Score", "Science", True, "Rendimiento en ciencias (escala 1-100 -> 1-10)"),
    ("Programming_Skill", "Programming", False, "Habilidad de programacion"),
    ("Communication_Skill", "Communication", False, "Capacidad de comunicacion"),
    ("Logical_Ability", "Logical", False, "Razonamiento logico"),
    ("R_score", "R", False, "RIASEC Realista (trabajo practico / tecnico)"),
    ("I_score", "I", False, "RIASEC Investigador (analitico / cientifico)"),
    ("A_score", "A", False, "RIASEC Artistico (creativo / diseno)"),
    ("S_score", "S", False, "RIASEC Social (docencia / salud / servicio)"),
    ("E_score", "E", False, "RIASEC Emprendedor (liderazgo / negocios)"),
    ("C_score", "C", False, "RIASEC Convencional (organizacion / contable)"),
]


def obtener_columnas_110():
    """Devuelve la lista con los 110 nombres de columnas binarias (v1 a v10 por variable)."""
    return [f"{pref}_v{b}" for _, pref, _, _ in VARIABLES for b in range(1, 11)]


def valor_a_bin(valor, div10):
    """
    Preprocesamiento en 2 pasos (TPFI Seccion 2):
    1. Escalamiento previo: si esta en escala 1-100, se divide por 10.
    2. Mapeo a bines discretos [0,1)->0 (v1), [1,2)->1 (v2), ..., [9,10]->9 (v10).
    """
    v = float(valor) / 10.0 if div10 else float(valor)
    return int(np.clip(np.floor(v), 0, 9))


def binarizar_dataframe(df):
    """Convierte el DataFrame original de 11 variables en una matriz binaria de 110 bits."""
    n = len(df)
    matriz = np.zeros((n, 110), dtype=int)

    for fila_idx, (_, fila) in enumerate(df.iterrows()):
        vals = [float(fila[col]) for col, _, _, _ in VARIABLES]
        if all(v <= 0 for v in vals):
            continue  # Vector nulo: queda en 0 para ser filtrado

        for var_idx, (col, _, div10, _) in enumerate(VARIABLES):
            b = valor_a_bin(fila[col], div10)
            matriz[fila_idx, var_idx * 10 + b] = 1

    # Filtrar vectores completamente nulos (todos 0), ya que anulan la capa F1
    validos = matriz.sum(axis=1) > 0
    nulos_filtrados = int(np.sum(~validos))
    df_bin = pd.DataFrame(matriz[validos], columns=obtener_columnas_110(), dtype=int)
    return df_bin, nulos_filtrados


def generar_metadatos_csv(ruta=None):
    """Documenta las 110 columnas binarias en formato CSV (sin usar JSON)."""
    if ruta is None:
        ruta = os.path.join(DATA_DIR, "metadatos_variables.csv")
    os.makedirs(os.path.dirname(ruta), exist_ok=True)

    filas = []
    bit = 1
    for orig, pref, div10, desc in VARIABLES:
        regla = "Dividir por 10 y mapear a [0,1)..[9,10]" if div10 else "Mapeo directo a [0,1)..[9,10]"
        for b in range(1, 11):
            rango = f"[{b - 1}, {b})" if b < 10 else "[9, 10]"
            filas.append({
                "Indice_Bit": bit,
                "Columna_Binaria": f"{pref}_v{b}",
                "Variable_Original": orig,
                "Bin": f"v{b}",
                "Intervalo": rango,
                "Regla_Escalamiento": regla,
                "Valores": "0 o 1",
                "Descripcion": desc,
            })
            bit += 1

    df_meta = pd.DataFrame(filas)
    df_meta.to_csv(ruta, index=False, encoding="utf-8")
    return df_meta


def _muestrear_por_carrera(df_raw, X, n_por_dataset=60):
    """Selecciona dos subconjuntos independientes y balanceados (A y B) desde career_data.csv."""
    carreras = sorted(df_raw["Career"].unique())
    cupo = n_por_dataset // len(carreras)
    idx_a, idx_b = [], []

    for carrera in carreras:
        indices_c = np.where(df_raw["Career"].to_numpy() == carrera)[0]
        Xc = X[indices_c]
        usados = set()
        pa, pb = [], []
        ref_a, ref_b = None, None
        sim = Xc @ Xc.T
        turno = 0

        while len(pa) < cupo or len(pb) < cupo:
            es_a = (turno % 2 == 0 and len(pa) < cupo) or len(pb) >= cupo
            destino = pa if es_a else pb
            ref = ref_a if es_a else ref_b
            mejor_score, mejor_grupo, mejor_p7 = (-1, -1), [], None

            for i in range(len(Xc)):
                if i in usados:
                    continue
                for j in np.where(sim[i] >= 7)[0]:
                    if j <= i or j in usados:
                        continue
                    bits = np.where((Xc[i] & Xc[j]) == 1)[0]
                    for comb in combinations(bits, 7):
                        p7 = np.zeros(110, dtype=int)
                        p7[list(comb)] = 1
                        grupo = [int(k) for k in np.where((Xc @ p7) == 7)[0] if int(k) not in usados]
                        solap = int(np.sum(p7 & ref)) if ref is not None else 0
                        if (len(grupo), solap) > mejor_score:
                            mejor_score = (len(grupo), solap)
                            mejor_grupo = grupo
                            mejor_p7 = p7

            if not mejor_grupo:
                restantes = [k for k in range(len(Xc)) if k not in usados]
                mejor_grupo = [restantes[0]]
                mejor_p7 = Xc[restantes[0]]

            if ref is None and mejor_p7 is not None:
                if es_a:
                    ref_a = mejor_p7
                else:
                    ref_b = mejor_p7

            for k in mejor_grupo[: cupo - len(destino)]:
                usados.add(k)
                destino.append(int(indices_c[k]))
            turno += 1

        idx_a.extend(pa[:cupo])
        idx_b.extend(pb[:cupo])

    return idx_a, idx_b


def generar_datasets_prueba(ruta_crudo=None, n_registros=60):
    """Genera data/dataset_A.csv, data/dataset_B.csv (>= 55 filas) y data/metadatos_variables.csv."""
    os.makedirs(DATA_DIR, exist_ok=True)
    if ruta_crudo is None:
        ruta_crudo = os.path.join(DATA_DIR, "career_data.csv")
        if not os.path.exists(ruta_crudo) and os.path.exists("career_data.csv"):
            ruta_crudo = "career_data.csv"

    ruta_meta = os.path.join(DATA_DIR, "metadatos_variables.csv")
    generar_metadatos_csv(ruta_meta)

    if not os.path.exists(ruta_crudo):
        raise FileNotFoundError(f"No se encontro el archivo base '{ruta_crudo}'.")

    df_raw = pd.read_csv(ruta_crudo)
    df_bin, _ = binarizar_dataframe(df_raw)

    idx_a, idx_b = _muestrear_por_carrera(df_raw, df_bin.to_numpy(dtype=int), n_registros)
    df_a = df_bin.iloc[idx_a].reset_index(drop=True)
    df_b = df_bin.iloc[idx_b].reset_index(drop=True)

    ruta_a = os.path.join(DATA_DIR, "dataset_A.csv")
    ruta_b = os.path.join(DATA_DIR, "dataset_B.csv")
    df_a.to_csv(ruta_a, index=False, encoding="utf-8")
    df_b.to_csv(ruta_b, index=False, encoding="utf-8")
    return ruta_a, ruta_b, ruta_meta


if __name__ == "__main__":
    a, b, m = generar_datasets_prueba()
    print(f"Archivos listos en /data: {a}, {b} y {m}.")

