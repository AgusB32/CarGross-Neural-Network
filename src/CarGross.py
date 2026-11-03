"""
src/CarGross.py - Sistema de Test Vocacional con Red Neuronal ART1 (Carpenter-Grossberg).

Trabajo Practico Final Integrador - Inteligencia Artificial (2026)
UADER - FCyT | Grupo 6: Benedetti V., Benedetti A., Frachia F., Nievas J., Saenz J.
Bibliografia base: Lippmann (1987) / Sanchez-Sinencio & Lau (1992), pp. 5-14 (Box 3).
"""

import argparse
import os
import sys

sys.dont_write_bytecode = True

import numpy as np
import pandas as pd

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SRC_DIR)
DATA_DIR = os.path.join(BASE_DIR, "data")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

N_BITS = 110  # 11 variables psicometricas RIASEC/aptitud x 10 bines discretos (v1..v10)

# Perfiles teoricos RIASEC (escala de bines 1..10 en las 11 variables) para validacion externa
PERFILES_RIASEC = {
    "Ingenieria de Software (Software Engineer)": np.array([8.36, 8.30, 5.47, 4.00, 5.50, 3.29, 8.52, 3.42, 3.48, 3.50, 7.03]),
    "Ciencia de Datos (Data Scientist)":          np.array([9.14, 8.32, 3.98, 3.99, 5.48, 3.44, 8.89, 3.52, 3.53, 3.39, 7.96]),
    "Medicina y Salud (Doctor)":                  np.array([8.28, 9.45, 4.01, 4.04, 3.98, 3.56, 7.98, 3.67, 7.97, 3.48, 3.35]),
    "Contaduria y Finanzas (Accountant)":         np.array([8.94, 8.37, 4.00, 4.03, 5.51, 3.50, 3.44, 3.59, 3.44, 3.55, 8.85]),
    "Emprendimiento y Gestion (Entrepreneur)":    np.array([8.27, 8.37, 3.92, 5.52, 3.95, 6.92, 3.48, 3.61, 3.46, 8.97, 3.64]),
    "Docencia y Educacion (Teacher)":             np.array([8.24, 8.28, 3.98, 5.48, 3.94, 3.44, 3.54, 7.10, 8.86, 3.58, 3.50]),
}
PESOS_VAR = np.array([1.5, 1.5, 4.0, 3.0, 3.0, 2.0, 2.5, 2.0, 2.5, 2.5, 2.5])

MAN_PAGE = """
================================================================================
CARGROSS(1)                  Manual de Usuario UNIX                  CARGROSS(1)
================================================================================
NOMBRE
    CarGross.py - Clasificador Vocacional No Supervisado basado en Red ART1

SINOPSIS
    python src/CarGross.py <archivo.csv> [--rho RHO] [--L L] [--epocas E] [-o SALIDA.csv]
    python src/CarGross.py --demo
    python src/CarGross.py --experimento
    python src/CarGross.py --preparar
    python src/CarGross.py --man

DESCRIPCION
    Implementa el algoritmo de Carpenter y Grossberg (Box 3, Lau pp. 12-14) para
    entradas binarias. La Capa de Comparacion F1 recibe vectores X de 110 bits
    (11 variables x 10 bines v1..v10) y la Capa de Reconocimiento F2 agrupa los
    perfiles dinamicamente segun el parametro de vigilancia rho en (0, 1].

REGLAS DE ENTRADA Y VALIDACION
    - Solo se admiten archivos .csv (prohibido el uso de .json).
    - Exactamente 110 columnas con valores binarios (0 y 1).
    - Los vectores nulos (todos 0, ||X|| = 0) se detectan y filtran automaticamente
      para no anular la estimulacion matematica en F1.

OPCIONES
    <archivo.csv> / -i  Ruta al dataset binario de entrada (ej. data/dataset_A.csv).
    -r, --rho FLOAT     Parametro de vigilancia en (0, 1] (por defecto: 0.6).
    --L FLOAT           Parametro de eleccion Bottom-Up L > 1 (por defecto: 2.0).
    -e, --epocas INT    Maximo de epocas hasta convergencia (por defecto: 50).
    -o, --output CSV    Exporta las asignaciones a un archivo CSV opcional.
    --demo              Ejecuta una prueba rapida verificando F1, F2, Reset y nulos.
    --experimento       Compara dataset_A.csv y dataset_B.csv con rho = 0.3, 0.6 y 0.85.
    --preparar          Regenera data/dataset_A.csv, data/dataset_B.csv y metadatos.
    --man, -h, --help   Muestra este manual de referencia.

EJEMPLOS
    python src/CarGross.py --demo
    python src/CarGross.py data/dataset_A.csv --rho 0.6
    python src/CarGross.py --experimento
================================================================================
""".strip()


def error_con_man(mensaje):
    """Informa el error detectado y despliega automaticamente el manual man."""
    print(f"\n[ERROR] {mensaje}\n", file=sys.stderr)
    print(MAN_PAGE)
    sys.exit(1)


class ParserConMan(argparse.ArgumentParser):
    """Intercepta errores de argumentos en consola y muestra el manual man."""

    def error(self, message):
        error_con_man(f"Argumentos invalidos: {message}")


class ART1:
    """
    Red Neuronal ART1 (Adaptive Resonance Theory 1 - Carpenter & Grossberg).
    Sigue los 7 pasos del algoritmo de Box 3 en Sanchez-Sinencio & Lau (1992, p. 12).
    """

    def __init__(self, n_features=N_BITS, rho=0.6, L=2.0, max_epochs=50):
        if n_features != N_BITS:
            raise ValueError(f"La capa F1 requiere exactamente {N_BITS} bits.")
        if not (0.0 < float(rho) <= 1.0):
            raise ValueError(f"El parametro rho debe estar en (0, 1]. Recibido: {rho}")
        if float(L) <= 1.0:
            raise ValueError(f"El parametro L debe ser > 1. Recibido: {L}")

        self.n = int(n_features)
        self.rho = float(rho)
        self.L = float(L)
        self.max_epochs = int(max_epochs)

        # Paso 1 (Inicializacion de pesos):
        # Top-Down T_ji(0) = 1.0 | Bottom-Up B_ij(0) = 1 / (1 + N)
        self.b0 = 1.0 / (1.0 + self.n)
        self.weights_td = np.empty((0, self.n), dtype=float)  # Matriz T_ji (M x N)
        self.weights_bu = np.empty((self.n, 0), dtype=float)  # Matriz B_ij (N x M)

    @property
    def n_categorias(self):
        return self.weights_td.shape[0]

    def entrenar_patron(self, x):
        """Procesa un vector binario X en F1 y devuelve el cluster J asignado en F2."""
        norma_x = float(np.sum(x))
        if norma_x == 0.0:
            return -1  # Vector nulo: se filtra para no dividir por cero en vigilancia

        inhibidas = set()
        while True:
            # 1. Activacion ascendente en F2: y_j = sum_i(B_ij * X_i)
            # Se incluye al final una neurona candidata no comprometida con peso b0
            n_cand = self.n_categorias + 1
            y = np.full(n_cand, -np.inf)
            if self.n_categorias > 0:
                y[: self.n_categorias] = x @ self.weights_bu
            y[self.n_categorias] = norma_x * self.b0

            for j_inh in inhibidas:
                y[j_inh] = -np.inf

            if np.all(np.isneginf(y)):
                return -1

            # 2. Seleccion competitiva (MAXNET): neurona ganadora J = argmax(y_j)
            J = int(np.argmax(y))

            # 3. Vector de expectativa en F1 (AND logico): S_i = X_i * T_Ji
            s = x * self.weights_td[J] if J < self.n_categorias else x.copy()

            # 4. Prueba de vigilancia: ||S|| / ||X|| >= rho (norma L1)
            if (float(np.sum(s)) / norma_x) + 1e-12 >= self.rho:
                # 5a. Resonancia: actualizar T_Ji y B_iJ (equivalente a Paso 7 Box 3 con L=2)
                t_nuevo = s.astype(float)
                b_nuevo = (self.L * t_nuevo) / (self.L - 1.0 + float(np.sum(t_nuevo)))

                if J < self.n_categorias:
                    self.weights_td[J] = t_nuevo
                    self.weights_bu[:, J] = b_nuevo
                else:
                    self.weights_td = np.vstack([self.weights_td, t_nuevo])
                    self.weights_bu = np.hstack([self.weights_bu, b_nuevo.reshape(-1, 1)])
                return J

            # 5b. Reset: no supera rho -> inhibir neurona J y buscar otra en F2
            inhibidas.add(J)

    def entrenar(self, X):
        """Entrena la red por epocas hasta que los clusters y pesos T_Ji se estabilicen."""
        asignaciones = np.full(len(X), -1, dtype=int)
        for epoca in range(1, self.max_epochs + 1):
            td_prev = self.weights_td.copy()
            cambios = 0
            for i, x in enumerate(X):
                cat = self.entrenar_patron(x)
                if cat != asignaciones[i]:
                    cambios += 1
                asignaciones[i] = cat

            estable = td_prev.shape == self.weights_td.shape and np.array_equal(td_prev, self.weights_td)
            if cambios == 0 and estable:
                break

        # Clasificacion final sobre los prototipos estabilizados
        finales, sim = self.predecir(X)
        mascara = finales == -1
        finales[mascara] = asignaciones[mascara]
        return finales, sim, epoca

    def predecir(self, X):
        """Evalua un lote de patrones X sin modificar los pesos aprendidos."""
        cats = np.full(len(X), -1, dtype=int)
        sims = np.zeros(len(X), dtype=float)
        if self.n_categorias == 0:
            return cats, sims

        for i, x in enumerate(X):
            norma_x = float(np.sum(x))
            if norma_x <= 0.0:
                continue
            orden = np.argsort(-(x @ self.weights_bu), kind="stable")
            for j in orden:
                s = x * self.weights_td[int(j)]
                cociente = float(np.sum(s)) / norma_x
                if cociente + 1e-12 >= self.rho:
                    cats[i], sims[i] = int(j), cociente
                    break
        return cats, sims


def resolver_ruta_csv(ruta):
    """Busca el archivo en la ruta indicada o dentro de la carpeta /data."""
    if os.path.isfile(ruta):
        return ruta
    candidato_data = os.path.join(DATA_DIR, os.path.basename(ruta))
    if os.path.isfile(candidato_data):
        return candidato_data
    return ruta


def cargar_csv_binario(ruta):
    """Valida formato .csv, dimension de 110 bits, valores {0, 1} y filtra vectores nulos."""
    ext = os.path.splitext(ruta)[1].lower()
    if ext == ".json":
        raise ValueError(f"Formato JSON prohibido ('{ruta}'). Use unicamente archivos .csv.")
    if ext != ".csv":
        raise ValueError(f"Extension invalida ('{ext}'). El archivo debe ser .csv.")

    ruta_efectiva = resolver_ruta_csv(ruta)
    if not os.path.isfile(ruta_efectiva):
        raise ValueError(f"El archivo '{ruta}' no existe (ni en ruta actual ni en /data).")

    df = pd.read_csv(ruta_efectiva)
    if df.empty or df.shape[1] != N_BITS:
        raise ValueError(f"'{ruta_efectiva}' tiene {df.shape[1]} columnas; se exigen exactamente {N_BITS} bits.")
    if df.isna().any().any():
        raise ValueError(f"'{ruta_efectiva}' contiene celdas vacias (NaN).")

    try:
        mat = df.to_numpy(dtype=float)
    except ValueError as exc:
        raise ValueError(f"'{ruta_efectiva}' contiene datos no numericos: {exc}") from exc

    if not np.all(np.isin(mat, [0.0, 1.0])):
        raise ValueError(f"'{ruta_efectiva}' contiene valores distintos de 0 y 1.")

    mat_int = mat.astype(int)
    sumas = mat_int.sum(axis=1)
    idx_nulos = [i + 1 for i, s in enumerate(sumas) if s == 0]
    idx_validos = [i + 1 for i, s in enumerate(sumas) if s > 0]

    if idx_nulos:
        print(f"  [AVISO] Se filtraron {len(idx_nulos)} vector(es) nulo(s) en '{ruta_efectiva}' (filas {idx_nulos}).")
    if not idx_validos:
        raise ValueError(f"Todos los registros de '{ruta_efectiva}' son vectores nulos (todos 0).")

    return mat_int[sumas > 0], list(df.columns), idx_validos, idx_nulos, len(mat_int)


def interpretar_cluster(t_j, columnas, sub_x):
    """Decodifica los bines activos del prototipo T_Ji y calcula su coherencia RIASEC."""
    activos = [columnas[i] for i in np.where(t_j >= 0.5)[0]]
    activos.sort(key=lambda c: int(c.split("_v")[-1]) if "_v" in c else 0, reverse=True)
    rasgos = f"T_Ji ({len(activos)} bits): " + ", ".join(activos) if activos else "Sin bits fijos"

    # Nivel promedio (1..10) por variable para contrastar con la matriz teorica RIASEC
    niveles = np.zeros(11)
    escala = np.arange(1, 11, dtype=float)
    for v in range(11):
        seg = t_j[v * 10 : (v + 1) * 10]
        if seg.sum() > 0:
            niveles[v] = np.dot(seg, escala) / seg.sum()
        else:
            seg_c = sub_x[:, v * 10 : (v + 1) * 10].mean(axis=0)
            niveles[v] = np.dot(seg_c, escala) / max(1e-9, seg_c.sum())

    dists = {
        carrera: float(np.sqrt(np.sum(PESOS_VAR * (niveles - proto) ** 2) / PESOS_VAR.sum()))
        for carrera, proto in PERFILES_RIASEC.items()
    }
    vocacion = min(dists, key=dists.get)
    coherencia = round(float(np.clip((1.0 - dists[vocacion] / 4.5) * 100.0, 50.0, 99.9)), 1)
    return rasgos, vocacion, coherencia


def evaluar_dataset(ruta_csv, rho, L=2.0, epocas=50, salida_csv=None):
    """Entrena ART1 sobre el CSV, imprime el reporte experimental y exporta si se pide."""
    X, cols, idx_val, idx_nul, total = cargar_csv_binario(ruta_csv)
    red = ART1(n_features=N_BITS, rho=rho, L=L, max_epochs=epocas)
    asig, sims, ep_usadas = red.entrenar(X)

    resumen = []
    for j in range(red.n_categorias):
        mask = asig == j
        cant = int(np.sum(mask))
        rasgos, voc, coh = interpretar_cluster(red.weights_td[j], cols, X[mask])
        resumen.append({
            "cat": j,
            "alumnos": cant,
            "porc": round(cant / len(X) * 100.0, 1),
            "bits": int(np.sum(red.weights_td[j] >= 0.5)),
            "coh": coh,
            "vocacion": voc,
            "rasgos": rasgos,
        })

    media_bits = float(np.mean([r["bits"] for r in resumen])) if resumen else 0.0
    coh_global = float(np.sum([r["coh"] * r["alumnos"] for r in resumen]) / len(X))

    print("\n" + "=" * 107)
    print(f"  REPORTE EXPERIMENTAL ART1 | Dataset: {os.path.basename(ruta_csv)} | Vigilancia rho = {rho:.2f}")
    print("=" * 107)
    print(f"  Registros leidos: {total} | Nulos filtrados: {len(idx_nul)} | Procesados: {len(X)} | Epocas: {ep_usadas}")
    print(f"  Categorias creadas en F2: {red.n_categorias} | Media ||T_J||_1: {media_bits:.1f}/11 bits | Coherencia RIASEC: {coh_global:.1f}%")
    print("-" * 107)
    print(f"  {'Cluster':<8} | {'Alumnos':>7} | {'% Total':>7} | {'Coher.':>7} | {'Orientacion Vocacional':<42} | {'Prototipo T_Ji'}")
    print("-" * 107)

    for r in sorted(resumen, key=lambda x: (-x["alumnos"], x["cat"]))[:20]:
        print(f"  Cat_{r['cat']:02d}   | {r['alumnos']:7d} | {r['porc']:6.1f}% | {r['coh']:6.1f}% | {r['vocacion']:<42} | {r['rasgos']}")
    if len(resumen) > 20:
        print(f"  ... ({len(resumen) - 20} clusters adicionales de 1 alumno) ...")
    print("=" * 107)

    if salida_csv:
        mapa_voc = {r["cat"]: r["vocacion"] for r in resumen}
        filas_out = [
            {
                "ID_Fila": idx_val[i],
                "Cluster_F2": f"Cat_{asig[i]:02d}",
                "Similitud_S_X": round(float(sims[i]), 4),
                "Orientacion_Vocacional": mapa_voc.get(int(asig[i]), ""),
            }
            for i in range(len(X))
        ]
        pd.DataFrame(filas_out).to_csv(salida_csv, index=False, encoding="utf-8")
        print(f"  [OK] Resultados exportados en: {salida_csv}")

    return {
        "dataset": os.path.basename(ruta_csv),
        "rho": rho,
        "procesados": len(X),
        "nulos": len(idx_nul),
        "categorias": red.n_categorias,
        "bits_td": round(media_bits, 2),
        "coherencia": round(coh_global, 1),
        "epocas": ep_usadas,
    }


def _asegurar_datos():
    ruta_a = os.path.join(DATA_DIR, "dataset_A.csv")
    ruta_b = os.path.join(DATA_DIR, "dataset_B.csv")
    if not (os.path.exists(ruta_a) and os.path.exists(ruta_b)):
        from preparar_datos import generar_datasets_prueba
        generar_datasets_prueba()
    return ruta_a, ruta_b


def modo_demo():
    """Prueba automatizada verificando F1, F2, vigilancia, Reset y filtrado de vector nulo."""
    print("\n" + "=" * 95)
    print("  MODO DEMO (--demo) - VERIFICACION DE LA RED ART1 (CARPENTER-GROSSBERG)")
    print("=" * 95)

    ruta_a, _ = _asegurar_datos()

    red = ART1(n_features=N_BITS, rho=0.6)
    p1 = np.zeros(N_BITS, dtype=int)
    p2 = np.zeros(N_BITS, dtype=int)
    p_nulo = np.zeros(N_BITS, dtype=int)
    p4 = np.zeros(N_BITS, dtype=int)

    for v in range(11):
        p1[v * 10 + 8] = 1
        p2[v * 10 + (8 if v < 9 else 7)] = 1  # Comparte 9/11 bits con p1 (0.82 >= 0.6)
        p4[v * 10 + 1] = 1                    # Opuesto (0/11 bits en comun -> Reset)

    print(f"  1. Patron 1 (Ingenieria A)  -> Cluster F2 asignado: {red.entrenar_patron(p1)} (Nuevo cluster)")
    print(f"  2. Patron 2 (Ingenieria B)  -> Similitud 9/11 >= 0.6 -> RESONANCIA en Cluster F2: {red.entrenar_patron(p2)}")
    print(f"  3. Patron 3 (Vector Nulo)   -> Filtrado sin estimular F1 (retorno: {red.entrenar_patron(p_nulo)})")
    print(f"  4. Patron 4 (Humanidades)   -> Similitud 0/11 < 0.6  -> RESET y nuevo Cluster F2: {red.entrenar_patron(p4)}")

    evaluar_dataset(ruta_a, rho=0.6)


def modo_experimento(salida_csv=None):
    """Compara data/dataset_A.csv y data/dataset_B.csv con rho = 0.3, 0.6 y 0.85."""
    ruta_a, ruta_b = _asegurar_datos()

    resultados = []
    for ds in (ruta_a, ruta_b):
        for r in (0.3, 0.6, 0.85):
            resultados.append(evaluar_dataset(ds, rho=r))

    print("\n" + "=" * 95)
    print("  RESUMEN COMPARATIVO EXPERIMENTAL (DATASET A vs DATASET B)")
    print("=" * 95)
    print(f"  {'Dataset':<14} | {'Rho':>5} | {'Proc.':>5} | {'Nulos':>5} | {'Clusters F2':>11} | {'Media ||T_J||':>13} | {'Coherencia':>10} | {'Epocas':>6}")
    print("-" * 95)
    for res in resultados:
        print(
            f"  {res['dataset']:<14} | {res['rho']:5.2f} | {res['procesados']:5d} | {res['nulos']:5d} | "
            f"{res['categorias']:11d} | {res['bits_td']:9.2f} bit | {res['coherencia']:9.1f}% | {res['epocas']:6d}"
        )
    print("=" * 95)

    if salida_csv:
        pd.DataFrame(resultados).to_csv(salida_csv, index=False, encoding="utf-8")


def main():
    try:
        if len(sys.argv) == 1:
            error_con_man("Faltan argumentos. Indique un archivo CSV o use --demo / --experimento.")

        parser = ParserConMan(prog="CarGross.py", add_help=False)
        parser.add_argument("dataset", nargs="?", default=None)
        parser.add_argument("-i", "--input", dest="input_csv", default=None)
        parser.add_argument("-r", "--rho", type=float, default=0.6)
        parser.add_argument("--L", type=float, default=2.0)
        parser.add_argument("-e", "--epocas", type=int, default=50)
        parser.add_argument("-o", "--output", dest="output_csv", default=None)
        parser.add_argument("--demo", action="store_true")
        parser.add_argument("--experimento", action="store_true")
        parser.add_argument("--preparar", action="store_true")
        parser.add_argument("--man", "-h", "--help", dest="man", action="store_true")

        args = parser.parse_args()

        if args.man:
            print(MAN_PAGE)
            return
        if args.preparar:
            from preparar_datos import generar_datasets_prueba
            a, b, m = generar_datasets_prueba()
            print(f"[OK] Generados en /data: {a}, {b} y {m}")
            return
        if args.demo:
            modo_demo()
            return
        if args.experimento:
            modo_experimento(salida_csv=args.output_csv)
            return

        ruta = args.dataset or args.input_csv
        if not ruta:
            error_con_man("Debe especificar un archivo CSV de entrada (ej. data/dataset_A.csv).")

        evaluar_dataset(ruta, rho=args.rho, L=args.L, epocas=args.epocas, salida_csv=args.output_csv)

    except Exception as exc:
        error_con_man(str(exc))


if __name__ == "__main__":
    main()

