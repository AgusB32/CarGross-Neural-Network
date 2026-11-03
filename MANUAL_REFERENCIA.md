# Manual de Referencia — CarGross (Red Neuronal ART1 de Carpenter-Grossberg)

Implementación en Python de una red neuronal **ART1 (*Adaptive Resonance Theory 1*)**, formulada por Stephen Grossberg y Gail Carpenter (1987) y especificada en el algoritmo de **Box 3** de Lippmann (1987) / Sánchez-Sinencio & Lau (1992, pp. 12–14), aplicada al agrupamiento no supervisado de perfiles vocacionales psicométricos bajo el modelo RIASEC.

- **Trabajo Práctico Final Integrador — Inteligencia Artificial (2026)**
- **Institución:** Universidad Autónoma de Entre Ríos (UADER) — Facultad de Ciencia y Tecnología (FCyT) — Museo IDTI Lab de Redes Neuronales
- **Docentes:** Dra. Daniela López De Luise & Alejandro Pérez
- **Grupo 6 — Integrantes:**
  - Benedetti, Valentino *(Responsable de Arquitectura)*
  - Benedetti, Agustín *(Responsable del Repositorio)*
  - Frachia, Franco *(Responsable del Test)*
  - Nievas, Juan José *(Líder / Documentación)*
  - Sáenz, Juan Ignacio *(Responsable del Código)*

---

## Tabla de Contenidos

1. [Alcances y Limitaciones](#1-alcances-y-limitaciones)
2. [Proceso de Instalación](#2-proceso-de-instalación)
3. [Estructura del Proyecto](#3-estructura-del-proyecto)
4. [Datasets de Uso y Archivo de Metadatos en CSV](#4-datasets-de-uso-y-archivo-de-metadatos-en-csv)
5. [Modo de Uso y Ejecución del Test Demo](#5-modo-de-uso-y-ejecución-del-test-demo)
6. [Salida Esperada](#6-salida-esperada)
7. [Parámetros de la Red](#7-parámetros-de-la-red)
8. [Preguntas Frecuentes (FAQ)](#8-preguntas-frecuentes-faq)
9. [REFERENCIAS](#referencias)

---

## 1. Alcances y Limitaciones

### 1.1 Alcances
- **Implementación fiel al algoritmo clásico ART1 (Box 3, Lau pp. 12–14):** Incluye inicialización de pesos *Bottom-Up* $B_{ij}(0) = \frac{1}{1 + N}$ y *Top-Down* $T_{ji}(0) = 1$, competencia lateral en la capa de reconocimiento $F_2$ (*winner-take-all* / MAXNET), cálculo del vector de intersección lógica $S = X \wedge T_J$ en la capa de comparación $F_1$, prueba de vigilancia $\frac{\|S\|_1}{\|X\|_1} \ge \rho$, inhibición selectiva (*Reset*) y adaptación resonante de pesos con asignación dinámica de neuronas.
- **Resolución del dilema de estabilidad-plasticidad:** Permite incorporar nuevos perfiles vocacionales creando categorías nuevas cuando ningún prototipo existente supera el umbral $\rho$, sin destruir ni olvidar las categorías previamente consolidadas.
- **Preprocesamiento y binarización automatizada (`src/preparar_datos.py`):** Escala variables numéricas continuas/ordinales a la escala $[0, 10]$ y las codifica en vectores binarios dispersos (*one-hot binning*) de **110 bits** (11 variables $\times$ 10 bines).
- **Documentación de metadatos en CSV puro:** Genera y utiliza `data/metadatos_variables.csv` para documentar el significado psicométrico, rango y regla de escalamiento de cada uno de los 110 bits, sin emplear archivos JSON.
- **Validación técnica y psicopedagógica integrada:** Reporta tanto métricas estructurales internas (cantidad de clústeres en $F_2$, épocas hasta convergencia y media de bits activos $\|T_J\|_1$) como validación externa de coherencia frente a la matriz teórica de perfiles RIASEC.
- **Manejo integral de errores y manual `man` automático:** Ante cualquier corrida infructuosa (argumentos faltantes o inválidos, archivos inexistentes, formatos prohibidos como `.json`, dimensiones incorrectas, valores no binarios o vectores nulos), el módulo informa la causa exacta y despliega automáticamente el manual de usuario estilo UNIX (`MAN_PAGE`).

### 1.2 Limitaciones
- **Exclusividad para entradas binarias ($\{0, 1\}$):** Por definición arquitectónica, ART1 procesa únicamente vectores booleanos. El procesamiento directo de variables continuas sin discretización previa requeriría arquitecturas como ART2 o Fuzzy ART.
- **Erosión monotónica de prototipos por intersección (`AND`):** Dado que la regla de aprendizaje $T_{Ji}(t+1) = T_{Ji}(t) \wedge X_i$ solo puede conservar o apagar bits en `1`, una vigilancia $\rho$ excesivamente baja frente a datos muy heterogéneos reduce progresivamente la cantidad de bits activos del ejemplar (*exemplar degradation*, Lau pp. 13–14).
- **Sensibilidad al orden de presentación:** Al tratarse de un algoritmo secuencial en línea (*online leader clustering*, Lau p. 12), alterar el orden de las filas de entrada puede variar ligeramente el índice o la cantidad total de categorías formadas, manteniendo la coherencia global del agrupamiento.
- **Interfaz exclusivamente por línea de comandos (CLI):** Conforme a los requerimientos de la cátedra, el sistema opera íntegramente por consola estándar sin interfaz gráfica de usuario (GUI).

---

## 2. Proceso de Instalación

### 2.1 Requisitos del Sistema
- **Sistema Operativo:** Windows, Linux o macOS.
- **Intérprete:** Python 3.8 o superior.
- **Gestor de paquetes:** `pip`.
- **Dependencias (`requirements.txt`):** `numpy>=1.21.0` y `pandas>=1.3.0`.

### 2.2 Pasos de Instalación

1. Abrir una terminal o consola de comandos y ubicarse en el directorio raíz del proyecto (`RedNeuronalNew`):
   ```bash
   cd RedNeuronalNew
   ```

2. Instalar las dependencias requeridas mediante `pip`:
   ```bash
   pip install -r requirements.txt
   ```

3. Verificar que la instalación sea correcta consultando el manual integrado del módulo:
   ```bash
   python src/CarGross.py --man
   ```

---

## 3. Estructura del Proyecto

```text
RedNeuronalNew/
├── src/
│   ├── CarGross.py                     # Módulo principal de la red neuronal ART1 y CLI
│   └── preparar_datos.py               # Pipeline de binarización (110 bits) y partición A/B
├── notebooks/
│   └── simulacion_calibracion_rho.py   # Barrido experimental de calibración de vigilancia (rho)
├── data/
│   ├── career_data.csv                 # Dataset psicométrico original (Kaggle RIASEC, 2400 filas)
│   ├── dataset_A.csv                   # Dataset de uso A binarizado (60 registros x 110 bits)
│   ├── dataset_B.csv                   # Dataset de uso B binarizado (60 registros x 110 bits)
│   └── metadatos_variables.csv         # Diccionario de metadatos de las 110 columnas en CSV
├── MANUAL_REFERENCIA.md                # Manual de referencia técnica y de usuario
├── INFORME.md                          # Informe experimental comparativo con ambos datasets
└── requirements.txt                    # Dependencias de Python (numpy, pandas)
```

---

## 4. Datasets de Uso y Archivo de Metadatos en CSV

### 4.1 Variables Originales y Binarización a 110 Bits
A partir del corpus psicométrico vocacional `data/career_data.csv` (11 variables de entrada), cada variable se escala al intervalo $[0, 10]$ y se codifica en **10 bines discretos** (`v1` a `v10`), generando vectores binarios de **110 columnas** donde cada registro válido posee exactamente $\|X\|_1 = 11$ bits activos en `1`:

| # | Variable Original | Prefijo Binario | Rango Original | Transformación Previa | Columnas Binarias (10 bines) |
|:-:|:------------------|:----------------|:--------------:|:----------------------|:-----------------------------|
| 1 | `Math_Score` | `Math` | 60 – 100 | División por 10 $\rightarrow [6.0, 10.0]$ | `Math_v1` .. `Math_v10` |
| 2 | `Science_Score` | `Science` | 60 – 100 | División por 10 $\rightarrow [6.0, 10.0]$ | `Science_v1` .. `Science_v10` |
| 3 | `Programming_Skill` | `Programming` | 1 – 5 | Mapeo directo a bin discreto | `Programming_v1` .. `Programming_v10` |
| 4 | `Communication_Skill` | `Communication` | 1 – 5 | Mapeo directo a bin discreto | `Communication_v1` .. `Communication_v10` |
| 5 | `Logical_Ability` | `Logical` | 1 – 5 | Mapeo directo a bin discreto | `Logical_v1` .. `Logical_v10` |
| 6 | `R_score` (Realista) | `R` | 0 – 10 | Mapeo directo a bin discreto | `R_v1` .. `R_v10` |
| 7 | `I_score` (Investigador) | `I` | 0 – 10 | Mapeo directo a bin discreto | `I_v1` .. `I_v10` |
| 8 | `A_score` (Artístico) | `A` | 0 – 10 | Mapeo directo a bin discreto | `A_v1` .. `A_v10` |
| 9 | `S_score` (Social) | `S` | 0 – 10 | Mapeo directo a bin discreto | `S_v1` .. `S_v10` |
| 10 | `E_score` (Emprendedor) | `E` | 0 – 10 | Mapeo directo a bin discreto | `E_v1` .. `E_v10` |
| 11 | `C_score` (Convencional) | `C` | 0 – 10 | Mapeo directo a bin discreto | `C_v1` .. `C_v10` |

### 4.2 Datasets Entregados (`dataset_A.csv` y `dataset_B.csv`)
Para cumplir con la exigencia de al menos 2 datasets de uso de al menos 50 registros y 5 variables:
- **`data/dataset_A.csv`:** **60 registros** independientes y balanceados (10 perfiles por cada una de las 6 orientaciones vocacionales) $\times$ **110 columnas binarias** (representando las **11 variables** psicométricas).
- **`data/dataset_B.csv`:** **60 registros** independientes y disjuntos respecto del Dataset A (10 perfiles por cada orientación vocacional) $\times$ **110 columnas binarias**.
- Ninguno de los dos archivos incluye la columna de etiqueta `Career`, garantizando que el entrenamiento de `CarGross.py` sea estrictamente no supervisado.

### 4.3 Archivo de Metadatos (`data/metadatos_variables.csv`)
El archivo `data/metadatos_variables.csv` documenta en formato CSV estándar las 110 posiciones del vector de entrada mediante las siguientes columnas:
- `Indice_Bit`: Posición entera del bit (1 a 110).
- `Columna_Binaria`: Identificador de columna en los datasets (`Math_v1` .. `C_v10`).
- `Variable_Original`: Nombre del atributo psicométrico de origen.
- `Bin`: Nivel discreto asignado (`v1` a `v10`).
- `Intervalo`: Rango numérico correspondiente (`[0, 1)` hasta `[9, 10]`).
- `Regla_Escalamiento`: Procedimiento aplicado antes de la binarización.
- `Valores`: Dominio permitido (`0 o 1`).
- `Descripcion`: Interpretación psicopedagógica de la variable.

---

## 5. Modo de Uso y Ejecución del Test Demo

### 5.1 Ejecución del Test Demo Rápido (`--demo`)
Verifica de forma automática el funcionamiento de las capas $F_1$ y $F_2$, la condición de resonancia, el disparo del mecanismo *Reset*, el filtrado de un vector nulo (`||X|| = 0`) y ejecuta el agrupamiento completo sobre `data/dataset_A.csv` con $\rho = 0.60$:

```bash
python src/CarGross.py --demo
```

### 5.2 Ejecución Individual sobre los Datasets de Uso (`dataset_A.csv` y `dataset_B.csv`)
Para correr el módulo sobre cada dataset indicando el parámetro de vigilancia $\rho$ y exportando opcionalmente las asignaciones a un archivo CSV:

```bash
# Corrida con Dataset A (rho = 0.60)
python src/CarGross.py data/dataset_A.csv --rho 0.60

# Corrida con Dataset B (rho = 0.60) exportando resultados a CSV
python src/CarGross.py data/dataset_B.csv --rho 0.60 -o data/resultados_B.csv
```

### 5.3 Ejecución del Experimento Comparativo Multi-Régimen (`--experimento`)
Evalúa automáticamente ambos datasets (`dataset_A.csv` y `dataset_B.csv`) bajo tres regímenes de vigilancia ($\rho \in \{0.30, 0.60, 0.85\}$) y presenta un cuadro comparativo consolidado:

```bash
python src/CarGross.py --experimento
```

### 5.4 Curva de Calibración del Parámetro de Vigilancia $\rho$
Para ejecutar el estudio completo de sensibilidad ($\rho \in \{0.20, 0.30, 0.45, 0.60, 0.75, 0.85, 0.95\}$) en ambos datasets:

```bash
python notebooks/simulacion_calibracion_rho.py
```

### 5.5 Regeneración de Datasets y Metadatos (`--preparar`)
Si se desea regenerar `data/dataset_A.csv`, `data/dataset_B.csv` y `data/metadatos_variables.csv` desde `data/career_data.csv`:

```bash
python src/CarGross.py --preparar
```

---

## 6. Salida Esperada

### 6.1 Salida Esperada del Test Demo (`python src/CarGross.py --demo`)

```text
===============================================================================================
  MODO DEMO (--demo) - VERIFICACION DE LA RED ART1 (CARPENTER-GROSSBERG)
===============================================================================================
  1. Patron 1 (Ingenieria A)  -> Cluster F2 asignado: 0 (Nuevo cluster)
  2. Patron 2 (Ingenieria B)  -> Similitud 9/11 >= 0.6 -> RESONANCIA en Cluster F2: 0
  3. Patron 3 (Vector Nulo)   -> Filtrado sin estimular F1 (retorno: -1)
  4. Patron 4 (Humanidades)   -> Similitud 0/11 < 0.6  -> RESET y nuevo Cluster F2: 1

=================================================================================================================
  REPORTE EXPERIMENTAL ART1 | Dataset: dataset_A.csv | Vigilancia rho = 0.60
=================================================================================================================
  Registros leidos: 60 | Nulos filtrados: 0 | Procesados: 60 | Epocas: 2
  Categorias creadas en F2: 19 | Media ||T_J||_1: 7.3/11 bits | Coherencia RIASEC: 83.2%
-----------------------------------------------------------------------------------------------------------------
  Cluster  | Alumnos | % Total |  Coher. | Orientacion Vocacional (Subperfil)               | Prototipo T_Ji
-----------------------------------------------------------------------------------------------------------------
  Cat_00   |       4 |    6.7% |   85.2% | Contaduria y Finanzas (Logico - Practico)        | T_Ji (7 bits): Math_v9, Logical_v6, R_v5, A_v4, S_v4, Communication_v3, E_v3
  Cat_03   |       4 |    6.7% |   83.3% | Ciencia de Datos (Matematico - Programacion)     | T_Ji (7 bits): Math_v10, I_v9, Science_v7, Programming_v5, Logical_v5, S_v4, Communication_v3
  Cat_04   |       4 |    6.7% |   83.0% | Ciencia de Datos (Matematico - Estructurado)     | T_Ji (7 bits): Math_v10, Science_v9, I_v9, C_v9, Logical_v6, Programming_v5, Communication_v5
  Cat_06   |       4 |    6.7% |   85.6% | Medicina y Salud (Investigador - Matematico)     | T_Ji (8 bits): Math_v9, Science_v9, I_v8, S_v7, Logical_v5, R_v4, A_v4, Communication_v3
  Cat_13   |       4 |    6.7% |   83.4% | Ingenieria de Software (Logico - Programacion)   | T_Ji (7 bits): I_v8, Logical_v6, Programming_v5, Communication_v4, S_v4, A_v2, E_v2
  Cat_16   |       4 |    6.7% |   83.9% | Docencia y Educacion (Cientifico - Comunicativo) | T_Ji (7 bits): Science_v10, S_v9, A_v7, Communication_v6, Programming_v3, Logical_v3, C_v3
  ...
=================================================================================================================
```

### 6.2 Salida Esperada del Resumen Comparativo (`python src/CarGross.py --experimento`)

```text
===============================================================================================
  RESUMEN COMPARATIVO EXPERIMENTAL (DATASET A vs DATASET B)
===============================================================================================
  Dataset        |   Rho | Proc. | Nulos | Clusters F2 | Media ||T_J|| | Coherencia | Epocas
-----------------------------------------------------------------------------------------------
  dataset_A.csv  |  0.30 |    60 |     0 |          14 |      4.64 bit |      82.8% |      3
  dataset_A.csv  |  0.60 |    60 |     0 |          19 |      7.32 bit |      83.2% |      2
  dataset_A.csv  |  0.85 |    60 |     0 |          59 |     10.98 bit |      78.8% |      2
  dataset_B.csv  |  0.30 |    60 |     0 |          16 |      5.69 bit |      83.3% |      2
  dataset_B.csv  |  0.60 |    60 |     0 |          22 |      7.82 bit |      82.7% |      2
  dataset_B.csv  |  0.85 |    60 |     0 |          58 |     10.97 bit |      78.7% |      2
===============================================================================================
```

---

## 7. Parámetros de la Red

| Parámetro | Flag CLI | Rango / Dominio | Valor por Defecto | Función en la Arquitectura ART1 |
|:----------|:---------|:---------------:|:-----------------:|:--------------------------------|
| **Vigilancia ($\rho$)** | `-r`, `--rho` | $(0.0, 1.0]$ | `0.60` | Umbral del mecanismo de *Reset*. Exige que la fracción de rasgos conservados $\frac{\|X \wedge T_J\|_1}{\|X\|_1}$ sea al menos $\rho$ para aceptar resonancia. |
| **Elección ($L$)** | `--L` | $L > 1.0$ | `2.0` | Constante de actualización de pesos *Bottom-Up* $B_{iJ} = \frac{L \cdot T_{Ji}}{L - 1 + \|T_J\|_1}$ (equivalente al paso 7 del Box 3 con $L = 2$). |
| **Épocas máximas** | `-e`, `--epocas` | $\mathbb{Z}^+$ | `50` | Límite de iteraciones completas sobre el dataset hasta alcanzar estabilidad de asignaciones y pesos $T_{Ji}$. |

---

## 8. Preguntas Frecuentes (FAQ)

### 1. ¿Cómo elegir el valor adecuado del parámetro de vigilancia ($\rho$)?
El parámetro $\rho$ controla el compromiso entre generalización y especificidad:
- **Valores bajos ($\rho \le 0.30$):** Provocan **sub-clasificación**, agrupando perfiles vocacionales distintos en pocas categorías amplias y erosionando los prototipos $T_{Ji}$ a ~3–5 bits activos.
- **Valores altos ($\rho \ge 0.85$):** Provocan **sobre-clasificación**, exigiendo coincidencia en 10 u 11 de los 11 rasgos activos y generando casi un clúster individual por alumno (58–59 categorías sobre 60 registros).
- **Punto óptimo ($\rho = 0.60$):** Exige al menos $\lceil 0.60 \times 11 \rceil = 7$ rasgos compartidos de 11, logrando entre 19 y 22 clústeres estables en 2 épocas con prototipos ricos en información (~7.3–7.8 bits activos).

### 2. ¿Qué ocurre si se ingresa un archivo con formato incorrecto, valores no binarios o sin argumentos?
El módulo `CarGross.py` intercepta cualquier corrida infructuosa mediante `ParserConMan` y `error_con_man`: imprime en `stderr` el diagnóstico específico del problema (por ejemplo, rechazo de extensión `.json`, presencia de valores distintos de `0` y `1`, o falta de ruta de entrada) e inmediatamente despliega en consola el manual completo estilo UNIX (`--man`), finalizando con código de salida `1`.

### 3. ¿Cómo trata la red los vectores nulos (filas compuestas íntegramente por ceros)?
Un vector nulo ($\|X\|_1 = 0$) anula la activación ascendente en la capa $F_1$ e indeterminaría el cociente de vigilancia $\frac{\|S\|_1}{\|X\|_1} = \frac{0}{0}$. Por ello, `cargar_csv_binario()` detecta y filtra automáticamente cualquier fila nula antes del entrenamiento (emitiendo un aviso con los números de fila descartados), y `entrenar_patron()` retorna `-1` de forma segura si recibe un vector nulo en forma directa.

### 4. ¿Cómo se interpretan los prototipos aprendidos $T_{Ji}$ en cada clúster?
Cada vector de pesos *Top-Down* $T_{Ji}$ almacena la intersección lógica (`AND`) de todos los alumnos que resonaron en la categoría $J$. Por ejemplo, si un clúster presenta activos los bits `Math_v10, I_v9, Science_v7, Programming_v5`, significa que todos los estudiantes agrupados en esa neurona comparten simultáneamente rendimiento sobresaliente en matemática (`[9, 10]`), perfil Investigador muy alto (`[8, 9)`), base científica alta (`[6, 7)`) y destreza en programación.

### 5. ¿Por qué no se incluye la columna `Career` dentro de `dataset_A.csv` y `dataset_B.csv`?
De acuerdo con la taxonomía de Lippmann (1987, pp. 7–8, Fig. 3), el clasificador de Carpenter-Grossberg es una red de **aprendizaje estrictamente no supervisado** análoga al algoritmo *Leader Clustering*: durante el entrenamiento jamás debe recibir etiquetas de clase. La validación psicopedagógica se realiza *a posteriori*, contrastando los prototipos emergentes $T_{Ji}$ contra la matriz teórica `PERFILES_RIASEC`.

---

## REFERENCIAS

[1] Sánchez-Sinencio, E., & Lau, C. (1992). *Artificial Neural Networks: Paradigms, Applications, and Hardware Implementations*. IEEE Press. *(Material de cátedra provisto como `Lau.pp5.a.11.pdf` y `Lau.pp12.a.14.pdf`)*.

[2] Lippmann, R. P. (1987). "An introduction to computing with neural nets." *IEEE ASSP Magazine*, 4(2), 4–22. *(Reimpreso en [1], incluyendo Box 3: Carpenter/Grossberg Net Algorithm, pp. 12–14)*.

[3] Carpenter, G. A., & Grossberg, S. (1987). "A massively parallel architecture for a self-organizing neural pattern recognition machine." *Computer Vision, Graphics, and Image Processing*, 37(1), 54–115.

[4] Holland, J. L. (1997). *Making Vocational Choices: A Theory of Vocational Personalities and Work Environments* (3rd ed.). Psychological Assessment Resources.

[5] Venkateshkumar, S. (2023). *Student Career Prediction Using RIASEC Dataset*. Kaggle Datasets. Disponible en: https://www.kaggle.com/datasets/svenkateshkumar/student-career-prediction-using-riasec-dataset

