# CarGross — Red Neuronal ART1 (Carpenter-Grossberg) para Orientación Vocacional

Este proyecto implementa desde cero en Python una red neuronal **ART1 (*Adaptive Resonance Theory 1*)**, creada por Stephen Grossberg y Gail Carpenter, aplicada a la **orientación vocacional de estudiantes** según el test psicométrico **RIASEC**.

- **Materia:** Inteligencia Artificial — Trabajo Práctico Final Integrador (2026)
- **Institución:** Universidad Autónoma de Entre Ríos (UADER) — Facultad de Ciencia y Tecnología (FCyT) — Museo IDTI Lab de Redes Neuronales
- **Docentes:** Dra. Daniela López De Luise & Alejandro Pérez
- **Grupo 6 — Integrantes:**
  - Benedetti, Valentino *(Responsable de Arquitectura)*
  - Benedetti, Agustín *(Responsable del Repositorio)*
  - Frachia, Franco *(Responsable del Test)*
  - Nievas, Juan José *(Líder / Documentación)*
  - Saenz, Juan Ignacio *(Responsable del Código)*

---

## Índice

1. [¿Cómo funciona la red?](#1-cómo-funciona-la-red)
2. [Alcances y Limitaciones](#2-alcances-y-limitaciones-manual-de-referencia)
3. [Proceso de Instalación](#3-proceso-de-instalación)
4. [Estructura del Proyecto y Datasets en CSV](#4-estructura-del-proyecto-y-datasets-en-csv)
5. [Modo de Uso y Cómo Correr el Test Demo](#5-modo-de-uso-y-cómo-correr-el-test-demo)
6. [Resultados Obtenidos en los Datasets e Interpretación](#6-resultados-obtenidos-en-los-datasets-e-interpretación)
7. [Preguntas Frecuentes (FAQ)](#7-preguntas-frecuentes-faq)
8. [REFERENCIAS](#referencias)

---

## 1. ¿Cómo funciona la red?

El objetivo del programa (`src/CarGross.py`) es **agrupar alumnos con perfiles vocacionales parecidos sin usar etiquetas previas** (aprendizaje no supervisado). La red no sabe de antemano qué carrera eligió cada estudiante: solo mira sus puntajes en 11 áreas (Matemática, Ciencias, Programación, Comunicación, Lógica y los 6 ejes vocacionales RIASEC) y descubre los grupos por sí sola.

Para que la red ART1 pueda leer estos datos, cada alumno se representa como una fila de **110 bits** (`0` y `1`):
- Hay **11 variables** en total, y cada una se divide en **10 niveles** (`v1` al `v10`, donde `v1` es puntaje muy bajo y `v10` es puntaje sobresaliente).
- Así, cada estudiante tiene exactamente **11 bits encendidos en `1`** (uno por cada variable) y 99 bits en `0`.

Cuando un alumno entra a la red, pasan tres cosas:
1. **Búsqueda de grupo ($F_1 \rightarrow F_2$):** La red compara el vector del alumno con los grupos (clústeres) que ya conoce y elige el más parecido.
2. **Prueba de vigilancia ($\rho$):** La red se fija cuántos rasgos en común (`AND` lógico) tiene el alumno con el molde o **prototipo $T_{Ji}$** de ese grupo. Si la proporción de coincidencias alcanza o supera el parámetro de vigilancia `rho` (por ejemplo, `0.60`, que exige coincidir en al menos 7 de las 11 variables), hay **resonancia** y el alumno entra al grupo.
3. **Aprendizaje o creación de un grupo nuevo (*Reset*):**
   - Si el alumno **entra al grupo**, el prototipo $T_{Ji}$ se actualiza conservando únicamente los bits en `1` donde coinciden todos los miembros del clúster.
   - Si **no alcanza** el mínimo de parecido que pide `rho`, se activa el mecanismo de **Reset**: la red descarta ese grupo, prueba con el siguiente y, si ninguno sirve, **crea un clúster nuevo** usando a ese alumno como molde inicial.

---

## 2. Alcances y Limitaciones (Manual de Referencia)

### 2.1 Alcances (¿Qué hace el sistema?)
- **Agrupamiento automático sin supervisión:** Crea la cantidad necesaria de clústeres de forma dinámica según qué tan exigente sea el parámetro de vigilancia `rho`, sin que haya que definir un número fijo de grupos de antemano (a diferencia de *K-Means*).
- **Aprende sin borrar lo anterior:** Si aparece un perfil vocacional completamente nuevo, la red abre una categoría nueva sin destruir ni olvidar los grupos que ya había formado.
- **Binarización y partición de datos incluidas (`src/preparar_datos.py`):** Toma el dataset original, lleva las 11 variables a una escala del 1 al 10, las transforma en vectores de 110 bits y genera dos datasets independientes listos para usar (`dataset_A.csv` y `dataset_B.csv`).
- **Interpretación de cada grupo (Prototipos $T_{Ji}$):** Una vez que la red termina de agrupar, el programa analiza el prototipo $T_{Ji}$ de cada clúster y muestra qué rasgos exactos tienen en común el 100% de los alumnos de ese grupo.
- **Validación de errores y ayuda automática (`--man`):** Si se ejecuta el programa sin parámetros, con un archivo que no existe, con formato prohibido (como `.json`), con valores que no sean `0` o `1`, o con filas llenas de ceros (vectores nulos), el programa avisa cuál fue el problema y muestra automáticamente en pantalla el manual de usuario completo.

### 2.2 Limitaciones (¿Qué límites tiene?)
- **Solo acepta entradas binarias (`0` y `1`):** El modelo ART1 original trabaja únicamente con unos y ceros. Por eso cualquier puntaje numérico (como una nota de 0 a 100) debe pasar primero por el paso de binarización por niveles (`v1` a `v10`).
- **Si `rho` es muy bajo, los prototipos pierden detalle:** Como el prototipo $T_{Ji}$ guarda la intersección (`AND`) de todos los alumnos del grupo, si se pone un `rho` demasiado bajo (por ejemplo `0.20`), entran al mismo grupo alumnos muy distintos entre sí y se van apagando bits en `T_Ji`, quedando pocos rasgos en común.
- **El orden de las filas influye levemente:** Como la red lee a los estudiantes uno por uno en orden secuencial, cambiar el orden de las filas del CSV puede cambiar ligeramente qué alumno inaugura cada grupo, aunque la consistencia general del agrupamiento se mantiene estable.
- **Uso exclusivo por consola (CLI):** Siguiendo las pautas de la cátedra, todas las pruebas, reportes y ayudas se ejecutan por terminal, sin interfaz gráfica.

---

## 3. Proceso de Instalación

### 3.1 Requisitos previos
- **Sistema Operativo:** Windows, Linux o macOS.
- **Python:** Versión 3.8 o superior instalada.
- **Librerías necesarias:** `numpy` y `pandas` (incluidas en `requirements.txt`).

### 3.2 Pasos de instalación

1. Abrir una terminal y entrar en la carpeta del proyecto:
   ```bash
   cd RedNeuronalNew
   ```

2. Instalar las dependencias con `pip`:
   ```bash
   pip install -r requirements.txt
   ```

3. Comprobar que todo funcione abriendo el manual integrado en consola:
   ```bash
   python src/CarGross.py --man
   ```

---

## 4. Estructura del Proyecto y Datasets en CSV

### 4.1 Organización de carpetas
```text
RedNeuronalNew/
├── src/
│   ├── CarGross.py                     # Código principal de la red ART1 y comandos de consola
│   └── preparar_datos.py               # Script que convierte los puntajes a 110 bits y crea los CSV
├── data/
│   ├── career_data.csv                 # Dataset vocacional original (2400 estudiantes)
│   ├── dataset_A.csv                   # Dataset de uso A binarizado (60 alumnos x 110 bits)
│   ├── dataset_B.csv                   # Dataset de uso B binarizado (60 alumnos x 110 bits)
│   └── metadatos_variables.csv         # Explicación de cada una de las 110 columnas en formato CSV
├── README.md                           # Este manual de referencia y guía de uso
└── requirements.txt                    # Librerías requeridas (numpy, pandas)
```

### 4.2 ¿Qué contienen `dataset_A.csv`, `dataset_B.csv` y `metadatos_variables.csv`?
Todos los datos y metadatos del proyecto están guardados en **formato CSV puro** (sin usar archivos JSON):

- **`data/dataset_A.csv` y `data/dataset_B.csv`:** Son los dos datasets de prueba. Cada uno tiene **60 alumnos distintos** (10 por cada una de las 6 orientaciones vocacionales) y **110 columnas binarias** que representan las **11 variables** del test. Ninguno de los dos incluye la columna con el nombre de la carrera (`Career`), para asegurar que la red agrupe totalmente a ciegas.
- **`data/metadatos_variables.csv`:** Es una tabla en CSV que explica columna por columna qué significa cada uno de los 110 bits (`Math_v1` hasta `C_v10`), de qué variable original sale, qué rango de puntaje representa y cómo se escaló:

| Variable Original | Prefijo en el CSV | Escala Original | Cómo se transforma a 10 niveles (`v1` .. `v10`) |
|:------------------|:-----------------:|:---------------:|:------------------------------------------------|
| `Math_Score` | `Math` | 60 a 100 | Se divide por 10 (queda entre 6.0 y 10.0) |
| `Science_Score` | `Science` | 60 a 100 | Se divide por 10 (queda entre 6.0 y 10.0) |
| `Programming_Skill` | `Programming` | 1 a 5 | Pasa directo al nivel `v1`..`v5` |
| `Communication_Skill` | `Communication` | 1 a 5 | Pasa directo al nivel `v1`..`v5` |
| `Logical_Ability` | `Logical` | 1 a 5 | Pasa directo al nivel `v1`..`v5` |
| `R_score` (Realista) | `R` | 0 a 10 | Pasa directo al nivel `v1`..`v10` |
| `I_score` (Investigador) | `I` | 0 a 10 | Pasa directo al nivel `v1`..`v10` |
| `A_score` (Artístico) | `A` | 0 a 10 | Pasa directo al nivel `v1`..`v10` |
| `S_score` (Social) | `S` | 0 a 10 | Pasa directo al nivel `v1`..`v10` |
| `E_score` (Emprendedor) | `E` | 0 a 10 | Pasa directo al nivel `v1`..`v10` |
| `C_score` (Convencional) | `C` | 0 a 10 | Pasa directo al nivel `v1`..`v10` |

---

## 5. Modo de Uso y Cómo Correr el Test Demo

### 1. Ejecución individual sobre un archivo
```bash
python src/CarGross.py <archivo.csv> [--rho RHO]
```
Es el modo principal para clasificar un dataset puntual con los parámetros deseados:
- **`<archivo.csv>`:** La ruta al CSV binario de 110 bits (ej. `data/dataset_A.csv` o `data/dataset_B.csv`).
- **`--rho RHO` (o `-r`):** El nivel de vigilancia $\rho \in (0, 1]$ (por defecto `0.60`). Controla qué tan estricta es la red para admitir alumnos en un mismo clúster.
- **`--L L`:** Parámetro de elección *Bottom-Up* (por defecto `2.0`, el valor canónico de Lippmann/Lau para $B_{ij}$).
- **`--epocas E` (o `-e`):** Cantidad máxima de pasadas sobre los datos hasta converger (por defecto `50`, aunque suele estabilizar en 2 o 3 épocas).
- **`-o SALIDA.csv` (opcional):** Guarda en un nuevo archivo CSV la lista de cada alumno con su clúster asignado y la similitud $\frac{\|S\|_1}{\|X\|_1}$.

### 2. Modo demostración
```bash
python src/CarGross.py --demo
```
Una prueba rápida automatizada (ideal para verificar el funcionamiento completo en segundos):
- Primero corre el test unitario con los 4 alumnos sintéticos (creación de clúster, resonancia, filtrado de vector nulo y *Reset*).
- Luego corre automáticamente la clasificación de `data/dataset_A.csv` con $\rho = 0.60$ mostrando el informe en pantalla.

### 3. Modo experimento comparativo
```bash
python src/CarGross.py --experimento
```
Ejecuta la batería de pruebas oficial exigida por la Sección 3 del TP:
- Evalúa los dos datasets (`dataset_A.csv` y `dataset_B.csv`) en los 3 regímenes de vigilancia:
  - $\rho = 0.30$ (Baja vigilancia — sub-clasificación / clústeres amplios).
  - $\rho = 0.60$ (Vigilancia intermedia — equilibrio óptimo).
  - $\rho = 0.85$ (Alta vigilancia — sobre-clasificación / micro-clústeres).
- Al final imprime la tabla comparativa global consolidando cantidad de clústeres, promedio de bits en los prototipos $T_{Ji}$.

### 4. Modo preparación de datos
```bash
python src/CarGross.py --preparar
```
Ejecuta el pipeline de preprocesamiento (`src/preparar_datos.py`):
- Lee el dataset original `data/career_data.csv`.
- Aplica los 2 pasos de preprocesamiento: divide las notas de 1 a 100 por 10 y mapea cada variable a sus 10 bines discretos ($11 \times 10 = 110$ columnas binarias).
- Regenera `data/dataset_A.csv` (60 registros), `data/dataset_B.csv` (60 registros) y el diccionario de datos `data/metadatos_variables.csv`.

### 5. Modo manual de referencia
```bash
python src/CarGross.py --man
```
Despliega en la terminal el manual de usuario completo estilo UNIX `man`:
- Explica la sintaxis, las opciones, el formato de entrada de 110 bits, las reglas de validación y ejemplos.
- **Dato clave:** Este manual también se dispara automáticamente si alguien ejecuta el script sin argumentos o con parámetros inválidos (para evitar *tracebacks* crudos de Python).

---

## 6. Resultados Obtenidos en los Datasets e Interpretación

### 6.1 Salida del Test Demo y Resultados sobre `dataset_A.csv` ($\rho = 0.60$)

Al ejecutar `python src/CarGross.py --demo` (o `python src/CarGross.py data/dataset_A.csv --rho 0.60`), la red converge en **2 épocas**, formando **19 categorías** en $F_2$ con un promedio de **7.3 bits activos** por prototipo $T_{Ji}$:

```text
===============================================================================================
  MODO DEMO (--demo) - VERIFICACION DE LA RED ART1 (CARPENTER-GROSSBERG)
===============================================================================================
  1. Patron 1 (Ingenieria A)  -> Cluster F2 asignado: 0 (Nuevo cluster)
  2. Patron 2 (Ingenieria B)  -> Similitud 9/11 >= 0.6 -> RESONANCIA en Cluster F2: 0
  3. Patron 3 (Vector Nulo)   -> Filtrado sin estimular F1 (retorno: -1)
  4. Patron 4 (Humanidades)   -> Similitud 0/11 < 0.6  -> RESET y nuevo Cluster F2: 1

===========================================================================================================
  REPORTE EXPERIMENTAL ART1 | Dataset: dataset_A.csv | Vigilancia rho = 0.60
===========================================================================================================
  Registros leidos: 60 | Nulos filtrados: 0 | Procesados: 60 | Epocas: 2
  Categorias creadas en F2: 19 | Media ||T_J||_1: 7.3/11 bits
-----------------------------------------------------------------------------------------------------------
  Cluster  | Alumnos | % Total | Prototipo T_Ji
-----------------------------------------------------------------------------------------------------------
  Cat_00   |       4 |    6.7% | T_Ji (7 bits): Math_v9, Logical_v6, R_v5, A_v4, S_v4, Communication_v3, E_v3
  Cat_03   |       4 |    6.7% | T_Ji (7 bits): Math_v10, I_v9, Science_v7, Programming_v5, Logical_v5, S_v4, Communication_v3
  Cat_04   |       4 |    6.7% | T_Ji (7 bits): Math_v10, Science_v9, I_v9, C_v9, Logical_v6, Programming_v5, Communication_v5
  Cat_06   |       4 |    6.7% | T_Ji (8 bits): Math_v9, Science_v9, I_v8, S_v7, Logical_v5, R_v4, A_v4, Communication_v3
  Cat_13   |       4 |    6.7% | T_Ji (7 bits): I_v8, Logical_v6, Programming_v5, Communication_v4, S_v4, A_v2, E_v2
  Cat_16   |       4 |    6.7% | T_Ji (7 bits): Science_v10, S_v9, A_v7, Communication_v6, Programming_v3, Logical_v3, C_v3
  Cat_01   |       3 |    5.0% | T_Ji (7 bits): Math_v9, Science_v9, C_v8, Logical_v6, Programming_v4, A_v4, E_v2
  Cat_02   |       3 |    5.0% | T_Ji (7 bits): Math_v9, Logical_v6, Programming_v5, Communication_v5, I_v3, S_v3, E_v3
  Cat_07   |       3 |    5.0% | T_Ji (7 bits): Science_v9, I_v9, Math_v8, Logical_v5, R_v4, E_v4, Communication_v3
  Cat_08   |       3 |    5.0% | T_Ji (7 bits): Math_v9, Science_v9, I_v8, S_v8, Communication_v5, Logical_v3, E_v2
  Cat_09   |       3 |    5.0% | T_Ji (7 bits): E_v10, R_v6, A_v6, Communication_v5, I_v4, S_v4, C_v3
  Cat_10   |       3 |    5.0% | T_Ji (7 bits): E_v10, Math_v9, Communication_v6, R_v6, S_v4, Programming_v3, C_v3
  Cat_11   |       3 |    5.0% | T_Ji (7 bits): E_v10, Math_v8, Communication_v5, C_v5, Logical_v4, I_v4, S_v4
  Cat_14   |       3 |    5.0% | T_Ji (7 bits): I_v10, Math_v9, Science_v7, Logical_v6, Programming_v5, S_v4, A_v3
  Cat_15   |       3 |    5.0% | T_Ji (7 bits): Science_v8, Logical_v6, Programming_v5, Communication_v5, R_v5, S_v4, A_v3
  Cat_17   |       3 |    5.0% | T_Ji (7 bits): S_v9, Science_v8, A_v7, Communication_v6, Logical_v5, E_v4, Programming_v3
  Cat_18   |       3 |    5.0% | T_Ji (7 bits): S_v9, Communication_v6, R_v6, A_v6, E_v4, Programming_v3, I_v2
  Cat_05   |       2 |    3.3% | T_Ji (8 bits): Math_v10, I_v9, C_v8, Logical_v5, S_v5, Programming_v3, Communication_v3, A_v3
  Cat_12   |       1 |    1.7% | T_Ji (11 bits): Science_v10, E_v10, Math_v7, R_v7, Communication_v6, Logical_v5, I_v4, Programming_v3, A_v3, S_v3, C_v1
===========================================================================================================
```

### 6.2 Resultados sobre `dataset_B.csv` ($\rho = 0.60$)

Al ejecutar `python src/CarGross.py data/dataset_B.csv --rho 0.60` sobre el segundo conjunto de 60 alumnos independientes, la red también converge en **2 épocas**, formando **22 categorías** en $F_2$ con un promedio de **7.8 bits activos** por prototipo $T_{Ji}$:

```text
===========================================================================================================
  REPORTE EXPERIMENTAL ART1 | Dataset: dataset_B.csv | Vigilancia rho = 0.60
===========================================================================================================
  Registros leidos: 60 | Nulos filtrados: 0 | Procesados: 60 | Epocas: 2
  Categorias creadas en F2: 22 | Media ||T_J||_1: 7.8/11 bits
-----------------------------------------------------------------------------------------------------------
  Cluster  | Alumnos | % Total | Prototipo T_Ji
-----------------------------------------------------------------------------------------------------------
  Cat_04   |       4 |    6.7% | T_Ji (7 bits): Math_v9, C_v9, Science_v7, Logical_v5, A_v4, Programming_v3, E_v3
  Cat_05   |       4 |    6.7% | T_Ji (7 bits): I_v10, Math_v9, Science_v8, Logical_v6, S_v4, A_v3, R_v2
  Cat_15   |       4 |    6.7% | T_Ji (7 bits): Science_v7, Programming_v6, Logical_v5, S_v4, E_v4, Communication_v3, A_v3
  Cat_00   |       3 |    5.0% | T_Ji (7 bits): Math_v9, Logical_v6, S_v4, Programming_v3, Communication_v3, I_v3, E_v2
  Cat_01   |       3 |    5.0% | T_Ji (7 bits): Math_v9, Science_v9, C_v9, Logical_v6, Programming_v5, Communication_v3, E_v2
  Cat_02   |       3 |    5.0% | T_Ji (8 bits): Math_v9, C_v9, Science_v8, Logical_v6, Programming_v5, Communication_v3, I_v3, E_v1
  Cat_07   |       3 |    5.0% | T_Ji (7 bits): Science_v9, Math_v8, Programming_v5, E_v5, R_v4, A_v3, C_v3
  Cat_08   |       3 |    5.0% | T_Ji (7 bits): Science_v9, S_v9, Math_v7, Programming_v4, Communication_v3, Logical_v3, C_v3
  Cat_09   |       3 |    5.0% | T_Ji (7 bits): Science_v9, S_v8, R_v4, Programming_v3, Communication_v3, Logical_v3, E_v3
  Cat_11   |       3 |    5.0% | T_Ji (7 bits): Math_v9, R_v8, Programming_v4, S_v4, C_v4, Logical_v3, A_v3
  Cat_12   |       3 |    5.0% | T_Ji (7 bits): Math_v8, R_v8, S_v4, C_v4, Programming_v3, Logical_v3, A_v2
  Cat_13   |       3 |    5.0% | T_Ji (7 bits): Math_v8, Communication_v5, Programming_v4, A_v4, S_v4, C_v4, I_v3
  Cat_16   |       3 |    5.0% | T_Ji (7 bits): C_v9, Math_v8, Science_v7, A_v6, Logical_v5, R_v4, Communication_v3
  Cat_17   |       3 |    5.0% | T_Ji (7 bits): Science_v9, Math_v8, Programming_v6, Communication_v5, Logical_v5, R_v5, E_v4
  Cat_18   |       3 |    5.0% | T_Ji (7 bits): S_v10, Science_v8, Math_v7, Programming_v5, Communication_v5, Logical_v3, I_v3
  Cat_19   |       3 |    5.0% | T_Ji (7 bits): Science_v8, S_v8, Math_v7, A_v7, Communication_v5, Programming_v3, I_v3
  Cat_20   |       3 |    5.0% | T_Ji (7 bits): S_v10, Math_v8, Programming_v5, Communication_v5, R_v5, E_v4, Logical_v3
  Cat_06   |       2 |    3.3% | T_Ji (8 bits): I_v10, Math_v9, C_v9, Science_v7, Logical_v6, S_v5, A_v4, R_v3
  Cat_03   |       1 |    1.7% | T_Ji (11 bits): Math_v9, C_v9, Science_v7, S_v6, Logical_v5, Programming_v4, Communication_v4, R_v4, I_v3, A_v2, E_v2
  Cat_10   |       1 |    1.7% | T_Ji (11 bits): Science_v9, I_v8, S_v8, Math_v7, Programming_v5, A_v5, Communication_v4, Logical_v4, E_v4, C_v4, R_v3
  ... (2 micro-clusters adicionales de 1 alumno) ...
===========================================================================================================
```

### 6.3 ¿Qué significa cada columna de la tabla?
- **`Cluster` (`Cat_00`, `Cat_03`, etc.):** Es el número de neurona (categoría) que se creó en la capa $F_2$ de la red.
- **`Alumnos` y `% Total`:** Cuántos estudiantes quedaron agrupados dentro de ese clúster y qué porcentaje del total representan.
- **`Prototipo T_Ji`:** Muestra **todos los bits que quedaron encendidos en `1`** en los pesos *Top-Down* ($T_{Ji}$) de esa categoría. Esos bits son exactamente las **características en las que coinciden el 100% de los alumnos de ese clúster**. Por ejemplo, en `dataset_A.csv` se ve que `Cat_03` tiene `Science_v7, Logical_v5, S_v4` mientras que `Cat_04` tiene `Science_v9, C_v9, Logical_v6, Communication_v5`, lo que permite distinguir de inmediato en qué se diferencian ambos grupos de Ciencia de Datos.

### 6.4 Comparación entre `dataset_A.csv` y `dataset_B.csv` en Distintos Niveles de Vigilancia (`--experimento`)

Al ejecutar `python src/CarGross.py --experimento`, el programa evalúa ambos datasets bajo tres niveles de vigilancia ($\rho \in \{0.30, 0.60, 0.85\}$) y arroja el siguiente cuadro comparativo:

```text
===============================================================================================
  RESUMEN COMPARATIVO EXPERIMENTAL (DATASET A vs DATASET B)
===============================================================================================
  Dataset        |   Rho | Proc. | Nulos | Clusters F2 | Media ||T_J|| | Epocas
---------------------------------------------------------------------------------
  dataset_A.csv  |  0.30 |    60 |     0 |          14 |      4.64 bit |      3
  dataset_A.csv  |  0.60 |    60 |     0 |          19 |      7.32 bit |      2
  dataset_A.csv  |  0.85 |    60 |     0 |          59 |     10.98 bit |      2
  dataset_B.csv  |  0.30 |    60 |     0 |          16 |      5.69 bit |      2
  dataset_B.csv  |  0.60 |    60 |     0 |          22 |      7.82 bit |      2
  dataset_B.csv  |  0.85 |    60 |     0 |          58 |     10.97 bit |      2
===============================================================================================
```

#### Interpretación y comparación de los resultados:
1. **Con vigilancia baja ($\rho = 0.30$ — Sub-clasificación):** La red solo exige que un alumno comparta 4 de los 11 bits activos ($\lceil 0.30 \times 11 \rceil = 4$) para entrar a un grupo. Por eso forma pocos clústeres grandes (14 en el Dataset A y 16 en el Dataset B), pero al mezclar alumnos más diversos la operación `AND` va apagando bits y los prototipos $T_{Ji}$ se erosionan a un promedio de **4.64 bits** (Dataset A) y **5.69 bits** (Dataset B).
2. **Con vigilancia intermedia ($\rho = 0.60$ — Punto de equilibrio óptimo):** La red exige al menos **7 rasgos compartidos de 11** ($\lceil 0.60 \times 11 \rceil = 7$). Ambos datasets logran un equilibrio muy consistente: se forman **19 clústeres** en el Dataset A y **22 clústeres** en el Dataset B en apenas **2 épocas**, conservando prototipos $T_{Ji}$ bien descriptivos (**7.32 y 7.82 bits activos** en promedio).
3. **Con vigilancia alta ($\rho = 0.85$ — Sobre-clasificación):** La red exige coincidir en al menos 10 de las 11 variables ($\lceil 0.85 \times 11 \rceil = 10$). Ante una condición tan estricta, el mecanismo de *Reset* se dispara casi con cada estudiante y la red termina creando prácticamente un clúster individual por alumno (**59 categorías** para 60 alumnos en el Dataset A y **58 categorías** en el Dataset B, con $\|T_J\|_1 \approx 10.98$ bits), perdiendo capacidad de generalización.
4. **Consistencia entre Dataset A y Dataset B:** A pesar de contener 60 alumnos completamente distintos cada uno, ambos conjuntos muestran exactamente la misma curva de comportamiento frente a $\rho$, la misma velocidad de convergencia (2 épocas en $\rho = 0.60$), confirmando que la red ART1 generaliza de manera estable sin sobreajustarse a una muestra en particular.

---

## 7. Preguntas Frecuentes (FAQ)

### 1. ¿Cómo elijo un buen valor para el parámetro de vigilancia (`--rho`)?
El valor de `rho` (entre `0.0` y `1.0`) define qué tan parecidos tienen que ser dos alumnos para entrar al mismo grupo:
- **Si se usa un valor bajo (ej. `0.20` o `0.30`):** La red es muy permisiva, junta alumnos bastante distintos en pocos grupos y los prototipos $T_{Ji}$ se quedan con pocos bits en común.
- **Si se usa un valor muy alto (ej. `0.85` o `0.95`):** La red exige que coincidan en casi las 11 variables, por lo que termina armando prácticamente un grupo individual por cada alumno.
- **Valor recomendado (`0.60`):** Pide que coincidan en al menos 7 de las 11 variables ($\lceil 0.60 \times 11 \rceil = 7$). En nuestras pruebas forma entre 19 y 22 grupos bien definidos en solo 2 épocas.

### 2. ¿Por qué la columna del molde de cada grupo se llama `Prototipo T_Ji`?
En la bibliografía original de Carpenter y Grossberg (Box 3 de Lippmann / Lau), la red tiene dos matrices de pesos:
- Los pesos **$B_{ij}$ (*Bottom-Up*)**, que van desde la entrada $F_1$ hacia arriba ($F_2$) para elegir qué neurona gana la competencia.
- Los pesos **$T_{ji}$ (*Top-Down*)**, que bajan desde la neurona ganadora $J$ de $F_2$ hacia los bits de entrada $i$ en $F_1$. Este vector $T_{Ji}$ guarda en `1` los rasgos que comparten todos los integrantes del clúster y actúa como la plantilla o "expectativa" contra la que se compara cada nuevo alumno.

### 3. ¿Qué pasa si le paso un archivo con errores, valores que no sean 0 y 1, o me olvido de poner argumentos?
El programa valida toda la entrada antes de entrenar. Si falta la ruta del archivo, si se intenta abrir un formato no permitido (como `.json`), si el CSV no tiene las 110 columnas o si algún dato no es `0` o `1`, muestra un mensaje claro explicando el error e imprime automáticamente en pantalla el manual completo (`--man`) para guiar al usuario.

### 4. ¿Qué hace la red si encuentra un alumno con todos los bits en cero (vector nulo)?
En ART1, una fila compuesta solo por ceros ($\|X\|_1 = 0$) no activa ninguna neurona y produciría una división por cero al calcular la vigilancia ($\|S\|_1 / \|X\|_1 = 0/0$). Por eso el cargador de datos detecta automáticamente cualquier fila nula, la descarta antes de entrenar avisando en qué número de fila estaba, y continúa procesando el resto del dataset sin fallar.

### 5. ¿Por qué `dataset_A.csv` y `dataset_B.csv` no tienen la columna `Career` y cómo sabe la red a qué vocación pertenece cada clúster?
Porque la red de Carpenter-Grossberg es un método de **aprendizaje estrictamente no supervisado**: durante el entrenamiento agrupa a los alumnos mirando únicamente sus 110 bits (`0` y `1`), descubriendo categorías naturales a partir de sus rasgos afines sin utilizar ninguna etiqueta previa ni contaminar el agrupamiento.

---

## REFERENCIAS

[1] Sánchez-Sinencio, E., & Lau, C. (1992). *Artificial Neural Networks: Paradigms, Applications, and Hardware Implementations*. IEEE Press. *(Material de cátedra: `Lau.pp5.a.11.pdf` y `Lau.pp12.a.14.pdf`)*.

[2] Lippmann, R. P. (1987). "An introduction to computing with neural nets." *IEEE ASSP Magazine*, 4(2), 4–22. *(Incluye el Box 3: Carpenter/Grossberg Net Algorithm, pp. 12–14)*.

[3] Carpenter, G. A., & Grossberg, S. (1987). "A massively parallel architecture for a self-organizing neural pattern recognition machine." *Computer Vision, Graphics, and Image Processing*, 37(1), 54–115.

[4] Holland, J. L. (1997). *Making Vocational Choices: A Theory of Vocational Personalities and Work Environments* (3rd ed.). Psychological Assessment Resources.

[5] Venkateshkumar, S. (2023). *Student Career Prediction Using RIASEC Dataset*. Kaggle Datasets. Disponible en: https://www.kaggle.com/datasets/svenkateshkumar/student-career-prediction-using-riasec-dataset
