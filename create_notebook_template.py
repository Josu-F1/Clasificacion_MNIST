import json
import nbformat as nbf

nb = nbf.v4.new_notebook()

# Metadatos del notebook
nb.metadata = {
    "language_info": {
        "name": "python",
        "version": "3.12"
    },
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3"
    }
}

cells = []

# Celda 1: Encabezado Institucional
cells.append(nbf.v4.new_markdown_cell("""# UNIVERSIDAD TÉCNICA DE AMBATO
### FACULTAD DE INGENIERÍA EN SISTEMAS, ELECTRÓNICA E INDUSTRIAL
**Carrera:** Ingeniería en Software / Computación  
**Ciclo Académico:** Julio – Diciembre 2026  
**Asignatura:** Inteligencia Artificial  
**Docente:** Ing. Rubén Nogales  

---
# INFORME DE GUÍA PRÁCTICA (TALLER APE)
## Tema: Clasificación de dígitos manuscritos con MNIST (Árbol de Decisión, Random Forest y k-NN)

**Alumnos Participantes (Orden Alfabético):**
1. [Nombre del Estudiante 1]
2. [Nombre del Estudiante 2]
"""))

# Celda 2: Objetivos
cells.append(nbf.v4.new_markdown_cell("""## Objetivos de la Práctica
* **General:** Comparar el desempeño de un árbol de decisión, Random Forest y k-NN al clasificar dígitos manuscritos de MNIST mediante un protocolo común de entrenamiento y prueba.
* **Específicos:**
  1. Preparar las imágenes de MNIST y reproducir los conjuntos de entrenamiento y prueba.
  2. Ajustar y evaluar los tres clasificadores con los mismos datos y calcular su exactitud y error en cada conjunto.
  3. Justificar la elección de un modelo a partir de las métricas de prueba y del análisis de sus errores.
"""))

# Celda 3: Entorno y librerías
cells.append(nbf.v4.new_markdown_cell("""## 1. Configuración del Entorno e Importación de Librerías"""))

cells.append(nbf.v4.new_code_cell("""import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

print(f"Versión de Python: {sys.version}")
print(f"Versión de Scikit-Learn: {sklearn.__version__}")
"""))

# Celda 4: Actividad 1
cells.append(nbf.v4.new_markdown_cell("""## 2. Actividad 1: Preparación y Exploración del Dataset MNIST
Se descarga el dataset `mnist_784` versión 1 (OpenML ID 554), conteniendo imágenes de $28 \\times 28$ píxeles en escala de grises. Se normalizan las intensidades al rango $[0, 1]$ dividiendo por 255.0.
"""))

cells.append(nbf.v4.new_code_cell("""# Carga del dataset MNIST desde OpenML
print("Cargando MNIST desde OpenML (data_id=554)...")
X, y = fetch_openml(data_id=554, as_frame=False, return_X_y=True)

# Normalización de intensidades a [0, 1] y conversión de tipos
X = X.astype("float32") / 255.0
y = y.astype("int64")

print(f"Dimensiones de X: {X.shape}, dimensiones de y: {y.shape}")
"""))

cells.append(nbf.v4.new_markdown_cell("""### Visualización de 5 Muestras Iniciales y sus Etiquetas"""))

cells.append(nbf.v4.new_code_cell("""fig, axes = plt.subplots(1, 5, figsize=(11, 2.8))
for i in range(5):
    axes[i].imshow(X[i].reshape(28, 28), cmap="gray")
    axes[i].set_title(f"Etiqueta: {y[i]}", fontsize=11, fontweight="bold")
    axes[i].axis("off")
plt.suptitle("Cinco ejemplos iniciales del conjunto MNIST", fontsize=12, fontweight="bold", y=1.05)
plt.tight_layout()
plt.show()
"""))

cells.append(nbf.v4.new_markdown_cell("""### Muestreo Estratificado (Entrenamiento y Prueba)
Se mantiene la partición canónica:
* Las primeras 60 000 observaciones corresponden al bloque de entrenamiento original.
* Las últimas 10 000 observaciones corresponden al bloque de prueba original.

De acuerdo a la guía, se toma un subconjunto estratificado con semilla 42:
* $N_{\\text{ent}} = 10\\,000$ observaciones.
* $N_{\\text{pru}} = 2\\,000$ observaciones.
"""))

cells.append(nbf.v4.new_code_cell("""def tomar_muestra(X_base, y_base, n):
    X_muestra, _, y_muestra, _ = train_test_split(
        X_base, y_base, train_size=n, stratify=y_base,
        random_state=42
    )
    return X_muestra, y_muestra

X_ent, y_ent = tomar_muestra(X[:60000], y[:60000], 10000)
X_pru, y_pru = tomar_muestra(X[60000:], y[60000:], 2000)

print(f"Dimensiones X_ent: {X_ent.shape}, y_ent: {y_ent.shape}")
print(f"Dimensiones X_pru: {X_pru.shape}, y_pru: {y_pru.shape}")
"""))

cells.append(nbf.v4.new_markdown_cell("""### Verificación de la Distribución por Dígito"""))

cells.append(nbf.v4.new_code_cell("""dist_ent = pd.Series(y_ent).value_counts().sort_index()
dist_pru = pd.Series(y_pru).value_counts().sort_index()

df_dist = pd.DataFrame({
    "Dígito": list(range(10)),
    "Cant. Entrenamiento": dist_ent.values,
    "Prop. Ent. (%)": (dist_ent.values / len(y_ent) * 100).round(2),
    "Cant. Prueba": dist_pru.values,
    "Prop. Pru. (%)": (dist_pru.values / len(y_pru) * 100).round(2)
})
df_dist
"""))

# Celda 5: Actividad 2
cells.append(nbf.v4.new_markdown_cell("""## 3. Actividad 2: Instanciación y Entrenamiento de Modelos
Se configuran los tres clasificadores con los hiperparámetros estipulados en la Tabla 1:
1. **Árbol de Decisión:** `DecisionTreeClassifier(max_depth=15, random_state=42)`
2. **Random Forest:** `RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42)`
3. **k-NN:** `KNeighborsClassifier(n_neighbors=5, metric="euclidean", weights="uniform")`

*Nota explicativa sobre `fit`:*
* **Árbol de decisión:** Aprende reglas jerárquicas binarias de partición sobre píxeles individuales para minimizar la impureza de Gini hasta una profundidad de 15.
* **Random Forest:** Construye de forma ensemble 100 árboles mediante bootstrapping y muestreo aleatorio de características (píxeles), reduciendo significativamente la varianza.
* **k-NN:** Conserva los vectores en memoria dentro de estructuras espaciales de búsqueda sin ajustar coeficientes (aprendizaje perezoso o lazy).
"""))

cells.append(nbf.v4.new_code_cell("""modelos = {
    "Árbol de decisión": DecisionTreeClassifier(max_depth=15, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42),
    "k-NN": KNeighborsClassifier(n_neighbors=5, metric="euclidean", weights="uniform")
}

tiempos_ent = {}
for nombre, mod in modelos.items():
    t0 = time.time()
    mod.fit(X_ent, y_ent)
    t_elap = time.time() - t0
    tiempos_ent[nombre] = t_elap
    print(f"[{nombre}] Ajuste fit completado en {t_elap:.2f} s")
"""))

# Celda 6: Actividad 3
cells.append(nbf.v4.new_markdown_cell("""## 4. Actividad 3: Evaluación de Desempeño
Se ejecutan predicciones con `predict` tanto en entrenamiento como en prueba para calcular:
* **Exactitud (%):** $100 \\times \\text{aciertos} / N$
* **Error de clasificación (%):** $100 - \\text{Exactitud}$
* **Brecha (p. p.):** $\\text{Exactitud}_{\\text{ent}} - \\text{Exactitud}_{\\text{pru}}$
"""))

cells.append(nbf.v4.new_code_cell("""resultados = []
predicciones_ent = {}
predicciones_pru = {}

for nombre, mod in modelos.items():
    y_pred_ent = mod.predict(X_ent)
    y_pred_pru = mod.predict(X_pru)
    
    predicciones_ent[nombre] = y_pred_ent
    predicciones_pru[nombre] = y_pred_pru
    
    acc_ent = accuracy_score(y_ent, y_pred_ent) * 100.0
    err_ent = 100.0 - acc_ent
    acc_pru = accuracy_score(y_pru, y_pred_pru) * 100.0
    err_pru = 100.0 - acc_pru
    brecha = acc_ent - acc_pru
    
    resultados.append({
        "Modelo": nombre,
        "Exactitud ent. (%)": round(acc_ent, 2),
        "Error ent. (%)": round(err_ent, 2),
        "Exactitud prueba (%)": round(acc_pru, 2),
        "Error prueba (%)": round(err_pru, 2),
        "Brecha (p. p.)": round(brecha, 2)
    })

df_tabla2 = pd.DataFrame(resultados)
print("Tabla 2. Resultados de entrenamiento y prueba:")
df_tabla2
"""))

# Celda 7: Actividad 4
cells.append(nbf.v4.new_markdown_cell("""## 5. Actividad 4: Matrices de Confusión y Análisis de Errores
Se construyen las matrices de confusión para cada clasificador en el conjunto de prueba (clases 0 a 9) y se calcula la confusión más repetida entre dígitos distintos.
"""))

cells.append(nbf.v4.new_code_cell("""fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
confusiones_max = {}

for idx, (nombre, mod) in enumerate(modelos.items()):
    cm = confusion_matrix(y_pru, predicciones_pru[nombre], labels=list(range(10)))
    
    # Análisis de confusión más frecuente fuera de la diagonal principal
    cm_sin_diag = cm.copy()
    np.fill_diagonal(cm_sin_diag, 0)
    max_casos = int(cm_sin_diag.max())
    max_indices = np.where(cm_sin_diag == max_casos)
    pares = [(int(max_indices[0][k]), int(max_indices[1][k])) for k in range(len(max_indices[0]))]
    
    confusiones_max[nombre] = {
        "max_casos": max_casos,
        "pares": pares
    }
    
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(range(10)))
    disp.plot(ax=axes[idx], cmap="Blues", colorbar=False)
    axes[idx].set_title(f"Matriz de Confusión\\n{nombre}", fontsize=11, fontweight="bold")
    axes[idx].set_xlabel("Predicción", fontsize=10)
    axes[idx].set_ylabel("Etiqueta Real", fontsize=10)

plt.tight_layout()
plt.show()

print("--- Confusiones más frecuentes entre dígitos distintos ---")
for nombre, info in confusiones_max.items():
    print(f"{nombre}: Máximo de {info['max_casos']} confusiones en los pares (Real, Predicho) = {info['pares']}")
"""))

cells.append(nbf.v4.new_markdown_cell("""### Análisis de un Ejemplo de Prueba Mal Clasificado"""))

cells.append(nbf.v4.new_code_cell("""# Seleccionar un índice de prueba donde haya discrepancias o fallos
errores_dt = (predicciones_pru["Árbol de decisión"] != y_pru)
errores_rf = (predicciones_pru["Random Forest"] != y_pru)
indices_fallo = np.where(errores_dt & errores_rf)[0]

idx_mostrar = indices_fallo[0]

plt.figure(figsize=(3.5, 3.5))
plt.imshow(X_pru[idx_mostrar].reshape(28, 28), cmap="gray")
plt.title(
    f"Muestra de Prueba #{idx_mostrar}\\n"
    f"Etiqueta Real: {y_pru[idx_mostrar]}\\n"
    f"Árbol: {predicciones_pru['Árbol de decisión'][idx_mostrar]} | "
    f"RF: {predicciones_pru['Random Forest'][idx_mostrar]} | "
    f"k-NN: {predicciones_pru['k-NN'][idx_mostrar]}",
    fontsize=11, fontweight="bold"
)
plt.axis("off")
plt.show()
"""))

cells.append(nbf.v4.new_markdown_cell("""## 6. Respuestas a la Comparación y Discusión (Sección 2.7)

### 1. ¿Qué modelo obtuvo la mayor exactitud y el menor error en prueba?
* **Respuesta:** *[Se completará automáticamente a partir de los datos calculados]*

### 2. ¿Qué modelo presentó la mayor brecha entre entrenamiento y prueba?
* **Respuesta:** *[Se completará automáticamente a partir de los datos calculados]*
* *Nota conceptual:* En k-NN, una observación de entrenamiento evaluada en entrenamiento tiene distancia cero consigo misma (su propio vecino más cercano), lo cual puede inflar la exactitud en entrenamiento sin implicar que el modelo tenga un sobreajuste tradicional paramétrico.

### 3. ¿Qué confusión común o diferente observaron entre los modelos? Justificación de recomendación.
* **Respuesta:** *[Se completará con la evidencia de las matrices de confusión]*

---
## 7. Habilidades Blandas Empleadas (Sección 2.8)
* **Pensamiento Crítico:** Aplicado al examinar las causas intrínsecas de los errores de predicción morfológica entre dígitos visualmente análogos (como 4 vs 9 o 7 vs 2), en lugar de limitarse a tabular porcentajes ciegos.
* **Trabajo en Equipo:** Distribución eficiente y coordinada de roles para la codificación modular, ejecución computacional y redacción técnica rigurosa.
* **Responsabilidad:** Cumplimiento estricto de las restricciones del protocolo experimental (fijación estricta de semillas, hiperparámetros y particiones).

---
## 8. Conclusiones y Recomendaciones (Secciones 2.9 y 2.10)
### Conclusiones:
1. **Desempeño en prueba:** Sustentado con las métricas cuantitativas obtenidas.
2. **Diferencia entre entrenamiento y prueba:** Análisis de la capacidad de generalización y efecto de la reducción de varianza.

### Recomendaciones:
* Proponer optimización de hiperparámetros mediante búsqueda en malla estratificada (*Stratified K-Fold Cross Validation*) o extracción previa de características espaciales (PCA/HOG) antes del clasificador.
"""))

nb.cells = cells

with open("Taller_APE_MNIST_Arbol_RandomForest_kNN.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print("Notebook generado exitosamente en Taller_APE_MNIST_Arbol_RandomForest_kNN.ipynb")

