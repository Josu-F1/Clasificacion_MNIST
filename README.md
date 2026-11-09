# Clasificación de Dígitos Manuscritos con MNIST (Árbol de Decisión, Random Forest y k-NN)

Este repositorio contiene el desarrollo del **Taller APE (Aprendizaje Práctico Experimental)** de la asignatura de **Inteligencia Artificial** de la **Universidad Técnica de Ambato (UTA)** - Facultad de Ingeniería en Sistemas, Electrónica e Industrial (FISEI).

## 👥 Integrantes
* **Fiallos Yanza Jonathan Josue**
* **Guevara Panimboza Steven Marlon**

* **Docente:** Ing. Rubén Nogales  
* **Ciclo Académico:** Julio – Diciembre 2026  
* **Nivel y Paralelo:** Octavo / A  

---

## 📌 Descripción del Proyecto

El objetivo de esta práctica es comparar el desempeño de tres clasificadores de aprendizaje supervisado al reconocer dígitos manuscritos del conjunto de datos **MNIST** (`mnist_784`, OpenML ID 554), bajo un protocolo estrictamente común de entrenamiento y prueba:
1. **Árbol de Decisión (`DecisionTreeClassifier`):** `max_depth=15`, `random_state=42`.
2. **Random Forest (`RandomForestClassifier`):** `n_estimators=100`, `max_depth=15`, `random_state=42`.
3. **k-NN (`KNeighborsClassifier`):** `n_neighbors=5`, `metric="euclidean"`, `weights="uniform"`.

### Protocolo Experimental
* **Normalización:** Las imágenes ($28 \times 28$ píxeles) se aplanan a vectores de 784 características escaladas al rango continuo $[0, 1]$ dividiendo por $255.0$ (`float32`).
* **Partición y Muestreo Estratificado (`random_state=42`):**
  * **Entrenamiento:** $N_{\text{ent}} = 10\,000$ instancias extraídas del bloque original de 60\,000.
  * **Prueba:** $N_{\text{pru}} = 2\,000$ instancias extraídas del bloque original de 10\,000.

---

## 📊 Resultados Obtenidos (Tabla 2)

| Modelo | Exactitud ent. (%) | Error ent. (%) | Exactitud prueba (%) | Error prueba (%) | Brecha (p. p.) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Árbol de decisión** | $99.02$ | $0.98$ | $81.80$ | $18.20$ | **$17.22$** |
| **Random Forest** | $99.93$ | $0.07$ | **$94.80$** | **$5.20$** | **$5.13$** |
| **k-NN** | $96.51$ | $3.49$ | **$94.55$** | **$5.45$** | **$1.96$** |

*Nota: La exactitud y el error suman exactamente el 100.00% en cada conjunto.*

---

## 📈 Visualizaciones y Análisis

* **Matrices de Confusión:**
  * Árbol de decisión: Mayor confusión en **3 predicho como 5 (14 casos)**.
  * Random Forest: Mayor confusión en **7 predicho como 9 (7 casos)**.
  * k-NN: Mayor confusión en **2 predicho como 7 (9 casos)**.
* **Modelo Recomendado:** **Random Forest** debido a su superioridad en precisión de generalización ($94.80\%$) y su alta eficiencia en inferencia ($0.62\text{ s}$ en prueba vs $2.65\text{ s}$ de k-NN).

---

## 📂 Estructura del Repositorio

```text
├── images/                                            # Figuras y matrices de confusión en alta resolución
│   ├── logo_uta.png                                   # Logo institucional Universidad Técnica de Ambato
│   ├── logo_fisei.png                                 # Logo institucional FISEI
│   ├── mnist_5_ejemplos.png                           # Visualización de 5 muestras iniciales de MNIST
│   ├── distribucion_digitos.png                       # Gráfico de distribución de clases estratificadas
│   ├── comparativa_metricas.png                       # Gráfico de barras de exactitud y brecha
│   ├── comparativa_tiempos.png                        # Gráfico de tiempos de ajuste e inferencia (log)
│   ├── matrices_confusion_comparativa.png             # Panel comparativo de matrices de confusión
│   ├── cm_arbol_de_decision.png                       # Matriz detallada del Árbol de Decisión
│   ├── cm_random_forest.png                           # Matriz detallada de Random Forest
│   ├── cm_knn.png                                     # Matriz detallada de k-NN
│   ├── ejemplo_mal_clasificado.png                    # Muestra de prueba #64 (Real 2, DT 6, RF 4, k-NN 2)
│   └── ejemplos_mal_clasificados_grid.png             # Galería de 6 casos complejos en prueba
├── informe_mnist.pdf                                  # Informe formal compilado en PDF (11 páginas)
├── informe_mnist.tex                                  # Código fuente de LaTeX
├── Taller_APE_MNIST_Arbol_RandomForest_kNN.ipynb      # Cuaderno Jupyter interactivo completamente ejecutado
├── experiment.py                                      # Script reproducible del experimento completo
├── build_rich_notebook.py                             # Script de construcción y ejecución del notebook
├── generate_rich_experiments.py                       # Generador de gráficos de alta resolución
├── resultados_experimento.json                        # Métricas y tiempos computacionales consolidados
├── .gitignore                                         # Archivos excluidos del control de versiones
└── README.md                                          # Documentación principal del proyecto
```

---

## 🚀 Requisitos e Instalación

Para replicar el experimento localmente:

```bash
git clone https://github.com/Josu-F1/Clasificacion_MNIST.git
cd Clasificacion_MNIST
pip install numpy pandas matplotlib seaborn scikit-learn nbformat nbclient
```

### Ejecutar el Cuaderno:
```bash
jupyter notebook Taller_APE_MNIST_Arbol_RandomForest_kNN.ipynb
```

### Compilar el Informe en LaTeX:
```bash
pdflatex informe_mnist.tex
pdflatex informe_mnist.tex
```
