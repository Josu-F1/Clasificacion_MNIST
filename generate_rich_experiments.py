import os
import sys
import time
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay

# Configuración estética
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['font.family'] = 'sans-serif'
os.makedirs("images", exist_ok=True)

print("1. Cargando MNIST desde OpenML...")
t0 = time.time()
X, y = fetch_openml(data_id=554, as_frame=False, return_X_y=True)
print(f"Cargado en {time.time() - t0:.2f} s. X={X.shape}, y={y.shape}")

# Preprocesamiento
X = X.astype("float32") / 255.0
y = y.astype("int64")

# Actividad 1: Visualizar cinco ejemplos iniciales con etiquetas
fig, axes = plt.subplots(1, 5, figsize=(11, 2.7))
for i in range(5):
    axes[i].imshow(X[i].reshape(28, 28), cmap="gray")
    axes[i].set_title(f"Dígito: {y[i]}", fontsize=12, fontweight="bold", pad=6)
    axes[i].axis("off")
plt.suptitle("Figura 1: Muestras representativas del conjunto MNIST (28x28 píxeles)", fontsize=13, fontweight="bold", y=1.03)
plt.tight_layout()
plt.savefig("images/mnist_5_ejemplos.png", dpi=300, bbox_inches="tight")
plt.close()

# Muestreo estratificado prescrito
def tomar_muestra(X_base, y_base, n):
    X_muestra, _, y_muestra, _ = train_test_split(
        X_base, y_base, train_size=n, stratify=y_base, random_state=42
    )
    return X_muestra, y_muestra

X_ent, y_ent = tomar_muestra(X[:60000], y[:60000], 10000)
X_pru, y_pru = tomar_muestra(X[60000:], y[60000:], 2000)

print(f"X_ent: {X_ent.shape}, X_pru: {X_pru.shape}")

# Gráfico de distribución de clases
dist_ent = pd.Series(y_ent).value_counts().sort_index()
dist_pru = pd.Series(y_pru).value_counts().sort_index()

fig, ax = plt.subplots(figsize=(9, 4))
x_idx = np.arange(10)
w = 0.38
rects1 = ax.bar(x_idx - w/2, dist_ent.values / len(y_ent) * 100, w, label='Entrenamiento (10 000)', color='#1f77b4', edgecolor='black', alpha=0.85)
rects2 = ax.bar(x_idx + w/2, dist_pru.values / len(y_pru) * 100, w, label='Prueba (2 000)', color='#ff7f0e', edgecolor='black', alpha=0.85)
ax.set_xlabel('Dígito', fontsize=11, fontweight='bold')
ax.set_ylabel('Proporción (%)', fontsize=11, fontweight='bold')
ax.set_title('Distribución Estratificada de Clases en Entrenamiento y Prueba', fontsize=12, fontweight='bold')
ax.set_xticks(x_idx)
ax.set_xticklabels([str(i) for i in range(10)])
ax.set_ylim(0, 15)
ax.legend(frameon=True, facecolor='white', framealpha=0.9)
for rect in rects1:
    h = rect.get_height()
    ax.annotate(f'{h:.1f}%', xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 2),
                textcoords="offset points", ha='center', va='bottom', fontsize=8)
for rect in rects2:
    h = rect.get_height()
    ax.annotate(f'{h:.1f}%', xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 2),
                textcoords="offset points", ha='center', va='bottom', fontsize=8)
plt.tight_layout()
plt.savefig("images/distribucion_digitos.png", dpi=300)
plt.close()
print("Guardado images/distribucion_digitos.png")

# Modelos
modelos = {
    "Árbol de decisión": DecisionTreeClassifier(max_depth=15, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42),
    "k-NN": KNeighborsClassifier(n_neighbors=5, metric="euclidean", weights="uniform")
}

tiempos_ent = {}
tiempos_pru = {}
predicciones_ent = {}
predicciones_pru = {}
resultados = []

for nombre, mod in modelos.items():
    print(f"Entrenando {nombre}...")
    t_start = time.time()
    mod.fit(X_ent, y_ent)
    tiempos_ent[nombre] = time.time() - t_start
    
    t_p_start = time.time()
    y_pred_ent = mod.predict(X_ent)
    y_pred_pru = mod.predict(X_pru)
    tiempos_pru[nombre] = time.time() - t_p_start
    
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

df_res = pd.DataFrame(resultados)
print(df_res.to_string(index=False))

# Gráfico de barras de métricas comparativas
fig, ax = plt.subplots(figsize=(9, 4.5))
mod_names = [r["Modelo"] for r in resultados]
x = np.arange(len(mod_names))
w = 0.25

r1 = ax.bar(x - w, [r["Exactitud ent. (%)"] for r in resultados], w, label='Exactitud Entrenamiento', color='#2ca02c', edgecolor='black', alpha=0.85)
r2 = ax.bar(x, [r["Exactitud prueba (%)"] for r in resultados], w, label='Exactitud Prueba', color='#1f77b4', edgecolor='black', alpha=0.85)
r3 = ax.bar(x + w, [r["Brecha (p. p.)"] for r in resultados], w, label='Brecha (Sobreajuste)', color='#d62728', edgecolor='black', alpha=0.85)

ax.set_xlabel('Modelo Clasificador', fontsize=11, fontweight='bold')
ax.set_ylabel('Porcentaje / Puntos Porcentuales', fontsize=11, fontweight='bold')
ax.set_title('Comparativa de Rendimiento y Brecha de Generalización', fontsize=12, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(mod_names, fontweight='bold')
ax.set_ylim(0, 115)
ax.legend(frameon=True, facecolor='white', framealpha=0.9, loc='upper right')

for rects in [r1, r2, r3]:
    for rect in rects:
        h = rect.get_height()
        ax.annotate(f'{h:.2f}%' if rects != r3 else f'{h:.2f} pp',
                    xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 2),
                    textcoords="offset points", ha='center', va='bottom', fontsize=8.5, fontweight='bold')

plt.tight_layout()
plt.savefig("images/comparativa_metricas.png", dpi=300)
plt.close()
print("Guardado images/comparativa_metricas.png")

# Gráfico de tiempos computacionales
fig, ax = plt.subplots(figsize=(8, 4))
t_ent_vals = [tiempos_ent[m] for m in mod_names]
t_pru_vals = [tiempos_pru[m] for m in mod_names]
w = 0.35
ax.bar(x - w/2, t_ent_vals, w, label='Tiempo Entrenamiento (fit)', color='#4a90e2', edgecolor='black', alpha=0.85)
ax.bar(x + w/2, t_pru_vals, w, label='Tiempo Inferencia (predict en 2 000 pru)', color='#e94e77', edgecolor='black', alpha=0.85)
ax.set_yscale('log')
ax.set_xlabel('Modelo Clasificador', fontsize=11, fontweight='bold')
ax.set_ylabel('Tiempo en segundos (Escala logarítmica)', fontsize=11, fontweight='bold')
ax.set_title('Comparativa de Costo Computacional (Entrenamiento vs Inferencia)', fontsize=12, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(mod_names, fontweight='bold')
ax.legend(frameon=True, facecolor='white')

for i, v in enumerate(t_ent_vals):
    ax.text(x[i] - w/2, v * 1.15, f'{v:.2f}s', ha='center', fontsize=8.5, fontweight='bold')
for i, v in enumerate(t_pru_vals):
    ax.text(x[i] + w/2, v * 1.15, f'{v:.2f}s', ha='center', fontsize=8.5, fontweight='bold')

plt.tight_layout()
plt.savefig("images/comparativa_tiempos.png", dpi=300)
plt.close()
print("Guardado images/comparativa_tiempos.png")

# Matrices de confusión
confusiones_max = {}
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
for idx, (nombre, mod) in enumerate(modelos.items()):
    cm = confusion_matrix(y_pru, predicciones_pru[nombre], labels=list(range(10)))
    cm_sin_diag = cm.copy()
    np.fill_diagonal(cm_sin_diag, 0)
    max_val = int(cm_sin_diag.max())
    max_indices = np.where(cm_sin_diag == max_val)
    parejas = [(int(max_indices[0][k]), int(max_indices[1][k])) for k in range(len(max_indices[0]))]
    confusiones_max[nombre] = {
        "max_casos": max_val,
        "parejas": parejas
    }
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(range(10)))
    disp.plot(ax=axes[idx], cmap="Blues", colorbar=False)
    axes[idx].set_title(f"Matriz de Confusión\n{nombre}", fontsize=11, fontweight="bold")
    axes[idx].set_xlabel("Predicción", fontsize=10)
    axes[idx].set_ylabel("Etiqueta Real", fontsize=10)

plt.tight_layout()
plt.savefig("images/matrices_confusion_comparativa.png", dpi=300)
plt.close()

# Matrices individuales detalladas con colorbar
for nombre, mod in modelos.items():
    safe_name = "cm_" + nombre.lower().replace(" ", "_").replace("á", "a").replace("-", "")
    cm = confusion_matrix(y_pru, predicciones_pru[nombre], labels=list(range(10)))
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(range(10)))
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    ax.set_title(f"Matriz de Confusión Detallada: {nombre}", fontsize=11, fontweight="bold", pad=10)
    ax.set_xlabel("Etiqueta Predicha", fontsize=10, fontweight="bold")
    ax.set_ylabel("Etiqueta Verdadera", fontsize=10, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"images/{safe_name}.png", dpi=300)
    plt.close()
    print(f"Guardado images/{safe_name}.png")

# GRID DE ERRORES: Múltiples ejemplos mal clasificados representativos
# Busquemos varios errores interesantes en el conjunto de prueba
errores_dt = (predicciones_pru["Árbol de decisión"] != y_pru)
errores_rf = (predicciones_pru["Random Forest"] != y_pru)
errores_knn = (predicciones_pru["k-NN"] != y_pru)

# Casos donde fallen dos o más modelos
fallos_varios = np.where((errores_dt & errores_rf) | (errores_rf & errores_knn) | (errores_dt & errores_knn))[0]

# Tomar 6 casos ilustrativos con distintos dígitos reales
seleccionados = []
digitos_vistos = set()
for idx in fallos_varios:
    d_real = int(y_pru[idx])
    if d_real not in digitos_vistos:
        seleccionados.append(idx)
        digitos_vistos.add(d_real)
    if len(seleccionados) == 6:
        break

fig, axes = plt.subplots(2, 3, figsize=(11, 7))
axes = axes.flatten()
for i, idx in enumerate(seleccionados):
    real = int(y_pru[idx])
    p_dt = int(predicciones_pru["Árbol de decisión"][idx])
    p_rf = int(predicciones_pru["Random Forest"][idx])
    p_knn = int(predicciones_pru["k-NN"][idx])
    
    axes[i].imshow(X_pru[idx].reshape(28, 28), cmap="gray")
    titulo = f"Muestra #{idx} | Real: {real}\nÁrbol: {p_dt} | RF: {p_rf} | k-NN: {p_knn}"
    axes[i].set_title(titulo, fontsize=10, fontweight='bold', pad=4)
    axes[i].axis("off")

plt.suptitle("Galería de Casos Complejos y Confusiones Interclase en Prueba", fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig("images/ejemplos_mal_clasificados_grid.png", dpi=300)
plt.close()
print("Guardado images/ejemplos_mal_clasificados_grid.png")

# Guardar caso individual #64
idx_64 = 64
fig, ax = plt.subplots(figsize=(3.8, 3.8))
ax.imshow(X_pru[idx_64].reshape(28, 28), cmap="gray")
ax.set_title(
    f"Muestra #{idx_64} | Etiqueta Real: {y_pru[idx_64]}\n"
    f"Árbol: {predicciones_pru['Árbol de decisión'][idx_64]} | "
    f"RF: {predicciones_pru['Random Forest'][idx_64]} | "
    f"k-NN: {predicciones_pru['k-NN'][idx_64]}",
    fontsize=10.5, fontweight="bold"
)
ax.axis("off")
plt.tight_layout()
plt.savefig("images/ejemplo_mal_clasificado.png", dpi=300)
plt.close()
print("Guardado images/ejemplo_mal_clasificado.png")

# Guardar informe completo en JSON
info_json = {
    "integrantes": [
        "Fiallos Yanza Jonathan Josue",
        "Guevara Panimboza Steven Marlon"
    ],
    "distribucion": [
        {"digito": int(i), "ent": int(dist_ent[i]), "pru": int(dist_pru[i])} for i in range(10)
    ],
    "resultados": resultados,
    "tiempos_ent": tiempos_ent,
    "tiempos_pru": tiempos_pru,
    "confusiones_max": confusiones_max,
    "ejemplos_analizados": [
        {
            "indice": int(idx),
            "real": int(y_pru[idx]),
            "pred_dt": int(predicciones_pru["Árbol de decisión"][idx]),
            "pred_rf": int(predicciones_pru["Random Forest"][idx]),
            "pred_knn": int(predicciones_pru["k-NN"][idx])
        } for idx in seleccionados
    ]
}

with open("resultados_experimento.json", "w", encoding="utf-8") as f:
    json.dump(info_json, f, indent=4, ensure_ascii=False)

print("¡Experimentos ricos y gráficos generados con éxito!")
