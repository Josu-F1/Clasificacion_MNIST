import os
import sys
import time
import json
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

print(f"Python version: {sys.version}")
print(f"Scikit-learn version: {sklearn.__version__}")

os.makedirs("images", exist_ok=True)

# 1. Cargar MNIST
print("Cargando MNIST...")
t0 = time.time()
X, y = fetch_openml(data_id=554, as_frame=False, return_X_y=True)
print(f"MNIST cargado en {time.time() - t0:.2f} s. X shape: {X.shape}, y shape: {y.shape}")

# Preprocesamiento
X = X.astype("float32") / 255.0
y = y.astype("int64")

# Actividad 1: Visualizar cinco ejemplos con sus etiquetas
np.random.seed(42)
sample_indices = [0, 1, 2, 3, 4]  # O índices representativos
fig, axes = plt.subplots(1, 5, figsize=(10, 2.5))
for idx, ax in zip(sample_indices, axes):
    ax.imshow(X[idx].reshape(28, 28), cmap="gray")
    ax.set_title(f"Etiqueta: {y[idx]}", fontsize=11, fontweight="bold")
    ax.axis("off")
plt.tight_layout()
plt.savefig("images/mnist_5_ejemplos.png", dpi=300)
plt.close()
print("Guardado images/mnist_5_ejemplos.png")

# Muestreo estratificado
def tomar_muestra(X_base, y_base, n):
    X_muestra, _, y_muestra, _ = train_test_split(
        X_base, y_base, train_size=n, stratify=y_base, random_state=42
    )
    return X_muestra, y_muestra

X_ent, y_ent = tomar_muestra(X[:60000], y[:60000], 10000)
X_pru, y_pru = tomar_muestra(X[60000:], y[60000:], 2000)

print(f"X_ent: {X_ent.shape}, y_ent: {y_ent.shape}")
print(f"X_pru: {X_pru.shape}, y_pru: {y_pru.shape}")

# Distribución por dígito
dist_ent = pd.Series(y_ent).value_counts().sort_index()
dist_pru = pd.Series(y_pru).value_counts().sort_index()
dist_df = pd.DataFrame({
    "Dígito": list(range(10)),
    "Entrenamiento": dist_ent.values,
    "Porcentaje Ent (%)": (dist_ent.values / len(y_ent) * 100).round(2),
    "Prueba": dist_pru.values,
    "Porcentaje Pru (%)": (dist_pru.values / len(y_pru) * 100).round(2)
})
print("\n--- Distribución de clases ---")
print(dist_df.to_string(index=False))

# Actividad 2: Entrenar modelos
modelos = {
    "Árbol de decisión": DecisionTreeClassifier(max_depth=15, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42),
    "k-NN": KNeighborsClassifier(n_neighbors=5, metric="euclidean", weights="uniform")
}

resultados = []
predicciones_ent = {}
predicciones_pru = {}
tiempos_ent = {}
tiempos_pru = {}

for nombre, mod in modelos.items():
    print(f"\nEntrenando {nombre}...")
    t_start = time.time()
    mod.fit(X_ent, y_ent)
    t_fit = time.time() - t_start
    tiempos_ent[nombre] = t_fit
    print(f"{nombre} entrenado en {t_fit:.2f} s")
    
    # Actividad 3: Evaluar
    t_p_start = time.time()
    y_pred_ent = mod.predict(X_ent)
    y_pred_pru = mod.predict(X_pru)
    t_pred = time.time() - t_p_start
    tiempos_pru[nombre] = t_pred
    
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
print("\n--- Tabla 2. Resultados de entrenamiento y prueba ---")
print(df_res.to_string(index=False))

# Actividad 4: Matrices de confusión y análisis de errores
confusiones_max = {}
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
for idx, (nombre, mod) in enumerate(modelos.items()):
    cm = confusion_matrix(y_pru, predicciones_pru[nombre], labels=list(range(10)))
    
    # Encontrar la confusión más frecuente entre dígitos distintos (i != j)
    cm_sin_diag = cm.copy()
    np.fill_diagonal(cm_sin_diag, 0)
    max_val = cm_sin_diag.max()
    max_indices = np.where(cm_sin_diag == max_val)
    parejas = [(max_indices[0][k], max_indices[1][k]) for k in range(len(max_indices[0]))]
    confusiones_max[nombre] = {
        "max_casos": int(max_val),
        "parejas": [(int(r), int(p)) for r, p in parejas]
    }
    
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(range(10)))
    disp.plot(ax=axes[idx], cmap="Blues", colorbar=False)
    axes[idx].set_title(f"Matriz de Confusión\n{nombre}", fontsize=12, fontweight="bold")
    axes[idx].set_xlabel("Etiqueta predicha", fontsize=10)
    axes[idx].set_ylabel("Etiqueta real", fontsize=10)

plt.tight_layout()
plt.savefig("images/matrices_confusion_comparativa.png", dpi=300)
plt.close()
print("Guardado images/matrices_confusion_comparativa.png")

# Guardar matrices individuales con barra de color para alta resolución
for nombre, mod in modelos.items():
    safe_name = nombre.replace(" ", "_").replace("Á", "A").replace("-", "").lower()
    cm = confusion_matrix(y_pru, predicciones_pru[nombre], labels=list(range(10)))
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(range(10)))
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    ax.set_title(f"Matriz de Confusión: {nombre}", fontsize=12, fontweight="bold")
    ax.set_xlabel("Etiqueta predicha")
    ax.set_ylabel("Etiqueta real")
    plt.tight_layout()
    plt.savefig(f"images/cm_{safe_name}.png", dpi=300)
    plt.close()

# Identificar imagen mal clasificada interesante
# Buscamos un ejemplo donde al menos dos modelos se equivoquen o donde haya error representativo
errores_dt = (predicciones_pru["Árbol de decisión"] != y_pru)
errores_rf = (predicciones_pru["Random Forest"] != y_pru)
errores_knn = (predicciones_pru["k-NN"] != y_pru)

# Índices donde falle el árbol y/o Random Forest y/o knn
indices_fallo_multiple = np.where(errores_dt & errores_rf)[0]
idx_ejemplo = indices_fallo_multiple[0]  # Primer ejemplo ilustrativo

y_real = int(y_pru[idx_ejemplo])
pred_dt = int(predicciones_pru["Árbol de decisión"][idx_ejemplo])
pred_rf = int(predicciones_pru["Random Forest"][idx_ejemplo])
pred_knn = int(predicciones_pru["k-NN"][idx_ejemplo])

fig, ax = plt.subplots(figsize=(4, 4))
ax.imshow(X_pru[idx_ejemplo].reshape(28, 28), cmap="gray")
ax.set_title(
    f"Etiqueta Real: {y_real}\n"
    f"Árbol: {pred_dt} | RF: {pred_rf} | k-NN: {pred_knn}",
    fontsize=11, fontweight="bold"
)
ax.axis("off")
plt.tight_layout()
plt.savefig("images/ejemplo_mal_clasificado.png", dpi=300)
plt.close()
print(f"Guardado images/ejemplo_mal_clasificado.png (Índice: {idx_ejemplo}, Real: {y_real}, DT: {pred_dt}, RF: {pred_rf}, kNN: {pred_knn})")

# Guardar todo en JSON para análisis
datos_completos = {
    "distribucion": dist_df.to_dict(orient="records"),
    "resultados": resultados,
    "confusiones_max": confusiones_max,
    "ejemplo_error": {
        "indice": int(idx_ejemplo),
        "real": y_real,
        "pred_dt": pred_dt,
        "pred_rf": pred_rf,
        "pred_knn": pred_knn
    },
    "tiempos_ent": tiempos_ent,
    "tiempos_pru": tiempos_pru,
    "sklearn_version": sklearn.__version__,
    "python_version": sys.version
}

with open("resultados_experimento.json", "w", encoding="utf-8") as f:
    json.dump(datos_completos, f, indent=4, ensure_ascii=False)

print("\n¡Experimento completado exitosamente y datos guardados!")

