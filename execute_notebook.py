import nbformat
from nbclient import NotebookClient

print("Abriendo notebook para ejecución...")
with open("Taller_APE_MNIST_Arbol_RandomForest_kNN.ipynb", "r", encoding="utf-8") as f:
    nb = nbformat.read(f, as_version=4)

client = NotebookClient(nb, timeout=600, kernel_name="python3")
print("Ejecutando todas las celdas del notebook...")
client.execute()

with open("Taller_APE_MNIST_Arbol_RandomForest_kNN.ipynb", "w", encoding="utf-8") as f:
    nbformat.write(nb, f)

print("¡Notebook ejecutado y guardado con todas sus salidas!")

