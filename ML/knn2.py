import os
import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer

from sklearn.decomposition import PCA

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import (
    classification_report,
    confusion_matrix
)

import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# RUTAS
# ============================================================
path_entrenamiento = "/home/user/Desktop/CSI_Procesado/ROUTER/laboratorio"

path_test = "/home/user/Desktop/CSI_Procesado/ROUTER/test"

# ============================================================
# CARGA DE DATOS
# ============================================================
def cargar_datos_de_carpeta(ruta):

    X_list = []
    y_list = []

    for file in os.listdir(ruta):

        if file.endswith(".txt"):

            # Label desde nombre archivo
            label = int(
                ''.join(filter(str.isdigit, file))
            )

            file_path = os.path.join(ruta, file)

            data = pd.read_csv(
                file_path,
                header=None
            )

            # ⚠️ ELIMINAR PRIMERA COLUMNA
            # porque es el label guardado
            X = data.values[:, 1:]

            X_list.append(X)

            y_list.extend([label] * len(X))

    if not X_list:
        return None, None

    return np.vstack(X_list), np.array(y_list)

# ============================================================
# CARGA
# ============================================================
print("Cargando entrenamiento...")

X_train_raw, y_train = cargar_datos_de_carpeta(
    path_entrenamiento
)

print("Cargando test...")

X_test_raw, y_test = cargar_datos_de_carpeta(
    path_test
)

# ============================================================
# IMPUTACIÓN
# ============================================================
imputer = SimpleImputer(
    strategy='constant',
    fill_value=0
)

X_train = imputer.fit_transform(X_train_raw)

X_test = imputer.transform(X_test_raw)

# ============================================================
# NORMALIZACIÓN
# ============================================================
normalizer = Normalizer(norm='l2')

X_train = normalizer.fit_transform(X_train)

X_test = normalizer.transform(X_test)

# ============================================================
# ESTANDARIZACIÓN
# ============================================================
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)

# ============================================================
# PCA
# ============================================================
pca = PCA(n_components=5)

X_train_pca = pca.fit_transform(X_train)

X_test_pca = pca.transform(X_test)

# ============================================================
# KNN
# ============================================================
knn = KNeighborsClassifier(
    n_neighbors=3
)

knn.fit(
    X_train_pca,
    y_train
)

# ============================================================
# PREDICCIÓN
# ============================================================
y_pred = knn.predict(X_test_pca)

# ============================================================
# RESULTADOS
# ============================================================
print("\n--- RESULTADOS ---")

print(
    classification_report(
        y_test,
        y_pred
    )
)

# ============================================================
# MATRIZ DE CONFUSIÓN
# ============================================================
plt.figure(figsize=(12, 8))

cm = confusion_matrix(
    y_test,
    y_pred
)

labels = sorted(np.unique(y_test))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=labels,
    yticklabels=labels
)

plt.title(
    "Matriz de Confusión CSI k-NN"
)

plt.xlabel("Predicción")

plt.ylabel("Real")

plt.show()
