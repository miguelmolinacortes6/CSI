import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold  # Agregados GridSearchCV y StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
# Técnica para balancear el dataset de forma inteligente
from imblearn.over_sampling import SMOTE 

# =====================================================================
# 1. CARGA DINÁMICA Y CONFIGURACIÓN
# =====================================================================
path_entrenamiento = "/home/user/Desktop/CSI_Procesado/ROUTER/training"

X_list = []
y_list = []

for file in os.listdir(path_entrenamiento):
    if file.endswith(".txt"):
        file_path = os.path.join(path_entrenamiento, file)
        
        # Cargamos el CSV procesado
        data = pd.read_csv(file_path, header=None)
        
        # ¡IMPORTANTE!: 
        # Columna 0 es la etiqueta guardada por el script anterior.
        # Columnas 1 en adelante son las características (Metadatos + Amp + Fase).
        labels_en_archivo = data.iloc[:, 0].values
        features_en_archivo = data.iloc[:, 1:].values
        
        X_list.append(features_en_archivo)
        y_list.extend(labels_en_archivo)

# Unificar matrices
X = np.vstack(X_list)
y = np.array(y_list, dtype=int)

print(f"📊 Dataset cargado original: {X.shape[0]} muestras, {X.shape[1]} características.")

# =====================================================================
# 2. PRE-PROCESAMIENTO AVANZADO
# =====================================================================
# Limpieza de seguridad por si queda algún NaN colgado
imputer = SimpleImputer(strategy='constant', fill_value=0)
X_imputed = imputer.fit_transform(X)

# Escalado estándar por columnas (Crucial para equilibrar Amplitud vs Fase)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# =====================================================================
# 3. REDUCCIÓN DE DIMENSIONALIDAD (PCA)
# =====================================================================
# 30 componentes para dar suficiente espacio geométrico al mezclar amplitud y fase
n_componentes_pca = 30
pca = PCA(n_components=n_componentes_pca, random_state=42) 
X_pca = pca.fit_transform(X_scaled)
print(f"📉 Dimensión reducida con PCA a {n_componentes_pca} componentes.")

# =====================================================================
# 4. DIVISIÓN TRAIN/TEST (Estratificada)
# =====================================================================
# stratify=y garantiza que tanto en entrenamiento como en test quede la misma 
# proporción de muestras de cada punto físico.
X_train, X_test, y_train, y_test = train_test_split(
    X_pca, y, test_size=0.2, random_state=42, stratify=y
)

# =====================================================================
# 5. BALANCEO ANTE EL SESGO DE CLASES (SMOTE)
# =====================================================================
# El K-NN sufre si un punto tiene 100 muestras y otro tiene 10. SMOTE equilibra el tablero.
smote = SMOTE(k_neighbors=3, random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
print(f"⚖️ Dataset de entrenamiento balanceado con SMOTE. Muestras finales: {X_train_res.shape[0]}")

# =====================================================================
# 6. BÚSQUEDA AUTOMÁTICA DE HIPERPARÁMETROS (GridSearchCV)
# =====================================================================
# Definimos el clasificador base sin parámetros fijos
knn_base = KNeighborsClassifier()

# Cuadrícula de parámetros a evaluar de forma cruzada
param_grid = {
    'n_neighbors': [3, 5, 7],                         # Números impares para evitar empates
    'weights': ['uniform', 'distance'],               # Peso plano vs Peso ponderado por cercanía
    'metric': ['manhattan', 'euclidean', 'chebyshev'] # Diferentes métricas de distancia spatio-temporal
}

# Validación cruzada de 5 pliegues estratificada para mitigar variaciones del WiFi
cv_estratificado = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("🔍 Buscando la combinación óptima de K-NN mediante validación cruzada...")
grid_search = GridSearchCV(
    estimator=knn_base, 
    param_grid=param_grid, 
    cv=cv_estratificado, 
    scoring='accuracy', 
    n_jobs=-1  # Usa toda la potencia de procesamiento disponible
)

# Ajustamos usando los datos balanceados por SMOTE
grid_search.fit(X_train_res, y_train_res)

# Extraemos el mejor modelo resultante de la combinación ganadora
knn_optimizado = grid_search.best_estimator_

print("\n🏆 ¡Configuración ganadora encontrada!")
print(grid_search.best_params_)

# =====================================================================
# 7. EVALUACIÓN Y CORRECCIÓN DE MATRIZ
# =====================================================================
# Predecimos con el clasificador optimizado
y_pred = knn_optimizado.predict(X_test)

print("\n" + "="*40)
print("🎯 NUEVOS RESULTADOS DEL MODELO OPTIMIZADO CON GRIDSEARCH")
print("="*40)
# Las clases únicas reales presentes en y_test para un reporte ordenado
clases_unicas = np.unique(np.concatenate((y_test, y_pred)))
print(classification_report(y_test, y_pred, labels=clases_unicas))

# Visualización de la Matriz de Confusión Mejorada
plt.figure(figsize=(14, 10))
cm = confusion_matrix(y_test, y_pred, labels=clases_unicas)

# Usamos las etiquetas reales en los ejes para saber exactamente qué punto es cuál
sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu', 
            xticklabels=clases_unicas, yticklabels=clases_unicas)

plt.title("Matriz de Confusión Optimizada (GridSearch + SMOTE + Fase CSI)", fontsize=14)
plt.xlabel("Predicción (Punto Clasificado)", fontsize=12)
plt.ylabel("Realidad (Punto Real)", fontsize=12)
plt.tight_layout()
plt.show()
