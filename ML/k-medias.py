import os
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.feature_selection import VarianceThreshold
import matplotlib.pyplot as plt


base_path = "/home/user/Desktop/CSI_Procesado/SNIFFER"
folders = [("laboratorio", 0), ("pasillo", 1)]

X_list = []
y_list = []


for folder_name, label in folders:
    path = os.path.join(base_path, folder_name, "training")
    if not os.path.exists(path): continue
    
    for file in os.listdir(path):
        if file.endswith(".txt"):
            file_path = os.path.join(path, file)
            data = pd.read_csv(file_path, header=None)
            X_list.append(data.values)
            y_list.extend([label] * len(data))

X = np.vstack(X_list)
y = np.array(y_list)


imputer = SimpleImputer(strategy='constant', fill_value=0)
X = imputer.fit_transform(X)


selector = VarianceThreshold(threshold=0) 
X = selector.fit_transform(X)
print(f"Subportadoras útiles restantes: {X.shape[1]}")


normalizer = Normalizer(norm='l2')
X_norm = normalizer.fit_transform(X)

scaler = StandardScaler()
X_final = scaler.fit_transform(X_norm)


pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_final)


kmeans = KMeans(n_clusters=2, random_state=42)
labels = kmeans.fit_predict(X_pca)


plt.figure(figsize=(10, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', alpha=0.5)
plt.title("Clusters K-means (CSI Normalizado)")
plt.xlabel(f"PCA 1 ({pca.explained_variance_ratio_[0]:.2%} varianza)")
plt.ylabel(f"PCA 2 ({pca.explained_variance_ratio_[1]:.2%} varianza)")
plt.colorbar(label='Cluster ID')
plt.show()
