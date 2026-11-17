import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

os.makedirs("results",exist_ok=True)

# B1 - Load the Wine dataset
wine=load_wine(as_frame=True)

X=wine.data
cultivar=wine.target

print(wine.DESCR)

print("\nDataset shape:")
print(X.shape)

print("\nDataset description:")
print(X.describe())

print("\nFeature names:")
print(X.columns.tolist())

# B2 - PCA on raw data
pca_raw=PCA()
pca_raw.fit(X)

print("\nB2 - Raw PCA")
print("PC1 explained variance ratio:",pca_raw.explained_variance_ratio_[0])

pc1_loadings=pd.Series(pca_raw.components_[0],index=X.columns)

print("\nPC1 loadings sorted:")
print(pc1_loadings.sort_values())

# B3 - Standardize the data
scaler=StandardScaler()
Z=scaler.fit_transform(X)

print("\nB3 - Standardization check")
print("Means:",Z.mean(axis=0))
print("Standard deviations:",Z.std(axis=0))

# PCA on standardized data
pca=PCA()
pca.fit(Z)

print("\nExplained variance ratios:")
print(pca.explained_variance_ratio_)

print("\nCumulative explained variance:")
print(np.cumsum(pca.explained_variance_ratio_))

print("\nPC1 and PC2 loadings:")
loadings=pd.DataFrame(
    pca.components_[:2].T,
    index=X.columns,
    columns=["PC1","PC2"]
)
print(loadings.sort_values("PC1"))

# Scree plot
explained=pca.explained_variance_ratio_
cumulative=np.cumsum(explained)

plt.figure(figsize=(9,5))
plt.bar(range(1,14),explained,label="Individual variance")
plt.plot(range(1,14),cumulative,marker="o",color="red",
         label="Cumulative variance")
plt.axhline(y=0.9,color="green",linestyle="--",label="90%")
plt.xlabel("Principal Component")
plt.ylabel("Explained Variance Ratio")
plt.title("PCA Scree Plot")
plt.xticks(range(1,14))
plt.legend()
plt.tight_layout()
plt.savefig("results/B_scree_plot.png")
plt.close()

# PC1-PC2 scatter plot
Z_pca=pca.transform(Z)

plt.figure(figsize=(8,6))
plt.scatter(Z_pca[:,0],Z_pca[:,1])
plt.xlabel("PC1 (36.20%)")
plt.ylabel("PC2 (19.21%)")
plt.title("Wine Samples: PC1 vs PC2")
plt.tight_layout()
plt.savefig("results/B_pca_scatter.png")
plt.close()

# B4 - Part A: Different random states
print("\nB4 - Part A: Random initialization")
for seed in range(8):
    km=KMeans(
        n_clusters=3,
        init="k-means++",
        n_init=1,
        max_iter=300,
        random_state=seed
    )
    km.fit(Z)
    print("Random state:",seed,"Inertia:",km.inertia_)

# Compare with n_init=10
km=KMeans(
    n_clusters=3,
    init="k-means++",
    n_init=10,
    max_iter=300,
    random_state=0
)
km.fit(Z)
print("\nInertia with n_init=10:",km.inertia_)

# B4 - Part B: Elbow method
inertias=[]
for k in range(1,11):
    km=KMeans(
        n_clusters=k,
        init="k-means++",
        n_init=10,
        max_iter=300,
        random_state=0
    )
    km.fit(Z)
    inertias.append(km.inertia_)
    print("K= ",k,"Inertia= ",km.inertia_)

plt.figure(figsize=(8,5))
plt.plot(range(1,11),inertias,marker="o")
plt.xlabel("Number of clusters (k)")
plt.ylabel("Inertia")
plt.title("KMeans Elbow Method")
plt.xticks(range(1,11))
plt.tight_layout()
plt.savefig("results/B_elbow.png")
plt.close()

# B5 - Cluster sizes, real-unit centres, and coloured PC1-PC2 scatter
km_best=KMeans(n_clusters=3,init="k-means++",n_init=10,max_iter=300,random_state=0)
km_best.fit(Z)

sizes=np.bincount(km_best.labels_)
print("\nB5 - Cluster sizes:",sizes)

centres_real=pd.DataFrame(
    scaler.inverse_transform(km_best.cluster_centers_),
    columns=X.columns
)
print("\nB5 - Cluster centres in real units:")
print(centres_real.round(2).to_string())

plt.figure(figsize=(8,6))
scatter=plt.scatter(Z_pca[:,0],Z_pca[:,1],c=km_best.labels_,cmap="viridis",edgecolors="k",alpha=0.85)
plt.xlabel("PC1 (36.20%)")
plt.ylabel("PC2 (19.21%)")
plt.title("Wine Clusters (k=3) on PC1 vs PC2")
plt.legend(*scatter.legend_elements(),title="Cluster")
plt.tight_layout()
plt.savefig("results/B_clusters_pca.png")
plt.close()

# B6 - The reveal
print("\nB6 - Crosstabs and Agreement Scores")

# (i) Scaled data (13 features)
ct_scaled=pd.crosstab(cultivar,km_best.labels_)
agree_scaled=ct_scaled.max(axis=0).sum()/len(cultivar)
print("\n(i) Scaled K-Means crosstab:")
print(ct_scaled)

# (ii) Raw unscaled data (13 features)
km_raw=KMeans(n_clusters=3,init="k-means++",n_init=10,max_iter=300,random_state=0)
km_raw.fit(X)
ct_raw=pd.crosstab(cultivar,km_raw.labels_)
agree_raw=ct_raw.max(axis=0).sum()/len(cultivar)
print("\n(ii) Raw K-Means crosstab:")
print(ct_raw)

# (iii) First two PCs of scaled data
km_pca2=KMeans(n_clusters=3,init="k-means++",n_init=10,max_iter=300,random_state=0)
km_pca2.fit(Z_pca[:,:2])
ct_pca2=pd.crosstab(cultivar,km_pca2.labels_)
agree_pca2=ct_pca2.max(axis=0).sum()/len(cultivar)
print("\n(iii) PCA-2D K-Means crosstab:")
print(ct_pca2)

agreement_df=pd.DataFrame({
    "Representation":["Scaled (13 features)","Raw unscaled (13 features)","Scaled PCA (PC1 & PC2)"],
    "Agreement":[agree_scaled,agree_raw,agree_pca2]
})
print("\nB6 - Agreement Summary:")
print(agreement_df.round(4).to_string(index=False))

agreement_df.to_csv("results/B_agreement.csv",index=False)