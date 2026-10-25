# K-Means Telecom Customer Segmentation

**Aim:** Group customers into three clusters, display cluster centroids and a cluster-profile heatmap.

**Jupyter setup (run once):**

```python
%pip install pandas numpy matplotlib seaborn scipy scikit-learn mlxtend networkx
```

**Code:**

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA

df = pd.read_csv("telecom.csv")  # Change filename
display(df.head())
features = df.select_dtypes(include="number")
features = features.loc[:, ~features.columns.str.lower().str.contains("id|cluster")]
features = features.loc[:, features.nunique() > 1]
X = SimpleImputer(strategy="median").fit_transform(features)
X = StandardScaler().fit_transform(X)
model = KMeans(n_clusters=3, random_state=42, n_init=10)
df["Cluster"] = model.fit_predict(X)
print("Customers in Each Cluster:\n", df["Cluster"].value_counts().sort_index())
display(df.groupby("Cluster")[features.columns].mean())
pca = PCA(n_components=2)
points = pca.fit_transform(X)
centers = pca.transform(model.cluster_centers_)
plt.figure(figsize=(10, 6))
for i, color in enumerate(["red", "blue", "green"]):
    mask = df["Cluster"].to_numpy() == i
    plt.scatter(points[mask, 0], points[mask, 1], c=color,
                label=f"Cluster {i}", alpha=.35, s=10)
plt.scatter(centers[:, 0], centers[:, 1], c="yellow", marker="X",
            s=250, edgecolors="black", label="Centroids")
plt.title("K-Means Customer Segmentation")
plt.xlabel("Principal Component 1"); plt.ylabel("Principal Component 2")
plt.legend(); plt.show()
means = features.groupby(df["Cluster"]).mean()
scaled = (means - features.mean()) / features.std().replace(0, 1)
sns.heatmap(scaled, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Customer Behaviour by Cluster"); plt.show()
df.to_csv("telecom_clusters.csv", index=False)
print("Experiment Completed Successfully!")
# Overlapping PCA clusters may be genuine; colors do not guarantee separation.
```
