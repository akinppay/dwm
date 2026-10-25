# Hierarchical Agglomerative Clustering

**Aim:** Sample at most 500 records to avoid memory errors, plot a dendrogram, and inspect three clusters.

**Jupyter setup (run once):**

```python
%pip install pandas numpy matplotlib seaborn scipy scikit-learn mlxtend networkx
```

**Code:**

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA

df = pd.read_csv("biological.csv")  # Change filename
display(df.head())
X = df.select_dtypes(include="number")
X = X.loc[:, ~X.columns.str.lower().str.contains("id|cluster")]
X = X.loc[:, X.nunique() > 1]
df = df.sample(n=min(500, len(df)), random_state=42).copy()
X = X.loc[df.index]
X = SimpleImputer(strategy="median").fit_transform(X)
X = StandardScaler().fit_transform(X)
Z = linkage(X, method="ward")
plt.figure(figsize=(12, 5))
dendrogram(Z, truncate_mode="lastp", p=20)
plt.title("Hierarchical Clustering Dendrogram")
plt.xlabel("Sample Groups"); plt.ylabel("Distance"); plt.show()
model = AgglomerativeClustering(n_clusters=3, linkage="ward")
df["Cluster"] = model.fit_predict(X)
print("Samples in Each Cluster:\n", df["Cluster"].value_counts().sort_index())
display(df.groupby("Cluster").mean(numeric_only=True))
points = PCA(n_components=2).fit_transform(X)
sns.scatterplot(x=points[:, 0], y=points[:, 1], hue=df["Cluster"],
                palette="Set1", s=50, alpha=.7)
plt.title("Hierarchical Clustering of Biological Samples"); plt.show()
df.to_csv("biological_clusters.csv", index=False)
print("Experiment Completed Successfully!")
# Clustering is performed on the sampled records only.
```
