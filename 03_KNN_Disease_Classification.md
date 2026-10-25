# KNN Disease Classification

**Aim:** Train and test a KNN classifier and evaluate its accuracy and confusion matrix.

**Jupyter setup (run once):**

```python
%pip install pandas numpy matplotlib seaborn scipy scikit-learn mlxtend networkx
```

**Code:**

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

df = pd.read_csv("diabetes.csv")  # Change filename
display(df.head())
target = "Outcome" if "Outcome" in df.columns else df.columns[-1]
X = df.drop(columns=[target]).select_dtypes(include=np.number)
X = X.loc[:, ~X.columns.str.lower().str.contains("id")]
y = df[target]
valid = y.notna()
X, y = X.loc[valid], y.loc[valid]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
imputer = SimpleImputer(strategy="median")
scaler = StandardScaler()
X_train = scaler.fit_transform(imputer.fit_transform(X_train))
X_test = scaler.transform(imputer.transform(X_test))
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
print("Accuracy:", round(accuracy_score(y_test, y_pred)*100, 2), "%")
print(classification_report(y_test, y_pred, zero_division=0))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues")
plt.title("KNN Confusion Matrix"); plt.show()
print("Test Patient Prediction:", knn.predict(X_test[:1])[0])
```
