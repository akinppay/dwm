# Decision Tree (ID3-style) Purchase Prediction

**Aim:** Use entropy-based Decision Tree classification to predict customer purchases and measure performance.

**Jupyter setup (run once):**

```python
%pip install pandas numpy matplotlib seaborn scipy scikit-learn mlxtend networkx
```

**Code:**

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

df = pd.read_csv("ecommerce.csv")  # Change filename
display(df.head())
target = "Purchased" if "Purchased" in df.columns else df.columns[-1]
X = df.drop(columns=[target])
X = X.loc[:, ~X.columns.str.lower().str.contains("id")].copy()
y = df[target]
for c in X.columns:
    X[c] = X[c].fillna(X[c].mode()[0] if not X[c].mode().empty else "Unknown")
X = pd.get_dummies(X)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
model = DecisionTreeClassifier(criterion="entropy", max_depth=4, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Accuracy:", round(accuracy_score(y_test, y_pred)*100, 2), "%")
print(classification_report(y_test, y_pred, zero_division=0))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix"); plt.show()
plt.figure(figsize=(15, 7))
plot_tree(model, feature_names=X.columns,
          class_names=[str(c) for c in model.classes_], filled=True, rounded=True, fontsize=8)
plt.show()
print("Customer Purchase Prediction:", model.predict(X_test.iloc[:1])[0])
# Entropy criterion is ID3-style, not the original ID3 implementation.
```
