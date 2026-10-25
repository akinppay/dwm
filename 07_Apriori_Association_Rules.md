# Apriori Frequent Itemsets and Association Rules

**Aim:** Find frequently purchased itemsets and association rules from one transaction per row.

**Jupyter setup (run once):**

```python
%pip install pandas numpy matplotlib seaborn scipy scikit-learn mlxtend networkx
```

**Code:**

```python
import pandas as pd
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori, association_rules

df = pd.read_csv("supermarket.csv", header=None)  # Change filename
# Assumes one transaction per row, with items in separate columns.
df.columns = df.columns.astype(str)
display(df.head())
df = df.dropna(how="all")
basket = pd.get_dummies(df.astype("string").fillna(""), prefix="", prefix_sep="")
basket = basket.T.groupby(level=0).max().T
basket = basket.drop(columns=[""], errors="ignore").astype(bool)
items = apriori(basket, min_support=0.02, use_colnames=True)
print("Frequent Itemsets:")
display(items.sort_values("support", ascending=False).head(10))
if not items.empty:
    rules = association_rules(items, metric="confidence", min_threshold=0.3)
    print("Association Rules:")
    display(rules.sort_values("lift", ascending=False).head(10))
else:
    rules = pd.DataFrame()
    print("No frequent itemsets found")
basket.sum().sort_values(ascending=False).head(10).plot(
    kind="bar", figsize=(10, 5), color="skyblue")
plt.title("Top 10 Frequently Purchased Items")
plt.xlabel("Items"); plt.ylabel("Frequency")
plt.xticks(rotation=45, ha="right"); plt.tight_layout(); plt.show()
rules.to_csv("association_rules.csv", index=False)
print("Experiment Completed Successfully!")
# If the CSV has a header, remove header=None.
```
