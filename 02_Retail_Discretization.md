# Retail Customer Discretization

**Aim:** Apply equal-width binning to numeric customer attributes, visualize distributions, and count segments.

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

df = pd.read_csv("retail.csv")  # Change filename
display(df.head())
display(df.describe())
num = [c for c in df.select_dtypes(include=np.number).columns
       if df[c].nunique() > 1 and 'id' not in c.lower()]
df[num] = df[num].fillna(df[num].median())
for c in num:
    df[c + "_Bin"] = pd.cut(df[c], bins=3, labels=["Low", "Medium", "High"])
display(df.head())
if num:
    df[num].hist(figsize=(12, 8), bins=10)
    plt.tight_layout(); plt.show()
for c in num:
    sns.countplot(data=df, x=c + "_Bin", order=["Low", "Medium", "High"])
    plt.title(c + " - Customer Distribution")
    plt.show()
    print(c, "\n", df[c + "_Bin"].value_counts())
df.to_csv("retail_processed.csv", index=False)
print("Experiment Completed Successfully!")
```
