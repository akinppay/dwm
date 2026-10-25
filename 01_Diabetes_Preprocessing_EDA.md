# Diabetes Data Preprocessing and EDA

**Aim:** Perform missing-value handling, descriptive statistics, outlier capping, normalization, transformation, and visualization.

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
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv("diabetes.csv")  # Change filename
display(df.head())
df = df.replace([np.inf, -np.inf], np.nan)
num = df.select_dtypes(include=np.number).columns
cat = df.select_dtypes(exclude=np.number).columns
# In a diabetes dataset, zero glucose/BMI/etc. may also mean missing.
df[num] = df[num].fillna(df[num].median())
for c in cat:
    df[c] = df[c].fillna(df[c].mode()[0] if not df[c].mode().empty else "Unknown")
print("Missing values:\n", df.isna().sum())
display(df.describe(include="all"))
# Exclude outcome/target columns from numeric transformations.
features = [c for c in num if c.lower() not in ("outcome", "target", "label")]
for c in features:
    q1, q3 = df[c].quantile([.25, .75])
    iqr = q3 - q1
    df[c] = df[c].clip(q1 - 1.5*iqr, q3 + 1.5*iqr)
if features:
    df[features] = MinMaxScaler().fit_transform(df[features])
    df[features] = np.log1p(df[features])
    df[features].hist(figsize=(12, 8))
    plt.tight_layout(); plt.show()
    sns.heatmap(df[features].corr(), annot=True, cmap="coolwarm")
    plt.show()
    df[features].plot(kind="box", figsize=(12, 5), rot=45)
    plt.show()
df.to_csv("processed_data.csv", index=False)
print("Experiment Completed Successfully!")
```
