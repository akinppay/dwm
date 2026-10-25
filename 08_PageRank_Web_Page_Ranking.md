# PageRank Web Page Ranking

**Aim:** Calculate and display only the ranked web-page importance table.

**Jupyter setup (run once):**

```python
%pip install pandas numpy matplotlib seaborn scipy scikit-learn mlxtend networkx
```

**Code:**

```python
import pandas as pd
import networkx as nx

# 1. Load dataset
df = pd.read_csv("webpages.csv")  # Change filename

# 2. Create directed graph
G = nx.from_pandas_edgelist(
    df, source=df.columns[0], target=df.columns[1], create_using=nx.DiGraph()
)

# 3. Apply PageRank
rank = nx.pagerank(G, alpha=0.85)

# 4. Display rankings
result = pd.DataFrame(rank.items(), columns=["Web Page", "PageRank Score"])
result = result.sort_values("PageRank Score", ascending=False)
print("Web Page Rankings:")
display(result)
```
