#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# STEP 1: Load dataset
df = pd.read_excel("fully_encoded_dataset_complete.xlsx")

# STEP 2: Select only numeric columns
df_numeric = df.select_dtypes(include='number')

# STEP 3: Compute correlation matrix
corr_matrix = df_numeric.corr()

# STEP 4: Plot heatmap
plt.figure(figsize=(18, 14))
sns.heatmap(
    corr_matrix,
    cmap='coolwarm',
    annot=False,
    fmt=".2f",
    linewidths=0.5
)
plt.title("Correlation Matrix — Climate Change Survey", fontsize=16)
plt.tight_layout()
plt.show()


# In[4]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# STEP 1 — Load dataset
df = pd.read_excel("fully_encoded_dataset_complete.xlsx")

# STEP 2 — Select numeric columns only
df_numeric = df.select_dtypes(include='number')

# STEP 3 — Compute full correlation matrix
corr_matrix = df_numeric.corr()

# STEP 4 — General heatmap
plt.figure(figsize=(18, 14))
sns.heatmap(corr_matrix, cmap='coolwarm', annot=False, fmt=".2f", linewidths=0.5)
plt.title("Correlation Matrix — Climate Survey", fontsize=16)
plt.tight_layout()
plt.show()

# STEP 5 — Top 20 absolute correlations (excluding diagonal)
top_corr = (
    corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    .stack()
    .reset_index()
    .rename(columns={'level_0': 'Variable 1', 'level_1': 'Variable 2', 0: 'Correlation'})
    .assign(AbsCorr=lambda x: x['Correlation'].abs())
    .sort_values(by='AbsCorr', ascending=False)
    .head(20)
)

print("\n Top 20 Correlations:")
print(top_corr)

# STEP 6 — Grouped heatmaps (by thematic blocks)
group_blocks = {
    "Climate Concern": [col for col in df_numeric.columns if "harm" in col.lower() or "worry" in col.lower()],
    "Policy Support": [col for col in df_numeric.columns if "should" in col.lower() or "support" in col.lower()],
    "Personal Behavior": [col for col in df_numeric.columns if "how often" in col.lower()],
    "Political Views": [col for col in df_numeric.columns if "patriotism" in col.lower() or "optimistic" in col.lower() or "politic" in col.lower()]
}

for group_name, cols in group_blocks.items():
    if len(cols) >= 2:
        plt.figure(figsize=(10, 8))
        sns.heatmap(df_numeric[cols].corr(), cmap='coolwarm', annot=True, fmt=".2f", linewidths=0.5)
        plt.title(f"Correlation Matrix — {group_name}", fontsize=14)
        plt.tight_layout()
        plt.show()


# In[ ]:




