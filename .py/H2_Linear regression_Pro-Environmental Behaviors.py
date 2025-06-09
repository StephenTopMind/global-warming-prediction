#!/usr/bin/env python
# coding: utf-8

# In[6]:


import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# STEP 1 — Load the full encoded dataset
df = pd.read_excel("fully_encoded_dataset_complete.xlsx")

# STEP 2 — Select columns for trust in institutions (Q20–Q25) and personal responsibility (Q34)
trust_cols = [
    '20. Do you think the following should be doing more or less to address global warming? Corporations and industry.',
    '21. The head of state (president, monarch,...) and the head of government (chancellor, first minister,...)',
    '22. The parliament/assembly/congress legislature of your country',
    '23. The governor (of your administrative region, province, state, county...)',
    '24. Your local officials (the mayor and the city councilmembers)',
    '25. The citizens themselves',
    '1. Do you think that global warming is happening?',
    '5. How worried are you about global warming?'
]

responsibility_col = '34. How responsible do you feel personally for helping reduce the effects of global warming on future generations?'

# STEP 3 — Prepare the data
df_model = df[trust_cols + [responsibility_col]].dropna()
df_model['Institutional_Trust_Score'] = df_model[trust_cols].mean(axis=1)

X = df_model[['Institutional_Trust_Score']]
y = df_model[responsibility_col]

# STEP 4 — Linear Regression
model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)

# STEP 5 — Evaluation
print("--- Linear Regression: Institutional Trust → Responsibility ---")
print("R^2 Score:", r2_score(y, y_pred))
print("Coefficient:", model.coef_[0])
print("Intercept:", model.intercept_)

# STEP 6 — Visualization
plt.figure(figsize=(6, 4))
sns.regplot(x='Institutional_Trust_Score', y=responsibility_col, data=df_model, ci=None, line_kws={"color": "red"})
plt.title("Institutional Trust vs. Responsibility")
plt.xlabel("Institutional Trust Score")
plt.ylabel("Responsibility (Q34)")
plt.tight_layout()
plt.show()


# In[ ]:





# In[ ]:




