#!/usr/bin/env python
# coding: utf-8

# In[11]:


import pandas as pd

# Load dataset
df = pd.read_excel("fully_encoded_dataset_final.xlsx")

# === Colonna 3. Percezione consenso scientifico ===
sci_consensus = {
    "Most scientists think global warming is happening": 3,
    "There is a lot of disagreement among scientists about whether or not global warming is happening": 2,
    "Most scientists think global warming is NOT happening": 1,
    "Most scientists think global warming is not happening": 1,
    "Don’t know enough to say": 0
}
df["3. Which comes closest to your own view about what most scientists think?"] = df["3. Which comes closest to your own view about what most scientists think?"].map(sci_consensus)

# === 28–29: Politiche energetiche → accordo/supporto Likert 1–4 ===
likert_support4 = {
    "Strongly disagree": 1, "Somewhat disagree": 2,
    "Somewhat agree": 3, "Strongly agree": 4,
    "Strongly support": 4, "Somewhat support": 3,
    "Somewhat oppose": 2, "Strongly oppose": 1
}
for i in [28, 29]:
    for col in df.columns:
        if col.startswith(f"{i}. "):
            df[col] = df[col].map(likert_support4)

# === 30–31: Frequenza informativa ===
freq_map = {
    "Never": 1, "Rarely": 2, "Occasionally": 3, "Often": 4,
    "Once a year or less often": 1, "Several times a year": 2,
    "At least once a month": 3, "At least once a week": 4
}
for i in [30, 31]:
    for col in df.columns:
        if col.startswith(f"{i}. "):
            df[col] = df[col].map(freq_map)

# === 32: Severità personale ===
severity_map = {
    "Not severe at all": 1, "Per niente gravi": 1,
    "Slightly severe": 2, "Poco gravi": 2,
    "Moderately severe": 3, "Moderatamente gravi": 3,
    "Very severe": 4, "Molto gravi": 4,
    "Don't know": 0, "Don't know / Not applicable": 0
}
df["32. How severe would you rate the personal effects of global warming that you’ve experienced?"] = df["32. How severe would you rate the personal effects of global warming that you’ve experienced?"].map(severity_map)

# === 1: Worry costi politiche ===
worry_map = {
    "Not at all concerned": 1, "Not at all worried": 1,
    "Little concerned": 2, "Not very concerned": 2, "Not very worried": 2,
    "Somewhat concerned": 3, "Somewhat worried": 3,
    "Very concerned": 4, "Very worried": 4,
    "Don't know": 0
}
df["1.  How concerned are you that policies to address global warming will increase your cost of living?"] = df["1.  How concerned are you that policies to address global warming will increase your cost of living?"].map(worry_map)

# === 2: Importanza innovazione ===
importance_map = {
    "Not important at all": 1, "Poco importante": 1,
    "Slightly important": 2,
    "Moderately important": 3, "Moderatamente importante": 3,
    "Very important": 4, "Molto importante": 4,
    "Extremely important": 5, "Estremamente importante": 5
}
df["2. How important is it for you that your country is at the forefront of developing new technologies and innovations?"] = df["2. How important is it for you that your country is at the forefront of developing new technologies and innovations?"].map(importance_map)

# === 3: Patriotismo ===
patriot_map = {
    "Strongly disagree": 1, "Somewhat disagree": 2,
    "Neither agree nor disagree": 3,
    "Somewhat agree": 4, "Strongly agree": 5,
    "Completamente d'accordo": 5
}
df["3.  Do you agree or disagree with the statement: True patriotism means leaving the country better than you found it."] = df["3.  Do you agree or disagree with the statement: True patriotism means leaving the country better than you found it."].map(patriot_map)

# === 4: Politica/società ===
political_map = {
    "Very conservative": 1, "Conservatore": 1,
    "Slightly conservative": 2, "Leggermente conservatore": 2,
    "Moderate / Middle of the road": 3, "Moderato / neutrale": 3,
    "Slightly liberal": 4, "Leggermente liberale": 4,
    "Liberal": 5, "Liberale": 5,
    "Very liberal": 6, "Molto liberale": 6,
    "Prefer not to answer": 0
}
df["4.  On social and political issues, where would you place yourself?"] = df["4.  On social and political issues, where would you place yourself?"].map(political_map)

# === 5: Struttura sociale ideale ===
hierarchy_map = {
    "Strongly oppose": 1, "Fortemente contrario": 1,
    "Somewhat oppose": 2,
    "Neutral": 3, "Neutrale": 3,
    "Somewhat favor": 4, "Somewhat support": 4,
    "Strongly favor": 5, "Fortemente favorevole": 5,
    "Somewhat agree": 4, "Somewhat disagree": 2
}
df["5. An ideal society requires some groups to be on top and others to be on the bottom."] = df["5. An ideal society requires some groups to be on top and others to be on the bottom."].map(hierarchy_map)

# === 6: Ottimismo futuro ===
optimism_map = {
    "Very pessimistic": 1, "Abbastanza pessimista": 2,
    "Somewhat pessimistic": 2,
    "Neither optimistic nor pessimistic": 3, "Né ottimista né pessimista": 3,
    "Somewhat optimistic": 4, "Abbstanza ottimista": 4,
    "Very optimistic": 5, "Molto ottimista": 5
}
df["6. Do you feel optimistic or pessimistic about the future?"] = df["6. Do you feel optimistic or pessimistic about the future?"].map(optimism_map)

# === 7: Politiche immigrazione ===
immigration_map = {
    "Reduced": 1, "Diminuire": 1,
    "Remain the same": 2, "ReNevern the same": 2, "Rimanere lo stesso": 2,
    "Increased": 3, "Aumentare": 3
}
df["7. Do you think the number of immigrants to your country should be increased, reduced, or reNevern the same?"] = df["7. Do you think the number of immigrants to your country should be increased, reduced, or reNevern the same?"].map(immigration_map)

# === 4: Fonti informative (semplificate) ===
info_sources_map = {
    "Social network": 1, "Internet": 2,
    "Newspapers": 3, "Giornali": 3,
    "Tv / radio": 4, "Televisione / radio": 4,
    "Scientific papers": 5, "Articoli scientifici": 5,
    "Broadcasting": 6, "Other": 0, "Altro": 0
}
df["4. Which are your Nevern sources of information?"] = df["4. Which are your Nevern sources of information?"].map(info_sources_map)

# === 5: Igiene personale ===
hygiene_map = {
    "Water": 1, "Acqua": 1,
    "Paper": 2, "Carta igienica": 2,
    "Both": 3, "Entrambi": 3,
    "Other": 0
}
df["5. When ‘’nature calls’’ what do you use for your hygiene?"] = df["5. When ‘’nature calls’’ what do you use for your hygiene?"].map(hygiene_map)

# Save
df.to_excel("fully_encoded_dataset_complete.xlsx", index=False)
print("✅ Final encoding complete → saved as fully_encoded_dataset_complete.xlsx")


# In[12]:


import pandas as pd

# Load the final encoded dataset
df = pd.read_excel("fully_encoded_dataset_complete.xlsx")

# Show the shape and first rows
print("Rows and columns:", df.shape)
df.head(10)


# In[13]:


# Mostra le colonne ancora non numeriche
non_numeric_cols = df.select_dtypes(exclude='number')
for col in non_numeric_cols.columns:
    print(f"\n🔍 {col}")
    print(df[col].dropna().unique())

