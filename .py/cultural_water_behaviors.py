#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd

# STEP 1 — Load the dataset
df = pd.read_excel("fully_encoded_dataset_complete.xlsx")

# STEP 2 — Create Cultural_Group from Region and Country
def map_cultural_group(row):
    country = str(row['What is your country of origin?']).strip().lower()
    region = str(row['Region']).strip().lower()

    if 'italy' in country:
        if any(r in region for r in ['Lombardia', 'Veneto', 'Friuli-Venezia Giulia', 'Trentino-Alto Adige / Trentino-Sudtirol', 'Piemonte', 'Emilia-Romagna', 'Liguria']):
            return 'Italy_North'
        elif any(r in region for r in ['Toscana', 'Lazio', 'Umbria', 'Marche', 'Sardegna']):
            return 'Italy_Center'
        elif any(r in region for r in ['Abruzzo', 'Molise', 'Campania', 'Puglia', 'Basilicata', 'Calabria', 'Sicilia']):
            return 'Italy_South'
        else:
            return 'Italy_Other'
    else:
        return 'Other_Countries'

df['Cultural_Group'] = df.apply(map_cultural_group, axis=1)

# STEP 3 — Rename columns for clarity
rename_map = {
    '2.  I turn off water while shampooing or soaping my body.': 'Turn_Off_Shower',
    '3.   How often do you usually run the washing machine with just a few items inside even if it isn’t completely full?': 'Washing_Half_Load',
    '5. When ‘’nature calls’’ what do you use for your hygiene?': 'Hygiene_Method',
    '6. Do you reuse the water that exits from your AC (Air Conditioner)?': 'Reuse_AC_Water',
    '9. Do you let the water run while brushing your teeth?': 'Teeth_Water_Run'
}
df = df.rename(columns=rename_map)

# STEP 4 — Transform to tidy format
relevant_cols = ['Cultural_Group'] + list(rename_map.values())
df_tidy = df[relevant_cols].melt(id_vars='Cultural_Group',
                                  var_name='Water_Behavior',
                                  value_name='Response')

# STEP 5 — Count frequencies
behavior_summary = df_tidy.groupby(['Cultural_Group', 'Water_Behavior', 'Response']).size().reset_index(name='Count')

# STEP 6 — Export to Excel
behavior_summary.to_excel("water_behavior_tidy_summary.xlsx", index=False)
print("✅ Tidy summary saved as: water_behavior_tidy_summary.xlsx")


# In[ ]:




