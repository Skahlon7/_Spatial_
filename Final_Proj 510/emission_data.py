#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
def import_data():
    emission_df = pd.read_csv('CO2_emission.csv')
    return emission_df #Relational Column: 'Country Name'


# In[4]:


import_data()

