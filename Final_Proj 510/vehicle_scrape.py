#!/usr/bin/env python
# coding: utf-8

# In[5]:


from bs4 import BeautifulSoup
import requests

def vehicle_scape():
    response = requests.get('https://en.wikipedia.org/wiki/List_of_countries_by_vehicles_per_capita')
    response.status_code

    #Parse Wiki Data
    soup = BeautifulSoup(response.content, 'html.parser')
    table = soup.find_all('table')[0]
    headers = [val.text.strip() for val in table.find_all('th')] #Headers

    table_rows = table.find_all('tbody')[0].find_all('tr')  #Aggregate Rows
    row_data = []
    for val in table_rows[1:]:
        for index in range(len(val.find_all('td'))):
            row_data.append(val.find_all('td')[index].text.strip())

    #Convert to DataFrame
    import pandas as pd
    auto_df = pd.DataFrame()
    for col_num in range(len(headers)):
        auto_df[headers[col_num]] = row_data[col_num::4]
    #sorted(auto_df['Country or region'].unique())
    return auto_df


# In[6]:


vehicle_scape()