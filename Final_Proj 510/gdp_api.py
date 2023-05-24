#!/usr/bin/env python
# coding: utf-8

# In[1]:


# GDP Data from api --> show 20000 per page and match with country name data below --> looks good w country
import requests
import json
import pandas as pd
import numpy as np

def gdp_scrape():
    form = 'json'
    indicator = 'NY.GDP.PCAP.CD'

    url = f'http://api.worldbank.org/v2/country/all/indicator/{indicator}?format={form}&per_page=20000'#&source=2'

    # Send a GET request to the API endpoint and store the response
    response = requests.get(url)

    data = response.json() #Status Code = 200

    country_dic = {}
    for index in range(len(data[1])):
        if data[1][index]['date'] == '2019':
            country_dic[data[1][index]['country']['value']] = [data[1][index]['date'], data[1][index]['value']]
            
    gdp_dic = {}
    for index in range(len(data[1])):
        if data[1][index]['date'] == '2019':
            gdp_dic[data[1][index]['country']['value']] = [data[1][index]['date'], data[1][index]['value']]
            
    gdp_df = pd.DataFrame(gdp_dic.values(), index=gdp_dic.keys(), columns=['Year', 'GDP'])
    gdp_df = gdp_df.reset_index().rename(columns={'index': 'Country'})
    return gdp_df


# In[2]:


gdp_scrape()

