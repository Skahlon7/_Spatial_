#!/usr/bin/env python
# coding: utf-8

# In[2]:


import argparse
from bs4 import BeautifulSoup
import requests
import pandas as pd
import numpy as np
import os
import sys

import gdp_api
import vehicle_scrape
import emission_data

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Scrapes data from a website.')
    #parser.add_argument('source', choices=['GDP','Vehicles','Emission'],help='Choose which data to load')
    parser.add_argument('--scrape', type=int, required=False, help='Number of pages to scrape')
    parser.add_argument('--save', type=str, required=False, help='File path to save the scraped data')
    args = parser.parse_args()

    # Use the scraped value from the command line
    n_pages_to_scrape = args.scrape

    # Use the save path from the command line
    save_path = args.save
    df = vehicle_scrape.vehicle_scape()
    if args.scrape is not None:
        print(df.head(n_pages_to_scrape))
    elif args.save is not None:
        df.to_csv(save_path,index=False)
    else:
        print(df)
