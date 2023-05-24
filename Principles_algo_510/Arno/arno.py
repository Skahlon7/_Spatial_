import argparse
import pandas as pd
import requests
from bs4 import BeautifulSoup
import shelter_data
import API_Mapdata
import population_data

if __name__ == '__main__':
    parse = argparse.ArgumentParser(description='Program to Scrape Data')
    parse.add_argument('--scrape', type=int, required=False)
    parse.add_argument('--save', type=str, required=False) #add path to save

    arguments = parse.parse_args()

    num = arguments.scrape

    path = arguments.save
    df = shelter_data.shelter_scrape()

    if arguments.scrape is not None:
        print(df.head(num))
    elif arguments.save is not None:
        shelter_data.shelter_scrape().to_csv(path, index=False)
    else:
        print(df)
