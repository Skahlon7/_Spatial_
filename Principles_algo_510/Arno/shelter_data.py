import pandas as pd
import requests
from bs4 import BeautifulSoup

def shelter_scrape():
    url = 'https://www.lapl.org/homeless-resources'
    response = requests.get(url)

    soup = BeautifulSoup(response.content, 'html.parser')

    tag = soup.select('.hrc:nth-child(3)')

    tag_list = []
    for i in range (len(tag)):
        tag_list.append(tag[i].text.strip())
        
    addy_ls, phone_ls = [], []

    for idx in range(len(tag_list)):
        try:
            addy_ls.append(tag_list[idx].split('|')[0].rstrip())
            phone_ls.append(tag_list[idx].split('|')[1])

        except:
            continue
    
    shelter_df = pd.DataFrame(list(zip(addy_ls, phone_ls)),
                         columns = ['name', 'phone'])
    zipcode_ls = [val[-6:].strip() for val in shelter_df.name]
    shelter_df['zipcode'] = zipcode_ls
    shelter_df = shelter_df[shelter_df['zipcode'] != 'Women']
    
    shelter_df1 =shelter_df.copy()
    shelter_df1['zipcode'] = shelter_df1['zipcode'].astype(int)
    
    return shelter_df1

