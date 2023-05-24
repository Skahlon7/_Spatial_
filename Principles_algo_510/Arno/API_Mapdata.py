import requests
import pandas as pd

def scrape_coords():
    df = pd.read_csv('./MergedDataset.csv')
    #cleanse for API
    df['name'] = df['name'].map(lambda x: x.replace("#", ""))
    API_key = 'AIzaSyAWXRek5w7WA4qithObH-inNzYL8UYGC0U'
    lat_list, long_list = [], []
    for num in range(len(df)):
        addy = df.name[num].replace(' ', '+')
        link = f'https://maps.googleapis.com/maps/api/geocode/json?address={addy}&key={API_key}'
        response = requests.get(link)
        addy_data = response.json()
        lat_list.append(addy_data['results'][0]['geometry']['location']['lat'])
        long_list.append(addy_data['results'][0]['geometry']['location']['lng'])
    
    coords_df = pd.DataFrame({'Address': list(df.name), 'Lat':lat_list, 'Lon': long_list})
    return coords_df
        