##for spot
import subprocess
import sys
from bs4 import BeautifulSoup
import requests

#Recieve Token to pull API Data
def get_token()-> str:

    command = f'curl -X POST "https://accounts.spotify.com/api/token" \
          -H "Content-Type: application/x-www-form-urlencoded" \
          -d "grant_type=client_credentials&client_id=c92544725ea24f009b2abc7fa2c38bdb&client_secret=2a14acc2799947f2be63138957c1b31f"'  # the terminal command you want to run

    output = subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT)

    length = None
    while length != 116:
        output = subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT)
        code = str(output[333:-42])[3:]
        token = code.replace("'", '')
        length = len(token)
    return token

token = get_token()

#Webscrape Billboard 100 Data from Wiki
response = requests.get('https://www.billboard.com/charts/hot-100/')

soup = BeautifulSoup(response.content, 'html.parser')
#soup.prettify()
chart_rows = soup.find_all('li')
li = soup.find_all('li')
h3s = [h3 for l in li for h3 in l.find_all('h3', {'id' : "title-of-a-story"})]
top100 = []
cnt=0
for h3 in h3s:
    cnt+=1
    top100.append(h3.text.strip())
top100 = top100[::2]

#Scrape top 100 Artists
artists_ls = []
for val in chart_rows:
    # Find the span element with the given class and print its text
    span_elem = val.find('span', {'class': "c-label a-no-trucate a-font-primary-s lrv-u-font-size-14@mobile-max u-line-height-normal@mobile-max u-letter-spacing-0021 lrv-u-display-block a-truncate-ellipsis-2line u-max-width-330 u-max-width-230@tablet-only u-font-size-20@tablet"})
    if span_elem:
        artists_ls.append(span_elem.text.strip())
        
for val in chart_rows:
    # Find the span element with the given class and print its text
    span_elem = val.find('span', {'class': "c-label a-no-trucate a-font-primary-s lrv-u-font-size-14@mobile-max u-line-height-normal@mobile-max u-letter-spacing-0021 lrv-u-display-block a-truncate-ellipsis-2line u-max-width-330 u-max-width-230@tablet-only"})
    if span_elem:
        artists_ls.append(span_elem.text.strip())
        
artists_ls = artists_ls[::2]

#GET SPOTIFY ID FOR SONGS TO FEED API
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

client_id = "c92544725ea24f009b2abc7fa2c38bdb"
client_secret = "2a14acc2799947f2be63138957c1b31f"

client_credentials_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
spotify = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

track_ids = []

for idx in range(len(top100)):
    query = top100[idx]
    results = spotify.search(q=query, type='track')

    track_id = results['tracks']['items'][0]['id']
    track_ids.append(track_id)

#Connect to API and query Results
import requests
import csv

dnc_ls, energy_ls, loud_ls, live_ls, tempo_ls, time_ls = [], [], [], [], [], []
# Spotify API endpoint for Tame Impala
for idx in range(len(track_ids)):

    url = f'https://api.spotify.com/v1/audio-features/{track_ids[idx]}'

    # Spotify authentication token
    token = f'Bearer  {get_token()}'

    headers = {'Authorization': token}
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        data_dict = response.json()
        dnc_ls.append(data_dict['danceability'])
        energy_ls.append(data_dict['energy'])
        loud_ls.append(data_dict['loudness'])
        live_ls.append(data_dict['liveness'])
        tempo_ls.append(data_dict['tempo'])
        time_ls.append(data_dict['duration_ms'])
        
import pandas as pd
songs_df = pd.DataFrame({'Artist': artists_ls, 'Song': top100, 'Danceability': dnc_ls,
                        'Energy': energy_ls, 'Loudness': loud_ls, 'Liveness': live_ls, 'Tempo': tempo_ls,
                        'Duration': time_ls})

songs_df['Loudness'] = abs(songs_df['Loudness'])
songs_df['Loudness'] = songs_df['Loudness'].map(lambda i: 1 if i>=10 else i/10)
print(songs_df.head(3))

#Analysis + Modeling
import plotly.graph_objects as go
import plotly.io as pio


fig = go.Figure()
colors = ['forestgreen', 'seagreen','lightslategrey', 'gold']

for idx in range(len(songs_df.columns[2:-2])):
    fig.add_trace(go.Bar(x=songs_df.head(5)['Song'], y=songs_df.head(5)[songs_df.columns[2:-2][idx]],
                        base=0,
                        marker_color=colors[idx],
                        name=songs_df.columns[2:-2][idx]))

fig.update_layout(title='Top 5 Billboard Tracks',
                  yaxis_title='Score',
                  xaxis=dict(title='Song Name', tickangle=-45),
                  xaxis2=dict(title='Artist Name', tickangle=-45,
                              overlaying='x', side='bottom'))

fig.show()
pio.write_image(fig, 'Top5_Billboard.png')