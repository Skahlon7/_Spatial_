###Yelp Sentiment Analysis (Restaraunts)
import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import folium
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from dmba import printTermDocumentMatrix
import warnings #Can Improve Viz colors,parameters, etc.
warnings.filterwarnings("ignore")
import plotly.graph_objects as go
import plotly.io as pio


if __name__ == "__main__":
    location = input('Enter a City to view restaraunts: ')

    url = 'https://api.yelp.com/v3/businesses/search' #ORIGINAL API QUERY

    # Set the request headers
    headers = {
        'Authorization': 'Bearer 9DSH-f85EF1qiZ-c5A6mS2gH_4S66oGKPPXsUlC0T25VZpDmZ2C04FofnBFX-4knlpxLmH_LEQJkcFMXdBBPmJ39MsFf-PC5SegE-_Xv6Dc-WaFLZmfL89_0LP54ZHYx',
    }

    params = {
        'location': location,
        'term': 'restaurants',
        'limit': 5,  # Number of results to retrieve
    }

    # Send the GET request
    response = requests.get(url, headers=headers, params=params)
    #print(response)
    while response.status_code !=200:
        location = input('Enter a VALID U.S City to view restaraunts: ')
        params = {
        'location': location,
        'term': 'restaurants',
        'limit': 5,  # Number of results to retrieve
        }
        response = requests.get(url, headers=headers, params=params)
    print(response)


    #Retrieve high-level info on business (rating, lat, lon, etc.)
    def bus_info():
        bus_array = np.empty((0, 6))
        json_search = response.json()

        for idx in range(len(json_search['businesses'])):
            row = [
                json_search['businesses'][idx]['name'],
                json_search['businesses'][idx]['id'],
                json_search['businesses'][idx]['rating'],
                json_search['businesses'][idx]['coordinates']['latitude'],
                json_search['businesses'][idx]['coordinates']['longitude'],
                json_search['businesses'][idx]['location']['display_address'][0] + ' ' +
                json_search['businesses'][idx]['location']['display_address'][1]
            ]
            bus_array = np.append(bus_array, [row], axis=0)

        df_businfo = pd.DataFrame(bus_array)

        # Optionally, set column names
        df_businfo.columns = ['Name', 'ID', 'Rating', 'Latitude', 'Longitude', 'Address']
        return df_businfo
    

    #Retrieve Reviews for Each Business via webscrape/API Config
    def get_reviews():
        json_search = response.json()

        bus_names, reviews = [], []
        for idx in range(len(json_search['businesses'])):
            url_bus = json_search['businesses'][idx]['url']
            response1 = requests.get(url_bus)
        #    print(json_search['businesses'][idx]['url'])
        #    print(json_search['businesses'][idx], '\n\n\n') 
            soup = BeautifulSoup(response1.content, 'html.parser')
            li = soup.find_all('li', {'class': 'margin-b5__09f24__pTvws border-color--default__09f24__NPAKY'})

            for idx1 in range(len(li)):
                try:#works for 10 but not all li have this span class
                    reviews.append(li[idx1].find_all('span', {'class': 'raw__09f24__T4Ezm'})[0].text)
                    bus_names.append(json_search['businesses'][idx]['name'])
        #            print('Review Added for {}'.format(json_search['businesses'][idx]['name']))
                except IndexError:
                    continue
        df_yelp = pd.DataFrame(data={'Bus_Name': bus_names, 'Reviews':reviews})
        return df_yelp

    #Spatial Analysis
    bus_df = bus_info()
    map = folium.Map(location=[bus_df['Latitude'].astype(float).mean(), 
                            bus_df['Longitude'].astype(float).mean()], zoom_start=12)
    # Define marker colors based on rating categories
    colors = {
        '4.0': 'orange',
        '4.5': 'green',
        '5.0': 'green',
        '3.5': 'orange',
        '3': 'yellow',
        '2': 'red',
        '1': 'red'
    }

    # Iterate over the rows of the DataFrame
    for index, row in bus_df.iterrows():
        rating = row['Rating']
        lat = row['Latitude']
        lon = row['Longitude']
        name = row['Name']
        addy = row['Address']
        
        # Create a marker with a color based on the rating
        color = colors.get(rating, 'gray')
        marker = folium.Marker([lat, lon], icon=folium.Icon(color=color))
        popup_text = f"<div style='width: 250px;'>Name: {name}<br>Rating: {rating} {'⭐️'*int(float(rating)//1)}<br>Address: {addy}</div>"
        marker = folium.Marker([lat, lon], icon=folium.Icon(color=color), popup=popup_text)
        # Add the marker to the map
        marker.add_to(map)

    map.save('map.html')
    map  

 
    # Word Bubble #
    df_r = get_reviews() #Dataframe for sentiment analysis + WordBubble
    df_r.head()

    #recieve common transition words in English
    response2 = requests.get('https://www.touro.edu/departments/writing-center/tutorials/transitional-words/#:~:text=and%2C%20again%2C%20and%20then%2C,first%20(second%2C%20etc.')

    #connect + scrape  to university website
    soup = BeautifulSoup(response2.content, 'html.parser')
    transition_wrds = soup.find_all('div', {'class': 'body'})[0].text.strip().split(' ')
    transition_wrds = [x[:-1] if ',' in x else x for x in transition_wrds]
    transition_wrds = transition_wrds +  ['we', 'were', 'are', 'our', 'you', 'me', 'had', 'their', 'just',
                                            'out', 'would', 'it', 'with', 't', 's', 'place', 'very', 'there',
                                            'like', 'come', 'also', 'came', 'ordered', 'if', 'no', 'us', 'can',
                                            'when', 'got', 'your', 'around', 'wait', 'will', 'because', 'what',
                                            'every', 'not', 'was', 'back', 'my', 'some', 'one', 'really', 'get',
                                        'only', 'got', ':', ')', 've', 'de']

    #add more words seen + Bus Name

#Iterate and Configure Viz
    for bus in df_r.Bus_Name.unique():
        count_vect = CountVectorizer(token_pattern='[a-zA-Z!,:)]+')
        tfidfTransformer = TfidfTransformer(smooth_idf=False, norm=None)
        counts = count_vect.fit_transform(df_r[df_r.Bus_Name==bus].Reviews)
        tfidf = tfidfTransformer.fit_transform(counts)
    #    printTermDocumentMatrix(count_vect, counts)
    #     print()
    #     print('END {}'.format(bus))
    #     print()
        feature_names = count_vect.get_feature_names()
        df_sent = pd.DataFrame(data=counts.toarray().T, columns=df_r[df_r.Bus_Name==bus].index)
        df_sent = df_sent.set_index(pd.Index(feature_names))
    #    print(df_sent)
        row_sums = df_sent.sum(axis=1) #sum word counts and plot
        word_cnt = {wrd:cnt for wrd,cnt in zip(row_sums.index, row_sums)}

        word_cnt = {k:v for k,v in word_cnt.items() if k.lower() not in [x.lower() for x in transition_wrds]}
        word_cnt = dict(sorted(word_cnt.items(), key=lambda i: i[1], reverse=True))

        # Extract words and their corresponding frequencies from the word_cnt dictionary
        words = list(word_cnt.keys())[:22]
        frequencies = list(word_cnt.values())[:23]

        # Define the data trace
        trace = go.Scatter(
            x=words,
            y=frequencies,
            mode='markers',
            marker=dict(
                size=frequencies,  # Use frequencies as the size of the markers
                sizemode='diameter',
                sizeref=max(frequencies) / 100,  # Adjust the size scale as per your preference
                sizemin=1,  # Minimum marker size
                color=frequencies,  # Use frequencies as the color scale
                colorscale='Viridis',  # Choose a color scale
                showscale=True  # Display the color scale
            )
        )

        # Define the layout
        layout = go.Layout(
            title='{} Word Frequencies in Reviews'.format(bus),
            xaxis=dict(title='Words'),
            yaxis=dict(title='Frequencies')
        )

        # Create the figure
        fig = go.Figure(data=[trace], layout=layout)
        # Display the figure
        fig.show()
        pio.write_html(fig, f'{bus}_WordCount.html')


        

