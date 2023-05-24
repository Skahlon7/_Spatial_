{\rtf1\ansi\ansicpg1252\cocoartf2639
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fswiss\fcharset0 Helvetica-Bold;\f1\fswiss\fcharset0 Helvetica;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\margl1440\margr1440\vieww19440\viewh13740\viewkind0
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural\partightenfactor0

\f0\b\fs36 \cf0 Economic Impact on Co2 Emissions\
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural\partightenfactor0
\cf0 \

\f1\b0\fs24 The project aims to identify global economic disparities and surpluses and it\'92s effects on per capita emissions. Per capita data inclusive of GDP, automobile count, and emissions were extracted from various public web sources via API scraping, web scraping, and CSV download. \
\

\f0\b\fs36 Structure & Configuration
\f1\b0\fs24 \
\
The code initially scrapes 2019 GDP data from the World Bank API source via an obtained API key leveraging the requests library. Further the scraping of automobile count per capita was scraped from Wikipedia leveraging the BeautifulSoup library. Finally, emission per capita data was downloaded from Kaggle and read into the python script via Pandas. All extrapolated data was initialized into a Pandas data frame and merged on the primary key of Country Name. Prior to the merge, the data was cleaned via the re library in order to optimize the number of inner joins due to misspellings or inconsistencies in country names over the three data frame tables. Two data frames are extracted from the merged data frame, one for 2019 analysis and the latter for regression modeling. The code begins by performing descriptive analysis on numeric data in the data inclusive of 2019 per capita statistics on GDP, automobile count, and emissions per capita. Further a correlation analysis is executed in order to identify highly correlated and associative features. The classes \'91topn\'92 and \'91worstn\'92 are developed to analyze the GDP and emissions per capita of a user specified \'92n\'92 number of best and worst nations respecting emissions. A user is also able to specify the year of analysis and selection of nations to analyze. An independent samples t-test was then developed to examine the hypothesis: \'91Is Emissions per capita significantly different between countries with higher than average gdp against countries with lower than average GDP\'92. Following, a global analysis was performed to identify continents and nations as a whole and their contribution to global Co2 emissions via a  plotly pie chart and choropleth map. Lastly, the a linear regression model was performed on basis of country input by the user to model the specified nation\'92s Co2 Emissions over the duration 1957-2019 (total years of data collection). The developed class allows a user to specify a nation to model and includes functionality in order to model the data, evaluate performance metrics, and predict with set 95% confidence. \
\

\f0\b\fs36 Execution\

\f1\b0\fs24 Following the download to the python script and CSV data source, the program can be ran via execution on the terminal. Navigating to the set directory, running the program will allow the user to receive figures surrounding descriptive analysis, GDP and emissions disparities of top and worst 5 nations, correlation matrix heat map, and a plotly pie chart along with a choropleth map highlighting nations and continents contribution to global emissions. Prior to the end of the script, the user is prompted to enter a country for regression modeling where any World nation is considered a valid input (try: India, Pakistan, Brazil), model statistics and a forecast for the year (A Future year entered by the user), will be automatically output as a jpg file along with the prior processed visualizations.}