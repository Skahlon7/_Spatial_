#!/usr/bin/env python
# coding: utf-8

# ## DSCI 510 Final Project

# ## I. Extract and Preprocess Data

# ### GDP Data by Country 
# - Source: World Bank (API)

# In[62]:


# MATPLOTLIB SAVE FIG!!!! FOR .PY


# In[1]:


# GDP Data from api --> show 20000 per page and match with country name data below --> looks good w country
import requests
import json
import pandas as pd
import numpy as np

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


# In[2]:


gdp_dic = {} #retrieve 2019 statistics on GDP
for index in range(len(data[1])):
    if data[1][index]['date'] == '2019':
        gdp_dic[data[1][index]['country']['value']] = [data[1][index]['date'], data[1][index]['value']]


# In[3]:


gdp_df = pd.DataFrame(gdp_dic.values(), index=gdp_dic.keys(), columns=['Year', 'GDP'])
gdp_df = gdp_df.reset_index().rename(columns={'index': 'Country'}) #Convert to dataframe
gdp_df.tail(3)


# ### Vehicle Per Capita Data
# - Source: Wikipedia (Webscrape)

# In[4]:


from bs4 import BeautifulSoup
import requests
response = requests.get('https://en.wikipedia.org/wiki/List_of_countries_by_vehicles_per_capita')
response.status_code


# In[5]:


#Parse Wiki Data
soup = BeautifulSoup(response.content, 'html.parser')
table = soup.find_all('table')[0]
headers = [val.text.strip() for val in table.find_all('th')] #Headers


# In[6]:


table_rows = table.find_all('tbody')[0].find_all('tr')  #Aggregate Rows
row_data = []
for val in table_rows[1:]:
    for index in range(len(val.find_all('td'))):
        row_data.append(val.find_all('td')[index].text.strip())


# In[7]:


#Convert to DataFrame
import pandas as pd
auto_df = pd.DataFrame()
for col_num in range(len(headers)):
    auto_df[headers[col_num]] = row_data[col_num::4]
#sorted(auto_df['Country or region'].unique())
auto_df.head(3) 


# ### Emission Data by World Nation
# - Source: Kaggle (CSV Download)

# In[8]:


emission_df = pd.read_csv('./CO2_emission.csv')
emission_df.head(2) #Relational Column: 'Country Name'


# 
# 

# 
# 
# 
# 
# 

# ## II. Data Cleansing
# - After inspecting data its observed that the gdp and co2 data are retrieved from sources that style the country name similar, the aim is to manipulate the auto data (smallest dataset) to match the maximized number of country names in co2 and gdp data (optimizing the # of values in the final merged dataframe)

# In[12]:


set_gdp, set_auto, set_co2 = set(gdp_df['Country']), set(auto_df['Country or region']),\
                    set(emission_df['Country Name'])

# print(sorted(set_auto.difference(set_gdp, set_co2)))#Elements in auto dataset not in others


# In[13]:


# print((set_co2.union(set_gdp))-set_auto) #Elements in gdp+c02 dataset not in auto dataset
#GDP and C02 dataset is singnificantly larger than auto dataset
# and contain regions and terrorities 


# In[11]:


#Cleansing of primary key column to ease the multiple table relational join process
auto_df['Country or region'] = auto_df['Country or region'].map(lambda col: col.replace('Iran', 
                                'Iran, Islamic Rep.').replace('Venezuela',
                                'Venezuela, RB').replace('Yemen', 'Yemen, Rep.').replace('Russia',
                                'Russian Federation').replace('Egypt',
                                'Egypt, Arab Rep.').replace('Turkey', 'Turkiye').replace('Syria',
                                'Syrian Arab Republic').replace('South Korea', 
                                'Korea, Rep.').replace('Democratic Republic of the Congo',
                                'Congo, Rep.').replace('Kyrgyzstan','Kyrgyz Republic'))

#Execute above 2 cells again to visualize commited changes done here


# ## III. Relational Join
# - 1st Join: `gdp` and `emissions` data on primary key `Country` and `Country Name` respectfully
# - 2nd Join: `Merged Dataset` above^ with `automobile` dataset: primary key: `GDP(Country) = Automobile(Country or Region)`
# - Data frames developed to perform EDA and analytic tasks for for 2019 data and linear regression modeling

# In[14]:


#Initial Merge
merged_df = gdp_df.merge(emission_df, how='inner', left_on='Country', right_on='Country Name')
##Second Merge - main data frame 
df = merged_df.merge(auto_df, how='inner', left_on='Country', right_on='Country or region')
df = df.rename(columns={str(x): f"{str(x)}_emissions" for x in range(1990, 2020)}) #Identify gdp columns
df.head(2)


# In[15]:


df.to_csv('Merged_Df.csv')


# #### 2019 Dataframe - to perform Analytic tasks and statistical tests

# In[15]:


import re
#Filter desired columns 
df_2019 = df[['Year_x', 'Country', 'GDP', 'Region', '2019_emissions', 'Motor vehiclesper 1,000 people']]

#Convert vehicle per capita column to integer 
df_2019 = df_2019.copy()
df_2019.loc[:, 'Motor vehiclesper 1,000 people']  = df_2019['Motor vehiclesper 1,000 people'].map(lambda x:\
                                                                                re.sub('[,]', '', str(x)))

df_2019.loc[:, 'Motor vehiclesper 1,000 people'] = df_2019['Motor vehiclesper 1,000 people'].astype(int)
df_2019.rename(columns = {'Motor vehiclesper 1,000 people': 'Vehicles_Capita'}, inplace=True)
df_2019.head(2) #A cleaner method to view and handle required data for analysis


# #### Modeling Dataframe - To perform `Linear Regression` on world & country-by-country emissions

# In[16]:


#Data transposed through nested looping to display country-country emissions data for years (1990- 2019)
df_modeling = df[[f'{str(x)}_emissions' for x in range(1990,2020)]].copy()
df_modeling['Country'] = df_2019['Country']
df_modeling #Create Dataframe to transpose data for LR Modeling

emissions_ls = []
for row_index in range(len(df_modeling)):
    for col_index in range(len(df_modeling.columns)-1):
        emissions_ls.append([df_modeling.iloc[row_index, col_index], df_modeling.columns[col_index].split('_')[0], 
               df_modeling.Country.unique()[row_index]])
        
emissions_df = pd.DataFrame(emissions_ls, columns = ['Emissions_Capita', 'Year', 'Country'])
emissions_df.loc[:,'Emissions_Capita'] = emissions_df['Emissions_Capita'].astype(float)
emissions_df.head(2)


# ## IV. Descriptive Analysis
# - Distribution and characteristics of Data 
# - Correlation Matrix to measure Pearson correlation of Per capita Emissions & per capita GDP + Automobiles
# - View the top and bottom 'n' countries with respect to emissions per capita via class: `world_emissions` along with visualizations
# - Statistical Tests
# - Aggregate regional statistics for capita emissions (by continent)

# #### Descriptive stats and distribution analysis

# In[17]:


import matplotlib.pyplot as plt #Change fig size
import numpy as np
import seaborn as sns

class descriptive_stats():
    def __init__(self):
        self.cols = df_2019.select_dtypes(include=['float', 'int']).columns.tolist() #numeric columns
        
    def stats(self):
        for idx in range(len(self.cols)): #calc descriptive stats for each numeric col
            x_bar = sum(df_2019[self.cols[idx]].dropna())/len(df_2019[self.cols[idx]].dropna())

            #Population STDEV
            std = (sum([(val-x_bar)**2\
                        for val in df_2019[self.cols[idx]].dropna()])/(len(df_2019[self.cols[idx]].dropna())))**.5

            variance = std**2

            minimum = min(df_2019[self.cols[idx]])
            maximum = max(df_2019[self.cols[idx]])
            q1, q2, q3, q4 = np.percentile(df_2019[self.cols[idx]].dropna(), [25, 50, 75, 99])
            
            print(self.cols[idx], 'Descriptive Stats:',
                  '\nMean:', round(x_bar,2),'\nSTDEV:', round(std,2), '\nVariance:', round(variance,2), '\nMin:',
              round(minimum,2), '\nMax:', round(maximum,2), '\nQuartile 1:', round(q1,2),'\nQuartile 2:',
              round(q2,2),'\nQuartile 3:', round(q3,2), '\nQuartile 99%:', round(q4,2), '\n')

    def plot(self):
        #Visualize
        title = 'Descriptive Statistics: ' + ', '.join(self.cols)
        fig, ax = plt.subplots(len(self.cols), 2, figsize=(22, 16))
        #fig.suptitle(f' Descriptive Statistics: {self.cols}', fontsize=15)
        fig.suptitle(title, fontsize=15)
        
        for index in range(len(self.cols)): #plot each numeric column box plot + distribution
            ax[index, 0].set_title(f'{self.cols[index]} Box Plot')              
            ax[index, 0].set_ylabel(self.cols[index])
            sns.boxplot(x=self.cols[index], data=df_2019, ax=ax[index, 0])
            ax[index, 0].set_xlabel('Value');

            ax[index, 1].set_title(f'{self.cols[index]} Distribution')
            ax[index, 1].set_xlabel(self.cols[index])
            sns.histplot(x=self.cols[index], data=df_2019, kde=True, ax=ax[index, 1])

            mean = np.mean(df_2019[self.cols[index]])
            std = np.std(df_2019[self.cols[index]])

            z_1_neg = mean - std #68%
            z_1_pos = mean + std 

            z_2_pos = mean + (std*2) #95%
            z_3_pos = mean + (std*3)

            ax[index, 1].axvline(mean, color='r', linestyle='--', linewidth=2, alpha=0.5)
            ax[index, 1].axvline(z_1_pos, color='black', linestyle='--', linewidth=2, alpha=0.5)
            ax[index, 1].axvline(z_1_neg, color='black', linestyle='--', linewidth=2, alpha=0.5)
            ax[index, 1].axvline(z_2_pos, color='black', linestyle='--', linewidth=2, alpha=0.5)
            ax[index, 1].axvline(z_3_pos, color='black', linestyle='--', linewidth=2, alpha=0.5)

            ax[index, 1].legend(['Distibution Curve', 'Mean', '68% (z=-1)', '68% (z=1)', '95% (z=2)', '99.7% (z=3)'])
            plt.savefig('Descriptive_Stats.jpg');
            


# In[18]:


t = descriptive_stats()
t.stats()
t.plot()


# #### Corelation Analysis

# In[19]:


import seaborn as sns
corr = df_2019.corr() #Correlation Analysis

print(f'Highest Correlated Features: {corr.unstack()[corr.unstack()!=1].idxmax()}')
print(f'Correlation: {max(corr.unstack()[corr.unstack()!=1])}')

# Heatmap for identifying highly correlated features
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.savefig('Corr_Matrix.jpg');


# ### Top 'n' and Worst 'n' nations respecting emissions per capita
# - Analysis of Emissions and GDP 

# In[25]:


class top_n(): 
    def __init__(self, df, year, n, *args):
        self.df = df
        self.year = year
        self.n = n
        self.args_ls = [x for x in args if x in self.df.Country.unique() or x.lower() == 'all']
        self.colors_co2 = ['firebrick', 'indianred', 'tomato', 'salmon', 'mistyrose']
        self.colors_gdp = ['darkgreen', 'green', 'forestgreen', 'seagreen', 'mediumseagreen']
        
        #validate parameter input to adgjust dataframe by year and country
        if str(self.year).lower()== 'all': pass
        else: self.df = self.df[self.df.Year.astype(int) == self.year]
        
        if len(self.args_ls) == 0 or 'all' in [str(x).lower() for x in self.args_ls]:pass
        else: self.df = self.df[self.df.Country.isin(self.args_ls)]
        
    def view(self):
        #aggregate top n emissions respecting filtered df
        self.grouped_df = self.df.groupby('Country').agg({'Emissions_Capita': 'mean'})
        #merge to include gdp statistics
        self.top_n = self.grouped_df.merge(df , how='left', left_on=self.grouped_df.index, right_on=df.Country)
        self.top_n = self.top_n[['Country', 'Year_x', 'GDP', 'Emissions_Capita']]
        self.top_n = self.top_n.sort_values(by='Emissions_Capita', ascending=True)[:self.n]
        return self.top_n.rename(columns = {'Emissions_Capita': f'Emissions per capita (Year: {self.year})'})\
                        [['Country', f'Emissions per capita (Year: {self.year})']]

    def plot(self):
        if len(self.top_n)<=7: #instantiate size of figure
            fig, axes = plt.subplots(1, 2, figsize=(18, 10))
        else: fig, axes = plt.subplots(1, 2, figsize=(26, 18))
        
        #custom colors
        fig.suptitle(f'Best {self.n} Countries Emissions per Capita Year ({self.year})', fontsize=14)
        axes[0].set_title('Emissions by Country')
        axes[0].bar(self.top_n.Country,self.top_n.Emissions_Capita, color=self.colors_co2[::-1])
        axes[0].set_ylabel('Emissions Per Capita')
        axes[0].grid(True, linestyle='--', color='grey', alpha=0.3);
        
        axes[1].set_title('GDP by Country')
        axes[1].set_ylabel('GDP per Capita')
        axes[1].bar(self.top_n.Country,self.top_n.GDP, color=self.colors_gdp[::-1])
        axes[1].grid(True, linestyle='--', color='grey', alpha=0.3)
        plt.savefig('best_n_emissions.jpg');
        plt.show()
    
    def get_args(self):
        return f'Countries Queried: {self.args_ls}'
    
class worst_n(top_n):
    def __init__(self, df, year, n, *args):
        super().__init__(df, year, n, *args)
        
    def view(self):
        self.grouped_df = self.df.groupby('Country').agg({'Emissions_Capita': 'mean'})
        self.top_n = self.grouped_df.merge(df , how='left', left_on=self.grouped_df.index, right_on=df.Country)
        
        
        self.top_n = self.top_n[['Country', 'Year_x', 'GDP', 'Emissions_Capita']]
        self.top_n = self.top_n.sort_values(by='Emissions_Capita', ascending=False)[:self.n]
        return self.top_n.rename(columns = {'Emissions_Capita': f'Emissions per capita (Year: {self.year})'})\
    [['Country', f'Emissions per capita (Year: {self.year})']]
    
    def plot(self):
        if len(self.top_n)<=7:
            fig, axes = plt.subplots(1, 2, figsize=(18, 10))
        else: fig, axes = plt.subplots(1, 2, figsize=(26, 18))
        
        fig.suptitle(f'Worst {self.n} Countries Emissions per capita Year ({self.year})', fontsize=14)
        axes[0].set_title('Emissions by Country')
        axes[0].bar(self.top_n.Country,self.top_n.Emissions_Capita, color=self.colors_co2)
        axes[0].set_ylabel('Emissions Per Capita')
        axes[0].grid(True, linestyle='--', color='grey', alpha=0.3);
        
        colors2 = ['pink', 'lightsteelblue', 'cornflowerblue', 'lavenderblush', 'thistle', 'mistyrose']
        axes[1].set_title('GDP by Country')
        axes[1].set_ylabel('GDP per Capita')
        axes[1].bar(self.top_n.Country,self.top_n.GDP, color=self.colors_gdp)
        axes[1].grid(True, linestyle='--', color='grey', alpha=0.3)
        plt.savefig('worst_n_emissions.jpg');
        plt.show()

    
t = worst_n(emissions_df, 'all', 5, 'all')
print(t.view())
t.plot()
#t.get_args()


# In[24]:


w = top_n(emissions_df, 'all', 5, 'all')
print(w.view())
w.plot()


# ## Statistical Tests
# ### Two-sample t-test
# 
# ### $ t = \frac{\bar{x}_1 - \bar{x}_2}{\sqrt{\frac{s_1}{n_1} + \frac{s_2}{n_2}}} $ 
# 

# - Q1: Is Emissions per capita significantly different between countries with higher than average gdp against countries with lower than average GDP

# In[26]:


from scipy.stats import t, ttest_ind


# In[27]:


sns.set_style("darkgrid")


# In[28]:


from scipy.stats import t, ttest_ind
df_2019.rename(columns={'2019_emissions': 'emissions'}, inplace=True)

#Group A - Nations with Higher than avg. GDP
group_a = df_2019[(df_2019.GDP > df_2019.GDP.mean()) & (df_2019.emissions.isna() == False)]
#Group B- Nations with Lower than avg. GDP
group_b = df_2019[(df_2019.GDP < df_2019.GDP.mean()) & (df_2019.emissions.isna() == False)]

t_table = t.ppf(1-.05/2, (len(group_a)+len(group_b))-2) #t critical value 95% - 2-TAILED test - dof 

t_num = (sum(group_a.emissions)/len(group_a)) - (sum(group_b.emissions)/len(group_b))

sd_a = group_a.emissions.std()
sd_b = group_b.emissions.std()

t_stat = t_num/(np.sqrt((sd_a**2/len(group_a))+ (sd_b**2/len(group_b))))

print('Ho: Mean_a == Mean_b', '\nH1: Mean_a != Mean_b\n')
print('T table Value=', t_table, '\nT stat Value=', t_stat, '\nAlpha=0.5\n')
print('We reject the null hypothesis with ', int((1-.05)*100), '% confidence and conclude there is a \
significance between the average number of Emissions per capita with countries higher than\
average GDP v.s. lower than average GDP per capita (tstat > ttable)', sep='')


# In[29]:


#Validate
t_statistic, p_value = ttest_ind(group_a.emissions, group_b.emissions, equal_var=False
                                )
t_statistic, t_stat


# In[30]:


#Visualize conclusion
plt.figure(figsize=(8,4))
sns.scatterplot(x=df_2019.GDP, y=df_2019.emissions, hue=df_2019.Region)
plt.legend(loc='best')
plt.title('GDP vs. Emissions per Capita (2019)')
plt.ylabel('Emissions per Capita')
plt.xlabel('GDP per Capita')
plt.savefig('t_test.jpg');


# - Q2: Is the number of Automobiles per capita significantly different between countries with higher than average gdp against countries with lower than average GDP

# In[31]:


alpha = 0.05

#Group A - Nations with Higher than avg. GDP
group_a = df_2019[df_2019.GDP > df_2019.GDP.mean()]

#Group B- Nations with Lower than avg. GDP
group_b = df_2019[df_2019.GDP < df_2019.GDP.mean()]

dof = (len(group_a) + len(group_b))-2 #Degrees of Freedom 
t_table = t.ppf(1-alpha/2, dof) #t critical value

t_num = (sum(group_a.Vehicles_Capita)/len(group_a.Vehicles_Capita)) -\
                                (sum(group_b.Vehicles_Capita)/len(group_b.Vehicles_Capita)) #numerater t-stat

sd_a = group_a.Vehicles_Capita.std() #sample sd of both groups
sd_b = group_b.Vehicles_Capita.std()

t_stat = (t_num)/((sd_a**2/len(group_a)) + (sd_b**2/len(group_b)))**.5

#Results
print('Ho: Mean_a == Mean_b', '\nH1: Mean_a != Mean_b\n')
print('T table Value=', t_table, '\nT stat Value=', t_stat, '\nAlpha=', alpha, '\n')
print('We reject the null hypothesis with ', int((1-alpha)*100), '% confidence and conclude there is a \
significance between the average number of automobiles per capita with countries higher than\
average gdp v.s. lower than average gdp per capita (tstat > ttable)', sep='')


# In[32]:


#Validate
t_statistic, p_value = ttest_ind(group_a.Vehicles_Capita, group_b.Vehicles_Capita, equal_var=False)


t_statistic, t_stat


# In[33]:


#Visualize conclusion
plt.figure(figsize=(8,4))
sns.scatterplot(x=df_2019.GDP, y=df_2019.Vehicles_Capita, hue=df_2019.Region)
plt.legend(loc='best')
plt.title('GDP vs. Vehicles per Capita (2019)')
plt.ylabel('Vehicles per Capita')
plt.xlabel('GDP per Capita');


# #### Aggregate Global statistics for capita emissions 

# In[34]:


cont_agg = df_2019.groupby('Region').agg({'emissions': 'mean'})
cont_agg['2019_emissions'] = [val/sum(cont_agg['emissions']) for val in cont_agg['emissions']]


# In[40]:


import plotly.express as px

fig = px.pie(cont_agg, values=cont_agg['2019_emissions'], names=np.array(cont_agg.index),
            title = 'Proportion of C02 Emissions by World Continent')
fig.write_image('pie_chart_emissions.jpg')
fig.show()


# In[41]:


import plotly.express as px

fig = px.choropleth(emissions_df, locations='Country', locationmode="country names", 
                    color='Emissions_Capita', scope='world', 
              labels={'Emissions_Capita': 'Emissions per Capita'}, color_continuous_scale='Redor',
                   projection='natural earth', animation_frame='Year')

fig.update_layout(title='Emissions per Capita by Country (1990-2019)')      
fig.write_image('emissions_map.jpg')
fig.show()


# ## Linear Regression
# ## $$ ŷ = β₀ + β₁x $$
# 
# ## $$ β₀ = ((Σy)(Σx²) - (Σx)(Σxy)) / (n(Σx²) - (Σx)²) $$
# 
# ## $$ β₁ = (n(Σxy) - (Σx)(Σy)) / (n(Σx²) - (Σx)²) $$
# 
# * Regression Model to predict Emissions Per capita by Country on a Global basis
# * y = Emissions Per Capita 
# * x = Year
# * Performance Evaluation

# In[42]:


from scipy.stats import t
class LR_Country():
    def __init__(self):
        self.country = input('Enter a Country to Model: ')
        #Validate User Input
        while self.country not in emissions_df.Country.unique():
            self.country = input('Enter a valid World Country to Model: ')
            
    def model(self):
        #Model Development
        cnt_df = emissions_df[(emissions_df.Country == self.country) &
                          (emissions_df.Emissions_Capita.isna() == False)]
            
        self.bo = ((sum(cnt_df.Emissions_Capita)*sum(cnt_df.Year.astype(int)**2)) - (sum(cnt_df.Year.astype(int))\
                    *sum(cnt_df.Emissions_Capita*cnt_df.Year.astype(int))))/((len(cnt_df)\
                    *sum(cnt_df.Year.astype(int)**2)) - (sum(cnt_df.Year.astype(int))**2))


        self.b1 = ((len(cnt_df)*sum(cnt_df.Emissions_Capita*cnt_df.Year.astype(int)))\
              - (sum(cnt_df.Year.astype(int))\
                    *sum(cnt_df.Emissions_Capita)))/ ((len(cnt_df)*sum(cnt_df.Year.astype(int)**2)) \
                    -sum(cnt_df.Year.astype(int))**2)

        print(f'bo: {round(self.bo,3)}, b1: {round(self.b1,5)}')
        
        #Predictions to current
        self.cnt_df_performance = cnt_df.copy()

        self.cnt_df_performance['y_pred'] = self.bo + self.b1*cnt_df['Year'].astype(int).values
        self.cnt_df_performance['e'] = self.cnt_df_performance.Emissions_Capita-self.cnt_df_performance.y_pred
        
        
        plt.figure(figsize=(8,4))
        sns.set_style('whitegrid')
        #sns.set_style("darkgrid")

        plt.title(f'{self.country} LR Model on C02 Emissions per Capita')
        sns.regplot(x=self.cnt_df_performance.Year.astype(int), y='Emissions_Capita', \
                    data=self.cnt_df_performance)
        plt.savefig('model_fit.jpg');
        
    def performance(self):
        self.mse = np.mean(self.cnt_df_performance.e**2)
        print('MAPE:',np.mean(abs((self.cnt_df_performance.Emissions_Capita - self.cnt_df_performance.y_pred)/\
               self.cnt_df_performance.Emissions_Capita)), '\nME:', np.mean(self.cnt_df_performance.e),
             '\nMSE:', self.mse, '\nMAD:', np.mean(abs(self.cnt_df_performance.e)))
        
        plt.figure(figsize=(16,8))
        
        plt.plot(self.cnt_df_performance.Year, self.cnt_df_performance.Emissions_Capita,
         marker='s', c='b', ls='-', lw=2, ms=8, mew=2, mec='cyan', label='Actual') 
        
        plt.plot(self.cnt_df_performance.Year, self.cnt_df_performance.y_pred,
         marker='s', c='r', ls='-', lw=2, ms=8, mew=4, mec='red', alpha = 0.7, label='Model')
        plt.title(f'{self.country} C02 Emissions vs LR Model')
        plt.legend()
        plt.savefig('model_performance.jpg');
        
    def predict(self, year):
        while year<=2019:
            year=int(input('Enter a future year: '))
            
        self.prediction = self.bo + self.b1*int(year)
        
        t_value = t.ppf(1-.05/2, df=len(self.cnt_df_performance)-2)
        
        x_var = np.var(self.cnt_df_performance.Year.astype(int),axis=0, ddof=1)
        #Standard Error -- check abs - remove if not working
        se = np.sqrt(abs(self.mse*(1+1/len(self.cnt_df_performance) + (self.prediction-\
                                np.mean(self.cnt_df_performance.Emissions_Capita)**2))/\
                              (len(self.cnt_df_performance)-1)  * x_var))
        
        co2, low_ci, up_ci, years = list(self.cnt_df_performance.y_pred), [], [],\
                                    list(self.cnt_df_performance.Year.astype(int))
        
        pred = self.bo+self.b1*year
        print(f'Prediction Year {year}: {round(pred, 3)} | 95% conf: ({round(pred-t_value*se, 3)}, {round(pred+t_value*se,3)})')
            
        cnt = 0 
        for yr in range(2020, year+1):
            pred = self.bo + self.b1*yr
            co2.append(pred)
            low_ci.append(pred-t_value*se)
            up_ci.append(pred+t_value*se)
            years.append(int(yr))
            cnt+=1
        
        #Plot Actuals
        plt.figure(figsize=(14,8))
        plt.plot(self.cnt_df_performance.Year.astype(int), self.cnt_df_performance.Emissions_Capita,
         marker='s', c='b', ls='-', lw=2, ms=8, mew=2, mec='cyan', label='Actual') 
    
        #Plot Regression Forecast
        plt.plot(years, co2, marker='s', c='r', ls='-', lw=1, ms=6, mew=4, mec='red', alpha = 0.7, label='Model')
        
        #Plot CI
        plt.plot(years[-cnt:], low_ci, c='#3f76a3', ls='--')#, alpha=0.9)
        plt.plot(years[-cnt:], up_ci, c='#3f76a3', ls='--', label='Confidence Interval Bound')#, alpha=0.9)
        plt.fill_between(years[-cnt:], low_ci, up_ci, color='#3f76a3', alpha=0.4)
        plt.legend()
        plt.savefig('model_pred.jpg');
        


# In[43]:


m = LR_Country() #try: India, Brazil, Pakistan
m.model() 


# In[44]:


m.performance()
yr = int(input('Enter a future year to predict emissions: '))
m.predict(yr)

# In[45]:



# In[125]:


# #Scikitlearn model --> Confirm manual results
# from sklearn.linear_model import LinearRegression

# model = LinearRegression().fit(np.array(cnt_df.Year.astype(int)).reshape(-1,1), cnt_df.Emissions_Capita)

# print(f'bo: {round(model.intercept_,3)}, b1: {round(model.coef_[0],5)}')

