#!/usr/bin/env python
# coding: utf-8

# ## Housing Price Estimates 
# 
# - Data Source: `Kaggle`
# 
# 
# #### Tasks:
# - **Data Pre-Processing**
# - **EDA**
# - **Merge Additional Data by geo/zip**
# - **K-Means Clustering**
# - **Estimate Price**

# ![300px-Washington_in_United_States.svg.png](attachment:300px-Washington_in_United_States.svg.png)

# In[1]:


#Import Libraries
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split as SPLIT
import seaborn as sns 
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import warnings
from scipy import stats
warnings.filterwarnings("ignore")


# In[2]:


df_raw = pd.read_csv('/Users/sandeepk/Desktop/_Spatial_/Final_550/WA_house_data.csv')
df_raw.head(2)


# ## Merge Additional Data
# - `Zip` or `city` as **Primary Key**
# - Crime Data, Population Data, etc.

# ![tables.png](attachment:tables.png)

# In[3]:


#Population Data #ON QI DATA
zip_df = pd.read_csv('population_by_zip_2010.csv').groupby('zipcode')['population'].max().reset_index() 
df_raw = df_raw.merge(zip_df, left_on='zipcode', right_on='zipcode', how='left') #Relational Join


# In[4]:


#School Data


# ## Cleansing 🧽
# - Null Values Omitted
# - Outliers exceding 3 Z-scores (99.7% data distribution) extracted 

# In[5]:


#Output z-score ranges for data distribution of a feature
def dist_range(df, col, z): 
    upper = round(df[col].mean() + (df[col].std())*z,4)
    lower = round(df[col].mean() - (df[col].std())*z,4)
    if lower<0:lower=0
    print(f'{col.title()} {z} Z-scores of Data: ({lower},{upper})') 


# In[6]:


dist_range(df_raw, 'bathrooms',3) #QI DATA
dist_range(df_raw, 'bedrooms',3)
dist_range(df_raw, 'sqft_living',3)
dist_range(df_raw, 'price',3)


# In[7]:


# Remove Outliers in Analysis Columns
df = df_raw.query('bathrooms <=5 and bedrooms <=7 and sqft_living <= 4500 and price<1500000')


# In[8]:


import scipy
#Kurtosis, Original + Cleansed
scipy.stats.kurtosis(np.array(df_raw['price'])),  scipy.stats.kurtosis(np.array(df['price'])) 


# In[9]:


#View Outlier Mgmt
fig, axes = plt.subplots(2, 2, figsize=(18, 10))

axes[0,0].set_title('Price Distribution (Before Clean)')
sns.histplot(df_raw['price'], kde=True, ax=axes[0,0]);

axes[1,0].set_title('Price Box-Whisker (Before Clean)')
sns.boxplot(x=df_raw['price'],ax=axes[1,0]);

axes[0,1].set_title('Price Distribution (After Clean)')
sns.histplot(df['price'], kde=True, ax=axes[0,1]);

axes[1,1].set_title('Price Box-Whisker (After Clean)')
sns.boxplot(x=df['price'], ax=axes[1,1]);


# In[ ]:


#Engineer Days since prev_sold and peep corr


# ## EDA 🔍
# - _Univariate_ Analysis of Price
# - _Bivariate_ Analysis
# - Correlation
# - Sig. Testing
# 
# #### Distributions, Correlation, Multidimensional Viz 

# In[10]:


#QI DATA
fig = plt.figure(figsize=(7,9))
ax = plt.axes(projection = '3d') #3D scatter plot

sampled_df = df.sample(n=1500) #Sample df for Viz

ax.scatter(sampled_df['bathrooms'], sampled_df['price'], sampled_df['sqft_living'])

ax.set_xlabel('Baths')
ax.set_ylabel('Price')
ax.set_zlabel('Size')
ax.set_title('3D Scatter Housing Properties')
plt.show()


# In[12]:


fig, axes = plt.subplots(2, 2, figsize=(25, 14))
sns.set_style("darkgrid")

axes[0,0].set_title('Bathrooms V. Price')
sns.boxplot(x="bathrooms", y="price", data=df_raw, palette="Set1", ax=axes[0,0])

axes[0,1].set_title('Bedrooms V. Price')
sns.boxplot(x="bedrooms", y="price", data=df_raw, palette="Set1", ax=axes[0,1])

axes[1,0].set_title('View V. Price')
sns.boxplot(x="view", y="price", data=df_raw, palette="Set1", ax=axes[1,0])

axes[1,1].set_title('Grade V. Price')
sns.boxplot(x="grade", y="price", data=df_raw, palette="Set1", ax=axes[1,1]);


# In[13]:


#Col Rating Engineered to better viz Grade
df_raw['rating'] = df_raw['grade'].apply(lambda i: 'low' if i <= 6 else 'avg' if i <= 8 else 'high')
sns.pairplot(data=df_raw[['sqft_living', 'rating', 'bedrooms','bathrooms', 'price']].sample(1000), hue='rating',
            palette={"low": "red", "avg": "blue", "high": "green"});


# In[14]:


from sklearn.linear_model import LinearRegression #SQFT_Living LR VIZ
lr = LinearRegression()
lr.fit(np.array(df_raw['sqft_living']).reshape(-1, 1), np.array(df_raw['price']).reshape(-1, 1))


# In[15]:


bo = lr.intercept_
b1 = lr.coef_
print('Intercept: {} \nSlope (Sqft): {}'.format(bo,b1))


# In[16]:


# Create a figure and set the background color
plt.figure(figsize=(8, 6))
 
sns.set_style("darkgrid", {'axes.facecolor': '0.95'})

sns.regplot(df_raw['sqft_living'], df_raw['price'], marker = 'p', scatter_kws={"color": "blue", "alpha": 0.18}, 
    line_kws={"color": "red", "linewidth": 1.5})
plt.xlabel('Sqft')
plt.ylabel('Price')
plt.title('SQFT V. Price')
plt.text(8500, 3700000, f'ŷ={round(bo[0],3)}+{round(b1[0][0],3)}x',fontsize=11, color='red')
plt.grid(True, linestyle='--', alpha=0.5);


# In[15]:


df_raw1 = df_raw.groupby(['zipcode', 'population']).agg({'price':'mean', 'yr_built':'mean'}).\
sort_values(by='price', ascending=False).iloc[0:50].reset_index()

yr_price = pd.DataFrame(df_raw.groupby('yr_built')['price'].mean()).reset_index().query('yr_built >= 1939 & yr_built <= 2000')

high_price_threshold = 60000

high_price_df = df_raw1[df_raw1['price'] >= df_raw1.iloc[10]['price']]

# Create a bubble chart
plt.figure(figsize=(12, 6))
plt.scatter(df_raw1['yr_built'], df_raw1['price'], s=df_raw1['population']/100, alpha=0.7, label='Zipcodes')


plt.scatter(high_price_df['yr_built'], high_price_df['price'], s=high_price_df['population'] / 100, color='red', label="High Price Zip Codes")

# Annotate high population zip codes
for i, row in high_price_df.iterrows():
    plt.annotate(int(row['zipcode']), (row['yr_built'], row['price']), fontsize=10.5, ha='center', va='bottom')

    
    
plt.plot(yr_price['yr_built'], yr_price['price'])
plt.xlabel('Year Built')
plt.ylabel('Average Price')
plt.title('Bubble Chart with High Population Zip Codes')
plt.legend()
plt.ylim(20000, 2400000)

plt.gca().set_facecolor('whitesmoke')

# Add a grid
plt.grid(color='grey', linestyle='--', linewidth=0.5)

plt.show()


# In[16]:


# Now, create a 'Decade' column by integer division to the nearest 10
df_raw['dc_built'] = (df_raw['yr_built'] // 10) * 10

# Create a figure with 2 subplots
fig, axs = plt.subplots(1, 2, figsize=(60, 20))  # 1 row, 2 columns

plt.rcParams.update({'font.size': 30})

# First subplot: Bar Graph
sns.barplot(x='dc_built', y='price', data=df_raw, lw=5, ax=axs[0])
axs[0].set_title('Price depending on Year Built') 
axs[0].set_xlabel('Decade Built')                 
axs[0].set_ylabel('Price')                        
axs[0].tick_params(axis='x')                      
axs[0].tick_params(axis='y')                      


# Second subplot: Pie Chart
# Count number of houses that have been renovated
renovated_cnt = len(df_raw[df_raw['yr_renovated'] != 0])

# Calculate the total number of houses
total = len(df_raw)

# Calculate percentage
percentage_renovated = (renovated_cnt / total) * 100

# Create a pie chart
labels = ['Renovated', 'Not Renovated']
sizes = [percentage_renovated, 100 - percentage_renovated]

axs[1].pie(sizes, labels=labels, autopct='%1.1f%%')
axs[1].set_title('Percentage of Houses Renovated')
axs[1].axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

# Show the plot
plt.show()


# In[17]:


# Now, create a 'Decade' column by integer division to the nearest 10
df_raw['dc_renovated'] = (df_raw['yr_renovated'] // 10) * 10
#Plot bar graph price depending on year built 
plt.rcParams.update({'font.size': 15})
plt.figure(figsize=(20, 5))
sns.barplot(x='dc_renovated', y='price', data=df_raw[df_raw['yr_renovated'] != 0], lw=5)
plt.title('Price depending on Year Renovated')
plt.xlabel('Year Renovated')
plt.ylabel('Price');


# In[22]:


sns.set_style('whitegrid')
plt.figure(figsize=(14,9))
sns.heatmap(df.drop(columns=['OID_', 'id']).corr(), cmap='coolwarm', annot=True, fmt='.2f');


# ## Hypothesis Testing 
# - **NHST**
# - `1-Tailed Independent Sample T-Test`
# - Statistical Justification for Data Cleansing

# In[8]:


def perform_ttest(df, column, split_value, condition = 'greater'):
    if condition == 'greater':
        group1 = df[df[column] > split_value]['price']
        group2 = df[df[column] <= split_value]['price']
    else:
        group1 = df[df[column] >= split_value]['price']
        group2 = df[df[column] < split_value]['price']

    t_stat, p_value = stats.ttest_ind(group1, group2, equal_var=False)
    
    print(f"T-test for {column} split at {split_value}:")
    if p_value < 0.05:
        print("Reject null hypothesis")
    else:
        print("Fail to reject null hypothesis")
    print(f"T-statistic: {t_stat}, P-value: {p_value}\n")


# In[9]:


perform_ttest(df_raw, 'grade', 4, 'greater')


# In[10]:


perform_ttest(df_raw, 'sqft_living', 3000, 'greater')


# In[11]:


perform_ttest(df_raw, 'bathrooms', 5, 'greater')


# # K-means Clustering 
# - `k = 5`
# > **_Luxury_:
# Label Homes as Low End, Average, and High on basis of clustering property features highest correlated w/ price to 5 randomly placed centroids on basis of Scaled Eucilidean Distance 
# `vars = price, house_size, bathrooms`**

# In[17]:


#Import Scaling+Clustering Libraries
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans 


# In[18]:


df.reset_index(inplace=True)
df_k = df.copy() #df for K-means processes


# ### _Normalize Features_

# In[19]:


#Normalize to Z-scores
scaler = StandardScaler()
df_k[['price_T', 'house_size_t', 'bath_t']] = scaler.fit_transform(df_k[['price', 'sqft_living', 'bathrooms']])


# ### _Elbow Function_
# - Identify optimal K value respecting SSE of Eucidlidean Distance to Respective Centroid

# In[20]:


def elbow(data, max_k): #Identify optimal K value respecting SSE
    k_vals = []
    sse = []
    
    for k in range(1, max_k+1):
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(data)
        
        k_vals.append(k)
        sse.append(kmeans.inertia_)
        
    #Elbow Plot
    fig = plt.subplots(figsize=(10,5))
    plt.plot(k_vals, sse, 'o-')
    plt.xlabel('K')
    plt.ylabel('SSE')
    plt.grid(True)
    plt.show()


# In[23]:


elbow(df_k[['price_T', 'house_size_t', 'bath_t']], 10) #QI DATA


# ## K-means Clustering Application
# - `Luxury`

# In[24]:


#Fit K-Means Cluster on highest Corr features w/ Price
kmeans = KMeans(n_clusters=5, random_state=0) #Fit data to Kmeans Algo
kmeans.fit(df_k[['price_T', 'bath_t', 'house_size_t']])


# In[25]:


df_k['rating_lux'] = kmeans.labels_ #Labels added to k_means df
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(10, 9))
ax = fig.add_subplot(111, projection='3d')  # 3D scatter plot

df_k_sampled = df_k.sample(700, random_state=100)  # Sample df for Viz

# Adjust marker style, color, size, and transparency for the scatter points
scatter = ax.scatter(df_k_sampled['price_T'], df_k_sampled['house_size_t'], df_k_sampled['bath_t'],
                     c=df_k_sampled['rating_lux'], cmap='viridis', s=20, marker='o', alpha=.36)

centers = kmeans.cluster_centers_
ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2], c='black', s=300, marker='o', edgecolors='k', label='Cluster Centers')

ax.set_xlabel('price_T')
ax.set_ylabel('house_size_t')
ax.set_zlabel('bath_t')
ax.set_title('K-Means Algo (k=5)')
ax.legend()

ax.view_init(elev=13, azim=140)
plt.show()


# In[26]:


df['rating_lux'] = kmeans.labels_ #Labels added to main df


# ### K-Means Summary

# In[27]:


print('SSE:', kmeans.inertia_)


# In[28]:


#Assign Weights to features in order to sort and rate by 'Luxury'
cluster_df = df_k.groupby('rating_lux').agg({'price_T':'mean', 'house_size_t':'mean', 'bath_t':'mean'})
cluster_df['Weighted_Val'] = cluster_df.price_T*.2 + cluster_df.house_size_t*.6 + cluster_df.bath_t*.2
cluster_df = cluster_df.sort_values(by='Weighted_Val')
cluster_df


# In[29]:


kmeans.cluster_centers_


# In[30]:


#Initialize Rating 
map_ = {k:v for k,v in zip(cluster_df.index, [5,4,3,2,1])}
df['rating_lux'] = df['rating_lux'].map(map_)


# # Modeling Tasks
# #### **Regression** - Predict Home Prices 💲🏡
# - `Multiple Linear Regression` - _Coefficient + Trend analysis_
# - `Gradient Boosted RF Regression`
# - `XGBoost`

# ### Multiple Linear Regression
# ### $$ y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \ldots + \beta_n x_n + \epsilon $$
# 
# - Coefficient Analysis between lineary associated variables
# - Analysis of R2 & Feature Significance

# In[68]:


import statsmodels.api as sm

#Initialize Predictors & Outcome
Y = df['price']
X = df[['bathrooms', 'bedrooms', 'sqft_living',
       'grade', 'view', 'floors', 'yr_built']]


# In[69]:


#OLS Model (Coef Analysis)
ols_model = sm.OLS(Y, sm.add_constant(X))

# Apply HAC Covariance to Adjust for Multicolinearity  
results = ols_model.fit(cov_type='HAC', cov_kwds={'maxlags': 5})
print(results.summary()) 


# In[70]:


import matplotlib.pyplot as plt
import numpy as np


coefficients = results.params[1:]

plt.figure(figsize=(8, 5))  

x = np.arange(len(coefficients.index[1:]))
y =  [x/100 for x in coefficients.values[1:]]

plt.bar(x, y, align='center', alpha=0.5)

plt.ylabel('Coefficient Value')
plt.title('Coefficient Values')

plt.xticks(x, coefficients.index[1:], rotation=45)

plt.yscale('symlog')
plt.show()


# ## Tree-Based Methods
# - `Random Forest Regression +Boosting`
# - `XGBoost`
# - `GWR`

# ![tree_img.png](attachment:tree_img.png)

# ## Random Forest Regressor 🌲
# - `Gradient Boosting`
# - _Optimize Hyperparameters_ - `GridSearchCV` (3-Fold Cross Validaiton)
# - Analyze Loss + Performance Metrics

# In[31]:


#Import ML Libraries
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import make_scorer, r2_score


# In[32]:


#Split - Train/Test
x_train, x_test, y_train, y_test = train_test_split(df[['bedrooms', 'bathrooms', 'sqft_living',
                                                        'floors', 'view', 'condition', 'grade',
                                                       'sqft_basement', 'yr_built', 'yr_renovated',
                                                       'population']], df['price'], test_size=.2,
                                                   random_state=0)


# In[61]:


#Search space 
param_map = {
    'n_estimators' : [100, 130, 170],
    'max_depth' : [3, 5, 8]
} ## 130 - 170 --> +tune learning rate/min_samples

scorer = make_scorer(r2_score)
gs= GridSearchCV(estimator=GradientBoostingRegressor(loss='squared_error'),
                           param_grid=param_map,
                           scoring=scorer,
                           cv=3,
                           verbose=4)

gs.fit(x_train, y_train)


# In[62]:


#Loss function Analysis
gs.best_params_ #Optimal HyperParameters


# In[36]:


gbr = GradientBoostingRegressor(loss='squared_error', n_estimators=170, max_depth=5)
gbr.fit(x_train, y_train) #Fit model


# <div class="alert alert-block alert-success">
# <b>Gradient Boosted Regression Summary:</b> *Best Parameters - {'max_depth': 5, 'n_estimators': 170}
# *Test R2 Score (3-Fold Cross Validation ) - .812
# </div>

# ## Loss Function
# - Mean Squared Error:
# ## $ \text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 $
# 
# > Estimaters apply gradient boosting on error predictions w/ learning rate 0.1 at each stage to minimize error and shift toward global minima

# In[71]:


train_errors = [mean_squared_error(y_train, pred) for pred in list(gbr.staged_predict(x_train))]
test_errors = [mean_squared_error(y_test, pred) for pred in list(gbr.staged_predict(x_test))]


plt.figure(figsize=(10, 6))
plt.plot(np.arange(1, gbr.n_estimators + 1), train_errors, label='Training Set', marker='o')
plt.plot(np.arange(1, gbr.n_estimators + 1), test_errors, label='Test Set', marker='o')
plt.xlabel('Number of Boosted Trees')
plt.ylabel('Mean Squared Error')
plt.title('Train/Test Loss')
plt.legend()
plt.show()


# ### Feature Importances

# In[72]:


#Sorted Feature Importances
f_imp = sorted(np.vstack((gbr.feature_importances_, gbr.feature_names_in_)).T,key=lambda i: i[0], reverse=True)


# In[75]:


import seaborn as sns
plt.figure(figsize=(8,6))
sns.barplot(x=[i[0] for i in f_imp], y=[i[1] for i in f_imp]);


# In[76]:


#Price Estimates Added 
df['price_est'] = gbr.predict(df[['bedrooms', 'bathrooms', 'sqft_living','floors', 'view', 'condition', 
                                  'grade', 'sqft_basement', 'yr_built', 'yr_renovated','population']])


# ### Performance Metrics
# - _Function to Return Modeling Metrics (Regress)_
# #### $ \text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 $
# #### $ \text{MAPE} = \frac{1}{n} \sum_{i=1}^{n} \left| \frac{y_i - \hat{y}_i}{y_i} \right| \times 100 $
# #### $ \text{ME} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i) $
# #### $ \text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2} $

# In[33]:


from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error

def performance(y_true, y_pred):
    return pd.DataFrame({'MAPE': [mean_absolute_percentage_error(y_true, y_pred)],
             'MSE':[mean_squared_error(y_true, y_pred)],
             'RMSE': [np.sqrt(mean_squared_error(y_true, y_pred))],
             'ME': [np.mean(y_true-y_pred)]}).rename(index={0:'Results'})


# In[78]:


performance(y_test, gbr.predict(x_test)) #Metrics for Gradient Boosted RF Regressor


# In[69]:


"""
To DO:
2.) Other model/gradient characteristics to show (Feature Importances)
4.) XGBoost
5.) Functionize/clean notebook
"""


# ### XGBoost

# In[34]:


from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV, KFold


# In[84]:


#Gridsearch Hyper-Parameters
param_map = {
    'n_estimators': [100, 150],
    'max_depth': [6, 9, 12],
    'gamma': [1, 10],
    'learning_rate': [.1, .3, .4]
}

kf = KFold(n_splits=3, shuffle=True, random_state=42)

scorer = make_scorer(r2_score)
gs= GridSearchCV(estimator=XGBRegressor(objective='reg:squarederror', booster='gbtree', 
                                         random_state=42),
                           param_grid=param_map,
                           scoring=scorer,
                           cv=kf,
                           verbose=4)
gs.fit(x_train, y_train)


# In[86]:


gs.best_score_, gs.best_params_ #View Optimized Results


# In[35]:


xgb = XGBRegressor(objective='reg:squarederror', n_estimators=100, booster='gbtree', gamma=1,
                   learning_rate=.3, max_depth=6, random_state=42) #Fit Optimized XGB

xgb.fit(x_train, y_train)


# In[36]:


performance(y_test, xgb.predict(x_test))


# In[37]:


#Add y-hat
df['price_est'] = xgb.predict(df[['bedrooms', 'bathrooms', 'sqft_living','floors', 'view', 'condition', 
                                  'grade', 'sqft_basement', 'yr_built', 'yr_renovated','population']])


# ### K-Means Clustering
# - `Flip-Value Score`

# In[38]:


#Engineer price margin column
df['price_margin'] = df['price'] - df['price_est']


# In[39]:


#Initialize the scaler
scaler = StandardScaler()

#Scale the features to add to the k-means
df_scaled = scaler.fit_transform(df[['price_margin', 'sqft_living', 'grade']])

#Fit the K-Means clustering
kmeans = KMeans(n_clusters=5, random_state=42)
kmeans.fit(df_scaled)

#Assign the cluster labels to df
df['flip_rating'] = kmeans.labels_

#Get the cluster centers
centers_scaled = kmeans.cluster_centers_

#Inverse transform the cluster centers to original...
centers = scaler.inverse_transform(centers_scaled)

#Downsample the data to see better in plot
df.reset_index(drop=True, inplace=True)

#Use min function to ensure sample size not greater than size of df
sample_size = min(500, len(df))

#Get random indeces for sampling
indices = np.random.choice(df.index, size=sample_size, replace=False)
df_sampled = df.iloc[indices]

#Inverse scale the sampled data for viz purposes
df_sampled_scaled = scaler.inverse_transform(df_scaled[indices])

#Plot the clusters, cluster centers, and the data points using sampled data
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(df_sampled_scaled[:, 0], df_sampled_scaled[:, 1], df_sampled_scaled[:, 2],
                     c=df_sampled['flip_rating'], cmap='viridis', s=20, marker='o', alpha=0.5)

ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2],
           c='black', s=300, marker='o', edgecolors='white', linewidths=2, label='Cluster Centers')


ax.set_xlim(-300000, 350000)

# Set axis labels and title
ax.set_xlabel('Price Margin')
ax.set_ylabel('Square Footage')
ax.set_zlabel('Grade')
ax.set_title('K-Means Clustering (k=5) on Sampled Data')
ax.legend()

plt.show()


# ### K-Means Summary

# In[40]:


print('SSE:', kmeans.inertia_)


# In[41]:


grouped = df.groupby('flip_rating').agg({'price_margin': 'mean', 'sqft_living': 'mean', 'grade': 'mean'})
grouped['Weighted_val'] = -.25*grouped.price_margin + .25*grouped.sqft_living + .5*grouped.grade
grouped = grouped.sort_values(by='Weighted_val')
grouped


# In[42]:


#Initialize Ranking change in df
map_ = {k:v for k,v in zip(grouped.index, [5,4,3,2,1])}

df['flip_rating'] = df['flip_rating'].map(map_)


# In[43]:


df.rename(columns={'rating_lux':'lux_rating'}, inplace=True)


# ## Mapping of Computed Scores 🌎
# - `Luxury & Flip Rating`
# > **1-5 stars ⭐️**

# In[60]:


import folium
from folium.plugins import HeatMap
from branca.element import Element, Template

#Each value in the rating column is represented as a different color.
def color_producer(rating):
    if rating == 5:
        return 'green'
    elif rating == 4:
        return 'blue'
    elif rating == 3:
        return 'yellow'
    elif rating == 2:
        return 'orange'
    else:
        return 'red'

#Setting up the heat map.
#Creating a fucntion for user to choose between luxury map and flipping map.
def heat_map(rating:int):
    if rating == 1:
        rating = "lux_rating"
        legend = "Luxury Score"
    if rating == 2: #for second heat map.
        rating = "flip_rating"
        legend = "Flip Score"
    m = folium.Map(location=[df['lat'].mean(), df['long'].mean()], zoom_start=12)

    #Setting up the map to display a heatmap based on the '5'rating. 
    heat_data = [[row['lat'], row['long'], row[f'{rating}']] for index, row in df.query(f"{rating} == 5").iterrows()]

    #Add interactive circle markers to display latitude, longitude, bedrooms, bathrooms, and sqft.
    for index, row in df.iterrows():
        folium.CircleMarker(
            location=[row['lat'], row['long']],
            radius=1.5,  # Can be adjusted
            color=color_producer(row[f'{rating}']),
            fill=True,
            fill_color=color_producer(row[f'{rating}']), popup = f"Latitude:{row['lat']}\n Longitude:{row['long']} Price:${row['price']} Bedrooms:{row['bedrooms']} Bathrooms:{row['bathrooms']} SQFT:{row['sqft_living']}"
        ).add_to(m)

    # Add the heat map layer
    HeatMap(heat_data).add_to(m)

    # Custom HTML for the legend. Add colors, labels, and background. Legend will change with function and map.
    legend_html = f'''
    <div style="position: fixed; 
         bottom: 50px; left: 50px; width: 150px; height: 130px; 
         background-color: white; border:2px solid grey; z-index:9999; font-size:14px;">
    &nbsp; {legend} <br>
    &nbsp; <i style="background: green; width: 15px; height: 15px; display: inline-block;"></i> 5 Star <br>
    &nbsp; <i style="background: blue; width: 15px; height: 15px; display: inline-block;"></i> 4 Star <br>
    &nbsp; <i style="background: yellow; width: 15px; height: 15px; display: inline-block;"></i> 3 Star <br>
    &nbsp; <i style="background: orange; width: 15px; height: 15px; display: inline-block;"></i> 2 Star <br>
    &nbsp; <i style="background: red; width: 15px; height: 15px; display: inline-block;"></i> 1 Star <br>
    </div>
    '''

    # Add the legend to the map
    legend = Element(legend_html)
    m.get_root().html.add_child(legend)
    
    # Display the map
    return m


# In[61]:


heat_map(1)


# In[46]:


heat_map(2)

