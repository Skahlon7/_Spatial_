import pandas as pd
import numpy as np

import matplotlib.pyplot as plt #Change fig size
import seaborn as sns

class descriptive_stats():
    def __init__(self, df):
        df = pd.read_csv(df)
        self.df = df
        self.cols = self.df.select_dtypes(include=['float', 'int']).columns.tolist() #numeric columns
        
    def stats(self):
        for idx in range(len(self.cols)): #calc descriptive stats for each numeric col
            x_bar = sum(self.df[self.cols[idx]].dropna())/len(self.df[self.cols[idx]].dropna())

            #Population STDEV
            std = (sum([(val-x_bar)**2\
                        for val in self.df[self.cols[idx]].dropna()])/(len(self.df[self.cols[idx]].dropna())))**.5

            variance = std**2

            minimum = self.df[self.cols[idx]].min(skipna=True)
            maximum = self.df[self.cols[idx]].max(skipna=True)
            q1, q2, q3, q4 = np.percentile(self.df[self.cols[idx]].dropna(), [25, 50, 75, 99])
            
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
            sns.boxplot(x=self.cols[index], data=self.df, ax=ax[index, 0])
            ax[index, 0].set_xlabel('Value');

            ax[index, 1].set_title(f'{self.cols[index]} Distribution')
            ax[index, 1].set_xlabel(self.cols[index])
            sns.histplot(x=self.cols[index], data=self.df, kde=True, ax=ax[index, 1])

            mean = np.mean(self.df[self.cols[index]])
            std = np.std(self.df[self.cols[index]])

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
            plt.savefig('Descriptive_Stats.png');

s = descriptive_stats('/Users/sandeepk/Desktop/DSCI_510/numpy_pandas/Salaries.csv')
s.stats()