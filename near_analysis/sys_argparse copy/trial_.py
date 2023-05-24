import argparse
import sys
import pandas as pd
import os
import numpy as np


#Get the current working directory
cwd = os.getcwd()
def average_(data, column):
    return round(np.mean(data[column]),2)

def descriptives(data):
    print(data.describe(), '\nVar:', end='')
    for col in data.describe().columns: print(round(np.std(data[col])**2, 6), '\t', end='')

#Import the Python files for scraping data
sys.path.append(os.path.join(cwd))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description= 'displays dataset and stats')
    parser.add_argument('--display', type=int, required=False, help='Displays n values of GapData')
    parser.add_argument('--average', type=str, required=False, help='Average Sales')
    parser.add_argument('--stats', type=str, required=False, help='Descriptive Stats on Data')
    args = parser.parse_args()

    n_display = args.display
    col = args.average
    file = args.stats

    df = pd.read_csv('Newcarsalesdata_copy.csv')

    if args.display is not None:
        print(df.head(n_display))
    elif args.average is not None:
        print(average_(df, col))
    elif args.stats is not None:
        df = pd.read_csv(file)
        print(descriptives(df))
    else:
        print(df)


