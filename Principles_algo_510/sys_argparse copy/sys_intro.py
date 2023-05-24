import sys
import pandas as pd
import argparse

#Custom function
import trial_

if __name__ == '__main__':
    df = pd.read_csv(sys.argv[1])
    print('1: View Descriptive Statistics on Feature(s)\
          \n2: Linear Regression')
    choice = input()

    # col = input('Enter Column to average: ')
    # while col not in df.columns.unique():
    #     print('Enter a Valid column:', df.columns.unique())
    #     col = input('Enter Column to average: ')

def print_csv(file):
    df = pd.read_csv(file)
    print(df)


def print_col(data, col):
    print('Column:', col)
    print(data[col])

print(trial_.descriptives(df))

#print(print_col(df, col))

#print_csv(sys.argv[1])

#print(trial_.average_(df, col))

