import pandas as pd

def read_pop():
    pop_df = pd.read_csv('./2010_Census_Populations_by_Zip_Code.csv')
    return pop_df