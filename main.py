import json

import pandas as pd

from processors.eda import dataframe_corr, hot_encode
with open(r'config/config.json') as f:
    config = json.load(f)
# pd.set_option("display.max_columns", None)

def train():
    df = pd.read_csv(config["TRAIN_PATH"])
    # print(df[df["Machine failure"]==1].head()) #Checking data for machine failure equipments
    # print(df.head())  

    df_he = hot_encode(df)
    corr = dataframe_corr(df_he, df_he.columns[2:])
    print(corr)
    

def test():
    pass


if __name__ == '__main__':
    options = {
        "train": train,
        "test": test
    }
    
    call_func = options[config["PROCESS"]]
    call_func()