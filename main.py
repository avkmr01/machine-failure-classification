import json
from pathlib import Path

import pandas as pd

from processors.eda import dataframe_corr, hot_encode, plot_scatter_matrix, hierarchical_clustering, plot_histograms, pie_distrbution, plot_distribution
with open(r'config/config.json') as f:
    config = json.load(f)
# pd.set_option("display.max_columns", None)

def eda():
    df = pd.read_csv(config["TRAIN_PATH"])
    # print(df[df["Machine failure"]==1].head()) #Checking data for machine failure equipments
    # print(df.head())  

    pie_distrbution(df, 'Type')
    df_he = hot_encode(df)
    # plot_histograms(df, df_he.columns[2:7], n_cols=3)
    plot_distribution(df, df.columns[2:8], "Type")
    # dataframe_corr(df_he, df_he.columns[2:])
    # plot_scatter_matrix(df_he, df_he.columns[2:7], "Machine failure")
    # hierarchical_clustering(df_he, df_he.columns[2:], "Machine failure")
    

def train():
    pass


def test():
    pass


if __name__ == '__main__':
    Path('figures').mkdir(exist_ok=True)

    options = {
        "eda": eda,
        "train": train,
        "test": test
    }
    
    call_func = options[config["PROCESS"]]
    call_func()