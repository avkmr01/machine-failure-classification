import json

import pandas as pd
from sklearn.preprocessing import OneHotEncoder

with open(r'config/config.json') as f:
    config = json.load(f)

def dataframe_corr(df: pd.DataFrame, columns: list):
    df = df[columns]
    print(df.corr())

def hot_encode(df: pd.DataFrame):
    one_hot_encoded_col_list = []
    categorical_cols = df.select_dtypes(include=['object']).columns
    encoder = OneHotEncoder(sparse_output=False)


    for col in categorical_cols:
        if df[col].nunique()/len(df)<config["HOT_ENCODE_CATEGORY_LIMIT"]:
            one_hot_encoded_col_list.append(col)
    
    one_hot_encoded_data = encoder.fit_transform(df[one_hot_encoded_col_list])
    one_hot_df = pd.DataFrame(one_hot_encoded_data, columns=encoder.get_feature_names_out(one_hot_encoded_col_list))
    df_encoded = pd.concat([df, one_hot_df], axis=1)
    df_encoded = df_encoded.drop(one_hot_encoded_col_list, axis=1)
    
    return df_encoded