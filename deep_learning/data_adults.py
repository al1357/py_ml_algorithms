# -*- coding: utf-8 -*-
"""
Preparation of adults earnings dataset. Predict if person's earnings are >50K or <=50K.
"""
import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder

data_file_path = "data/hidden/adult/adult.data"
columns = ["age", "workclass", "fnlwgt", "education", "education-num", "martial-status", "occupation", "relationship", "races", "sex", "capital-gain", "capital-loss", "hours-per-week", "native-ctry", "salary"]
# 1) Manually create list of columns to be converted, sample: {"col_name" : "ordinal"||"oh"}
cols_conversion_map = {"workclass":"oh", "education":"oh", "occupation":"ordinal"}

def convert_categorical(data_file_path, data_file_columns, conversion_map):
    data = pd.read_csv(data_file_path, names=columns)
    
    # 2) Iterate through columns and check if they're set; create separate lists of ordinal and oh columns to encode.
    missing_cols = []
    oh_cols = []
    ordinal_cols = []
    for col, transform in conversion_map.items():
        if col not in data.columns.values:
            missing_cols.append(col)
            continue
        
        if transform == "oh":
            oh_cols.append(col)
        elif transform == "ordinal":
            ordinal_cols.append(col)
            
            
    if len(missing_cols) > 0:
        print("These columns do not exist: ", missing_cols)
        
        
    # 3) Create Ord and OH encoders; fit, collect uniques values for each col and their indexes and transform.
    data_enc = data.copy()
    if len(oh_cols) > 0:
        oh_enc = OneHotEncoder(handle_unknown = "ignore", sparse = False)
        oh_enc.fit(data[oh_cols])
        oh_data = oh_enc.transform(data[oh_cols])
        oh_labels = []    
        for col_i, col in enumerate(oh_cols):
            for cat in oh_enc.categories_[col_i]:
                oh_labels.append(col+"-"+cat.strip())
    
    # 4) Convert resulting data structures to Pandas DataFrames; add headers to this data frames using [oh|ord]_cols
        oh_data_df = pd.DataFrame(oh_data, columns=oh_labels, dtype=int)
    # 5) Drop columns from original data frams, inplace=False; replace with encoded data
        data_enc = data_enc.drop(columns=oh_cols)
        data_enc = pd.concat([data_enc, oh_data_df], axis=1)
    
    # *) Optionally save category mapping in file
            
    
    if len(ordinal_cols) > 0:
        ord_enc = OrdinalEncoder() # handle_unknown = "use_encoded_value")
        ord_enc.fit(data[ordinal_cols])
        ord_data = ord_enc.transform(data[ordinal_cols])
        ord_data_df = pd.DataFrame(ord_data, columns=ordinal_cols, dtype=int)
        data_enc = data_enc.drop(columns=ordinal_cols)
        data_enc = pd.concat([data_enc, ord_data_df], axis=1)
    
    return data_enc

data_enc = convert_categorical(data_file_path, columns, cols_conversion_map)