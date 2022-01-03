import numpy as np    #version 1.9.2
import pandas as pd
import seaborn as sns

def read_data():

    # Load and prepare data
    ## pandas.core.frame.DataFrame
    data = pd.read_csv("data/iris.csv")
    
    print("\n\n---\nHead \n", data.head())
    data_stats = data.describe()
    print("\n\n---\nStats \n", data_stats)
    
    print("\n\n---\nNa values \n")
    print(data.isna().sum())
    data = data.dropna()
    
    sns.pairplot(data[["PetalLengthCm", "PetalWidthCm"]])
    sns.pairplot(data[["SepalLengthCm", "SepalWidthCm"]])
    
    # 1) Encode features categories into numbers
    # Q1: How to decode? Is it needed - if you feed the data into algorithm why would you want to take them out?
    # A1.1: encoding data "by hand" before prediction - only for small datasets and testing.
    # A1.2: adding features that have range outside of learning features. 
    #       Then either add learning examples with these features or create a separate category "other".
    from sklearn.preprocessing import OrdinalEncoder
    oe = OrdinalEncoder()
    
    
    
    
    
    ## 1c) Categorical variables with >2 unique values are best encoded using OneHotEncoder - Region data
    ## OrdinalEncoder test
    ## =============================================================================
    ## Predict data 1:  [[31.    0.   25.74  0.    0.  ]]
    ## Predict result 1:  [[4234.97751295]]
    ## Real result 1:  [3756.6216]
    ## =============================================================================
    ## oe.fit(np.array(data.region.drop_duplicates().array).reshape(-1, 1))
    ## data.region = oe.transform(np.array(data.region.array).reshape(-1, 1))
     
    ## OneHotEncoder
    ## =============================================================================
    ## Predict data 1:  [[31.    0.   25.74  0.    0.    0.    0.    1.    0.  ]]
    ## Predict result 1:  [[3719.82579905]]
    ## Real result 1:  [3756.6216]
    ## =============================================================================
    from sklearn.preprocessing import OneHotEncoder
    ohe = OneHotEncoder()
    
    ### create an array of all unique values of a given feature 
    ### pandas.core.series.Series -> pandas.core.arrays.numpy_.PandasArray
    labels_unique = data.Species.drop_duplicates().array
    ### 1D numpy Array
    labels_na_1d = np.array(labels_unique)
    ### 2D numpy Array
    labels_na_2d = labels_na_1d.reshape(-1, 1)
    ohe.fit(labels_na_2d)
    
    labels_one_hot = ohe.transform(np.array(data.Species.array).reshape(-1, 1)).toarray()
    
# =============================================================================
#     # Alternative way to OH encode without sklearn
#     data_region = data.pop("region")
#     data["region_1"] = (data_region == "r1")*1.0
#     data["region_2"] = (data_region == "r2")*1.0
#     data["region_n"] = (data_region == "rn")*1.0
# =============================================================================
    
    x = np.array(data.drop(["Species"], axis = 1).iloc[:, :])
    
    return (x, labels_one_hot, data_stats)