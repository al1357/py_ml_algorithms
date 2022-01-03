import numpy as np    #version 1.9.2
import pandas as pd
import seaborn as sns

def read_data():

    # Load and prepare data
    ## pandas.core.frame.DataFrame
    data = pd.read_csv("data/health_insurance.csv")
    
    print("\n\n---\nHead \n", data.head())
    data_stats = data.describe()
    print("\n\n---\nStats \n", data_stats)
    
    print("\n\n---\nNa values \n")
    print(data.isna().sum())
    data = data.dropna()
    
    sns.pairplot(data[["age", "bmi"]])
    
    # 1) Encode features categories into numbers
    # Q1: How to decode? Is it needed - if you feed the data into algorithm why would you want to take them out?
    # A1.1: encoding data "by hand" before prediction - only for small datasets and testing.
    # A1.2: adding features that have range outside of learning features. 
    #       Then either add learning examples with these features or create a separate category "other".
    from sklearn.preprocessing import OrdinalEncoder
    oe = OrdinalEncoder()
    
    
    
    ## 1a) Binary sex data
    ### unique values - pandas.core.series.Series
    sex_unique = data.sex.drop_duplicates()
    ### pandas.core.arrays.numpy_.PandasArray
    snd_npa = sex_unique.array
    ### 1D numpy Array
    snd_na_1d = np.array(snd_npa)
    ### 2D numpy Array
    snd_na_2d = snd_na_1d.reshape(-1, 1)
    oe.fit(snd_na_2d)
    
    ### pandas.core.arrays.numpy_.PandasArray
    features_sex = data.sex.array
    ### 1D numpy Array
    fs_na_1d = np.array(features_sex)
    ### 2D numpy Array
    fs_na_2d = fs_na_1d.reshape(-1, 1)
    data.sex = oe.transform(fs_na_2d)
    
    
    
    ## 1b) Binary smoker data
    oe.fit(np.array(data.smoker.drop_duplicates().array).reshape(-1, 1))
    data.smoker = oe.transform(np.array(data.smoker).reshape(-1, 1))
    
    
    
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
    ### create an array of all unique regions 
    ### pandas.core.series.Series -> pandas.core.arrays.numpy_.PandasArray
    regions_unique = data.region.drop_duplicates().array
    ### 1D numpy Array
    reg_na_1d = np.array(regions_unique)
    ### 2D numpy Array
    reg_na_2d = reg_na_1d.reshape(-1, 1)
    ohe.fit(reg_na_2d)
    
    region_one_hot = ohe.transform(np.array(data.region.array).reshape(-1, 1)).toarray()
    
# =============================================================================
#     # Alternative way to OH encode without sklearn
#     data_region = data.pop("region")
#     data["region_1"] = (data_region == "r1")*1.0
#     data["region_2"] = (data_region == "r2")*1.0
#     data["region_n"] = (data_region == "rn")*1.0
# =============================================================================
    
    data = data.drop(['region'], axis=1)
    x = np.array(data.drop(['charges'], axis = 1).iloc[:, :])
    x = np.concatenate((x, region_one_hot), axis=1)
    y = np.array(data.charges.iloc[:])
    y = np.reshape(y, (-1, 1))
    
    return (x, y, data_stats)