import numpy as np    #version 1.9.2
import pandas as pd
#import seaborn as sns

pd.options.display.width = 0

def read_data():
    data = pd.read_csv("data/admission/Admission_Predict.csv")
    data.columns = data.columns.str.strip()
    data.columns = data.columns.str.replace(" ", "_")
    
    print("\n\n---\nHead \n", data.head())
    data_stats = data.describe()
    print("\n\n---\nStats \n", data_stats)
    
    print("\n\n---\nNa values \n")
    print(data.isna().sum())
    data = data.dropna()
        
    x = np.array(data.drop(["Chance_of_Admit"], axis=1))
    y = np.array(data["Chance_of_Admit"]).reshape(-1, 1)
    
    return (x, y, data_stats)