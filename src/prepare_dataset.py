from sklearn.datasets import load_breast_cancer
from ucimlrepo import fetch_ucirepo 
import numpy as np
import pandas as pd 


   


def prepare_dataset(name, train_ratio = 0.6):
 
    if name == "breast_cancer":


        data = load_breast_cancer()
        X = data["data"]
        y =  data["target"].reshape(-1,1)
    
    
    if name == "wine":
                
        wine = fetch_ucirepo(id=109) 
        
        X = wine.data.features.values
        y = wine.data.targets.values - 1 ##  to 1,2,3 to 0,1,2

    
    if name == "toxicity":
        
        toxicity = fetch_ucirepo(id=728) 
        
        X = toxicity.data.features .values
        y = toxicity.data.targets

        y = y.replace("Toxic", 1)
        y  = y.replace("NonToxic", 0)
        y = y.values

    if name == "covertype":

        covertype = fetch_ucirepo(id=31) 
        
        X = covertype.data.features.values
        y = covertype.data.targets.values - 1 ##  to 1,2.., K to 0,1,2, K-1

    if name == "boston":

        data_url = "http://lib.stat.cmu.edu/datasets/boston"
        raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None).dropna(axis = 0)
        y = raw_df.values[:, 2]
        y = y[..., None]
        X = np.concatenate([raw_df.values[:, :2], raw_df.values[:, 3:]], axis = 1)




    if name == "synthetic":
        return None, None


    N, d = X.shape

    indices = np.random.RandomState(seed=42).permutation(N)

    # indices = np.random.permutation(N)
    X = X[indices]
    y = y[indices]
    

    N_train = int(train_ratio * N)

    N_train = int(train_ratio * N)
    X_train = X[:N_train]
    y_train = y[:N_train, 0]

    X_test = X[N_train:]
    y_test = y[N_train:, 0]
    


    train_dataset = (X_train, y_train)
    test_dataset = (X_test, y_test)




    return train_dataset, test_dataset
