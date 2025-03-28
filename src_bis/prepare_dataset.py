from sklearn.datasets import load_breast_cancer
from ucimlrepo import fetch_ucirepo 
import numpy as np


  


def prepare_dataset(name, train_ratio = 0.6):
 
    if name == "breast_cancer":


        data = load_breast_cancer()
        X = data["data"]
        y =  data["target"].reshape(-1,1)
    
    
    if name == "wine":
                
        wine = fetch_ucirepo(id=109) 
        
        X = wine.data.features.values
        y = wine.data.targets.values - 1 ##   to 0 and 1

    
    if name == "toxicity":
        
        toxicity = fetch_ucirepo(id=728) 
        
        X = toxicity.data.features .values
        y = toxicity.data.targets

        y = y.replace("Toxic", 1)
        y  = y.replace("NonToxic", 0)
        y = y.values


    N, d = X.shape

    indices = np.random.permutation(N)
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
