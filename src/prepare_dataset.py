from sklearn.datasets import load_breast_cancer, fetch_openml
from sklearn.preprocessing import OneHotEncoder
from ucimlrepo import fetch_ucirepo 
import numpy as np
import pandas as pd 


   

import os

def prepare_dataset(name, train_ratio = 0.6, mnist_digits = None, mnist_reduced = False, mnist_super_reduced = False):
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

    if name == "mnist":
        if os.path.exists("mnist/X.npy") and os.path.exists("mnist/y.npy"):
            X = np.load("mnist/X.npy")
            y = np.load("mnist/y.npy")
        else:
            # Load MNIST dataset from OpenML
            mnist = fetch_openml('mnist_784', version=1, as_frame=False)
            X = mnist.data  # Images are already flattened as 784-dimensional vectors
            y = mnist.target.astype(np.int8)  # Convert labels to int
            # Normalize pixel values to [0, 1]
            # One-hot encode the labels
            os.mkdir("mnist")
            np.save("mnist/X.npy", X)
            np.save("mnist/y.npy", y)

        # Filter to keep only digits 1 and 4
        if len(mnist_digits) > 0:
            mask = np.isin(y, mnist_digits)
            X = X[mask]
            y = y[mask]
            X = X / 255.0

        # 1-hot encode the labels
        encoder = OneHotEncoder(sparse_output=False)
        y = encoder.fit_transform(y.reshape(-1, 1))

        # Reshape the flattened images to 28x28
        if mnist_reduced or mnist_super_reduced:
            X_reshaped = X.reshape(-1, 28, 28)

            # Select every other pixel in both dimensions (reducing to 14x14)
            X_reduced = X_reshaped[:, ::2, ::2]
            if mnist_super_reduced:
                X_reduced = X_reduced[:, ::2, ::2]
                X_reduced_flat = X_reduced.reshape(-1, 7*7)
            else:
                # Flatten the reduced images back to 1D arrays (196 dimensions)
                X_reduced_flat = X_reduced.reshape(-1, 14*14)

            X = X_reduced_flat

        N, _ = X.shape
        indices = np.random.RandomState(seed=42).permutation(N)
        X = X[indices]
        y = y[indices]
        N_train = int(train_ratio * N)

        X_train = X[:N_train]
        y_train = y[:N_train]

        X_test = X[N_train:]
        y_test = y[N_train:]

        train_dataset = (X_train, y_train)
        test_dataset = (X_test, y_test)
        return train_dataset, test_dataset

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
