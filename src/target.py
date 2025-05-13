from src.gmm import GMM
from src.logreg import LogReg, MultiClassLogReg
from src.linreg import LinReg, LinReg_BNN

from src.funnel import Funnel

from matplotlib import  pyplot as plt
import numpy as np 
from  einops import rearrange 
import math
from scipy.stats import norm



class Target:
    def __init__(self, name = "gmm",
                mode = "diag", means = None, covariances =  None,  weights = None, n_components = 3, ### gmm traget param 
                dataset_train = None, dataset_test = None,  d = 2, s = 10, scale = 2, n_samples  = 100, Z = 100, meanShift = 1, cov_lg  = None, seed = 1, prior_mean = None, prior_eps= None, ### logreg traget param 
                n_classes = 3, ### multiclass logreg traget param 
                hidden_layers = [10], sigma = 1, n_layers = 1): ### lin reg param

        
        self.name = name 
        if self.name == "gmm":
            self.model = GMM(variational=False, mode = mode, weights = weights, means = means, covs=covariances, n_components = n_components, d = d, s = s, scale = scale)
        

        elif self.name == "logreg":
            self.model = LogReg(dataset_train= dataset_train, dataset_test = dataset_test, n_samples =  n_samples, d = d, Z = Z,  meanShift=meanShift, cov =  cov_lg, seed = seed, prior_eps=prior_eps, prior_mean=prior_mean)

        elif self.name == "funnel":
            self.model = Funnel()


        # elif self.name == "bnn":
        #     self.model = LogReg_withBNN(dataset_train = dataset_train, dataset_test = dataset_test, n_samples=n_samples, d_data=d, Z = Z, meanShift=meanShift, cov =cov_lg, seed=seed, prior_eps=prior_eps, prior_mean=prior_mean )

        elif self.name == "mlogreg":
            self.model = MultiClassLogReg(dataset_train  = dataset_train, dataset_test =dataset_test,  n_samples =  n_samples, d = d, Z = Z,  meanShift=meanShift, cov =  cov_lg, seed = seed, prior_eps=prior_eps, prior_mean=prior_mean, n_classes = n_classes, hidden_layers=hidden_layers)


        elif self.name == "linreg":
            self.model = LinReg(dataset_train  = dataset_train, dataset_test=dataset_test,  prior_eps=prior_eps, prior_mean=prior_mean)

        elif self.name == "linreg_bnn":
            self.model = LinReg_BNN(dataset_train  = dataset_train, dataset_test=dataset_test,  prior_eps=prior_eps, prior_mean=prior_mean, hidden_layers = hidden_layers, sigma = sigma, n_layers = n_layers)



        self.dim = self.model.dim

        self.contours = None

    
    def plot(self, bounds = (-20, 20), grid_size = 100, color = "black", label = None, ncols = 10):
        


        if self.dim == 2:
            fig, ax = plt.subplots()


            if self.contours:
                ax.contour(self.contours[0],self.contours[1], self.contours[2], levels=10, cmap="viridis")

            else:
                x = np.linspace(bounds[0], bounds[1], grid_size)
                y = np.linspace(bounds[0], bounds[1], grid_size)
                X, Y = np.meshgrid(x, y)
                pos = np.dstack((X, Y))[:, :, None, :]
                if self.name in ["funnel",  "logreg"]:
                    pos = rearrange(pos[:,:, 0], "h w d -> (h w) d")

                Z = self.model.prob(pos)

                if self.name in ["funnel",  "logreg", "mlogreg"]:
                    Z = rearrange(Z, "(h w) -> h w", h  = grid_size)

                    if self.name in ["logreg", "mlogreg"]:
                        Z = Z/Z.sum()
                
                if self.name == "gmm":
                    Z = Z[:,0,:]
                print(Z.shape)

                ax.contour(X, Y, Z, levels=20, cmap="viridis")
                self.contours = (X,Y,Z)

        elif self.dim == 1:
            fig, ax = plt.subplots()
            x = np.linspace(bounds[0], bounds[1], grid_size)
            y = self.model.prob(x[:,None, None])
            
            ax.plot(x, y)


        else:

            if not self.name == "gmm": 
                return 
            
            fig, ax = self.model.compute_marginals(fig = None, axes = None, bounds = bounds, grid_size=grid_size, ncols=ncols , label = label, color=color)


        return fig, ax

        


            

