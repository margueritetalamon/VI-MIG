from src_bis.gmm import GMM
from src_bis.logreg import LogReg
from src_bis.funnel import Funnel

from matplotlib import  pyplot as plt
import numpy as np 
from  einops import rearrange 


class Target:
    def __init__(self, name = "gmm", mode = "diag", means = None, covariances =  None,  weights = None, n_components = 3, dataset = None, d = 2, s = 10, scale = 2, n_samples  = 100, Z = 100, meanShift = 1, cov_lg  = None, seed = 1):

        
        self.name = name 
        if self.name == "gmm":
            self.model = GMM(variational=False, mode = mode, weights = weights, means = means, covs=covariances, n_components = n_components, d = d, s = s, scale = scale)
        

        elif self.name == "logreg":
            self.model = LogReg(dataset, n_samples =  n_samples, d = d, Z = Z,  meanShift=meanShift, cov =  cov_lg, seed = seed)

        elif self.name == "funnel":
            self.model = Funnel()


        self.dim = self.model.dim

        self.contours = None

    
    def plot(self, bound = 20, grid_size = 100):
        fig, ax = plt.subplots()


        if self.dim == 2:


            if self.contours:
                ax.contour(self.contours[0],self.contours[1], self.contours[2], levels=10, cmap="viridis")

            else:
                x = np.linspace(-bound, bound, grid_size)
                y = np.linspace(-bound, bound, grid_size)
                X, Y = np.meshgrid(x, y)
                pos = np.dstack((X, Y))[:, :, None, :]
                if self.name == "funnel":
                    pos = rearrange(pos[:,:, 0], "h w d -> (h w) d")

                Z = self.model.prob(pos)

                if self.name == "funnel":
                    Z = rearrange(Z, "(h w) -> h w", h  = grid_size)
                    

                ax.contour(X, Y, Z, levels=20, cmap="viridis")
                self.contours = (X,Y,Z)

        elif self.dim == 1:
            x = np.linspace(-bound, bound, grid_size)
            y = self.model.prob(x[:,None, None])
            
            ax.plot(x, y)

        return ax

        


            

