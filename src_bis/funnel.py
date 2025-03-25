from scipy.stats import multivariate_normal 
import torch
from torch.distributions import MultivariateNormal
from einops import rearrange 
import numpy as np 



class Funnel:
    def __init__(self, means = (0,0),  sigma = 1.2):
        self.sigma = sigma
        self.means = means
        self.dim = 2
        self.g1 = multivariate_normal(means[0], sigma)
        

    def prob(self, x):
        ### x size b 2 = d

        b = x.shape[0]

        means_tensor = torch.full((b, 1), self.means[1])
        cov_tensor = torch.exp(torch.as_tensor(x[:, 0])/2).unsqueeze(-1).unsqueeze(-1)  # shape becomes (b, 1, 1)
        g2 = MultivariateNormal(means_tensor, covariance_matrix=cov_tensor)
        return  self.g1.pdf(x[:, 0])*g2.log_prob(torch.as_tensor(x[:, 1][:, None])).exp().numpy()          
    
    def log_prob(self, x):
        return np.log(self.prob(x))
    
    def gradient_log_density(self, x):
        ### grad first variable 
        grad1 = - x[:,0]/self.sigma - 0.25 + ((x[:, 1]**2)/(2* np.exp(x[:, 0])))
        grad2 =  - x[:, 1]/np.exp(x[:, 0])

        return np.stack((grad1, grad2)).T
    

    def compute_KL(self, vgmm, noise = None, component_indices = None , B = 1000):
        samples = vgmm.sample(B, noise, component_indices)

        return (vgmm.log_prob(samples[:,None]) - self.log_prob(samples)).mean()

        