from scipy.stats import multivariate_normal 
import numpy as np 



class Funnel:
    def __init__(self, means = (0,0),  sigma = 1.2):
        self.sigma = sigma
        self.means = means
        self.dim = 2
        self.g1 = multivariate_normal(means[0], sigma)
        

    def prob(self, x):
      
        return np.exp(self.log_prob(x))
    
    def log_prob(self, x):
        x = x.astype( dtype=np.float64 )
        return -np.log(2*np.pi *self.sigma)/2 - x[:,0]**2/ (2*self.sigma) - np.log(2*np.pi)/2 - x[:,1]**2 * np.exp(-x[:,0])/2 - x[:,0]/2
    
    def gradient_log_density(self, x):
        x = x.astype( dtype=np.float64 )
        grad1 = - x[:,0] / self.sigma  + x[:,1]**2 * np.exp(-x[:,0]) / 2 - 0.5        
        grad2 =  - x[:, 1]*np.exp(-x[:,0])

        return np.stack((grad1, grad2)).T
    

    def compute_KL(self, vgmm, noise = None, component_indices = None , B = 1000):
        samples = vgmm.sample(B, noise, component_indices)

        return (vgmm.log_prob(samples[:,None]) - self.log_prob(samples)).mean()

        