import numpy as np
from sklearn.preprocessing import StandardScaler
import numpy.linalg as LA 
from scipy.stats import multivariate_normal 
from scipy.stats import special_ortho_group
from einops import rearrange




class LinReg:
    def __init__(self, dataset=None, n_samples=100, d=2, Z = 100, meanShift = 1, cov = None, seed = 1, prior_mean = None, prior_eps = None ):
        
        self.scaler = StandardScaler()
        self.Z = Z
        self.fixed_theta = None
       
        self.load_data(dataset)
 
        self.prior_mean = prior_mean if prior_mean is not None else np.zeros(self.dim)
        self.prior_eps  = prior_eps if prior_eps is not None else 1

        self.prior = multivariate_normal(self.prior_mean, self.prior_eps * np.eye(self.dim))

        print(self.dim)

        

    def load_data(self, dataset):
        X, y = dataset
        self.X = self.scaler.fit_transform(X)
        # self.y = y.reshape(-1, 1)
        self.y = y
        self.n_samples, self.dim = X.shape
    

    def log_likelihood(self, theta): ###  log likelihood for the parameter theta theta of  shape B, d
        logits = np.einsum("bd,nd->bn", theta, self.X) # B, n_samples

        lll = (-((self.y - logits)**2)/(2*self.prior_eps) - np.log(2*np.pi * self.prior_eps)/2).sum(axis = -1)

        return lll
    
    
    def gradient_log_likelihood(self, theta): 
        ### theta can be a sample so of shape B, d

        logits = np.einsum("bd,nd->bn", theta, self.X) # B, n_samples
        
        return ((self.y - logits)[..., None]*self.X[None]/self.prior_eps).sum(axis = 1) ### shape  B, d 
    
    def grad_log_prior(self, theta):
        ### theta of shape B,d 
        return - (theta - self.prior_mean)/self.prior_eps
    
        
    def gradient_log_density(self, theta): 
        ### theta can be a sample so of shape B, d

        return self.gradient_log_likelihood(theta) + self.grad_log_prior(theta)
    
    
    # def prob(self, X): ### for a theta (fixed) gives the prediction y in function of X
    #     ####  inference function after OPTIM 
    #     #### TO BE CHECKED

    #     X_scaled = self.scaler.transform(X)
    #     logits = np.dot(X_scaled, self.theta) 
    #     return self.sigmoid(logits)
    

    def prob(self, X):  ###
        probs = self.log_prob(X)
        return np.exp(probs)


    def log_prob(self, theta): ###  log density of the posterior UNORMALIZED
        
        return self.log_likelihood(theta) + self.prior.logpdf(theta)



    def compute_KL(self, vgmm, noise = None, component_indices = None , B = 1000):
        samples = vgmm.sample(B, noise, component_indices)

        return (vgmm.log_prob(samples[:,None]) - self.log_prob(samples)).mean()

        # return (GM_entropy - self.unormalized_logpdf(samples)).mean()
