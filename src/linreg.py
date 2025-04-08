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
        ones = np.ones((X.shape[0], 1))
        X = self.scaler.fit_transform(X)

        X = np.concatenate([X, ones], axis=1)
        self.X = X
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


import numpy as np

class SmallNeuralNetwork:
    def __init__(self, hidden_units, data_dim):

        self.data_dim = data_dim
        self.hidden_units = hidden_units
        self.param_dim = self.data_dim * self.hidden_units + self.hidden_units + self.hidden_units

    def forward(self, params , x):
        W,  v, b  = self.unpack_params(params)

        z = np.matmul(W, x.T).transpose(0, 2, 1) + b[:, None, :]  # (B, n, h)
        a = np.maximum(0, z)  # (B, n, h)
        f = np.sum(v[:, None, :] * a, axis=-1)  # (B, n)
        return f
    

    def unpack_params(self, params):
        idx = 0
        W = rearrange((params[:, idx : idx + self.data_dim * self.hidden_units]), "b (h d) -> b h d", h = self.hidden_units)
        idx += self.data_dim * self.hidden_units
        v =  params[:, idx : idx + self.hidden_units]
        idx+= self.hidden_units
        b =  params[:, idx : idx + self.hidden_units]

        return W, v, b 
    
    def flatten_params(self, params):
        W, v, b = params

        W = rearrange(W, "b h d -> b (h d)")
        return np.concatenate([W, v, b], axis=-1) #B , param_dim
    

    def compute_gradients(self, params, x):

        W, v, b = self.unpack_params(params)
        z = np.matmul(W, x.T).transpose(0, 2, 1) + b[:, None, :]  # (B, n, h)
        a = np.maximum(0, z)  # (B, n, h)
        indicator = (z > 0).astype(float)  # (B, n, h)
        
        grad_v = a  # (B, n, h)
        grad_b = v[:, None, :] * indicator  # (B, n, h)
        grad_W = (v[:, None, :] * indicator)[:, :, :, None] * x[None, :, None, :]  # (B, n, h, d)

        gradient = self.flatten_gradients(grad_W, grad_v, grad_b)# B n dim_params
        
        return gradient
    

    def flatten_gradients(self, grad_W, grad_v, grad_b):
        return np.concatenate([rearrange(grad_W, "b n h d -> b n (h d)"), grad_v, grad_b], axis = -1)
    




class LinReg_BNN:
    def __init__(self, dataset=None, n_samples=100, d=2, Z = 100, meanShift = 1, cov = None, seed = 1, prior_mean = None, prior_eps = None , hidden_units =  50):
        
        self.scaler = StandardScaler()
        self.Z = Z
        self.fixed_theta = None
       
        self.load_data(dataset)

        self.neural_network = SmallNeuralNetwork(hidden_units=hidden_units, data_dim = self.dim )
        self.dim = self.neural_network.param_dim

        self.prior_mean = prior_mean if prior_mean is not None else np.zeros(self.dim)
        self.prior_eps  = prior_eps if prior_eps is not None else 1

        self.prior = multivariate_normal(self.prior_mean, self.prior_eps * np.eye(self.dim))

        print(self.dim)

    

    def batchized_data(self, M = 32):

        indices = np.random.permutation(len(self.X))[:M]
        # print(indices[:2])
        # print(indices.shape)
        X = self.X[indices]
        y = self.y[indices]
        return X, y

        

    def load_data(self, dataset):
        X, y = dataset
        # ones = np.ones((X.shape[0], 1))
        X = self.scaler.fit_transform(X)
        # X = np.concatenate([X, ones], axis=1) ### no need in Bnn there is the bias 
        self.X = X
        # self.y = y.reshape(-1, 1)
        self.y = y
        self.n_samples, self.dim = X.shape
    

    def log_likelihood(self, theta): ###  log likelihood for the parameter theta theta of  shape B, d
        logits = self.neural_network.forward(theta, self.X) # B, n , 

        lll = (-((self.y - logits)**2)/(2*self.prior_eps) - np.log(2*np.pi * self.prior_eps)/2).sum(axis = -1)

        return lll
    
    
    def gradient_log_likelihood(self, theta): 
        ### theta can be a sample so of shape B, d
        X,y  = self.batchized_data()

        logits = self.neural_network.forward(theta, X) # B, n_samples
        gradients = self.neural_network.compute_gradients(theta, X) # B, n , dim_params
        
        return ((y - logits)[..., None]*gradients/self.prior_eps).sum(axis = 1) ### shape  B, dim_params
    
    def grad_log_prior(self, theta):
        ### theta of shape B,d 
        return - (theta - self.prior_mean)/self.prior_eps
    
        
    def gradient_log_density(self, theta): 
        ### theta can be a sample so of shape B, d

        return self.gradient_log_likelihood(theta) + self.grad_log_prior(theta)
    

    def prob(self, X):  ###
        probs = self.log_prob(X)
        return np.exp(probs)


    def log_prob(self, theta): ###  log density of the posterior UNORMALIZED
        
        return self.log_likelihood(theta) + self.prior.logpdf(theta)



    def compute_KL(self, vgmm, noise = None, component_indices = None , B = 1000):
        samples = vgmm.sample(B, noise, component_indices)

        return (vgmm.log_prob(samples[:,None]) - self.log_prob(samples)).mean()

