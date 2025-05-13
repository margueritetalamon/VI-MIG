import numpy as np
from sklearn.preprocessing import StandardScaler
import numpy.linalg as LA 
from scipy.stats import multivariate_normal , gamma
from scipy.stats import special_ortho_group
from einops import rearrange
import torch
from src.simple_mlp import SimpleMLP

# from src.neural_network import MultiLayerNeuralNetwork


 
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
    def __init__(self, dataset_train=None, dataset_test=None, n_samples=100,
                 d=2, Z=100, meanShift = 1, cov=None, seed=1,
                 prior_mean=None, prior_eps=None , hidden_units=50, sigma=1, n_layers=1):
        self.scaler = StandardScaler()
        self.Z = Z
        self.fixed_theta = None
       
        self.load_data(dataset_train)

        self.X_test = self.scaler.transform(dataset_test[0])
        self.y_test = dataset_test[1]
        self.data_dim  = self.X.shape[-1]

        hidden_sizes = [hidden_units] * 1
        self.neural_network = SimpleMLP(data_dim=self.dim, hidden_sizes=hidden_sizes, output_dim= 1, task='regression')
        self.bnn_dim = self.neural_network.param_dim
        self.dim = self.bnn_dim

        #### PRIOR ON PARAMETERS OF THE BNN
        self.prior_mean = prior_mean if prior_mean is not None else np.zeros(self.bnn_dim)
        self.prior_eps  = prior_eps if prior_eps is not None else 1
        self.prior = multivariate_normal(self.prior_mean, self.prior_eps * np.eye(self.bnn_dim))

        # ##### PRIOR ON SIGMA the scale of the additive noise in the linear regression. 
        # self.alpha_sigma = alpha
        # self.lambda_sigma = slambda
        # self.prior_sigma = gamma(self.alpha_sigma, self.lambda_sigma)
        self.sigma = sigma

        print("Number of parameters in model: ", self.dim)

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
    
    def log_likelihood(self, thetas): ###  log likelihood for the parameter theta theta of  shape B, d
        B, _ = thetas.shape
        X_ = torch.from_numpy(self.X)
        y_ = torch.from_numpy(self.y)
        lll = np.zeros(B)
        for i in range(B):
            theta = thetas[i]
            theta_ = torch.from_numpy(theta)
            self.neural_network.set_weights_from_vector(theta_)
            # forward
            with torch.no_grad():
                y_hat_ = self.neural_network.forward(X_).squeeze(1)
                # compute lll
                diff = y_ - y_hat_
                lll_ = (-torch.dot(diff, diff) / (2 * self.sigma) - torch.log(torch.tensor(2 * np.pi * self.sigma))/2).sum()
                lll[i] = lll_
        return lll
    
    def unpack(self, theta):
        """
        theta: array of shape (B, dim)
        returns: parameters of the NN (w), sigma
        """
        w = theta[:, :self.bnn_dim]
        sigma = theta[:, -1]
        
        # unpack the W,v,b from w as you had before
        # W, v, b = self.neural_network.unpack_params(w)
        return w, sigma

    def gradient_log_likelihood(self, thetas):
        ### theta can be a sample so of shape B, d
        B, d = thetas.shape
        X, y  = self.batchized_data()
        X_ = torch.from_numpy(X)
        y_ = torch.from_numpy(y)

        grad_thetas = np.zeros((B, d))
        for i in range(B):
            theta = thetas[i]
            theta_ = torch.from_numpy(theta)
            # zero the gradients
            self.neural_network.clear_grads()
            # load theta param
            self.neural_network.set_weights_from_vector(theta_)
            # forward
            y_hat_ = self.neural_network.forward(X_).squeeze(1)
            # backward
            diff = y_ - y_hat_
            log_likelihood = -torch.dot(diff, diff) / (2 * self.sigma) - torch.log(torch.tensor(2 * np.pi * self.sigma)) / 2
            lll_ = log_likelihood.sum()
            lll_.backward()
            # store gradient
            grad = self.neural_network.get_gradients_as_vector().numpy()
            grad_thetas[i] = grad
        return grad_thetas
    
    def grad_log_prior(self, theta):
        ### theta of shape B,d 
        # w, sigma = self.unpack(theta)
        # grad_w = - (w - self.prior_mean)/self.prior_eps
        # grad_sigma  = (self.alpha_sigma-1)/sigma - self.lambda_sigma

        # gradients = np.concatenate([grad_w, grad_sigma[:,None]], axis=1)

        gradients = - (theta - self.prior_mean)/(self.prior_eps * self.prior_eps)
        # print("grad prior", gradients.shape)
        return gradients
    
    def gradient_log_density(self, thetas):
        ### theta can be a sample so of shape B, d
        return self.gradient_log_likelihood(thetas) + self.grad_log_prior(thetas)
    
    def prob(self, X):  ###
        probs = self.log_prob(X)
        return np.exp(probs)

    def log_prob(self, thetas): ###  log density of the posterior UNORMALIZED
        # w, sigma = self.unpack(theta)
        # return self.log_likelihood(theta) + self.prior.logpdf(w) + self.prior_sigma.logpdf(sigma)
        return self.log_likelihood(thetas) + self.prior.logpdf(thetas)

    def compute_KL(self, vgmm, noise = None, component_indices = None , B = 1000):
        samples = vgmm.sample(B, noise, component_indices)
        return (vgmm.log_prob(samples[:,None]) - self.log_prob(samples)).mean()
