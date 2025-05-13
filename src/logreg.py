import numpy as np
from sklearn.preprocessing import StandardScaler
import numpy.linalg as LA 
from scipy.stats import multivariate_normal 
from scipy.stats import special_ortho_group
from einops import rearrange
 

import torch
from src.simple_mlp import SimpleMLP
from src.gmm import GMM
import torch.distributions as dist

import  math

class LogReg:
    def __init__(self, dataset_train=None, dataset_test = None, n_samples=100, d=2, Z = 100, meanShift = 1, cov = None, seed = 1, prior_mean = None, prior_eps = None ):
        
        self.scaler = StandardScaler()
        self.Z = Z
        self.fixed_theta = None
        self.meanShift = meanShift
        self.cov  = cov
        self.seed = seed

        if dataset_train is None:
            self.data_dim = d
            self.n_samples = n_samples
            self.generate_data(n_samples)
        else:
            self.load_data(dataset_train)
 
        self.dim = self.data_dim
        self.prior_mean = prior_mean if prior_mean is not None else np.zeros(self.dim)
        self.prior_eps  = prior_eps if prior_eps is not None else 1

        self.prior = multivariate_normal(self.prior_mean, self.prior_eps * np.eye(self.dim))

        print(f"Parameters dim = {self.dim}")

        self.X_test = self.scaler.transform(dataset_test[0])
        self.y_test = dataset_test[1]
        self.output_dim = self.y.shape[1]
        self.n_classes = self.output_dim if self.output_dim > 1 else 2

    def generate_cov(self,c = 1, scale = 1, rotate = True, normalize = True ):

        vec = (1/np.arange(1,self.dim+1)**c)*scale**2
        if normalize:
            vec=vec/np.linalg.norm(vec)**2
        cov = np.diag(vec)

        if rotate:
            local_rng = np.random.RandomState(self.seed)
            Q = special_ortho_group.rvs(dim=self.dim,  random_state=local_rng)
            cov=np.transpose(Q).dot(cov).dot(Q)

        return cov 

    def generate_data(self, n_samples):

        state = np.random.get_state()
        np.random.seed(self.seed)
        mean = np.random.rand(self.dim)
        np.random.set_state(state)  # Restore previous RNG state

        mean /= LA.norm(mean)
        self.mean0 = mean*self.meanShift/2
        self.mean1 = -mean*self.meanShift/2
        # self.cov = GMM.generate_random_covs( n = 1, d = self.dim, mode = "iso", scale=np.sqrt(self.dim))[0]
        # self.cov = np.eye(self.dim)*self.dim
        np.random.seed(self.seed)
        self.cov = self.cov if self.cov is not None  else self.generate_cov()
       

        X0 = np.random.multivariate_normal(self.mean0, self.cov, int(n_samples/2))
        X1 = np.random.multivariate_normal(self.mean1, self.cov, int(n_samples/2))
        X = np.concatenate((X0,X1))
        X = self.scaler.fit_transform(X)

        y0 = np.zeros((int(n_samples/2),1))
        y1 = np.ones((int(n_samples/2),1))
        y = np.concatenate((y0,y1))

        data = list(zip(y, X))
        np.random.shuffle(data)
        self.y, self.X = zip(*data)
        self.y = np.array(self.y)[:,0]
        self.X = np.array(self.X)


    def load_data(self, dataset):
        X, y = dataset
        self.X = self.scaler.fit_transform(X)
        # self.y = y.reshape(-1, 1)
        self.y = y
        self.n_samples, self.data_dim = X.shape
        
    def sigmoid(self, x):
        return 1/(1+np.exp(-x))
    

    def log_likelihood(self, theta, split = "train"): ###  log likelihood for the parameter theta theta of  shape B, d

        if split == "train":
            X = self.X
            y = self.y
        else: 
            X = self.X_test
            y = self.y_test


        logits = np.dot(X, theta.T)
        lll = (y[:,None] * logits - np.log(1 + np.exp(logits))).sum(axis =  0)
        return lll
    

    
    
    def gradient_log_likelihood(self, theta): 
        ### theta can be a sample so of shape B, d

        logits = np.dot(self.X, theta.T)
        residuals  =  (self.y[:,None] - self.sigmoid(logits))[:,:, None]
        return (self.X[:,None,:] *  residuals).sum(axis = 0) ### shape  B, d 
    
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

    
    def evaluate_accuracy(self, vgmm, B = 1000,  noise = None, component_indices = None, jump = 1, split = "test"):

        if split == "test":
            X = self.X_test
            y = self.y_test
        else: 
            X = self.X
            y = self.y


        acc = []

        n_iter = len(vgmm.optimized_means)
        #### MEAN A POSTERIORI 
        for t in range(0, n_iter, jump ):   

            beta = vgmm.sample(B = B, noise = noise, component_indices=component_indices, t = t)
            logits = np.einsum("nd,bd->nb" , X, beta)
            probs = (1/(1 + np.exp(-logits)) )
            hat_y = (probs.mean(axis = -1)>0.5)*1
            acc.append((hat_y == y).mean())

        return np.array(acc)
    
    def evaluate_lll(self, vgmm, B = 1000, noise = None, component_indices = None, jump = 1, split = "train"):

        n_iter = len(vgmm.optimized_epsilons) 
        lll = []
        for t in range(0, n_iter, jump):
            samples =  vgmm.sample(B = B, noise=noise, component_indices = component_indices,   t = t) 
            lll.append(self.log_likelihood(samples, split=split))  # shape B
        
        return np.array(lll) 





class MultiClassLogReg(LogReg):
    def __init__(self, hidden_layers, n_classes=3, **kwargs):
        super().__init__(**kwargs)

        self.name = "mlogreg"

        self.init_model(hidden_layers)
        print("Data dim: ", self.data_dim)
        print("Parameters dim: ", self.param_dim)
        print("Num classes: ", self.n_classes)

        self.prior_mean = np.zeros(self.param_dim)
        self.prior_eps  = 100

        # self.prior = multivariate_normal(self.prior_mean, self.prior_eps * np.eye(self.param_dim))
        # self.prior = IsotropicGaussian(self.prior_mean, self.prior_eps)
        base = dist.Normal(loc=torch.as_tensor(self.prior_mean), scale=torch.tensor(self.prior_eps**(1/2)))
        self.prior = dist.Independent(base, 1)

    def init_model(self, hidden_layers):
        # hidden layers, output dim
        self.neural_network = SimpleMLP(data_dim = self.data_dim, hidden_sizes=hidden_layers, output_dim=self.output_dim, task='classification')
        self.param_dim = self.neural_network.param_dim
        self.dim = self.param_dim

    def generate_data(self, n_samples):
        np.random.seed(self.seed)
        n_samples_per_class = n_samples // self.n_classes
        
        # Ensure a covariance matrix is available.
        self.cov = self.cov if self.cov is not None else self.generate_cov()
        
        # Create class means:
        if self.dim == 2:
            # For 2D, equally space the means around a circle.
            angles = np.linspace(0, 2 * np.pi, self.n_classes, endpoint=False)
            self.means = [np.array([np.cos(a), np.sin(a)]) * self.meanShift for a in angles]
        else:
            # For d > 2, generate random directions on the unit hypersphere.
            self.means = []
            for _ in range(self.n_classes):
                vec = np.random.randn(self.dim)
                vec /= LA.norm(vec)
                self.means.append(vec * self.meanShift)
        
        # Generate data for each class.
        X_list = []
        y_list = []
        for i, mean in enumerate(self.means):
            X_i = np.random.multivariate_normal(mean, self.cov, n_samples_per_class)
            y_i = np.full(n_samples_per_class, i)
            X_list.append(X_i)
            y_list.append(y_i)
        
        # Concatenate all classes and shuffle.
        X = np.concatenate(X_list)
        y = np.concatenate(y_list)
        X = self.scaler.fit_transform(X)
        
        data = list(zip(y, X))
        np.random.shuffle(data)
        self.y, self.X = zip(*data)
        self.y = np.array(self.y)
        self.X = np.array(self.X)

    def log_likelihood(self, thetas, split = "train"):
        # theta is of shape B, d, K 
        if split == "train":
            X = self.X
            y = self.y
        else:
            X = self.X_test
            y = self.y_test

        B, _ = thetas.shape
        X_ = torch.from_numpy(X)
        y_ = torch.from_numpy(y)
        lll = np.zeros(B)
        for i in range(B):
            theta = thetas[i]
            theta_ = torch.from_numpy(theta)
            self.neural_network.set_weights_from_vector(theta_)
            # forward
            with torch.no_grad():
                probs_ = self.neural_network.forward(X_) # logits come from softmax of nn
                # compute lll
                lll_ = ((y_ * torch.log(torch.clip(probs_, min=1e-3))).sum(dim = 1)).sum()
                lll[i] = lll_.numpy()
        return lll

        # if theta.ndim == 2:
        #     theta = self.unpack_theta(theta)
        # logits = np.einsum("nd,bdk->bnk", X, theta) # B, n_samples, K 
        # exp_logits = np.exp(logits)
        # sum_exp_logits = exp_logits.sum(axis =  -1) # B, n_samples
        # return ((logits - np.log(sum_exp_logits)[..., None])[:, (y[..., None] == np.arange(0, K))]).sum(axis = -1) ### B 

    def grad_log_prior(self, theta):
        ### theta of shape B, d*k 
        # if theta.ndim == 3: # theta B, d , k
            # theta = self.flatten_theta(theta)
        return - (theta - self.prior_mean)/self.prior_eps

    def unpack_theta(self, theta):
        ## theta of shape B, d * k 
        return rearrange(theta, "B (d k) -> B d k", k = self.n_classes )
    
    def flatten_theta(self, theta):
        ### theta of  shape B, d, k
        return rearrange(theta, "B d k -> B (d k)")      

    # def gradient_log_likelihood(self, theta):
    #     
    #     if theta.ndim == 2: # theta B, (d * k)
    #         theta = self.unpack_theta(theta)
    #
    #     ##  else theta already theta B, d , k
    #     logits= np.einsum("nd,bdk->bnk", self.X, theta) # B, n_samples, K 
    #     exp_logits = np.exp(logits)
    #
    #     denominator = exp_logits.sum(axis =  -1)
    #     probs = exp_logits/denominator[..., None]
    #
    #     indicatrix = (self.y[..., None] == np.arange(0, self.n_classes))
    #     grad_forall_k = np.einsum("nd,bnk->bdk", self.X, (indicatrix[None, :, :] - probs))
    #     grad_flatten = rearrange(grad_forall_k, "B d k -> B (d k)" )
    #     return  grad_flatten

    def batchized_data(self, M = 32):
        indices = np.random.permutation(len(self.X))[:M]
        X = self.X[indices]
        y = self.y[indices]
        return X, y

    def gradient_log_likelihood(self, thetas):
        B, d = thetas.shape
        X, y = self.batchized_data()
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
            probs_ = self.neural_network.forward(X_) # logits come from softmax of nn
            # compute lll and backward
            lll_ = ((y_ * torch.log(torch.clip(probs_, min=1e-3))).sum(dim = 1)).sum()
            lll_.backward()
            # store gradient
            grad = self.neural_network.get_gradients_as_vector().numpy()
            grad_thetas[i] = grad
        return grad_thetas
    
    def gradient_log_density(self, theta): 
        ### theta can be a sample so of shape B, d*k or B, d, k 
        gll = self.gradient_log_likelihood(theta)
        # print(f"{np.isnan(gll).any()=}")
        glp = self.grad_log_prior(theta)
        # print(f"{np.isnan(glp).any()=}")
        return  gll + glp  ### flatten so of shape B , (d*k)
    
    def log_prob(self, theta): ###  log density of the posterior UNORMALIZED
        return self.log_likelihood(theta) + self.prior.log_prob(torch.as_tensor(theta)).numpy()

    def evaluate_accuracy(self, vgmm, B=1000, noise=None, component_indices=None, jump=1, split="test"):

        if split == "test":
            X = self.X_test
            y = self.y_test
        else: 
            X = self.X
            y = self.y


        acc = []

        n_iter = len(vgmm.optimized_means)

        for t in range(0, n_iter, jump):
            thetas = vgmm.sample(B = B, noise = noise, component_indices = component_indices, t = t)
            N, _ = X.shape
            X_ = torch.from_numpy(X)
            K = self.n_classes
            probs = np.zeros((B, N, K))
            for i in range(B):
                theta = thetas[i]
                theta_ = torch.from_numpy(theta)
                self.neural_network.set_weights_from_vector(theta_)
                # forward
                with torch.no_grad():
                    probs_ = self.neural_network.forward(X_) # logits come from softmax of nn
                    probs[i] = probs_.numpy()

            y_hat = probs.mean(axis = 0).argmax(axis = -1) # mean over all samples B, then get class with argmax
            assert y_hat.shape[0] == N
            assert y_hat.size == N
            #
            y_argmax = y.argmax(axis = -1)
            assert y_argmax.shape[0] == N
            assert y_argmax.size == N
            #
            acc.append((y_hat == y_argmax).mean())
        return acc



      
    
    
