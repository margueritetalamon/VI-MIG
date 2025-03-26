import numpy as np
from sklearn.preprocessing import StandardScaler
import numpy.linalg as LA 
from scipy.stats import multivariate_normal 
from scipy.stats import special_ortho_group


from src_bis.gmm import GMM

import  math

class LogReg:
    def __init__(self, dataset=None, n_samples=100, d=2, Z = 100, meanShift = 1, cov = None, seed = 1, prior_mean = None, prior_eps = None ):
        
        self.scaler = StandardScaler()
        self.Z = Z
        self.fixed_theta = None
        self.meanShift = meanShift
        self.cov  = cov
        self.seed = seed

        if dataset is None:
            self.dim = d
            self.n_samples = n_samples
            self.generate_data(n_samples )
        else:
            self.load_data(dataset)
 
        self.prior_mean = prior_mean if prior_mean is not None else np.zeros(self.dim)
        self.prior_eps  = prior_eps if prior_eps is not None else 1

        self.prior = multivariate_normal(self.prior_mean, self.prior_eps * np.eye(self.dim))



        print(self.dim)


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
        self.n_samples, self.dim = X.shape
        
    def sigmoid(self, x):
        return 1/(1+np.exp(-x))
    

    def log_likelihood(self, theta): ###  log likelihood for the parameter theta theta of  shape B, d
        logits = np.dot(self.X, theta.T)
        lll = (self.y[:,None] * logits - np.log(1 + np.exp(logits))).sum(axis =  0)
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