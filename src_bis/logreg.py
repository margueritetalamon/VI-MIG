import numpy as np
from sklearn.preprocessing import StandardScaler
import numpy.linalg as LA 
from scipy.stats import multivariate_normal 
from scipy.stats import special_ortho_group
from einops import rearrange


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











class NeuralNetwork:
    def __init__(self, input_dim, hidden_units=50, output_dim=1, init_scale=0.01, prior_variance=1.0):
        """
        Initializes the network parameters for a batch (num_samples) of networks.
        
        Args:
            input_dim: Dimensionality of the input features.
            hidden_units: Number of neurons in the hidden layer (default 50).
            output_dim: Dimensionality of the output (default 1, for binary classification).
            init_scale: Scaling factor for random weight initialization.
            prior_variance: Variance for the normal prior on parameters.
            num_samples: Number of parameter samples (i.e. networks) to use.
        """
        self.input_dim = input_dim 
        self.hidden_units = hidden_units 
        self.output_dim = output_dim
        
        self.prior_variance = prior_variance

        # Initialize parameters with an extra dimension for the parameter samples.
        # self.params = {
        #     'W1': np.random.randn(num_samples, input_dim, hidden_units) * init_scale,
        #     'b1': np.zeros((num_samples, 1, hidden_units)),
        #     'W2': np.random.randn(num_samples, hidden_units, output_dim) * init_scale,
        #     'b2': np.zeros((num_samples, 1, output_dim))
        # }
        self.dim_params = input_dim * self.hidden_units + 2 * self.hidden_units + 1


    def flatten_parameters(self):


        B  = self.params["W1"].shape[0]
        return np.concatenate([self.params["W1"].reshape(B, -1), 
                               self.params["b1"].reshape(B, -1), 
                               self.params["W2"].reshape(B, -1), 
                               self.params["b2"].reshape(B, -1)], axis  =  -1)
    
    def unpack_parameters(self, flat_params):


        B = flat_params.shape[0]
        idx = 0
        W1 = flat_params[:, idx : idx + self.input_dim*self.hidden_units].reshape(B, self.input_dim, self.hidden_units)
        idx += self.input_dim*self.hidden_units


        b1 = flat_params[:, idx : idx + self.hidden_units].reshape(B, 1, self.hidden_units)
        idx += self.hidden_units

        W2 = flat_params[:, idx : idx + self.output_dim*self.hidden_units].reshape(B,  self.hidden_units, self.output_dim)
        idx += self.output_dim*self.hidden_units


        b2 = flat_params[:, idx : ].reshape(B, 1, self.output_dim)

        return W1, b1, W2, b2


    def update_params(self, samples_params):

        W1, b1, W2, b2  = self.unpack_parameters(samples_params)
        self.params = {
            'W1': W1,
            'b1': b1,
            'W2': W2,
            'b2': b2
        }
        







    
    def relu(self, x):
        """Applies the ReLU activation function elementwise."""
        return np.maximum(0, x)
    
    def relu_grad(self, x):
        """Computes the gradient of the ReLU function elementwise."""
        return (x > 0).astype(float)
    
    def sigmoid(self, x):
        """Applies the sigmoid activation function elementwise."""
        return 1 / (1 + np.exp(-x))
    
    def flatten_gradient(self, gradient):
        """
        Flattens the gradients for each sample into a single vector.
        
        Args:
            gradient: A dictionary with keys 'W1', 'b1', 'W2', and 'b2', each of shape 
                      (B, ...).
                      
        Returns:
            A numpy array of shape (B, total_params) where total_params is the
            sum of all parameter sizes.
        """
        B = gradient["W1"].shape[0]

        flat = np.concatenate([
            gradient["W1"].reshape(B, -1),
            gradient["b1"].reshape(B, -1),
            gradient["W2"].reshape(B, -1),
            gradient["b2"].reshape(B, -1)
        ], axis=1)
        return flat
    
    def unpack_gradient(self, flat_gradient):
        """
        Unpacks a flattened gradient array into its components.
        
        Args:
            flat_gradient: A numpy array of shape (B, total_params).
            
        Returns:
            Tuple of gradients (W1, b1, W2, b2) with shapes matching the original network.
        """
        idx = 0
        B = flat_gradient.shape[0]
        total_W1 = self.input_dim * self.hidden_units
        W1 = flat_gradient[:, idx: idx + total_W1].reshape(B, self.input_dim, self.hidden_units)
        idx += total_W1
        
        total_b1 = 1 * self.hidden_units
        b1 = flat_gradient[:, idx: idx + total_b1].reshape(B, 1, self.hidden_units)
        idx += total_b1
        
        total_W2 = self.hidden_units * self.output_dim
        W2 = flat_gradient[:, idx: idx + total_W2].reshape(B, self.hidden_units, self.output_dim)
        idx += total_W2
        
        total_b2 = 1 * self.output_dim
        b2 = flat_gradient[:, idx: idx + total_b2].reshape(B, 1, self.output_dim)


        return W1, b1, W2, b2

    def forward(self, x):
        """
        Performs a forward pass through the network for each parameter sample.
        
        Args:
            x: Input data of shape (m, input_dim) where m is the number of observations and input_dim the data dimension
        
        Returns:
            out: Network outputs after sigmoid activation, of shape (B, m, output_dim).
            cache: A tuple (x, z1, a1, z2, out) storing intermediate values for backpropagation.
                   Note: x is kept unbatched; the others include the parameter sample dimension.
        """
        # Unpack parameters (each with shape (num_samples, ...))
        W1, b1, W2, b2 = self.params['W1'], self.params['b1'], self.params['W2'], self.params['b2']
        m = x.shape[0]
        
        # Compute hidden layer pre-activation for each sample:
        # Using einsum to compute for each sample: z1[s] = x.dot(W1[s])
        z1 = np.einsum("mi,sih->smh", x, W1) + b1  # Shape: (B, m, hidden_units)
        a1 = self.relu(z1)  # Apply ReLU elementwise
        
        # Compute output layer:
        # For each sample, a1[s] dot W2[s]
        z2 = np.matmul(a1, W2) + b2  # Shape: (num_samples, m, output_dim)
        out = self.sigmoid(z2)      # Apply sigmoid elementwise
        
        cache = (x, z1, a1, z2, out)
        return out, cache
    
    def compute_gradients(self, x, y):
        """
        Computes gradients of the network parameters using backpropagation for each parameter sample.
        
        The loss is the sum of the negative log likelihood (binary cross-entropy) and 
        the negative log prior (normal prior -> L2 regularization).
        
        Args:
            x: Input data of shape (m, input_dim).
            y: True binary labels of shape (m, output_dim) with values 0 or 1.
        
        Returns:
            grads: A dictionary with gradients for 'W1', 'b1', 'W2', and 'b2',
                   each of shape matching the batched parameters.
        """
        m = x.shape[0]
        # Forward pass for each parameter sample:
        y_pred, cache = self.forward(x)
        x, z1, a1, z2, out = cache  # y_pred has shape (num_samples, m, output_dim)
        
        # Broadcast y to (num_samples, m, output_dim) and compute derivative of loss:
        dz2 = (y_pred - y[None, :, :]) / m  # Shape: (num_samples, m, output_dim)
        
        # Gradients for the output layer:
        # For each sample: dW2[s] = a1[s].T dot dz2[s]
        dW2 = np.matmul(a1.transpose(0, 2, 1), dz2)  # (num_samples, hidden_units, output_dim)
        db2 = np.sum(dz2, axis=1, keepdims=True)       # (num_samples, 1, output_dim)
        
        # Backpropagate to the hidden layer:
        # For each sample: da1[s] = dz2[s] dot W2[s].T
        dW2_T = self.params['W2'].transpose(0, 2, 1)  # (num_samples, output_dim, hidden_units)
        da1 = np.matmul(dz2, dW2_T)  # (num_samples, m, hidden_units)
        dz1 = da1 * self.relu_grad(z1)  # (num_samples, m, hidden_units)
        
        # Gradients for the hidden layer:
        # For each sample: dW1[s] = x.T dot dz1[s]
        dW1 = np.einsum("mi,smh->sih", x, dz1)  # (num_samples, input_dim, hidden_units)
        db1 = np.sum(dz1, axis=1, keepdims=True)  # (num_samples, 1, hidden_units)
        
        # Add gradient contributions from the negative log prior (L2 regularization)
        dW2 += self.params['W2'] / self.prior_variance
        db2 += self.params['b2'] / self.prior_variance
        dW1 += self.params['W1'] / self.prior_variance
        db1 += self.params['b1'] / self.prior_variance
        
        grads = {'W1': dW1, 'b1': db1, 'W2': dW2, 'b2': db2}
        return grads


class LogReg_withBNN:
    def __init__(self, dataset=None, n_samples=100, d_data=2,  Z = 100, meanShift = 1, cov = None, seed = 1, prior_mean = None, prior_eps = None):
        
        self.scaler = StandardScaler()
        self.Z = Z
        self.fixed_theta = None
        self.meanShift = meanShift
        self.cov  = cov
        self.seed = seed
        self.name = "LogRegBNN"

        if dataset is None:
            self.dim_data = d_data
            self.n_samples = n_samples
            self.generate_data(n_samples )
        else:
            self.load_data(dataset)


        self.BNN = NeuralNetwork(self.dim_data, hidden_units=50, output_dim=1)

        self.dim_params = self.BNN.dim_params
        self.dim = self.BNN.dim_params

        self.prior_mean = prior_mean if prior_mean is not None else np.zeros(self.dim_params)
        self.prior_eps  = prior_eps if prior_eps is not None else 1

        self.prior = multivariate_normal(self.prior_mean, self.prior_eps * np.eye(self.dim_params))



        print("Dim Theta", self.dim_params)
        print("Dim data", self.dim_data)


    def generate_cov(self,c = 1, scale = 1, rotate = True, normalize = True ):

        vec = (1/np.arange(1,self.dim_data+1)**c)*scale**2
        if normalize:
            vec=vec/np.linalg.norm(vec)**2
        cov = np.diag(vec)

        if rotate:
            local_rng = np.random.RandomState(self.seed)
            Q = special_ortho_group.rvs(dim=self.dim_data,  random_state=local_rng)
            cov=np.transpose(Q).dot(cov).dot(Q)

        return cov 

    def generate_data(self, n_samples):

        state = np.random.get_state()
        np.random.seed(self.seed)
        mean = np.random.rand(self.dim_data)
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
        self.n_samples, self.dim_data = X.shape
        
    def sigmoid(self, x):
        return 1/(1+np.exp(-x))
    

    def log_likelihood(self, theta): ###  log likelihood for the parameter theta theta of  shape B, d

        self.BNN.update_params(theta)
        pred , _ = self.BNN.forward(self.X)


        lll = ((self.y[None, :,None] * np.log(pred)) +  ((1 - self.y)[None, :, None] * np.log(1  - pred)))      # B, nobs, 1  
        lll = lll.sum(axis =  1)
        return lll[:, 0]
        
    def grad_log_prior(self, theta):
        ### theta of shape B,d 
        return - (theta - self.prior_mean)/self.prior_eps
    
        
    def gradient_log_density(self, theta): 
        ### theta can be a sample so of shape B, d
        self.BNN.update_params(theta)

        grad_log_pi = self.BNN.compute_gradients(self.X, self.y[..., None])
        grad_log_pi = self.BNN.flatten_gradient(grad_log_pi)

        return grad_log_pi
    
    
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







class MultiClassLogReg(LogReg):
    def __init__(self, n_classes=3, **kwargs):
        super().__init__(**kwargs)



        self.n_classes = len(set(kwargs["dataset"][-1].tolist())) if kwargs["dataset"] is not None else n_classes

        self.data_dim = self.dim
        self.param_dim = self.data_dim * self.n_classes
        self.dim = self.param_dim
        self.name = "Multi_LogReg"


        self.prior_mean = np.zeros(self.dim)
        self.prior_eps  = 1


        self.prior = multivariate_normal(self.prior_mean, self.prior_eps * np.eye(self.dim))



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


        ###  loglikelihood 
    def log_likelihood(self, theta):
        # theta is of shape B, d, K 

        if theta.ndim == 2:
            theta = self.unpack_theta(theta)

        K = self.n_classes

        logits = np.einsum("nd,bdk->bnk", self.X, theta) # B, n_samples, K 
        exp_logits = np.exp(logits)
        sum_exp_logits = exp_logits.sum(axis =  -1) # B, n_samples
        ((logits - np.log(sum_exp_logits)[..., None])[:, (self.y[..., None] == np.arange(0, K))]) ### B, n_samples

        return ((logits - np.log(sum_exp_logits)[..., None])[:, (self.y[..., None] == np.arange(0, K))]).sum(axis = -1) ### B 




    def grad_log_prior(self, theta):
            ### theta of shape B, d*k 

            if theta.ndim == 3: # theta B, d , k
                theta = self.flatten_theta(theta)


            return - (theta - self.prior_mean)/self.prior_eps


    def unpack_theta(self, theta):
        ## theta of shape B, d * k 
        return rearrange(theta, "B (d k) -> B d k", k = self.n_classes )
    
    def flatten_theta(self, theta):
        ### theta of  shape B, d, k
        return rearrange(theta, "B d k -> B (d k)")      
    


    def gradient_log_likelihood(self, theta):
        
        if theta.ndim == 2: # theta B, (d * k)
            theta = self.unpack_theta(theta)

        ##  else theta already theta B, d , k
        logits= np.einsum("nd,bdk->bnk", self.X, theta) # B, n_samples, K 
        exp_logits = np.exp(logits)

        denominator = exp_logits.sum(axis =  -1)
        probs = exp_logits/denominator[..., None]

        indicatrix = (self.y[..., None] == np.arange(0, self.n_classes))
        grad_forall_k = np.einsum("nd,bnk->bdk", self.X, (indicatrix[None, :, :] - probs))
        grad_flatten = rearrange(grad_forall_k, "B d k -> B (d k)" )
        return  grad_flatten
    

    def gradient_log_density(self, theta): 
        ### theta can be a sample so of shape B, d*k or B, d, k 

        return self.gradient_log_likelihood(theta) + self.grad_log_prior(theta)  ### flatten so of shape B , (d*k)
    

    def log_prob(self, theta): ###  log density of the posterior UNORMALIZED

        return self.log_likelihood(theta) + self.prior.logpdf(theta)



      
    