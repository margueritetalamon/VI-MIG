import numpy as np
import torch.distributions as dist
import torch
from einops import rearrange
from matplotlib import pyplot as plt



class GMM:
    def __init__(self, variational = True, mode ="full", weights = None, means = None, covs = None, n_components = 3, d = 2, s = 10, scale = 2):

        self.variational = variational
        self.mode = mode
      
        self.init_gmm(weights , means , covs , n_components, d = d, s = s, scale = scale)


        if self.variational:
            self.optimized_means = [self.means]
            self.optimized_covs = [self.covariances]
            if self.mode == "iso":
                self.optimized_epsilons = [self.epsilons]



    
    def init_gmm(self, weights = None, means = None, covs = None, n_components = 3 , d = 2, s = 10, scale = 2):
        
        if means is not None: 
            self.n_components, self.dim = means.shape
        else:
            self.n_components, self.dim = n_components, d
        
        self.weights = weights if weights is not None else self.generate_random_weights(self.n_components)
        self.means = means if means is not None else self.generate_random_means(s)
        self.covariances = covs if covs is not None else self.generate_random_covs(self.n_components, self.dim, self.mode, scale, self.variational)
        self.invcov = np.array([np.linalg.inv(cov) for cov in self.covariances])

        if self.mode == "iso":
            self.epsilons = self.covariances[:,0,0]  ### this is already squared N(0, epsilon* Id)


        self.gaussians = dist.MultivariateNormal(torch.as_tensor(self.means), covariance_matrix=torch.as_tensor(self.covariances)) 



    def generate_random_means(self, s):
        return np.random.uniform(low=-s, high=s, size=(self.n_components, self.dim))

    @staticmethod
    def generate_random_covs(n, d, mode, scale, variational = False):
        covs = []
  

        for _ in range(n):
            if mode == "full":
                A = np.random.randn(d, d)
                cov = A @ A.T  # Ensures positive semi-definiteness
                cov /= np.max(np.abs(cov))  # Normalize to prevent large values
                cov *= scale

            elif mode == "diag":
                diag = np.random.uniform(1, 10, d) 
                cov = np.diag(diag)
                cov /= np.max(np.abs(cov))  
                cov *= scale

            elif mode == "iso":
                if variational:
                    coef = scale
                else:
                    coef = np.random.randn(1)**2 + scale
                cov = coef*np.eye(d)
            else:
                raise ValueError("Covariance type not defined, please choose between : ['diag', 'iso', 'full']")


            covs.append(cov)
        
        return np.array(covs)

    def generate_random_weights(self, n):

        if self.variational:
            weights = np.ones((n,))/n
        else:
            weights = np.random.randint(low = 1, high = n*2, size = (n,))
            weights = weights /  weights.sum()

        return weights
    

    

    def prob(self, x):
        ### x needs to be B, 1, d

        if not isinstance(x, torch.Tensor):
            x = torch.as_tensor(x)

        return (self.gaussians.log_prob(x).exp() * self.weights).sum(dim = -1).numpy()
    
    def log_prob(self,x):
        return np.log(self.prob(x))
    

    def sample(self, B, noise = None, component_indices = None):

        if noise is None:
            print("Generating noise")
            noise = torch.randn(B, self.dim) 

        if component_indices is None:
            print("Generating noise")
            component_indices = np.random.choice(self.n_components, size=B, p=self.weights)


        selected_means = torch.as_tensor(self.means)[component_indices]  # (B, d)
        L = torch.linalg.cholesky(torch.as_tensor(self.covariances))
        selected_covs = L[component_indices]  # (B, d, d)
        samples = selected_means +  torch.einsum("bij,bj->bi", selected_covs.float(), noise.float())

        return samples.numpy()
    


    # def sample_from_each_gaussian(self, noise = None, B = 1):

    #     if noise is None :
    #         noise = np.random.randn(B, self.d)
    #     return self.means[:,None,:] +  (np.sqrt(self.epsilons[:,None, None]) * noise)
    

    def gradient_log_density(self, x): #### grad of the log density, ie - grad V 

        invcov_by_centered = np.einsum("ndd,bnd->bnd", self.invcov, (x[:,None] - self.means[None])) ### gives B, N, d.  B, 1, d - 1, N, d -> B, N, d
        probs = self.gaussians.log_prob(torch.as_tensor(x[:,None])).exp().numpy()[..., None] ### gives B,N,1
        numerator = - (self.weights[None,:,None] * invcov_by_centered * probs).sum(axis = 1) ### B, d
        denominator = self.prob(x[:, None])[:, None]

        return numerator/denominator
    
    
    def get_means_evolution(self):
        return np.array(self.optimized_means)

    def get_covs_evolution(self):
        return np.array(self.optimized_covs)
    
    def get_params_evolution(self):
        return self.get_means_evolution(), self.get_covs_evolution()
    
    def negative_entropy(self, noise = None, B = 1000):
        ### mc estimation of the neg entropy
        samples = self.sample(B = B, noise = noise)
        return self.log_prob(samples[:,None]).mean()

    def compute_KL(self, vgmm, noise = None, component_indices = None ,  B = 1000):
        
        samples = vgmm.sample(B, noise, component_indices)
   
        # return np.log(vgmm.prob(samples[:,None]) / self.prob(samples[:,None])).mean()
        return (vgmm.log_prob(samples[:,None]) - self.log_prob(samples[:,None])).mean()
    



class IGMM(GMM):
    def __init__(self, means = None, covs = None, n_components = 3, d = 2, s = 10, scale = 1):

        self.mode = "iso"
        self.variational = True
        self.init_gmm(n_components=n_components, means = means,  covs =  covs, s = s, scale = scale, d = d)

        self.optimized_means = [self.means]
        self.optimized_covs = [self.covariances]
        self.optimized_epsilons = [self.epsilons]

    

    def sample(self, B, noise = None, component_indices = None):

        if noise is None:
            noise = np.random.randn(B, self.dim) 
            print("Generating noise")
        if component_indices is None:
            component_indices = np.random.choice(self.n_components, size=B, p=self.weights)


        selected_means = self.means[component_indices]  # (B, d)
        selected_epsilons = self.epsilons[component_indices]  # (B, d, d)
        samples = selected_means +  (np.sqrt(selected_epsilons[:,None]) * noise)

        return samples
    


    def sample_from_each_gaussian(self, noise = None, B = 1):

        if noise is None:
            # print("Generating noise")
            noise = np.random.randn(B, self.dim)

        return self.means[:,None,:] +  (np.sqrt(self.epsilons[:,None, None]) * noise)
        # return self.gaussians.sample((B,)).numpy() ### shape B, N, d
    
    ### unifrom weights and isotropic
    def gradient_log_density(self, x): #### grad of the log density, ie - grad V 
        
        invcov_by_centered = ((x[:,None] - self.means[None]) / self.epsilons[None,:,None]) ### gives B, N, d.  B, 1, d - 1, N, d -> B, N, d
        probs = self.gaussians.log_prob(torch.as_tensor(x[:,None])).exp().numpy()[..., None] ### gives B,N,1
        # print(type(probs))
        # print(type(self.weights))
        # print(type(invcov_by_centered))
        numerator = - (self.weights[None,:,None] * invcov_by_centered * probs).sum(axis = 1) ### B, d
        denominator = self.prob(x[:, None])[:, None]

        return numerator/denominator
    

    def update(self, new_means, new_epsilons):

        
        self.optimized_means.append(new_means)
        self.optimized_epsilons.append(new_epsilons)

        new_covs = (new_epsilons[:,None,None] * np.eye(self.dim))
        self.optimized_covs.append(new_covs)


        self.epsilons = new_epsilons
        self.means = new_means
        self.covariances = new_covs

        self.gaussians = dist.MultivariateNormal(torch.as_tensor(self.means), covariance_matrix=torch.as_tensor(self.covariances)) 


    def get_epsilons_evolution(self):
        return np.array(self.optimized_epsilons)
    

    def get_means_evolution(self):
        return np.array(self.optimized_means)
    

    def get_params_evolution(self):
        return self.get_means_evolution(), self.get_epsilons_evolution()

    
    def compute_grads_iso(self, target, noise = None, B = 1, optim_epsilon = True):

        samples = self.sample_from_each_gaussian(noise = noise, B = B) # n, b, d

        samples_flat = rearrange(samples, "n b d -> (n b) d")

        grad_log_vgmm = self.gradient_log_density(samples_flat)
        grad_log_pi = target.gradient_log_density(samples_flat)
     
        centered_samples = samples - self.means[:,None] ### n, b, d


        # print(grad_log_pi.shape)
        # print(grad_log_vgmm.shape)

        grad_log_pi = rearrange(grad_log_pi , "(n b) d -> n b d", b = B)
        grad_log_vgmm = rearrange(grad_log_vgmm , "(n b) d -> n b d", b = B)

        grad_means = (grad_log_vgmm - grad_log_pi).mean(axis = 1)/self.n_components


        if optim_epsilon:

            E_grad_log_pi_centered_s = (grad_log_pi*centered_samples).sum(axis = -1).mean(axis = -1) ### dot product and mean, sum over dimension, mean over batch
            E_grad_log_vgmm_centered_s = (grad_log_vgmm*centered_samples).sum(axis = -1).mean(axis = -1) ### dot product and mean, sum over dimension, mean over batch 

            grad_epsilons = (E_grad_log_vgmm_centered_s - E_grad_log_pi_centered_s)/(2*self.n_components*self.epsilons) ### n, b
        else:
            grad_epsilons = None

        return grad_means, grad_epsilons


    def plot_circle(self, t, ax, bound = 20):

        for i in range(self.n_components):
            center = self.optimized_means[t][i]
            circle = plt.Circle(
                (center[0], center[1]),
                radius=self.optimized_epsilons[t][i],
                color = "black",
                fill=False,
                linewidth=1,
                zorder=10
            )
        
            ax.add_patch(circle)

        ax.set_xlim(-bound, bound)
        ax.set_ylim(-bound, bound)
        ax.set_aspect('equal')

    
    def plot_evolution(self, ax, jump = 100, bound = 20):

        for t in range(0, len(self.optimized_means), jump):
            self.plot_circle(t, ax, bound)







