import numpy as np
import torch.distributions as dist
import torch
from einops import rearrange
from matplotlib import pyplot as plt
from scipy.stats import norm 
from matplotlib.patches import Ellipse
import math 


class GMM: 
    def __init__(self, variational = True, mode ="full", weights = None, means = None, covs = None, n_components = 3, d = 2, s = 10, scale = 2, seed = None):

        self.variational = variational
        self.mode = mode
        self.contours = None
        self.num_stab = 0

      
        self.init_gmm(weights , means , covs , n_components, d = d, s = s, scale = scale, seed = seed)


        if self.variational:
            self.optimized_means = [self.means]
            if self.mode == "iso":
                self.optimized_epsilons = [self.epsilons]
            elif self.mode == "full":
                self.optimized_covs = [self.covariances]




    
    def init_gmm(self, weights = None, means = None, covs = None, n_components = 3 , d = 2, s = 10, scale = 2, seed = None):
        
        if means is not None: 
            self.n_components, self.dim = means.shape
        else:
            self.n_components, self.dim = n_components, d
        
        self.weights = weights if weights is not None else self.generate_random_weights(self.n_components, seed = seed)
        self.means = means if means is not None else self.generate_random_means(s, seed = seed)
        self.covariances = covs if covs is not None else self.generate_random_covs(self.n_components, self.dim, self.mode, scale, self.variational, seed = seed)
        self.invcov = np.array([np.linalg.inv(cov) for cov in self.covariances])

        if self.mode == "iso":
            self.epsilons = self.covariances[:,0,0]  ### this is already squared N(0, epsilon* Id)


        self.gaussians = dist.MultivariateNormal(torch.as_tensor(self.means), covariance_matrix=torch.as_tensor(self.covariances)) 



    def generate_random_means(self, s, seed = None):
        rng = np.random.RandomState(seed=seed)

        return rng.uniform(low=-s, high=s, size=(self.n_components, self.dim))

    @staticmethod
    def generate_random_covs(n, d, mode, scale, variational = False, seed = None):
        covs = []

        rng = np.random.RandomState(seed=seed)

        for _ in range(n):
            if mode == "full":
                A = rng.randn(d, d)
                cov = A @ A.T  # Ensures positive semi-definiteness
                cov /= np.max(np.abs(cov))  # Normalize to prevent large values
                cov *= scale
                cov = (cov + cov.T) / 2
                cov += np.eye(d) * 1e-6 

            elif mode == "diag":
                diag = rng.uniform(1, 10, d) 
                cov = np.diag(diag)
                cov /= np.max(np.abs(cov))  
                cov *= scale

            elif mode == "iso":
                if variational:
                    coef = scale
                else:
                    coef = rng.randn(1)**2 + scale
                cov = coef*np.eye(d)
            else:
                raise ValueError("Covariance type not defined, please choose between : ['diag', 'iso', 'full']")


            covs.append(cov)
        
        return np.array(covs)

    def generate_random_weights(self, n, seed = None):
        rng = np.random.RandomState(seed=seed)

        if self.variational:
            weights = np.ones((n,))/n
        else:
            weights = rng.randint(low = 1, high = n*2, size = (n,))
            weights = weights /  weights.sum()

        return weights
    


    def prob(self, x):
        ### x needs to be B, 1, d

        log_prob = self.gaussians.log_prob(torch.as_tensor(x[:,None]))

        log_prob_c = np.clip(log_prob , -700, 700).numpy()
        prob_c = np.exp(log_prob_c)

        return (self.weights[None] * prob_c).sum(axis = -1)
    

    def log_prob(self,x):
        return np.log(self.prob(x) + self.num_stab)
    

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
    
    def gradient_log_density(self, x): #### PREVENT NUMERICAL INTABILITY

        invcov_by_centered = np.einsum("ndd,bnd->bnd", self.invcov, (x[:,None] - self.means[None])) ### gives B, N, d.  B, 1, d - 1, N, d -> B, N, d
        log_prob = self.gaussians.log_prob(torch.as_tensor(x[:,None]))[..., None].numpy()# B, Ntarget
        log_prob_c = np.clip(log_prob , -700, 700, dtype = np.float64)
        prob_c = np.exp(log_prob_c)

        numerator = - (self.weights[None,:,None] * invcov_by_centered * prob_c).sum(axis = 1)
        denominator = (self.weights[None,:,None]*prob_c).sum(axis = 1)

        return (numerator + self.num_stab)/(denominator + self.num_stab)
    
    
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

    def compute_KL(self, vgmm, noise = None, component_indices = None ,  B = 1000 ):
        samples = vgmm.sample(B, noise, component_indices)   
        return (vgmm.log_prob(samples[:,None] + self.num_stab) - self.log_prob(samples[:,None] + self.num_stab)).mean()
    


    
    def compute_marginals(self,fig = None,  axes = None, t = -1,  x1= 5,x2 = 5, grid_size =  100, ncols = 10, label = None, color = "black", lw = 2):
        x_grid = np.linspace(x1, x2, grid_size)

        nrows = math.ceil(self.dim / ncols)

        if self.variational:
            means = self.optimized_means[t]
            if self.mode == "iso":
                covs = self.optimized_epsilons[t][:, None, None] * np.eye(self.dim)[None]
            else:
                covs = self.optimized_covs[t]
        else:
            means = self.means
            covs = self.covariances

        if axes is None:

            fig, axes = plt.subplots(
                nrows=nrows,
                ncols=ncols,
                figsize=(2.5 * ncols, 2.0 * nrows),
                sharex=True,
                sharey=False,
                constrained_layout=True
            )
            axes = axes.flatten()

        for j in range(self.dim):

        
            target_marginals = [
                norm.pdf(x_grid, loc=pim[j], scale=np.sqrt(pic[j, j]))
                for pim, pic in zip(means, covs)
            ]
            Z_target = (np.array(target_marginals) * self.weights[..., None]).sum(axis = 0)

            axes[j].plot(x_grid, Z_target, label=label, color=color, linestyle="--", linewidth=lw)
            axes[j].set_yticks([])

        handles, plot_labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, plot_labels, loc='upper right', fontsize=12)

        return fig, axes




        

    
    def plot_estimated(self, grid_size = 100,  x1 = -10, x2 = 10, y1=-10, y2 = 10):

        fig, ax = plt.subplots()


        if self.dim == 2:


            if self.contours:
                ax.contour(self.contours[0],self.contours[1], self.contours[2], levels=10, cmap="viridis")

            else:
                x = np.linspace(x1, x2, grid_size)
                y = np.linspace(y1, y2, grid_size)
                X, Y = np.meshgrid(x, y)
                pos = np.dstack((X, Y))[:, :, None, :]
                Z = self.prob(pos)
                Z = Z[:,0,:]
                ax.contour(X, Y, Z, levels=20, cmap="viridis")
                self.contours = (X,Y,Z)

        return fig, ax

    


class IGMM(GMM):
    def __init__(self, means = None, covs = None, n_components = 3, d = 2, s = 10, scale = 1):

        self.mode = "iso"
        self.variational = True
        self.contours = None
        self.num_stab = 0

        self.init_gmm(n_components=n_components, means = means,  covs =  covs, s = s, scale = scale, d = d)

        self.optimized_means = [self.means]
        self.optimized_covs = [self.covariances]
        self.optimized_epsilons = [self.epsilons]

    

    def sample(self, B, noise = None, component_indices = None, t = None):

        if noise is None:
            noise = np.random.randn(B, self.dim) 
        if component_indices is None:
            component_indices = np.random.choice(self.n_components, size=B, p=self.weights)

        if t is not None:
            means = self.optimized_means[t]
            epsilons = self.optimized_epsilons[t]
        
        else :
            means = self.means
            epsilons = self.epsilons


        selected_means = means[component_indices]  # (B, d)
        selected_epsilons = epsilons[component_indices]  # (B, d, d)
        samples = selected_means +  (np.sqrt(selected_epsilons[:,None]) * noise)

        return samples
    


    def sample_from_each_gaussian(self, noise = None, B = 1):

        if noise is None:
            noise = np.random.randn(B, self.dim)

        return self.means[:,None,:] +  (np.sqrt(self.epsilons[:,None, None]) * noise)
    
    
    def gradient_log_density(self, x): #### grad of the log density, ie - grad V  PREVENT NUMERICAL INSTABILITY
        
        invcov_by_centered = ((x[:,None] - self.means[None]) / self.epsilons[None,:,None]) ### gives B, N, d.  B, 1, d - 1, N, d -> B, N, d


        log_prob = self.gaussians.log_prob(torch.as_tensor(x[:,None], dtype = torch.float64))[..., None].numpy()        
        log_prob_c = np.clip(log_prob , -700, 700, dtype = np.float64)
        prob_c = np.exp(log_prob_c)
    
    
        numerator = - (self.weights[None,:,None] * invcov_by_centered * prob_c).sum(axis = 1) ### B, d
        denominator = (self.weights[None,:,None] * prob_c).sum(axis = 1)

        return (numerator + self.num_stab)/(denominator + self.num_stab)



  

    def update(self, new_means, new_epsilons):

        new_covs = (new_epsilons[:,None,None] * np.eye(self.dim))
        # self.optimized_covs.append(new_covs)
        # self.optimized_epsilons.append(new_epsilons)
        # self.optimized_means.append(new_means) ### Stop storing, too heavy when d big


        self.epsilons = new_epsilons
        self.means = new_means
        self.covariances = new_covs

        self.gaussians = dist.MultivariateNormal(torch.as_tensor(self.means), covariance_matrix=torch.as_tensor(new_covs)) 


    def get_epsilons_evolution(self):
        return np.array(self.optimized_epsilons)
    

    def get_means_evolution(self):
        return np.array(self.optimized_means)
    

    def get_params_evolution(self):
        return self.get_means_evolution(), self.get_epsilons_evolution()

    
    def compute_grads(self, target, noise = None, B = 1, optim_epsilon = True): ### GRAD FOR ISOTROPIC, these are euclidan grads

        samples = self.sample_from_each_gaussian(noise = noise, B = B) # n, b, d

        samples_flat = rearrange(samples, "n b d -> (n b) d")

        grad_log_vgmm = self.gradient_log_density(samples_flat)
        grad_log_pi = target.gradient_log_density(samples_flat)
     
        centered_samples = samples - self.means[:,None] ### n, b, d



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
                radius=np.sqrt(self.optimized_epsilons[t][i]),
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


        




class FGMM(GMM):
    def __init__(self, means = None, covs = None, n_components = 3, d = 2, s = 10, scale = 1):

        self.mode = "iso"
        self.variational = True
        self.contours = None

        self.init_gmm(n_components=n_components, means = means,  covs =  covs, s = s, scale = scale, d = d)

        self.num_stab = 0
        self.mode = "full"

        self.epsilons = None
        self.optimized_means = [self.means]
        self.optimized_covs = [self.covariances]
        self.optimized_epsilons = [self.epsilons]

    

    def sample(self, B, noise = None, component_indices = None, t = None):

        if noise is None:
            noise = np.random.randn(B, self.dim) 
            print("Generating noise")
        if component_indices is None:
            component_indices = np.random.choice(self.n_components, size=B, p=self.weights)


        if t is not None:
            means = self.optimized_means[t]
            covariances = self.optimized_covs[t]
        
        else :
            means = self.means
            covariances = self.covariances

        selected_means = means[component_indices]  # (B, d)
        selected_covariances = covariances[component_indices]  # (B, d, d)


        L = np.linalg.cholesky(selected_covariances)     # shape (B, d, d)

        samples = selected_means + np.einsum('bij,bj->bi', L, noise)

        return samples
    


    def sample_from_each_gaussian(self, noise = None, B = 1):

        if noise is None:
            # print("Generating noise")
            noise = np.random.randn(B, self.dim)

        Ls = np.linalg.cholesky(self.covariances)  # shape: (n_components, d, d)
    
   
        samples = self.means[:, None, :] + np.einsum('cij,bj->cbi', Ls, noise)

        return samples


    

    

    def update(self, new_means, new_covs):

        
        # self.optimized_means.append(new_means)
        # self.optimized_covs.append(new_covs)

        self.means = new_means
        self.covariances = new_covs
        self.invcov = np.array([np.linalg.inv(cov) for cov in self.covariances])


        self.gaussians = dist.MultivariateNormal(torch.as_tensor(self.means), covariance_matrix=torch.as_tensor(self.covariances)) 


    def get_epsilons_evolution(self):
        print("No epsilons for Full GMM")
        return np.array(self.optimized_epsilons)
    

    def get_covariances_evolution(self):
        return np.array(self.optimized_covs)
    

    def get_means_evolution(self):
        return np.array(self.optimized_means)
    

    def get_params_evolution(self):
        return self.get_means_evolution(), self.get_covariances_evolution()

    
    def compute_grads(self, target, noise = None, B = 1, optim_epsilon = True): ### GRADIENT FOR FULL COVS

        samples = self.sample_from_each_gaussian(noise = noise, B = B) # n, b, d

        samples_flat = rearrange(samples, "n b d -> (n b) d")

        grad_log_vgmm = self.gradient_log_density(samples_flat)
        grad_log_pi = target.gradient_log_density(samples_flat)
     
        centered_samples = samples - self.means[:,None] ### n, b, d



        grad_log_pi = rearrange(grad_log_pi , "(n b) d -> n b d", b = B)
        grad_log_vgmm = rearrange(grad_log_vgmm , "(n b) d -> n b d", b = B)

        grad_means = (grad_log_vgmm - grad_log_pi).mean(axis = 1)/self.n_components



        if optim_epsilon:  ## TODO grad covs

            E_grad_log_pi_centered_s = np.einsum('nbi,nbj->nbij', grad_log_pi, np.einsum('nij,nbj->nbi', self.invcov, centered_samples)).mean(axis = 1)
            E_grad_log_vgmm_centered_s = np.einsum('nbi,nbj->nbij', grad_log_vgmm, np.einsum('nij,nbj->nbi', self.invcov, centered_samples)).mean(axis = 1)

            grad_covs = (E_grad_log_vgmm_centered_s - E_grad_log_pi_centered_s)/(2*self.n_components) ### n, b

        else:
            grad_covs = None

        return grad_means, grad_covs


    def plot_circle(self, t, ax, bound = 20):

        for i in range(self.n_components):
            center = self.optimized_means[t][i]
            cov = self.optimized_covs[t][i]  
            eigvals, eigvecs = np.linalg.eigh(cov)
            order = eigvals.argsort()[::-1]
            eigvals, eigvecs = eigvals[order], eigvecs[:, order]
            angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
            width = 2  * np.sqrt(eigvals[0])
            height = 2 * np.sqrt(eigvals[1])

            ellip = Ellipse(xy=center, width=width, height=height, angle=angle,
                        edgecolor='black', fc='None', lw=1, zorder=10)
            
            ax.add_patch(ellip)



        ax.set_xlim(-bound, bound)
        ax.set_ylim(-bound, bound)
        ax.set_aspect('equal')

    
    def plot_evolution(self, ax, jump = 100, bound = 20):

        for t in range(0, len(self.optimized_means), jump):
            self.plot_circle(t, ax, bound)


    