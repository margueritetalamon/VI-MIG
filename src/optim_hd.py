import torch 
import tqdm 
from einops import rearrange, repeat

from src.utils_hd import gaussian_kernel_HD, compute_grads, new_grads
from src.utils import grad_V
import math
import numpy as np 

def cosine_annealing_scheduler(iteration, base_lr, total_iterations):
    return base_lr * (0.5 * (1 + math.cos(math.pi * iteration / total_iterations)))


def optim_mu_epsi_HD_ML(mu_locs, epsilon, pi_mean, pi_cov, learning_rate_mu=0.01, learning_rate_eps = 0.001, num_iterations=1000, B = 100):
    
    clippin = 1e-40

    n, d = mu_locs.shape
    # torch.manual_seed(1)
    # y_01 = torch.tensor([[[1.62, -0.611]], [[1.62, -0.611]]])
    # print(y_01.shape)
    E , M = [], []
    GM, GE = [], []
    # y_01  = torch.randn(n, B, d)
    # np.random.seed(1)
    y_01 = torch.tensor(np.random.multivariate_normal(np.zeros(d, ), np.identity(d), size=(n,B,)))
    # print(y_01)

    # print("samples", y_01)

    for i in tqdm.tqdm(range(num_iterations)):
        

        grad_locs, grad_eps = compute_grads(mu_locs, epsilon , pi_mean, pi_cov, y_01, optim_eps = True, clippin = 1e-40)
        # grad_locs, grad_eps = new_grads(mu_locs, epsilon , pi_mean, pi_cov, y_01, optimize_eps = True, clipval = 1e-40)
        # GM.append(grad_locs)
        # GE.append(grad_eps)
        ############# GRADIENT DESCENT #################
        mu_locs = mu_locs - learning_rate_mu * grad_locs


        # print("GRAD M", grad_locs)
        # print("GRAD E", grad_eps)
        ####################### ISOTROPIC BW #######################
        # print()
        epsilon = ((1 - 2*learning_rate_eps*grad_eps/d)**2) * epsilon**2
        epsilon = epsilon.sqrt()

        E.append(epsilon)
        M.append(mu_locs)

    # torch.save(torch.stack(GM), "gradients_mu.pt")
    # torch.save(torch.stack(GE), "gradients_epsilon.pt")
    return mu_locs, epsilon , M, E




def optim_mu_epsi_HD(mu_locs, epsilon, pi_mean, pi_cov, learning_rate_mu=0.01, learning_rate_eps = 0.001, num_iterations=1000, B = 100):
    
    clippin = 1e-40

    n, d = mu_locs.shape
    y_01  = torch.randn(n, B, d)
    E , M = [], []
    gradsM = []
    gradsE = []

    for i in tqdm.tqdm(range(num_iterations),leave=False):

       

        grad_locs, grad_eps = compute_grads(mu_locs, epsilon , pi_mean, pi_cov, y_01, optim_eps = True)

        ############# GRADIENT DESCENT #################
        mu_locs = mu_locs - learning_rate_mu * grad_locs

        ######################## MIRROR DESCENT #########################
        epsilon = (epsilon**2) * torch.exp(-learning_rate_eps * grad_eps )
        epsilon = epsilon.sqrt()

        if epsilon.isnan().any().item():
            print(epsilon)
            print(grad_locs)
            raise ValueError


        E.append(epsilon)
        M.append(mu_locs)
        gradsM.append(grad_locs)
        gradsE.append(grad_eps)
        # learning_rate_eps = cosine_annealing_scheduler(i, learning_rate_eps, num_iterations)
        # learning_rate_mu = cosine_annealing_scheduler(i, learning_rate_mu, num_iterations)

    # return mu_locs, epsilon , M, E,  gradsM, gradsE
    return mu_locs, epsilon , M, E









def optim_mu_only_HD(mu_locs, epsilon, pi_mean, pi_cov, learning_rate_mu=0.01, learning_rate_eps = 0.001, num_iterations=1000, B = 100):
    
    clippin = 1e-40

    n, d = mu_locs.shape
    y_01  = torch.randn(n, B, d)
    M = []

    for i in tqdm.tqdm(range(num_iterations)):
        grad_locs, grad_eps = compute_grads(mu_locs, epsilon , pi_mean, pi_cov, y_01, optim_eps = False)

        ############# GRADIENT DESCENT #################
        mu_locs = mu_locs - learning_rate_mu * grad_locs

        M.append(mu_locs)

    return mu_locs , M
