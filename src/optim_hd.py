import torch 
import tqdm 
from einops import rearrange, repeat

from src.utils_hd import gaussian_kernel_HD
from src.utils import grad_V




def optim_mu_epsi_HD_ML(mu_locs, epsilon, pi_mean, pi_cov, learning_rate_mu=0.01, learning_rate_eps = 0.001, num_iterations=1000, B = 100):
    
    clippin = 10e-40

    n, d = mu_locs.shape
    y_01  = torch.randn(n, B, d)
    E , M = [], []

    for i in tqdm.tqdm(range(num_iterations)):
        mixture_epsilon = repeat(epsilon, "n -> n b", b = B ) # n b 
        mixture_means = repeat(mu_locs, "n d -> n b d", b= B) # n b d
        Y = ((y_01 * mixture_epsilon[..., None]) + mixture_means) # n, B , d
        Y_flat = rearrange(Y, "n b d -> (n b) d")

        ### OPTIM MU
        GV = grad_V(Y_flat,  pi_mean, pi_cov)
        first_term = rearrange(GV, "(n b) d -> n b d", n  = n).mean(dim = -2)


        y_mu_div_eps = ((Y_flat[:, None]- mu_locs[None])/epsilon[None,:, None]**2) # b, n ,d (b is n * b ) 
        num = (y_mu_div_eps  * gaussian_kernel_HD(Y_flat, mu_locs,epsilon)[..., None]).sum(dim  = -2)
        den =  gaussian_kernel_HD(Y_flat, mu_locs,epsilon).sum(dim = -1)

        num1 = num + clippin
        den1 = den + clippin

        second_term = rearrange(num1/den1[:,None], "(n b) d -> n b d", n = n ).mean(dim = -2)


        grad_locs = (first_term - second_term) # n, d 

        # print("optim mu")

        ### Optim eps
        yi_mi_div_ei = rearrange(((Y - mixture_means)/epsilon[..., None,None]**2), "n b d -> (n b) d")
        num1 = (yi_mi_div_ei*num).sum(dim = -1) + clippin ### scalar product Tr(uvT)= uTv
        den1 = den + clippin 

        first_term = rearrange(num1/den1, "(n b) -> n b", n = n).mean(dim  = -1)
        second_term = rearrange((yi_mi_div_ei*GV).sum(dim = -1), "(n b) -> n b", n = n).mean(dim  = -1)
        grad_eps  = (second_term - first_term)/(2*n)

        mu_locs = mu_locs - learning_rate_mu * grad_locs
        mu_locs = mu_locs.detach().clone()

        # print(grad_eps)

        # Update epsilons with gradient descent step
        ### epsilon is the "ecart type", however we perform GD on the variance, thus we need to substract from the "variance"
        epsilon = (1 - learning_rate_eps*grad_eps/d)**2 * epsilon**2
        epsilon = epsilon.sqrt()

        E.append(epsilon)
        M.append(mu_locs)

    return mu_locs, epsilon , M, E




def optim_mu_epsi_HD(mu_locs, epsilon, pi_mean, pi_cov, learning_rate_mu=0.01, learning_rate_eps = 0.001, num_iterations=1000, B = 100):
    
    clippin = 10e-40

    n, d = mu_locs.shape
    y_01  = torch.randn(n, B, d)
    E , M = [], []

    for i in tqdm.tqdm(range(num_iterations)):
        mixture_epsilon = repeat(epsilon, "n -> n b", b = B ) # n b 
        mixture_means = repeat(mu_locs, "n d -> n b d", b= B) # n b d
        Y = ((y_01 * mixture_epsilon[..., None]) + mixture_means) # n, B , d
        Y_flat = rearrange(Y, "n b d -> (n b) d")

        ### OPTIM MU
        GV = grad_V(Y_flat,  pi_mean, pi_cov)
        first_term = rearrange(GV, "(n b) d -> n b d", n  = n).mean(dim = -2)


        y_mu_div_eps = ((Y_flat[:, None]- mu_locs[None])/epsilon[None,:, None]**2) # b, n ,d (b is n * b ) 
        num = (y_mu_div_eps  * gaussian_kernel_HD(Y_flat, mu_locs,epsilon)[..., None]).sum(dim  = -2)
        den =  gaussian_kernel_HD(Y_flat, mu_locs,epsilon).sum(dim = -1)

        num1 = num + clippin
        den1 = den + clippin

        second_term = rearrange(num1/den1[:,None], "(n b) d -> n b d", n = n ).mean(dim = -2)


        grad_locs = (first_term - second_term) # n, d 

        # print("optim mu")

        ### Optim eps
        yi_mi_div_ei = rearrange(((Y - mixture_means)/epsilon[..., None,None]**2), "n b d -> (n b) d")
        num1 = (yi_mi_div_ei*num).sum(dim = -1) + clippin ### scalar product Tr(uvT)= uTv
        den1 = den + clippin 

        first_term = rearrange(num1/den1, "(n b) -> n b", n = n).mean(dim  = -1)
        second_term = rearrange((yi_mi_div_ei*GV).sum(dim = -1), "(n b) -> n b", n = n).mean(dim  = -1)
        grad_eps  = (second_term - first_term)/(2*n)

        mu_locs = mu_locs - learning_rate_mu * grad_locs
        mu_locs = mu_locs.detach().clone()

        # print(grad_eps)

        # Update epsilons with gradient descent step
        ### epsilon is the "ecart type", however we perform GD on the variance, thus we need to substract from the "variance"
        epsilon = (epsilon**2) * torch.exp(-learning_rate_eps * grad_eps )
        epsilon = epsilon.sqrt()

        E.append(epsilon)
        M.append(mu_locs)

    return mu_locs, epsilon , M, E







def optim_mu_only_HD(mu_locs, epsilon, pi_mean, pi_cov, learning_rate_mu=0.01, learning_rate_eps = 0.001, num_iterations=1000, B = 100):
    
    clippin = 10e-40

    n, d = mu_locs.shape
    y_01  = torch.randn(n, B, d)
    M = []

    for i in tqdm.tqdm(range(num_iterations)):
        mixture_epsilon = repeat(epsilon, "n -> n b", b = B ) # n b 
        mixture_means = repeat(mu_locs, "n d -> n b d", b= B) # n b d
        Y = ((y_01 * mixture_epsilon[..., None]) + mixture_means) # n, B , d
        Y_flat = rearrange(Y, "n b d -> (n b) d")

        ### OPTIM MU
        GV = grad_V(Y_flat,  pi_mean, pi_cov)
        first_term = rearrange(GV, "(n b) d -> n b d", n  = n).mean(dim = -2)


        y_mu_div_eps = ((Y_flat[:, None]- mu_locs[None])/epsilon[None,:, None]**2) # b, n ,d (b is n * b ) 
        num = (y_mu_div_eps  * gaussian_kernel_HD(Y_flat, mu_locs,epsilon)[..., None]).sum(dim  = -2)
        den =  gaussian_kernel_HD(Y_flat, mu_locs,epsilon).sum(dim = -1)

        num1 = num + clippin
        den1 = den + clippin

        second_term = rearrange(num1/den1[:,None], "(n b) d -> n b d", n = n ).mean(dim = -2)


        grad_locs = (first_term - second_term) # n, d 


        mu_locs = mu_locs - learning_rate_mu * grad_locs
        mu_locs = mu_locs.detach().clone()

        # print(grad_eps)

        # Update epsilons with gradient descent step
        ### epsilon is the "ecart type", however we perform GD on the variance, thus we need to substract from the "variance"

        M.append(mu_locs)

    return mu_locs , M
