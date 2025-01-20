import torch 
import tqdm 
import math

from src.utils import monte_carlo_kl_approximation, grad_V, gaussian_kernel, V_function
from src.plot import plot_evolution_1d, plot_evolution_2d
from src.hessian.utils import hessian_V, hessian_ln_mixture



def gradient_descent_mu(mu_locs, epsilon, pi_mean, pi_cov, learning_rate=0.01, num_iterations=1000):
    """
    Perform gradient descent on KL(K * mu | pi) with respect to the locations of Dirac deltas in mu.
    """

    n, d = mu_locs.shape
    mu_locs = mu_locs.clone()
    kls = []
    means = []

    pi_dist = [torch.distributions.MultivariateNormal(mean, pi_cov) for mean in pi_mean]

    y_01  = torch.randn(100, d)
    print("critte")
    
    for iteration in tqdm.tqdm(range(num_iterations)):


        means.append(mu_locs.clone())
        
        grad_locs = torch.zeros_like(mu_locs)

        for i, loc in enumerate(mu_locs):


            # y_samples = y_01 * epsilon[i] + loc


            y_samples = torch.distributions.MultivariateNormal(loc, torch.eye(d) * epsilon[i]**2).sample((100,)) # B, d

            first_term = torch.mean(grad_V(y_samples, pi_mean, pi_cov), dim = 0)

            numerator = torch.sum(
                torch.stack([
                    -gaussian_kernel(y_samples, loc_i, epsilon[i])[..., None] * (y_samples - loc_i) / (epsilon[i] ** 2)
                    for loc_i in mu_locs
                ]),
                dim=0
            )

            denominator = torch.sum(
                torch.stack([gaussian_kernel(y_samples, loc_i, epsi) for loc_i,epsi in zip(mu_locs, epsilon)]),
                dim=0
            ) 

            denominator = torch.clamp(denominator, 1e-30)
            second_term = torch.mean((numerator / denominator[..., None]), dim = 0 )


            # Total gradient for location `i`
            grad_locs[i] = first_term + second_term

        # Update locations with gradient descent step
        mu_locs = mu_locs - learning_rate * grad_locs
        mu_locs = mu_locs.detach().clone()  

        if iteration % (num_iterations//2) == 0 or iteration == num_iterations-1:
            kl_div = monte_carlo_kl_approximation(mu_locs, epsilon, pi_dist)
            # print(kl_div)
            kls.append(kl_div)
            if d == 1: 
                plot_evolution_1d(pi_dist, mu_locs, epsilon) 

            if d == 2:
                plot_evolution_2d(pi_mean, pi_cov, mu_locs, epsilon)
            print(f"Iteration {iteration}, KL divergence: {kl_div}")


    return mu_locs, kls, means




def gradient_descent_mu_epsi(mu_locs, epsilon, pi_mean, pi_cov, learning_rate_mu=0.01, learning_rate_eps = 0.001, num_iterations=1000, B = 100):
    """
    Perform gradient descent on KL(K * mu | pi) with respect to the locations of Dirac deltas in x_i and epsilon_i
    """
    # Ensure that all variables are tensors

    n , d = mu_locs.shape
    mu_locs = mu_locs.clone() # n, d
    epsilon = epsilon.clone() # n, d
    kls = []
    means = []
    E = []


    pi_dist = [torch.distributions.MultivariateNormal(mean, pi_cov) for mean in pi_mean]

    # y_01  = torch.randn(B, d)
    
    for iteration in tqdm.tqdm(range(num_iterations)):

        means.append(mu_locs.clone())

        E.append(epsilon.clone())
        
        grad_locs = torch.zeros_like(mu_locs)
        grad_eps = torch.zeros_like(epsilon)



        # compute grad for each x_i and eps_i
        for i in range(len(epsilon)):            

            # y_samples = y_01 * epsilon[i] + mu_locs[i]
            y_samples = torch.distributions.MultivariateNormal(mu_locs[i], torch.eye(d) * epsilon[i]**2).sample((100*d,)) # B, d

            #### OPTIM OF MUS
            first_term = torch.mean(grad_V(y_samples, pi_mean, pi_cov), dim = 0)

            numerator = torch.stack([
                                        - gaussian_kernel(y_samples, loc_i, epsi)[..., None] * (y_samples - loc_i) / (epsi ** 2)
                                        for loc_i, epsi in zip(mu_locs, epsilon)
                                    ]).sum(dim = 0)
            
            denominator = torch.stack([gaussian_kernel(y_samples, loc_i, epsi) for loc_i,epsi in zip(mu_locs, epsilon)]).sum(dim = 0) 
            denominator = torch.clamp(denominator, 1e-40)
        
            second_term = torch.mean((numerator / denominator[..., None]), dim = 0 ) 
 
            grad_locs[i] = first_term + second_term


            ### OPTIM EPSI
            V = V_function(y_samples, pi_mean, pi_cov)
            norm = ((y_samples - mu_locs[i])**2).sum(dim = -1) /(2*epsilon[i]**4)
            cste = d*math.pi/(2*math.pi*epsilon[i]**2)

            first_term = (V*(norm-cste)).mean()/n
            # print("FT", first_term)

            ln_term = (torch.stack([
                                    gaussian_kernel(y_samples, loc_i, epsi) for loc_i,epsi in zip(mu_locs, epsilon)
                                    ]).sum(dim = 0)/n).log() + 1
            # print(ln_term.shape)

            second_term = ((norm - cste)*ln_term).mean()/n
            # print("ST", second_term)

            ### grad epsi
            grad_eps[i] = first_term+second_term
        


        # Update locations with gradient descent step
        mu_locs = mu_locs - learning_rate_mu * grad_locs
        mu_locs = mu_locs.detach().clone()

        # print(grad_eps)
        
        # Update epsilons with gradient descent step
        ### epsilon is the "ecart type", however we perform GD on the variance, thus we need to substract from the "variance"
        epsilon = (epsilon**2) * torch.exp(-learning_rate_eps * grad_eps )
        epsilon = epsilon.sqrt()

        print(epsilon)
  
        
        # if iteration % (num_iterations//2) == 0 or iteration == num_iterations-1:
            # kl_div = monte_carlo_kl_approximation(mu_locs, epsilon, pi_dist, B = 100)
            # kls.append(kl_div)
           
            # if d == 1: 
            #     plot_evolution_1d(pi_dist, mu_locs, epsilon) 

            # if d == 2:
            #     plot_evolution_2d(pi_mean, pi_cov, mu_locs, epsilon)
            # print(f"Iteration {iteration}, KL divergence: {kl_div}")


    return mu_locs, kls, means, epsilon, E






def gradient_descent_mu_epsi_new_update(mu_locs, epsilon, pi_mean, pi_cov, learning_rate_mu=0.01, learning_rate_eps = 0.001, num_iterations=1000):
    """
    Perform gradient descent on KL(K * mu | pi) with respect to the locations of Dirac deltas in x_i and epsilon_i
    """
    # Ensure that all variables are tensors

    n , d = mu_locs.shape
    mu_locs = mu_locs.clone() # n, d
    epsilon = epsilon.clone() # n, 1
    kls = []
    means = []
    E = []


    pi_dist = [torch.distributions.MultivariateNormal(mean, pi_cov) for mean in pi_mean]

    
    for iteration in tqdm.tqdm(range(num_iterations)):

        means.append(mu_locs.clone())

        E.append(epsilon.clone())
        
        grad_locs = torch.zeros_like(mu_locs)
        grad_epsilons = torch.zeros_like(epsilon)

        # compute grad for each x_i and eps_i
        for i in range(len(epsilon)):            

            y_samples = torch.distributions.MultivariateNormal(mu_locs[i], torch.eye(d) * epsilon[i]**2).sample((100,)) # B, d

            #### OPTIM OF MUS
            first_term = torch.mean(grad_V(y_samples, pi_mean, pi_cov), dim = 0)

            numerator = torch.stack([
                                        - gaussian_kernel(y_samples, loc_i, epsi)[..., None] * (y_samples - loc_i) / (epsi ** 2)
                                        for loc_i, epsi in zip(mu_locs, epsilon)
                                    ]).sum(dim = 0)
            
            denominator = torch.stack([gaussian_kernel(y_samples, loc_i, epsi) for loc_i,epsi in zip(mu_locs, epsilon)]).sum(dim = 0) 
            denominator = torch.clamp(denominator, 1e-40)
        
            second_term = torch.mean((numerator / denominator[..., None]), dim = 0 ) 
 
            grad_locs[i] = first_term + second_term


            ### OPTIM EPSI
            first_term = hessian_ln_mixture(y_samples, mu_locs, epsilon)

            second_term = (y_samples - mu_locs[0])[:,:, None] @ grad_V(y_samples, pi_mean, pi_cov)[:,None, :] / (epsilon[i]**2)

            # grad_epsilons[i] = (first_term + second_term).mean(dim = 0).diag().sum() / (2*n*d)
            grad_epsilons[i] = (first_term + second_term).mean(dim = 0).diag().sum() / (n*d)

            
        


        # Update locations with gradient descent step
        mu_locs = mu_locs - learning_rate_mu * grad_locs
        mu_locs = mu_locs.detach().clone()

        # print(grad_eps)
        
        # Update epsilons with gradient descent step
        ### epsilon is the "ecart type", however we perform GD on the variance, thus we need to perform the update on the "variance"
        # print( "HESSIAN V", hess_V)
        # print( "HESSIAN P", hess_P)
        epsilon = (1 -  learning_rate_eps * grad_epsilons)**2 * epsilon**2 ### maybe add d
        epsilon = epsilon.sqrt()
  
        
        if iteration % (num_iterations//10) == 0 or iteration == num_iterations-1:
            kl_div = monte_carlo_kl_approximation(mu_locs, epsilon, pi_dist, B = 100)
            kls.append(kl_div)
           
            if d == 1: 
                plot_evolution_1d(pi_dist, mu_locs, epsilon) 

            if d == 2:
                plot_evolution_2d(pi_mean, pi_cov, mu_locs, epsilon)
            print(f"Iteration {iteration}, KL divergence: {kl_div}")
            print("Epsilon values", epsilon)


    return mu_locs, kls, means, epsilon, E





from einops import rearrange, repeat



def gaussian_kernel_HD(Y_flat, mu_locs, epsilon):
    ### apply gaussian kernel Ze * exp to each y of of Y flat for each pair mu_locs, epsilon (there are n pairs)
    n, d = mu_locs.shape
    return ((-((Y_flat[:, None]- mu_locs[None])**2).sum(dim = -1)/(2*epsilon[None]**2)).exp()/ ((2*math.pi*epsilon[None]**2)**(d/2)))  # b, 1, d - 1,n,d = b, n, d -> b,n 


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