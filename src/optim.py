import torch 
import tqdm 
import math

from src.utils import monte_carlo_kl_approximation, grad_V, gaussian_kernel, V_function
from src.plot import plot_evolution_1d, plot_evolution_2d




def gradient_descent_mu(mu_locs, epsilon, pi_mean, pi_cov, learning_rate=0.01, num_iterations=1000):
    """
    Perform gradient descent on KL(K * mu | pi) with respect to the locations of Dirac deltas in mu.
    """

    n, d = mu_locs.shape
    mu_locs = mu_locs.clone()
    kls = []
    means = []

    pi_dist = [torch.distributions.MultivariateNormal(mean, pi_cov) for mean in pi_mean]

    
    for iteration in tqdm.tqdm(range(num_iterations)):


        means.append(mu_locs.clone())
        
        grad_locs = torch.zeros_like(mu_locs)

        for i, loc in enumerate(mu_locs):

            y_samples = torch.distributions.MultivariateNormal(loc, torch.eye(d) * epsilon**2).sample((100,)) # B, d

            first_term = torch.mean(grad_V(y_samples, pi_mean, pi_cov), dim = 0)

            numerator = torch.sum(
                torch.stack([
                    -gaussian_kernel(y_samples, loc_i, epsilon)[..., None] * (y_samples - loc_i) / (epsilon ** 2)
                    for loc_i in mu_locs
                ]),
                dim=0
            )

            denominator = torch.sum(
                torch.stack([gaussian_kernel(y_samples, loc_i, epsilon) for loc_i in mu_locs]),
                dim=0
            ) 

            denominator = torch.clamp(denominator, 1e-30)
            second_term = torch.mean((numerator / denominator[..., None]), dim = 0 )


            # Total gradient for location `i`
            grad_locs[i] = first_term + second_term

        # Update locations with gradient descent step
        mu_locs = mu_locs - learning_rate * grad_locs
        mu_locs = mu_locs.detach().clone()  

        if iteration % (num_iterations//10) == 0 or iteration == num_iterations-1:
            kl_div = monte_carlo_kl_approximation(mu_locs, epsilon, pi_dist)
            # print(kl_div)
            kls.append(kl_div)
            if d == 1: 
                plot_evolution_1d(pi_dist, mu_locs, epsilon) 

            if d == 2:
                plot_evolution_2d(pi_mean, pi_cov, mu_locs, epsilon)
            print(f"Iteration {iteration}, KL divergence: {kl_div}")


    return mu_locs, kls, means




def gradient_descent_mu_epsi(mu_locs, epsilon, pi_mean, pi_cov, learning_rate_mu=0.01, learning_rate_eps = 0.001, num_iterations=1000):
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

    
    for iteration in tqdm.tqdm(range(num_iterations)):

        means.append(mu_locs.clone())

        E.append(epsilon.clone())
        
        grad_locs = torch.zeros_like(mu_locs)
        grad_eps = torch.zeros_like(epsilon)

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
  
        
        if iteration % (num_iterations//10) == 0 or iteration == num_iterations-1:
            kl_div = monte_carlo_kl_approximation(mu_locs, epsilon, pi_dist, B = 100)
            kls.append(kl_div)
           
            if d == 1: 
                plot_evolution_1d(pi_dist, mu_locs, epsilon) 

            if d == 2:
                plot_evolution_2d(pi_mean, pi_cov, mu_locs, epsilon)
            print(f"Iteration {iteration}, KL divergence: {kl_div}")


    return mu_locs, kls, means, epsilon, E



def gradient_descent_mui_then_epsi(mu_locs, epsilon, pi_mean, pi_cov, learning_rate_mu=0.01, learning_rate_eps = 0.001, num_iterations=1000):
    """
    Perform gradient descent on KL(K * mu | pi) with respect to the locations of of x_i and THEN on eps_i 
    """
    n , d = mu_locs.shape
    mu_locs = mu_locs.clone()
    epsilon = epsilon.clone()
    kls = []
    means = []
    E = []


    pi_dist = [torch.distributions.MultivariateNormal(mean, pi_cov) for mean in pi_mean]

    
    for iteration in tqdm.tqdm(range(num_iterations)):



        means.append(mu_locs.clone())

        E.append(epsilon.clone())
        
        grad_locs = torch.zeros_like(mu_locs)
        grad_eps = torch.zeros_like(epsilon)

        #### OPTIM OF x_i
        for i in range(len(mu_locs)):
            
            y_samples = torch.distributions.MultivariateNormal(mu_locs[i], torch.eye(d) * epsilon[i]**2).sample((1000,)) # B, d

            first_term = torch.mean(grad_V(y_samples, pi_mean, pi_cov), dim = 0)
            numerator = torch.stack([
                                        - gaussian_kernel(y_samples, loc_i, epsi)[..., None] * (y_samples - loc_i) / (epsi ** 2)
                                        for loc_i, epsi in zip(mu_locs, epsilon)
                                    ]).sum(dim = 0)
            
            denominator = torch.stack([gaussian_kernel(y_samples, loc_i, epsi) for loc_i,epsi in zip(mu_locs, epsilon)]).sum(dim = 0) 
            denominator = torch.clamp(denominator, 1e-40)
        
            second_term = torch.mean((numerator / denominator[..., None]), dim = 0 ) 
 
            grad_locs[i] = first_term + second_term

        mu_locs = mu_locs - learning_rate_mu * grad_locs
        mu_locs = mu_locs.detach().clone()

        
        ### OPTIM EPSI



        for i in range(len(epsilon)):


            y_samples = torch.distributions.MultivariateNormal(mu_locs[i], torch.eye(d) * epsilon[i]**2).sample((1000,)) # B, d
            
            V = V_function(y_samples, pi_mean, pi_cov)
            norm = ((y_samples - mu_locs[i])**2).sum(dim = -1) /(2*epsilon[i]**4)
            cste = d*math.pi/(2*math.pi*epsilon[i]**2)
            first_term = (V*(norm-cste)).mean()/n
       
            ln_term = (torch.stack([gaussian_kernel(y_samples, loc_i, epsi) for loc_i,epsi in zip(mu_locs, epsilon)]).sum(dim = 0)/n).log() + 1
            second_term = ((norm - cste)*ln_term).mean()/n

            grad_eps[i] = first_term+second_term
        


        # Update locations with gradient descent step
        
        
        epsilon = (epsilon**2) * torch.exp( - learning_rate_eps * grad_eps )
        epsilon = epsilon.sqrt()

        
        if iteration % (num_iterations//10) == 0 or iteration == num_iterations-1:
            kl_div = monte_carlo_kl_approximation(mu_locs, epsilon, pi_dist, B = 100)
            kls.append(kl_div)
           
            if d == 1: 
                plot_evolution_1d(pi_dist, mu_locs, epsilon) 

            if d == 2:
                plot_evolution_2d(pi_mean, pi_cov, mu_locs, epsilon)
            print(f"Iteration {iteration}, KL divergence: {kl_div}")


    return mu_locs, kls, means, epsilon, E
