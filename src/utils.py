import math
import torch



def gaussian_kernel(sample, mu, epsilon): ## CHECKED
    d = sample.shape[-1]
    return (-((sample - mu)**2).sum(dim = -1)/(2*epsilon**2)).exp()/((2*math.pi*epsilon**2)**(d/2))


def V_function(x, pi_mean, pi_cov):
    ### x :  B, d
    ### out : B


    return - torch.clamp(torch.stack([-((x - pim)**2).sum(dim = 1)/(2*pi_cov[0,0]) for pim in pi_mean]).exp().sum(dim = 0), 1e-40).log()




def monte_carlo_kl_approximation(mu_locs, epsilon, pi_dist, B=100): ### CHECKED
    KL = 0
    n, d = mu_locs.shape
    mixture_dist =  [torch.distributions.MultivariateNormal(mean, torch.eye(d) * epsi**2) for mean, epsi in zip(mu_locs,epsilon)]
    
    for i in range(len(mu_locs)):
        
        y_samples = mixture_dist[i].sample((B,))
        nu_n = torch.stack([gaussian_kernel(y_samples, loc_i, epsi) for loc_i,epsi in zip(mu_locs, epsilon)]).sum(dim = 0)/len(mixture_dist)

        pi_density = torch.stack([p.log_prob(y_samples).exp() for p in pi_dist]).sum(dim = 0)/len(pi_dist)
        pi_density = pi_density 
        KL += torch.log(nu_n/pi_density).sum()

    KL = KL/(len(mu_locs)*B)

    
   
    return KL



def  grad_V(x, pi_mean, pi_cov):
    ### numerator B, d
    clippin = 0
    numerator = torch.stack([(-((x - pim)**2).sum(dim = 1)/(2*pic[0,0])).exp()[..., None]*(x-pim)/pic[0,0] for pim, pic in zip(pi_mean, pi_cov)]).sum(dim = 0) 
    numerator = numerator + clippin
    ### denom B, 1
    denominator = torch.stack([(-((x - pim)**2).sum(dim = 1)/(2*pic[0,0])).exp() for pim, pic in zip(pi_mean, pi_cov)]).sum(dim = 0)[..., None]
    denominator = denominator + clippin


    return numerator/denominator # B , d

def  grad_V_diag(x, pi_mean, pi_cov):
    ### numerator B, d
    clippin = 1e-40

    L = torch.linalg.cholesky(pi_cov)  # Shape: (n, d, d)
    L_inv = torch.linalg.inv(L)  # Shape: (n, d, d)

    pi_cov_inv = L_inv.transpose(-1, -2) @ L_inv  # Shape: (n, d, d)


    numerator = torch.stack([(-((x - pim)**2).sum(dim = 1)/2).exp()[..., None]*(x-pim)/pic[0,0] for pim, pic in zip(pi_mean, pi_cov)]).sum(dim = 0) 
    numerator = numerator + clippin
    ### denom B, 1
    denominator = torch.stack([(-((x - pim)**2).sum(dim = 1)/(2*pic[0,0])).exp() for pim, pic in zip(pi_mean, pi_cov)]).sum(dim = 0)[..., None]
    denominator = denominator + clippin


    return numerator/denominator # B , d




#### sample from a gaussian mixture 
def sample_mixture(means, epsilon , M):

    N, d = means.shape
    component_indices = torch.randint(0,N,(M,))

    samples = torch.zeros((M, d))
    cov = epsilon**2 * torch.eye(d)

    for i in range(N):
        count = (component_indices == i).sum().item()
        if count > 0:
            # Sample from N(mus[i], sigmas[i])
            dist = torch.distributions.MultivariateNormal(loc=means[i], covariance_matrix=cov)
            samples_from_i = dist.sample((count,))  
            samples[component_indices == i] = samples_from_i
            

    return samples