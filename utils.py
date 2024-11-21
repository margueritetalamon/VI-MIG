import math
import torch



def gaussian_kernel(sample, mu, epsilon): ## CHECKED
    d = sample.shape[-1]
    return (-((sample - mu)**2).sum(dim = -1)/(2*epsilon**2)).exp()/((2*math.pi*epsilon**2)**(d/2))


def V_function(x, pi_mean, pi_cov):
    ### x :  B, d
    ### out : B
    return torch.clamp(torch.stack([-((x - pim)**2).sum(dim = 1)/(2*pi_cov[0,0]) for pim in pi_mean]).exp().sum(dim = 0), 1e-40).log()




def monte_carlo_kl_approximation(mu_locs, epsilon, pi_dist, B=100): ### CHECKED
    KL = 0
    n, d = mu_locs.shape
    mixture_dist =  [torch.distributions.MultivariateNormal(mean, torch.eye(d) * epsi**2) for mean, epsi in zip(mu_locs,epsilon)]
    
    for i in range(len(mu_locs)):
        
        y_samples = mixture_dist[i].sample((B,))
        nu_n = torch.stack([gaussian_kernel(y_samples, loc_i, epsi) for loc_i,epsi in zip(mu_locs, epsilon)]).sum(dim = 0)/len(mixture_dist)

        pi_density = torch.stack([p.log_prob(y_samples).exp() for p in pi_dist]).sum(dim = 0)/len(pi_dist)
        pi_density = torch.clamp(pi_density, 10e-40)
        KL += torch.log(nu_n/pi_density).sum()

    KL = KL/(len(mu_locs)*B)

    
   
    return KL


def  grad_V(x, pi_mean, pi_cov):
    ### numerator B, d
    numerator = torch.stack([(-((x - pim)**2).sum(dim = 1)/(2*pi_cov[0,0])).exp()[..., None]*(x-pim)/pi_cov[0,0] for pim in pi_mean]).sum(dim = 0) 
    ### denom B, 1
    denominator = torch.stack([(-((x - pim)**2).sum(dim = 1)/(2*pi_cov[0,0])).exp() for pim in pi_mean]).sum(dim = 0)[..., None]
    denominator = torch.clamp(denominator, 10e-40)

    return numerator/denominator # B , d


