import math 
import torch
from einops import rearrange, repeat
import tqdm 

def gaussian_kernel_HD(Y_flat, mu_locs, epsilon):
    ### apply gaussian kernel Ze * exp to each y of of Y flat for each pair mu_locs, epsilon (there are n pairs)
    n, d = mu_locs.shape
    return ((-((Y_flat[:, None]- mu_locs[None])**2).sum(dim = -1)/(2*epsilon[None]**2)).exp()/ ((2*math.pi*epsilon[None]**2)**(d/2)))  # b, 1, d - 1,n,d = b, n, d -> b,n 



def MC_KL(mu, epsilon, pi_dist, y_01):
    n, B,  d = y_01.shape

    # print(epsilon.shape)
    if torch.isnan(epsilon).any():
        print("EPSILON", epsilon)
    mixture_epsilon = repeat(epsilon, "n -> n b", b = B ) # n b 
    mixture_means = repeat(mu, "n d -> n b d", b= B) # n b d
    Y = ((y_01 * mixture_epsilon[..., None]) + mixture_means) # n, B , d
    clippin = 10e-40
    Y_flat = rearrange(Y, "n b d -> (n b) d")
    mixt = gaussian_kernel_HD(Y_flat, mu, epsilon).mean(dim=-1) ## to all the samples of each gaussian of the mixture im applying k * mu
    pi_dens = pi_dist.log_prob(Y_flat.unsqueeze(1)).exp().mean(dim = -1)
    pi_dens = pi_dens + clippin 
    mixt = mixt + clippin
    return (mixt/pi_dens).log().mean()

def kl_evolution(pi_dist, M, E = None, B = 100): 

    n , d = M[0].shape
    if E is None:
        E = [torch.ones(n,)*1.5 for _ in range(len(M))]

    y_01 = torch.randn(n, B, d) 
    KLS = []
    for m, eps in tqdm.tqdm(zip(M, E)):
        KLS.append( MC_KL(m, eps, pi_dist, y_01) ) 
    
    return KLS