import math 
import torch
from einops import rearrange, repeat
import tqdm 
from src.utils import grad_V

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




def compute_grads(mu_locs, epsilon, pi_mean , pi_cov, y_01, optim_eps = True, clippin = 1e-40):


    n, B, d = y_01.shape

    ################### PREP ########################################
    mixture_epsilon = repeat(epsilon, "n -> n b", b = B ) # n b 
    mixture_means = repeat(mu_locs, "n d -> n b d", b= B) # n b d
    Y= ((y_01 * mixture_epsilon[..., None]) + mixture_means) # n, B , d
    Y_flat = rearrange(Y, "n b d -> (n b) d")

    #################### GRAD xj ####################################
    GV = rearrange(grad_V(Y_flat,  pi_mean, pi_cov), "(n b) d -> n b d", n  = n)
    first_term = GV.mean(dim = -2)

    y_mu_div_eps = ((Y_flat[:, None]- mu_locs[None])/epsilon[None,:, None]**2) # b, n ,d (b is n * b ) 
    GK = gaussian_kernel_HD(Y_flat, mu_locs,epsilon)
    num = (y_mu_div_eps  * GK[..., None]).sum(dim  = -2)
    den =  GK.sum(dim = -1)
    den = rearrange(den, "(n b) -> n b", n = n)
    num = rearrange(num, "(n b) d -> n b d", n = n )

    num1 = num + clippin
    den = den + clippin
    second_term = (num1/den[..., None]).mean(dim = 1)


    grad_locs = (first_term - second_term) # n, d 

    ################### GRAD ej #####################################
    if optim_eps:
        Yj_muj_epsj = ((Y - mixture_means)/epsilon[..., None,None]**2)
        num2 = (Yj_muj_epsj * num).sum(dim = -1) #PRODUIT SCALAIRE

        first_term = (num2/den).mean(dim = -1)

        second_term = (Yj_muj_epsj * GV).sum(dim = -1).mean(dim = -1)
        grad_eps  = (second_term - first_term)/(2*n)
    else: 
        grad_eps = None


    return grad_locs, grad_eps





def grad_V2(Y_flat, pi_mean, pi_cov, pi_dist, clippin = 1e-40):
    pic = pi_cov[0,0]
    y_pim_pic = ((Y_flat[:, None]- pi_mean[None])/pic)  # (n * b), N_target, d
    pi_prob = pi_dist.log_prob(Y_flat.unsqueeze(1)).exp()
    numerator = (y_pim_pic * pi_prob[..., None]).sum(dim = 1)
    denominator = pi_prob.sum(dim = -1)
    numerator += clippin
    denominator += clippin
    return (numerator/denominator[:,None]) # shape of Y flat , d



import torch
def sample_mixture(means, epsilon , M):

    N, d = means.shape
    component_indices = torch.randint(0, N, (M,))
    covs = (epsilon[..., None, None] ** 2) * torch.eye(d).unsqueeze(0)  
    mvn = torch.distributions.MultivariateNormal(loc=means, covariance_matrix=covs)

    all_samples = mvn.sample((M,))  

    samples = all_samples[torch.arange(M), component_indices]

    return samples


def compute_kl(m, e, pi_dist, M):
    y = sample_mixture(m, e, M = 100)
    n,d = m.shape
    N_target = pi_dist.batch_shape[0]
    clippin = 1e-40
    log_p = (torch.distributions.MultivariateNormal(m,
                                                   e[..., None, None]**2 * torch.eye(2)[None] 
                                                   ).log_prob(y.unsqueeze(1)).exp().sum(dim = -1) + clippin).log() - torch.log(torch.tensor([n]))
    log_q = (pi_dist.log_prob(y.unsqueeze(1)).exp().sum(dim = -1) + clippin).log() - torch.log(torch.tensor([N_target]))
    return (log_p - log_q).mean()