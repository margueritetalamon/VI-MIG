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
    clippin = 0
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

    torch.manual_seed(1)

    y_01 = torch.randn(n, B, d) 
    KLS = []
    for m, eps in tqdm.tqdm(zip(M, E)):
        KLS.append( MC_KL(m, eps, pi_dist, y_01) ) 
    
    return KLS




def compute_grads(mu_locs, epsilon, pi_mean , pi_cov, y_01, optim_eps = True, clippin = 0):

    clippin = 0
    n, B, d = y_01.shape
    # print("MU", mu_locs)
    # print("EPS", epsilon)
    ################### PREP ########################################
    mixture_epsilon = repeat(epsilon, "n -> n b", b = B ) # n b 
    # print("epsilon", mixture_epsilon)
    mixture_means = repeat(mu_locs, "n d -> n b d", b= B) # n b d
    # print("means", mixture_means)

    Y= ((y_01 * mixture_epsilon[..., None]) + mixture_means) # n, B , d
    Y_flat = rearrange(Y, "n b d -> (n b) d")

    #################### GRAD xj ####################################
    # print("Y_flat", Y_flat)
    GV = rearrange(grad_V(Y_flat,  pi_mean, pi_cov), "(n b) d -> n b d", n  = n)
    # print("GV", GV.shape)
    # print("GV", GV)
    first_term = GV.mean(dim = 1)
    # print("EgV", first_term)
    y_mu_div_eps = ((Y_flat[:, None]- mu_locs[None])/(epsilon[None,:, None]**2)) # b, n ,d (b is n * b ) 
    # print("y_mu_div_eps", y_mu_div_eps.shape)
    # print(y_mu_div_eps)
    GK = gaussian_kernel_HD(Y_flat, mu_locs,epsilon)

    # print("GK", GK.shape)
    num = (y_mu_div_eps  * GK[..., None]).sum(dim  = -2)
    # print("num", num.shape)
    # num = (y_mu_div_eps  * GK[..., None]).sum(dim  = 0)
    den =  GK.sum(dim = -1)
    # den =  GK.sum(dim =0)
    
    den = rearrange(den, "(n b) -> n b", n = n)
    num = rearrange(num, "(n b) d -> n b d", n = n )

    num1 = num + clippin
    den = den + clippin
    second_term = (num1/den[..., None]).mean(dim = 1)

    # print("second term", second_term)
    grad_locs = (first_term - second_term)/n # n, d 

    # print(grad_locs)
    # grad_locs = (first_term - second_term)# n, d 

    ################### GRAD ej #####################################
    if optim_eps:
        Yj_muj_epsj = ((Y - mixture_means)/(epsilon[..., None,None]**2))
        num2 = (Yj_muj_epsj * num).sum(dim = -1) #PRODUIT SCALAIRE
        num2 += clippin 
        first_term = (num2/den).mean(dim = -1)
        # print("FIRSST TERM  EPS", first_term)


        second_term = (Yj_muj_epsj * GV).sum(dim = -1).mean(dim = -1)
        grad_eps  = (second_term - first_term)/(2*n)
        # print("SECOND TERM  EPS", second_term)
        # print("GRAD EPS", grad_eps)
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
def sample_mixture(noise, mean, eps , component_indices):

    N, d = mean.shape
    # print("MEAN",mean.shape)
    mean = mean[component_indices]
    # print(mean.shape)

    # print("EPS",eps.shape)
    eps = eps[component_indices].unsqueeze(1)
    # print(eps.shape)
    # print(noise.shape)

    samples = mean + eps * noise

    return samples


def compute_kl(m, e, pi_dist, noise, component_indices):

    y = sample_mixture(noise, m, e, component_indices)
    n,d = m.shape
    N_target = pi_dist.batch_shape[0]
    clippin = 1e-40
    log_p = (torch.distributions.MultivariateNormal(m,
                                                   e[..., None, None]**2 * torch.eye(2)[None] 
                                                   ).log_prob(y.unsqueeze(1)).exp().sum(dim = -1) + clippin).log() - torch.log(torch.tensor([n]))
    log_q = (pi_dist.log_prob(y.unsqueeze(1)).exp().sum(dim = -1) + clippin).log() - torch.log(torch.tensor([N_target]))
    return (log_p - log_q).mean()



def compte_all_kls(means, epsilons, pi_dist, M =100):


    n, d = means[0].shape
    noise = torch.randn(M,d)
    component_indices = torch.randint(0, n, (M,))


    KLS =  []
    for m,e in zip(means,epsilons): 
        KLS.append(compute_kl(m,e, pi_dist, noise, component_indices))

    return KLS






import torch
import math

def new_grads(mu_locs, epsilons, pi_mean, pi_cov, y_init, 
                      optimize_eps=True, clipval=1e-40):
    """
    mu_locs:   shape (n, d)
    epsilons:  shape (n,)
    pi_mean, pi_cov: parameters for grad_V (same shape as in your code)
    y_init:    shape (n, b, d) -- the base samples y_01
    optimize_eps: bool, whether or not to compute grad_eps
    clipval:   small constant for numerical stability

    returns:
      grad_locs of shape (n, d)
      grad_eps  of shape (n,) or None
    """

    # sizes
    n, d = mu_locs.shape
    _, b, _ = y_init.shape
    
    ################################################################
    # 1) build the sample points: y_j^i = mu_j + eps_j * y_init_j^i
    ################################################################
    # shape (n, b, d)
    Y = y_init * epsilons[:, None, None] + mu_locs[:, None, :]

    ################################################################
    # 2) define the gaussian kernel and log-kernel helpers
    ################################################################
    # k_j(y) = (1 / (2 pi eps_j^2)^(d/2)) * exp( -||y - mu_j||^2 / (2 eps_j^2) )
    # we compute in a vectorized way:
    def new_gaussian_kernel(Y, mu, eps):
        # Y:  (n, b, d)
        # mu: (n,    d)
        # eps:(n,)
        # return shape (n, b)
        diff = Y - mu[:, None, :]  # (n,b,d)
        sqnorm = (diff**2).sum(dim=-1)  # (n,b)
        denom = (2 * math.pi * eps**2) ** (0.5 * d)  # shape (n,)
        return torch.exp(-0.5 * sqnorm / (eps[:, None]**2)) / denom[:, None]

    # gradient of V at each y; must be vectorized
    # in your code, pi_mean & pi_cov are lists or something similar.
    # below is one possible vectorized version if you had a single mixture:
    def new_grad_V(Y, pi_mean, pi_cov):
        """
        Y:  shape (n, b, d)
        pi_mean: shape (m, d)  (if it's a mixture of m Gaussians)
        pi_cov:  shape (m, d, d) or (m, 1, 1) if diagonal
        returns: shape (n, b, d)
        """
        # example: sum of exp(...) over each mixture component
        # you can adapt this to however pi_mean/pi_cov are shaped in your code
        # for illustration, we do the same logic you had, but in a vectorized style
        clip = clipval

        # broadcast shapes carefully: we want to compare each Y to each pi_mean
        # Y: (n,b,d) -> (n,b,1,d)
        # pi_mean: (m,d) -> (1,1,m,d)
        # assume pi_cov is also broadcastable
        Y_expanded = Y.unsqueeze(2)       # (n,b,1,d)
        pi_mean_exp = pi_mean.unsqueeze(0).unsqueeze(0)  # (1,1,m,d)
        diff = Y_expanded - pi_mean_exp   # (n,b,m,d)
        # if pi_cov is diagonal, do something like:
        #  sqnorm = diff^2 / pi_cov
        # for simplicity, assume a single scalar in each pi_cov (like your code):
        # pi_cov: shape (m,1,1) -> (1,1,m,1)
        pi_cov_exp = pi_cov.unsqueeze(0).unsqueeze(0) # (1,1,m,1)
        sqnorm = (diff**2).sum(dim=-1) / pi_cov_exp.squeeze(-1)  # (n,b,m)
        w = torch.exp(-0.5 * sqnorm) + clip
        # denominator
        denom = w.sum(dim=2, keepdim=True) + clip
        # numerator for gradient: sum( w * (diff/pi_cov) ), but we must keep shape aligned
        numerator = (w[..., None] * diff / pi_cov_exp).sum(dim=2) + clip

        # final shape (n,b,d)
        return numerator / denom

    ################################################################
    # 3) compute ∇V at each sample, then the first term for ∇_{mu}
    ################################################################
    # shape (n, b, d)
    GV = new_grad_V(Y, pi_mean, pi_cov)

    # average over b
    first_term = GV.mean(dim=1)  # shape (n, d)

    ################################################################
    # 4) compute the kernel-based term: E_{k_j}[ (y - mu_j)/eps_j^2 ]
    ################################################################
    # shape (n, b)
    K = new_gaussian_kernel(Y, mu_locs, epsilons)
    den = K.sum(dim=1) + clipval           # (n,)
    # shape (n, b, d)
    diff_over_eps2 = (Y - mu_locs[:, None, :]) / (epsilons[:, None, None]**2)
    # shape (n, d)
    num = (diff_over_eps2 * K[..., None]).sum(dim=1)
    second_term = num / den[:, None]  # (n, d)

    ################################################################
    # 5) combine the two terms for grad_locs
    ################################################################
    grad_locs = (first_term - second_term) / float(n)  # shape (n,d)

    ################################################################
    # 6) if requested, compute grad_eps
    ################################################################
    if optimize_eps:
        # we want a piece that looks like
        #   1/(2n) * E_{k_j}[ (grad_V(y) - something) dot ((y - mu)/eps^2 ) ]
        # the code below follows your original approach
        # shape (n, b, d)
        #   we already have diff_over_eps2, and we also have the num from above
        #   = K(...) * diff_over_eps2
        # note we do a dot product with grad_V or with the kernel portion
        # first:  (diff_over_eps2 * GV).sum(dim=-1)
        # second: (diff_over_eps2 * num/den).sum(dim=-1)
        # then average
        dot_gv = (diff_over_eps2 * GV).sum(dim=-1).mean(dim=1)  # (n,)
        # note num is shape (n,d), so we have to broadcast it back to (n,b,d):
        # but we only want the fraction:  (num / den[:,None]) => shape (n,d)
        # expand to match (n,b,d):
        frac_expanded = (num / den[:, None])[:, None, :]  # (n,1,d)
        # shape (n,b)
        dot_frac = (diff_over_eps2 * frac_expanded).sum(dim=-1).mean(dim=1)
        # combine
        grad_eps = 0.5 * (dot_gv - dot_frac) / float(n)
    else:
        grad_eps = None

    return grad_locs, grad_eps
