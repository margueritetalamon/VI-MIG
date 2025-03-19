import math 
import torch



def target_gmm(N_target, d, s):
    """ Define the target Gaussian mixture distribution. """
    # torch.manual_seed(4)
    # epsilon = torch.randn((N_target,))
    epsilon =  torch.ones((N_target,))
    epsilon *= math.sqrt(d)

    print(epsilon)
    # epsilon = torch.ones((N_target,)) * math.sqrt(d)
    # sig = torch.eye(d) * (math.sqrt(d)*5)**2

    # mvn_dist = torch.distributions.MultivariateNormal(torch.zeros(d), covariance_matrix=sig)
    # pi_mean = mvn_dist.sample((N_target,))

    pi_mean = torch.empty(N_target, d).uniform_(-s, s)
    # pi_mean = mvn_dist.sample((N_target,))

    
    # pi_mean = -10 + 20 * torch.rand((N_target,d))
    # pi_mean = sample_uniform_hypersphere(N_target,d)*10
    # pi_mean = torch.randint(-15,15, size = (N_target,d))

    pi_cov = epsilon[..., None, None]**2 * torch.eye(d)[None]
    print(pi_cov.shape)

    # print(pi_cov.shape)
    # print(pi_cov[0])
    # print(pi_cov[-1])
    # pi_cov = random_SDP(N_target, d) 
    
    return pi_mean, pi_cov

def target_gmm_paperLin(N_target, d):
    """ Define the target Gaussian mixture distribution. """

    s = 4.3 * math.sqrt(d)

    epsilon =  torch.ones((N_target,)) * math.sqrt(d)
    pi_mean = torch.empty(N_target, d).uniform_(-s, s)
    pi_cov = epsilon[..., None, None]**2 * torch.eye(d)[None]
    
    return pi_mean, pi_cov



def random_SDP(N,d, scale = 1):

    A = torch.randn(N, d, d)
    spd_matrices = torch.matmul(A.transpose(-1, -2), A)
    spd_matrices *= scale

    return spd_matrices


def sample_uniform_hypersphere(N, d, radius=1.0):

    directions = torch.randn(N, d)
    directions /= torch.norm(directions, dim=1, keepdim=True)  # Normalize to unit length
    radii = torch.rand(N).pow(1 / d) * radius  # Correct scaling of radius

    samples = directions * radii.unsqueeze(1)
    return samples




def target_gmm_4modes(N_target, d):
    """ Define the target Gaussian mixture distribution. """
    epsilon = torch.randn((N_target,))
    epsilon =  torch.ones((N_target,))
    print(epsilon)
    # epsilon = torch.ones((N_target,)) * math.sqrt(d)
    pt = 2.5
    # pi_mean = -10 + 20 * torch.rand((N_target,d))
    pi_mean = torch.tensor([[pt,pt], [-pt,pt], [-pt,-pt], [pt,-pt]])
    print(pi_mean.shape)
    # pi_mean = sample_uniform_hypersphere(N_target,d)*10
    # pi_mean = torch.randint(-15,15, size = (N_target,d))
    pi_cov = epsilon[..., None, None]**2 * torch.eye(d)[None]
    print(pi_cov.shape)

    # print(pi_cov.shape)
    # print(pi_cov[0])
    # print(pi_cov[-1])
    # pi_cov = random_SDP(N_target, d) 
    
    return pi_mean, pi_cov


def sanity_target_1():

    pi_mean = torch.tensor([[5,3]])
    pi_cov = torch.tensor([[10, 0], [0, 1]])
    
    return pi_mean, pi_cov

def sanity_init_1():

    mu_locs = torch.tensor([[-8,-2]])
    epsilon = torch.tensor([[2, 0], [0, 2]])
    
    return mu_locs, epsilon


def sanity_target_2():

    pt = 3
    pi_mean = torch.tensor([[pt,pt], [-pt,pt], [-pt,-pt], [pt,-pt]])
    pi_cov =  torch.ones((4,))[..., None, None]* torch.eye(2)[None]*2
    
    return pi_mean, pi_cov


def sanity_init_2():

    mu_locs = torch.tensor([[-10,0], [5,3]])
    epsilon = torch.ones((2,))
    
    return mu_locs, epsilon


def initLin(N_mixture, d):

    mvn_dist = torch.distributions.MultivariateNormal(torch.zeros(d), covariance_matrix=torch.eye(d)*10**2)
    mu_init = mvn_dist.sample((N_mixture,))

    epsilon_init = torch.ones(N_mixture) * 10

    return mu_init, epsilon_init