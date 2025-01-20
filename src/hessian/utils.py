import torch 
from src.utils import gaussian_kernel, sample_mixture, V_function


def compute_squared_norm(y, mean):
    return ((y - mean)**2).sum(dim = -1)

def compute_keps_mu(y,means, epsilon):
    N, _ = means.shape
    return torch.stack([gaussian_kernel(y  , means[i], epsilon) for i in range(N)])/N


def estimate_hessian(means, epsilon, pi_mean, pi_cov, M = 1000):

    N, d = means.shape
    y = sample_mixture(means, epsilon, M)
    

    first_term = ((torch.stack([gaussian_kernel(y  , means[i], epsilon) * 
                                ( 
                                    (compute_squared_norm(y, means[i]) / (2*epsilon**4)) -  (d/(2*epsilon**2))
                                ) 
                            for i in range(N)])/N) / torch.clamp(compute_keps_mu(y, means, epsilon), 10e-40) )**2
    
    second_term = ((torch.clamp(compute_keps_mu(y, means, epsilon), 10e-40).log() - V_function(y, pi_mean, pi_cov) + 1) * torch.stack([

                    gaussian_kernel(y  , means[i], epsilon) * 
                ( 
                    (compute_squared_norm(y, means[i])**2 / (4*epsilon**8)) -  ((d+2)*compute_squared_norm(y, means[i]) / (4*epsilon**6)) + (d*(d+2)/(4*epsilon**4)) 
                ) 
                                        for i in range(N)])/N) / torch.clamp(compute_keps_mu(y, means, epsilon), 10e-40)


    return (first_term+second_term).mean()


def exponential(y, pim, pi_epsilon):
    ### compute exp(- norm (y - pim)**2 / 2*epsilon**2) 
    ### ie exponential term of the gaussian without the normalization constant 

    return (-((y-pim)**2).sum(dim = 1)/(2*pi_epsilon**2)).exp()



def hessian_V(y, means, pi_cov):

    ### it compute hessian of ln of a gaussian mixture with same epsilon. it changes a a bit because if epsilon is the same 
    ### pi cov is epsilon**2 * Id 
    ### for each components we can move the constante in x outside the sum and it simplifies
    ### which we cannot when epsilon is different for each components
    ### out b, d , d 
    ### b is the number of y 

    epsilon = pi_cov[0,0].sqrt()

    N, d = means.shape
    
    first_numerator = torch.stack([(exponential(y, m, epsilon)[..., None, None] * ((torch.eye(d) - ((y-m)[:,:,None]@(y-m)[:,None,:]/epsilon**2))/epsilon**2)) for m in means]).sum(dim = 0)

    sum_exp_term = torch.stack([(exponential(y, m, epsilon)[...,None] * (y - m)/epsilon**2) for m in means]).sum(dim  = 0)
    second_numerator = (sum_exp_term[:,:, None] @ sum_exp_term[:, None, :])

    denominator = torch.clamp(torch.stack([gaussian_kernel(y, m, epsilon) for m in means]).sum(dim = 0), 10e-40)


    return (first_numerator + second_numerator)/denominator[..., None, None]


# def hessian_ln_mixture(y, means, epsilons):

#     ### it compute hessian of ln of a gaussian mixture with different epsilon. it changes a a bit because if epsilon is the same 
#     ### it is still for a simple form of the covariance matrix eps * ID 
#     ### for each components we can move the constante in x outside the sum and it simplifies
#     ### which we cannot when epsilon is different for each components
#     ### out b, d , d 
#     ### b is the number of y 

#     N, d = means.shape
    
#     first_numerator = torch.stack([(gaussian_kernel(y, m, eps)[..., None, None] * ((torch.eye(d) - ((y-m)[:,:,None]@(y-m)[:,None,:]/eps**2))/eps**2)) for m, eps in zip(means, epsilons)]).sum(dim = 0)

#     sum_exp_term = torch.stack([(gaussian_kernel(y, m, eps)[...,None] * (y - m)/eps**2) for m,eps in zip(means, epsilons)]).sum(dim  = 0)
#     second_numerator = (sum_exp_term[:,:, None] @ sum_exp_term[:, None, :])

#     denominator = torch.clamp(torch.stack([gaussian_kernel(y, m, eps) for m, eps in zip(means, epsilons)]).sum(dim = 0), 10e-40)


#     return (first_numerator + second_numerator)/denominator[..., None, None]




# def hessian_ln_mixture(y, means, epsilons):

#     ### it compute hessian of ln of a gaussian mixture with different epsilon. it changes a a bit because if epsilon is the same 
#     ### it is still for a simple form of the covariance matrix eps * ID 
#     ### for each components we can move the constante in x outside the sum and it simplifies
#     ### which we cannot when epsilon is different for each components
#     ### out b, d , d 
#     ### b is the number of y 

#     N, d = means.shape
    
#     first_numerator = torch.stack([(gaussian_kernel(y, m, eps)[..., None, None] * (((y-m)[:,:,None]@(y-m)[:,None,:]) - eps**2 * torch.eye(d))/ eps**4) for m, eps in zip(means, epsilons)]).sum(dim = 0).mean(dim = 0).diag().sum()

#     first_denominator = (torch.stack([gaussian_kernel(y, m, eps) for m, eps in zip(means, epsilons)]).sum(dim = 0)).mean()
#     first_denominator = torch.clamp(first_denominator, 1e-40)

#     sum_exp_term = torch.stack([(gaussian_kernel(y, m, eps)[...,None] * (y - m) / eps**2) for m,eps in zip(means, epsilons)]).sum(dim  = 0)
#     second_numerator = (sum_exp_term[:,:, None] @ sum_exp_term[:, None, :]).mean(dim = 0 ).diag().sum()


#     second_denominator = ((torch.stack([gaussian_kernel(y, m, eps) for m, eps in zip(means, epsilons)]).sum(dim = 0))**2).mean()
#     second_denominator = torch.clamp(second_denominator, 1e-40)


#     return first_numerator/first_denominator - second_numerator/second_denominator




def hessian_ln_mixture(y, means, epsilons):

    ### it compute hessian of ln of a gaussian mixture with different epsilon. it changes a a bit because if epsilon is the same 
    ### it is still for a simple form of the covariance matrix eps * ID 
    ### for each components we can move the constante in x outside the sum and it simplifies
    ### which we cannot when epsilon is different for each components
    ### out b, d , d 
    ### b is the number of y 
    clippin = 1e-40
    N, d = means.shape
    
    first_numerator = torch.stack([(gaussian_kernel(y, m, eps)[..., None, None] * (((y-m)[:,:,None]@(y-m)[:,None,:]) - eps**2 * torch.eye(d))/ eps**4) for m, eps in zip(means, epsilons)]).sum(dim = 0)
    first_numerator = first_numerator + clippin

    first_denominator = (torch.stack([gaussian_kernel(y, m, eps) for m, eps in zip(means, epsilons)]).sum(dim = 0))
    first_denominator = first_denominator + clippin

    sum_exp_term = torch.stack([(gaussian_kernel(y, m, eps)[...,None] * (y - m) / eps**2) for m,eps in zip(means, epsilons)]).sum(dim  = 0)
    second_numerator = (sum_exp_term[:,:, None] @ sum_exp_term[:, None, :]) 
    second_numerator = second_numerator + clippin


    second_denominator = ((torch.stack([gaussian_kernel(y, m, eps) for m, eps in zip(means, epsilons)]).sum(dim = 0))**2)
    second_denominator = second_denominator + clippin


    return first_numerator/first_denominator[..., None, None] - second_numerator/second_denominator[..., None, None]


