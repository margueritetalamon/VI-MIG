import torch 
from matplotlib import pyplot as plt 
from scipy.stats import multivariate_normal
import numpy as np 
import math 


def plot_evolution_1d(pi_dist, mu_locs, epsilon):
    n_components , d = mu_locs.shape
    x_range = (-5, 5)
    num_points = 100
    x = torch.linspace(x_range[0], x_range[1], num_points)

    target = torch.exp(pi_dist.log_prob(x[..., None]))
    muK = torch.zeros_like(x)
    for loc in mu_locs:
        component_density = torch.exp(-((x - loc)**2) / (2 * epsilon**2)) / (math.sqrt(2 * torch.pi * epsilon**2))
        muK +=  component_density
    muK /= n_components
    plt.plot(x.numpy(), muK.detach().numpy(), label='K * mu density')
    plt.plot(x.numpy(), target.detach().numpy(), label='pi density')
    plt.scatter(mu_locs.detach().numpy(), torch.zeros_like(mu_locs).numpy(), color='red', label='mu locations')
    plt.legend()
    plt.show()

def plot_evolution_2d(pi_mean, pi_cov, mu_locs, epsilon):
    n, d = mu_locs.shape
    N_target = pi_mean.shape[0]
    p_dist = [multivariate_normal(mean=mean, cov=pi_cov) for mean in pi_mean]
    x = np.linspace(-40, 40, 100)
    y = np.linspace(-40, 40, 100)
    X, Y = np.meshgrid(x, y)
    pos = np.dstack((X, Y)) 
    Z = np.dstack([p.pdf(pos) for p in p_dist]).sum(axis = -1)/N_target
    colors = plt.cm.tab10(range(n))  # Assign unique colors for each mean

    plt.figure(figsize=(5,5))
    plt.contour(X, Y, Z, levels=10, cmap="viridis")
    plt.axis("off")
    for i in range(len(mu_locs)):
        ellipse = plt.Circle(mu_locs[i], epsilon[i]**2, color=colors[i], fill=False, linewidth=2)
        plt.gca().add_patch(ellipse)
    
    plt.show()