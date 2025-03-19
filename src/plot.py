import torch 
from matplotlib import pyplot as plt 
from scipy.stats import multivariate_normal
import numpy as np 
import math 
import os 
import glob 
from matplotlib.cm import get_cmap
from scipy.stats import norm  # 1D normal for marginals


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
    # colors = plt.cm.tab10(range(n))  # Assign unique colors for each mean
    colors = plt.cm.nipy_spectral(np.linspace(0, 1, n))


    plt.figure(figsize=(5,5))
    plt.contour(X, Y, Z, levels=10, cmap="viridis")
    plt.axis("off")
    for i in range(len(mu_locs)):
        ellipse = plt.Circle(mu_locs[i], epsilon[i]**2, color=colors[i], fill=False, linewidth=2)
        plt.gca().add_patch(ellipse)
    
    plt.show()





def plot_kl_evolutions_byN(
    base_dir,
    nvals
):


    # Choose a color palette. Seaborn offers many: "deep", "muted", "pastel",
    # "bright", "dark", "colorblind", etc.
    # If we only have one line to plot, we can just pick the first color.
    cmap = get_cmap('viridis')

    colors = [cmap(0.0), cmap(0.3), cmap(0.8)] 
    file_pattern = os.path.join(base_dir, "**", "*kls_evolution*.pt")
    all_candidates = glob.glob(file_pattern, recursive=True)

    # 2. Filter to only those containing "N{nval}"
    #    e.g. "N1" if nval = 1. We also ensure we have "kls_evolution" in the filename.

    for i,nval in enumerate(nvals):
        matched_files = [
            fpath for fpath in all_candidates
            if f"_N{nval}_" in os.path.basename(fpath)
        ]

        kl_evolutions = []
        for fpath in matched_files:
            
            kl_evolutions.append(torch.tensor(torch.load(fpath))) 
            
        kl_evolutions = torch.stack(kl_evolutions)
        steps = torch.arange(kl_evolutions.shape[1])
        print(kl_evolutions.shape)
        kl_mean = kl_evolutions.mean(dim = 0)
        kl_std  = kl_evolutions.std(dim = 0)


        plt.semilogy(steps, kl_mean, label = f"N = {nval}", linewidth=2, color = colors[i])
        plt.fill_between(steps,
            kl_mean - .2*kl_std,
            kl_mean + .2*kl_std,
            alpha=0.1, 
            color = colors[i]
        )
        plt.xlabel("Iteration")
        plt.ylabel("KL")
        # plt.grid(True)

        legend = plt.legend(
                fontsize=10,             # Make legend text bigger
                loc='upper right',       # Place legend in the upper right corner
                frameon=True,            # Add a box around the legend
                fancybox=True,           # Round the box corners
                borderpad=0.5,             # Increase padding inside the box
            )
        legend.get_frame().set_linewidth(1)  # Set the border thickness
        legend.get_frame().set_edgecolor('gray')  # Set the border color
        legend.get_frame().set_alpha(1)
        plt.ylabel("KL value", fontsize = 12)
        plt.xlabel("step", fontsize = 12)
    os.makedirs(f"{base_dir}/plots", exist_ok=True)
    plt.savefig(f"{base_dir}/plots/kls_evolutions_N11050.pdf", format="pdf", bbox_inches="tight")

    plt.show()





def plot_kl_evolutions_byopt(
    base_dir, N
):

    optims = ["mirror", "ibw", "optim_wo_eps"]
    O = {"mirror" : "MD", "ibw" : "IBW", "optim_wo_eps" : "FIXe"}


    # Choose a color palette. Seaborn offers many: "deep", "muted", "pastel",
    # "bright", "dark", "colorblind", etc.
    # If we only have one line to plot, we can just pick the first color.
    cmap = get_cmap('viridis')

    colors = [cmap(0.0), cmap(0.3), cmap(0.8)] 
    file_pattern = os.path.join(base_dir, "**", "*kls_evolution*.pt")
    all_candidates = glob.glob(file_pattern, recursive=True)

    # 2. Filter to only those containing "N{nval}"
    #    e.g. "N1" if nval = 1. We also ensure we have "kls_evolution" in the filename.
    print(all_candidates)
    for i,opt in enumerate(optims):
        matched_files = [
            fpath for fpath in all_candidates
            if f"{opt}" in fpath and f"_N{N}_" in fpath
        ]

        kl_evolutions = []
        print(matched_files)
        for fpath in matched_files:
            # print("", fpath)
            
            kl_evolutions.append(torch.tensor(torch.load(fpath))) 
            
        kl_evolutions = torch.stack(kl_evolutions)
        print(kl_evolutions.shape)
        steps = torch.arange(kl_evolutions.shape[1])
        kl_mean = kl_evolutions.mean(dim = 0)
        kl_std  = kl_evolutions.std(dim = 0)


        plt.semilogy(steps, kl_mean, label = f"{O[opt]}", linewidth=2, color = colors[i])
        plt.fill_between(steps,
            kl_mean - 0.2*kl_std,
            kl_mean + 0.2*kl_std,
            alpha=0.2, 
            color = colors[i]
        )
        plt.xlabel("Iteration")
        plt.ylabel("KL")
        # plt.grid(True)

        legend = plt.legend(
                fontsize=10,             # Make legend text bigger
                loc='upper right',       # Place legend in the upper right corner
                frameon=True,            # Add a box around the legend
                fancybox=True,           # Round the box corners
                borderpad=0.5,             # Increase padding inside the box
            )
        legend.get_frame().set_linewidth(1)  # Set the border thickness
        legend.get_frame().set_edgecolor('gray')  # Set the border color
        legend.get_frame().set_alpha(1)
        plt.ylabel("KL value", fontsize = 12)
        plt.xlabel("step", fontsize = 12)
    os.makedirs(f"{base_dir}/plots", exist_ok=True)
    plt.savefig(f"{base_dir}/plots/N_{N}kls_evolutions.pdf", format="pdf", bbox_inches="tight")

    plt.show()





def plot_target_and_initial_gaussians(
    filename,
    pi_means,          # shape (N_target, d)
    pi_cov,            # shape (d, d)
    mu_inits,          # shape (N_mixture, d)
    epsilon,           # shape (N_mixture)
    bounds=(-20, 20),  # plotting region (xmin,xmax) = (ymin,ymax)
    grid_size=100,     # number of points for mesh in each dimension
):
   

   
    N_target , d = pi_means.shape
    
    
    mvn_dist = torch.distributions.MultivariateNormal(pi_means, covariance_matrix=pi_cov)

    x = np.linspace(bounds[0], bounds[1], grid_size)
    y = np.linspace(bounds[0], bounds[1], grid_size)
    X, Y = np.meshgrid(x, y)
    pos = np.dstack((X, Y))  # shape (grid_size, grid_size, 2)

    pos_torch = torch.tensor(pos, dtype=torch.float32)  # Convert grid to torch tensor
    pdf_values = mvn_dist.log_prob(pos_torch[:, :, None, :])  # Shape (grid_size, grid_size, N)
    pdf_values = torch.exp(pdf_values)  # Convert log-probs to probs

    # Sum over components and normalize
    Z = pdf_values.sum(dim=-1).numpy() / N_target

    # 4) Plot the resulting contour
    plt.figure(figsize=(5, 5))
    plt.contour(X, Y, Z, levels=10, cmap="viridis")

    # 5) Plot circles for each initialized Gaussian
    #    Using radius = epsilon**2 as in your snippet
    N_mixture = mu_inits.shape[0]
    colors = plt.cm.nipy_spectral(np.linspace(0, 1, N_mixture))
    ax = plt.gca()
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    for i in range(N_mixture):
        center = mu_inits[i]
        circle = plt.Circle(
            (center[0], center[1]),
            radius=epsilon[i]**2,
            # color=colors[i],
            color = "black",
            fill=False,
            linewidth=2,
            zorder=10
        )
        ax.add_patch(circle)

    # 6) Final formatting
    plt.xlim(bounds)
    plt.ylim(bounds)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.tight_layout()

    plt.savefig(filename, format="pdf", bbox_inches="tight")
    


def plot_estimated(
    filename,
    mus_final,          # shape (N_mixture, d)
    epsilon_final,        # scalar, e.g. sqrt(d)
    bounds=(-20, 20),  # plotting region (xmin,xmax) = (ymin,ymax)
    grid_size=100):
   

   
    N , d = mus_final.shape

    
    mvn_dist = torch.distributions.MultivariateNormal(mus_final, covariance_matrix=(epsilon_final[..., None, None]**2 * torch.eye(d)[None]))


    x = np.linspace(bounds[0], bounds[1], grid_size)
    y = np.linspace(bounds[0], bounds[1], grid_size)
    X, Y = np.meshgrid(x, y)
    pos = np.dstack((X, Y))  # shape (grid_size, grid_size, 2)

    pos_torch = torch.tensor(pos, dtype=torch.float32)  # Convert grid to torch tensor
    pdf_values = mvn_dist.log_prob(pos_torch[:, :, None, :])  # Shape (grid_size, grid_size, N)
    pdf_values = torch.exp(pdf_values)  # Convert log-probs to probs

    # Sum over components and normalize
    Z = pdf_values.sum(dim=-1).numpy() / N 

    # 4) Plot the resulting contour
    plt.figure(figsize=(5, 5))
    plt.contour(X, Y, Z, levels=10, cmap="viridis")

    # 6) Final formatting
    plt.xlim(bounds)
    plt.ylim(bounds)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.tight_layout()
    # plt.show()
    plt.savefig(filename, format="pdf", bbox_inches="tight")





def plot_marginals(mu_final, epsilon_final, pi_mean, pi_cov, filename, bounds=(-30, 30), grid_size=100, grid_cols = 3):
   

    # Dimension of the data
    N, d = mu_final.shape
    
    # Grid for plotting
    x_grid = np.linspace(bounds[0], bounds[1], grid_size)

    # Prepare the figure
    grid_rows = int(np.ceil(d / grid_cols))
    fig, axes = plt.subplots(grid_rows, grid_cols, figsize=(5 * grid_cols, 4 * grid_rows))
    axes = axes.flatten()  # Flatten for easy indexing

    os.makedirs(os.path.dirname(filename), exist_ok = True)

    for j in range(d):
        # Marginal for dimension j: Estimated distribution
        estimated_marginals = [
            norm.pdf(x_grid, loc=mu[j], scale=epsilon)  # 1D Gaussian PDF
            for mu, epsilon in zip(mu_final, epsilon_final)
        ]
        Z_estimated = np.sum(estimated_marginals, axis=0) / len(estimated_marginals)

        # Marginal for dimension j: Target distribution
        target_marginals = [
            norm.pdf(x_grid, loc=pim[j], scale=np.sqrt(pic[j, j]))
            for pim, pic in zip(pi_mean, pi_cov)
        ]
        Z_target = np.sum(target_marginals, axis=0) / len(target_marginals)

        axes[j].plot(x_grid, Z_estimated, label="estimated", color="blue", linewidth=2)
        axes[j].plot(x_grid, Z_target, label="target", color="red", linestyle="--", linewidth=2)
        axes[j].set_title(f"dim {j}", fontweight='bold')
      
        # Plot both on the same axis
        if j == 0:
            # axes[j].plot(x_grid, Z_estimated, label="estimated", color="blue", linewidth=2)
            # axes[j].plot(x_grid, Z_target, label="target", color="red", linestyle="--", linewidth=2)
            axes[j].set_ylabel("density")
            # legend = .legend()

            legend = axes[j].legend(
                    fontsize=10,             # Make legend text bigger
                    loc='upper right',       # Place legend in the upper right corner
                    frameon=True,            # Add a box around the legend
                    fancybox=True,           # Round the box corners
                    borderpad=0.5,             # Increase padding inside the box
                )
            legend.get_frame().set_linewidth(1)  # Set the border thickness
            legend.get_frame().set_edgecolor('gray')  # Set the border color
            legend.get_frame().set_alpha(1)
            axes[j].tick_params(axis='y', length=5)
            axes[j].set_xticks([])
        
        elif j%3  ==  0:
            axes[j].tick_params(axis='y', length=5)
            if j != 6:
                axes[j].set_xticks([])

        elif j in [6,7,8] :
            axes[j].tick_params(axis='x', length=5)
            axes[j].set_yticks([])

        else: 
            axes[j].set_xticks([])
            axes[j].set_yticks([])


        

    plt.savefig(filename, format = "pdf")

    plt.show()