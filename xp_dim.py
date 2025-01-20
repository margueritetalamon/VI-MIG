import argparse
import os
import math
import torch
import tqdm
from datetime import datetime
from src.optim_hd import optim_mu_epsi_HD, optim_mu_only_HD, optim_mu_epsi_HD_ML
from src.utils_hd import  kl_evolution
from src.target import target_gmm, target_gmm_4modes

from src.saving import save_results






def run_xp(xp, pi_mean, pi_cov, mu_init, epsilon_init, folder_name, hyperparam = {}):

    N_mixture, d = mu_init.shape
    if xp == "mirror":
        subfolder_name = os.path.join(folder_name, xp, str(hyperparam["xp"]))
        mu_optim, epsilon_optim, M, E = optim_mu_epsi_HD(mu_init, 
                                                         epsilon_init , 
                                                         pi_mean, pi_cov, 
                                                         learning_rate_mu = hyperparam["lr_mu"],
                                                         learning_rate_eps = hyperparam["lr_eps"], 
                                                         num_iterations = hyperparam["n_iter"], 
                                                         B = hyperparam["B"])
        
        pi_dist = torch.distributions.MultivariateNormal(loc=pi_mean, covariance_matrix=pi_cov)
        kls = kl_evolution(pi_dist, M, E, hyperparam["B"])
        file_prefix = os.path.join(subfolder_name ,f"d{d}_N{N_mixture}_lrmu{hyperparam['lr_mu']}_lre{hyperparam['lr_eps']}_it{hyperparam['n_iter']}_B{hyperparam['B']}")
        save_results(subfolder_name, file_prefix, mu_optim, M, kls, epsilon_optim, E)


    if xp == "optim_wo_eps":
        subfolder_name = os.path.join(folder_name, xp, str(hyperparam["xp"]))
        mu_optim, M = optim_mu_only_HD(mu_init, 
                                    epsilon_init , 
                                    pi_mean, pi_cov, 
                                    learning_rate_mu = hyperparam["lr_mu"],
                                    learning_rate_eps = hyperparam["lr_eps"], 
                                    num_iterations = hyperparam["n_iter"], 
                                    B = hyperparam["B"])
        
        pi_dist = torch.distributions.MultivariateNormal(loc=pi_mean, covariance_matrix=pi_cov)
        kls = kl_evolution(pi_dist, M, B =  hyperparam["B"])
        file_prefix = os.path.join(subfolder_name ,f"d{d}_N{N_mixture}_lrmu{hyperparam['lr_mu']}_it{hyperparam['n_iter']}_B{hyperparam['B']}")
        save_results(subfolder_name, file_prefix, mu_optim, M, kls)

    
    if xp == "ibw":
        subfolder_name = os.path.join(folder_name, xp,str(hyperparam["xp"]))
        mu_optim, epsilon_optim, M, E = optim_mu_epsi_HD_ML(mu_init, 
                                                            epsilon_init , 
                                                            pi_mean, pi_cov, 
                                                            learning_rate_mu = hyperparam["lr_mu"],
                                                            learning_rate_eps = hyperparam["lr_eps"], 
                                                            num_iterations = hyperparam["n_iter"], 
                                                            B = hyperparam["B"])
        
        pi_dist = torch.distributions.MultivariateNormal(loc=pi_mean, covariance_matrix=pi_cov)
        kls = kl_evolution(pi_dist, M, E, B =  hyperparam["B"])
        file_prefix = os.path.join(subfolder_name ,f"d{d}_N{N_mixture}_lrmu{hyperparam['lr_mu']}_lre{hyperparam['lr_eps']}_it{hyperparam['n_iter']}_B{hyperparam['B']}")
        save_results(subfolder_name, file_prefix, mu_optim, M, kls, epsilon_optim, E)




def parse_args():
    """ Parse command-line arguments. """
    parser = argparse.ArgumentParser(description="Run Gaussian mixture optimization experiments.")
    parser.add_argument("--d", type=int, default=10, help="Dimensionality of the data (d).")
    parser.add_argument("--lr_mu", type=float, default=0.1, help="Learning rate for mu")
    parser.add_argument("--lr_eps", type=float, default=0.1, help="Learning rate for epsilon")
    parser.add_argument("--B", type=int, default=100, help="Batch size for Monte Carlo estimation.")
    parser.add_argument("--n_iter", type=int, default=1000, help="Number of iterations")
    parser.add_argument("--nxp", type=int, default=5, help="Number of time to do the same xp")
    parser.add_argument("--n_values", nargs="+", type=int, default=[1, 10, 50, 100], 
                        help="List of values for the number of mixture components (N_mixture).")
    parser.add_argument("--exp_name", type=str, default="", help="Name for the parent folder of the experiment.")
    return parser.parse_args()



def main(args):
    d = args.d 
    n_values = args.n_values
    exp_name = args.exp_name
    nb_xps = args.nxp

    hyperparam = {"lr_mu" : args.lr_mu,
                  "lr_eps" : args.lr_mu,
                  "n_iter" : args.n_iter,
                  "B" : args.B,

    }


    current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M")
    folder_name = os.path.join("saved" , exp_name, current_datetime)

    # N_target = 100
    N_target = 20
    # pi_mean, pi_cov = target_gmm_4modes(N_target, d)
    pi_mean, pi_cov = target_gmm(N_target, d)

    init_folder_name = os.path.join(folder_name, "init")
    os.makedirs(init_folder_name, exist_ok=True)
    
    torch.save(pi_mean, f"{folder_name}/init/pi_mean.pt")
    torch.save(pi_cov, f"{folder_name}/init/pi_cov.pt")
    XPS = ["mirror", "optim_wo_eps", "ibw"]

    for N_mixture in n_values:
        for nxp in range(nb_xps):

            mu_init = torch.empty(size = (0,d) )

            mvn_dist = torch.distributions.MultivariateNormal(torch.zeros(d), covariance_matrix=torch.eye(d) * 5**2)
            mu_init = torch.cat([mu_init, mvn_dist.sample((N_mixture - mu_init.shape[0],))])
            # epsilon_init = torch.ones(N_mixture)*1.5 * math.sqrt(d)
            epsilon_init = torch.ones(N_mixture)

            torch.save(mu_init, f"{folder_name}/init/mu_init_N{N_mixture}_xp{nxp}.pt")
            torch.save(epsilon_init, f"{folder_name}/init/epsilon_init_N{N_mixture}_xp{nxp}.pt")


            for xp in XPS: 
                hyperparam["xp"] = nxp
                run_xp(xp, pi_mean, pi_cov, mu_init, epsilon_init, folder_name, hyperparam)




if __name__ == "__main__":
    args = parse_args()
    main(args)