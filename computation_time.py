
from src.optim import VI_GMM
from src.target import Target


import json
import argparse
import os
from datetime import datetime
import numpy as np 
import time
from einops import repeat 

# np.random.seed(1)





def parse_args():
    """ Parse command-line arguments. """
    parser = argparse.ArgumentParser(description="Run Gaussian mixture optimization experiments.")
    parser.add_argument("--d", nargs="+", type=int, default=[1, 10, 50, 80, 100, 200], help="Dimensionality of the data (d).")
    parser.add_argument("--lr", type=float, default=1, help="Learning rate for mu")
    parser.add_argument("--B_gradients", type=int, default=10, help="Batch size for Monte Carlo estimation.")
    parser.add_argument("--B_kls", type=int, default=1000, help="Batch size for Monte Carlo estimation.")
    parser.add_argument("--n_iter", type=int, default=10000, help="Number of iterations")
    parser.add_argument("--compute_kl", type=int, default=1000, help="compute KL every iter")
    parser.add_argument("--n_values", nargs="+", type=int, default=[1, 5], 
                        help="List of values for the number of mixture components (N_mixture).")
    parser.add_argument("--exp_name", type=str, default="", help="Name for the parent folder of the experiment.")
   



    return parser.parse_args()



def main(args):
    d_values = args.d 
    n_values = args.n_values
    exp_name = args.exp_name





    current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M")
    folder_name = os.path.join(exp_name, current_datetime)
    os.makedirs(folder_name, exist_ok=True)
 
    target_gmm_mode =  "diag" # "full", "diag"
    n_target = 5
    target_sample_boule = 10
    target_scale_cov  = 10
        

    weights = means = covariances = None


    vgmm_sample_boule = 20
    vgmm_scale_cov  = 100
    hyperparam = {
                    "lr" : args.lr,
                    "n_iter" : args.n_iter,
                    "B_kls" : args.B_kls,
                    "B_gradients" : args.B_gradients,
                    "target_gmm_mode" : target_gmm_mode, 
                    "n_target": n_target,
                    "target_sample_boule" : target_sample_boule,
                    "vgmm_sample_boule" : vgmm_sample_boule,
                    "target_scale_cov" : target_scale_cov,
                    "vgmm_scale_cov" : vgmm_scale_cov
        }

        


        # Save the hyperparameters as a JSON file
    with open(os.path.join(folder_name , "hp.json"), "w") as outfile:
        json.dump(hyperparam, outfile, indent=4)


    for d in d_values:
        lr = 0.1 / d
        vgmm_sample_boule = 100 / d
        target_sample_boule = 100 / d

        target_scale_cov = 5
        vgmm_scale_cov = 10


        folder_xp = os.path.join(folder_name, f"d{d}")
        os.makedirs(folder_xp, exist_ok=True)

        target = Target( "gmm" , target_gmm_mode, n_components = n_target, s = target_sample_boule, scale = target_scale_cov,  d = d)
    
        np.save( f"{folder_xp}/pi_mean.npy", target.model.means)
        np.save( f"{folder_xp}/pi_cov.npy", target.model.covariances)
        np.save( f"{folder_xp}/pi_weights.npy", target.model.weights)

        vi = None

        for N_mixture in n_values:      
            init_means, init_covs = None, None

            print("Optim IBW")
            vi = VI_GMM(target, mode = "iso", learning_rate= lr, n_iterations= args.n_iter,
                        n_components = N_mixture, scale = vgmm_scale_cov, BG = args.B_gradients, 
                        BKL = args.B_kls, s = vgmm_sample_boule, 
                        means = init_means, covs = init_covs) 
            os.makedirs(f"{folder_xp}/N{N_mixture}", exist_ok=True)

            if not os.path.exists((f"{folder_xp}/N{N_mixture}/vgmm_mean.npy")):
                init_means = vi.vgmm.means
                init_covs = vi.vgmm.covariances
                np.save( f"{folder_xp}/N{N_mixture}/vgmm_mean.npy", init_means)
                np.save( f"{folder_xp}/N{N_mixture}/vgmm_cov.npy", init_covs )

            vi.optimize(bw = True, md  = False, lin = False,  means_only=False, plot_iter=10000, gen_noise=True, scheduler = False, compute_kl=args.compute_kl) 
            folder_xp = os.path.join(folder_xp, f"N{N_mixture}", "ibw")
            os.makedirs(folder_xp, exist_ok=True)
            np.save(f"{folder_xp}/kls.npy", vi.kls)
            np.save(f"{folder_xp}/time.npy", vi.time)
            np.save( f"{folder_xp}/mean.npy", vi.vgmm.means)
            np.save( f"{folder_xp}/epsilons.npy", vi.vgmm.epsilons )

            folder_xp = os.path.join(folder_name, f"d{d}")

            print("OPTIM BW")
            vi = VI_GMM(target, mode = "full", learning_rate= lr, n_iterations= args.n_iter,
                        n_components = N_mixture, scale = vgmm_scale_cov, BG = args.B_gradients, 
                        BKL = args.B_kls, s = vgmm_sample_boule,
                        means = init_means,
                        covs = init_covs) 
            
            vi.optimize(bw = True, md  = False, lin = False,  means_only=False, plot_iter=10000, gen_noise=True, scheduler = False, compute_kl=args.compute_kl) 
            folder_xp = os.path.join(folder_xp, f"N{N_mixture}", "bw")
            os.makedirs(folder_xp, exist_ok=True)
            np.save(f"{folder_xp}/kls.npy", vi.kls)
            np.save(f"{folder_xp}/time.npy", vi.time)
            np.save( f"{folder_xp}mean.npy", vi.vgmm.means)
            np.save( f"{folder_xp}/epsilons.npy", vi.vgmm.covariances )





    print("FOLDER NAME", folder_name)






  










if __name__ == "__main__":
    args = parse_args()
    main(args)








