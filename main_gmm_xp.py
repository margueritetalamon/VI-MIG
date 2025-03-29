
from src_bis.optim import VI_GMM
from src_bis.target import Target

import json
import argparse
import os
from datetime import datetime
import numpy as np 







def parse_args():
    """ Parse command-line arguments. """
    parser = argparse.ArgumentParser(description="Run Gaussian mixture optimization experiments.")
    parser.add_argument("--d", type=int, default=10, help="Dimensionality of the data (d).")
    parser.add_argument("--target", type=str, default="gmm", help="Target type")
    parser.add_argument("--lr_mu", type=float, default=1, help="Learning rate for mu")
    parser.add_argument("--lr_eps", type=float, default=1, help="Learning rate for epsilon")
    parser.add_argument("--B_gradients", type=int, default=100, help="Batch size for Monte Carlo estimation.")
    parser.add_argument("--B_kls", type=int, default=1000, help="Batch size for Monte Carlo estimation.")
    parser.add_argument("--n_iter", type=int, default=1000, help="Number of iterations")
    parser.add_argument("--nxp", type=int, default=1, help="Number of time to do the same xp")
    parser.add_argument("--n_values", nargs="+", type=int, default=[1, 10, 20], 
                        help="List of values for the number of mixture components (N_mixture).")
    parser.add_argument("--exp_name", type=str, default="", help="Name for the parent folder of the experiment.")
    parser.add_argument("--full", type=str, default="n", help="Optim full cov matrices")
    parser.add_argument("--others", type=str, default="n", help="Optim iso others methods")
    return parser.parse_args()



def main(args):
    d = args.d 
    n_values = args.n_values
    exp_name = args.exp_name




    current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M")
    folder_name = os.path.join("REBUTTAL" , exp_name, current_datetime)

   

    target_gmm_mode =  "iso" # "full", "diag"
    n_target = 2
    target_sample_boule = 4 * np.sqrt(d) 
    target_scale_cov  = 8 * np.sqrt(d)  


    vgmm_sample_boule = 5 * np.sqrt(d) 
    vgmm_scale_cov  = 10 * np.sqrt(d) 


    target = Target( args.target , target_gmm_mode, n_components = n_target, s = target_sample_boule, scale = target_scale_cov,  d = d)

    hyperparam = {
                  "lr" : args.lr_eps,
                  "n_iter" : args.n_iter,
                  "B_kls" : args.B_kls,
                  "B_gradients" : args.B_gradients,
                  "target" :  args.target, 
                  "target_gmm_mode" : target_gmm_mode, 
                  "n_target": n_target,
                  "target_sample_boule" : target_sample_boule,
                  "vgmm_sample_boule" : vgmm_sample_boule,
                  "target_scale_cov" : target_scale_cov,
                  "vgmm_scale_cov" : vgmm_scale_cov
    }

    

    folder_name = os.path.join(folder_name)
    os.makedirs(folder_name, exist_ok=True)

    # Save the hyperparameters as a JSON file
    with open(os.path.join(folder_name , "hp.json"), "w") as outfile:
        json.dump(hyperparam, outfile, indent=4)

    
    np.save( f"{folder_name}/pi_mean.npy", target.model.means)
    np.save( f"{folder_name}/pi_cov.npy", target.model.covariances)

    vi = None
    # XPS = ["mirror", "optim_wo_eps", "ibw"]

    for N_mixture in n_values:  
            
                

       
        #### BASIC OPTIM  BW ISO 

        vi = VI_GMM(target, mode = "iso", learning_rate= args.lr_eps, n_iterations= args.n_iter,
                    n_components = N_mixture, scale = vgmm_scale_cov, BG = args.B_gradients, 
                    BKL = args.B_kls, s = vgmm_sample_boule) 
        
        previous_init = vi.vgmm.means 
        


        os.makedirs(f"{folder_name}/N{N_mixture}", exist_ok=True)

        np.save( f"{folder_name}/N{N_mixture}/vgmm_mean.npy", vi.vgmm.means)
        np.save( f"{folder_name}/N{N_mixture}/vgmm_cov.npy", vi.vgmm.covariances)

        vi.optimize(bw = True, md  = False, lin = False,  means_only=False, plot_iter=1000, gen_noise=True) 
        folder_xp = os.path.join(folder_name, f"N{N_mixture}", "ibw")
        os.makedirs(folder_xp, exist_ok=True)

        vi.save(folder_xp)




        vi = VI_GMM(target, mode = "iso", learning_rate= args.lr_eps, n_iterations= args.n_iter,
                    n_components = N_mixture, scale = vgmm_scale_cov, BG = args.B_gradients, 
                    BKL = args.B_kls, s = vgmm_sample_boule) 


        #### BASIC OPTIM  MD

        vi.optimize(bw = False, md  = True, lin = False,  means_only=False, plot_iter=1000, gen_noise=True) 
        folder_xp = os.path.join(folder_name, f"N{N_mixture}", "md")
        os.makedirs(folder_xp, exist_ok=True)

        vi.save(folder_xp)


        #### FULL COV OPTIM
        if args.full  == "y": 
            vi = VI_GMM(target, mode = "full", learning_rate= args.lr_eps, n_iterations= args.n_iter,
                        n_components = N_mixture, scale = vgmm_scale_cov, BG = args.B_gradients, 
                        BKL = args.B_kls, s = vgmm_sample_boule, means = previous_init) 


            #### BASIC OPTIM  BW FULL 
            vi.optimize(bw = True, md  = False, lin = False,  means_only=False, plot_iter=1000, gen_noise=True) 
            folder_xp = os.path.join(folder_name, f"N{N_mixture}", "bw")
            os.makedirs(folder_xp, exist_ok=True)

            vi.save(folder_xp)

        #### LIN
        if args.others  == "y": 
            vi = VI_GMM(target, mode = "iso", learning_rate= args.lr_eps, n_iterations= args.n_iter,
                        n_components = N_mixture, scale = vgmm_scale_cov, BG = args.B_gradients, 
                        BKL = args.B_kls, s = vgmm_sample_boule, means = previous_init) 


            #### BASIC OPTIM  LIN
            vi.optimize(bw = False, md  = False, lin = True,  means_only=False, plot_iter=1000, gen_noise=True) 
            folder_xp = os.path.join(folder_name, f"N{N_mixture}", "lin")
            os.makedirs(folder_xp, exist_ok=True)

            vi.save(folder_xp)



  










if __name__ == "__main__":
    args = parse_args()
    main(args)








