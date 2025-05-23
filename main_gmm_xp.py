
from src.optim import VI_GMM
from src.target import Target

import json
import argparse
import os
from datetime import datetime
import numpy as np 
import time

# np.random.seed(1)




def parse_args():
    """ Parse command-line arguments. """
    parser = argparse.ArgumentParser(description="Run Gaussian mixture optimization experiments.")
    parser.add_argument("--d", type=int, default=10, help="Dimensionality of the data (d).")
    parser.add_argument("--target", type=str, default="gmm", help="Target type")
    parser.add_argument("--target_mode", type=str, default="diag", help="Target type")
    parser.add_argument("--n_target", type=int, default=5, help="When GMM number of components")
    parser.add_argument("--target_sample_boule", type=int, default=5, help="Target_sample_boule")
    parser.add_argument("--target_scale_cov", type=int, default=5, help="target_scale_cov")
    parser.add_argument("--vgmm_sample_boule", type=int, default=5, help="vgmm sample boule")
    parser.add_argument("--vgmm_scale_cov", type=int, default=5, help="vgmm scale cov")
    parser.add_argument("--lr", type=float, default=1, help="Learning rate for mu")
    parser.add_argument("--B_gradients", type=int, default=10, help="Batch size for Monte Carlo estimation.")
    parser.add_argument("--B_kls", type=int, default=1000, help="Batch size for Monte Carlo estimation.")
    parser.add_argument("--n_iter", type=int, default=1000, help="Number of iterations")
    parser.add_argument("--compute_kl", type=int, default=1000, help="compute KL every iter")
    parser.add_argument("--nxp", type=int, default=1, help="Number of time to do the same xp")
    parser.add_argument("--seed", type=int, default=None, help="seed to create the target")
    parser.add_argument("--n_values", nargs="+", type=int, default=[1, 10, 20], 
                        help="List of values for the number of mixture components (N_mixture).")
    parser.add_argument("--exp_name", type=str, default="", help="Name for the parent folder of the experiment.")
    parser.add_argument("--full", action="store_true", help="Optim full cov matrices")
    parser.add_argument("--ngd", action="store_true", help="Optim iso NGD method")
    parser.add_argument("--means_only", action="store_true", help="Optim iso GD method")
    parser.add_argument("--scheduler", action="store_true", help="Enable scheduler in the optimizer")
    parser.add_argument("--nf", action="store_true", help="Optim normalizing flow")
    parser.add_argument("--n_iter_nf", type=int, default=1000, help="Number of iterations NF")
    parser.add_argument("--lr_nf", type=float, default=0.001, help="Learning rate NF")
    parser.add_argument("--hidden_dim_nf", type=int, default=124, help="Hidden dim NF")
    parser.add_argument("--n_blocks",  nargs="+", type=int, default=[1, 10, 20], 
                        help="List of values for the number of blocks in NF")



    return parser.parse_args()



def main(args):
    d = args.d 
    n_values = args.n_values
    exp_name = args.exp_name
    plot_iter = 10000



    current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M")
    folder_name = os.path.join(exp_name, current_datetime)
    os.makedirs(folder_name, exist_ok=True)


   
    n_target = args.n_target
    target_sample_boule = args.target_sample_boule
    target_scale_cov  = args.target_scale_cov
        
    ## 4modes target
    # pt = 3
    # means = np.array([[-pt, -pt], [pt, pt], [-pt, pt], [pt, -pt]])
    # weights = np.ones(n_target)/n_target
    # # weights = np.array([0.1, 0.3, 0.3, 0.3])
    # covariances = repeat(np.eye(d), "h w -> n h w", n = n_target)*2

    weights = means = covariances = None

    vgmm_sample_boule = args.vgmm_sample_boule
    vgmm_scale_cov  = args.vgmm_scale_cov


    target = Target( args.target , args.target_mode, n_components = n_target, s = target_sample_boule, scale = target_scale_cov,  d = d, weights = weights, means = means, covariances=covariances, seed = args.seed)
    means = None


    if target.name == "funnel":
        fun_sig = target.model.sigma 
    else:
        fun_sig = None



    hyperparam = {
                  "lr" : args.lr,
                  "n_iter" : args.n_iter,
                  "B_kls" : args.B_kls,
                  "B_gradients" : args.B_gradients,
                  "target" :  args.target, 
                  "target_mode" : args.target_mode, 
                  "n_target": n_target,
                  "target_sample_boule" : target_sample_boule,
                  "vgmm_sample_boule" : vgmm_sample_boule,
                  "target_scale_cov" : target_scale_cov,
                  "vgmm_scale_cov" : vgmm_scale_cov,
                  "funnel_sigma" : fun_sig
    }

    


    # Save the hyperparameters as a JSON file
    with open(os.path.join(folder_name , "hp.json"), "w") as outfile:
        json.dump(hyperparam, outfile, indent=4)

    if target.name == "gmm": 
        np.save( f"{folder_name}/pi_mean.npy", target.model.means)
        np.save( f"{folder_name}/pi_cov.npy", target.model.covariances)
        np.save( f"{folder_name}/pi_weights.npy", target.model.weights)

    vi = None

    for N_mixture in n_values:  
            
        init_means, init_covs = None, None

        for xp in range(args.nxp) :
            print("Optim IBW")
            vi = VI_GMM(target, mode = "iso", learning_rate= args.lr, n_iterations= args.n_iter,
                        n_components = N_mixture, scale = vgmm_scale_cov, BG = args.B_gradients, 
                        BKL = args.B_kls, s = vgmm_sample_boule, 
                        means = init_means, covs = init_covs) 
            os.makedirs(f"{folder_name}/N{N_mixture}/xp{xp}", exist_ok=True)

            if not os.path.exists((f"{folder_name}/N{N_mixture}/vgmm_mean.npy")):
                init_means = vi.vgmm.means
                init_covs = vi.vgmm.covariances
                np.save( f"{folder_name}/N{N_mixture}/vgmm_mean.npy", init_means)
                np.save( f"{folder_name}/N{N_mixture}/vgmm_cov.npy", init_covs )

            vi.optimize(bw = True, md  = False, ngd = False,  means_only=False, plot_iter=plot_iter, gen_noise=True, scheduler = args.scheduler, compute_kl=args.compute_kl) 
            folder_xp = os.path.join(folder_name, f"N{N_mixture}/xp{xp}", "ibw")
            os.makedirs(folder_xp, exist_ok=True)
            vi.save(folder_xp)

            print("Optim MD")
            vi = VI_GMM(target, mode = "iso", learning_rate= args.lr, n_iterations= args.n_iter,
                        n_components = N_mixture, scale = vgmm_scale_cov, BG = args.B_gradients, 
                        BKL = args.B_kls, s = vgmm_sample_boule,
                        means = init_means,
                        covs = init_covs) 
            vi.optimize(bw = False, md  = True, ngd = False,  means_only=False, plot_iter=plot_iter, gen_noise=True,  scheduler = args.scheduler, compute_kl=args.compute_kl) 
            folder_xp = os.path.join(folder_name, f"N{N_mixture}/xp{xp}", "md")
            os.makedirs(folder_xp, exist_ok=True)
            vi.save(folder_xp)



            if args.full: 
                print("Optim BW")
                vi = VI_GMM(target, mode = "full", learning_rate= args.lr, n_iterations= args.n_iter,
                            n_components = N_mixture, scale = vgmm_scale_cov, BG = args.B_gradients, 
                            BKL = args.B_kls, s = vgmm_sample_boule, 
                            means = init_means, covs = init_covs) 
                vi.optimize(bw = True, md  = False, ngd = False,  means_only=False, plot_iter=plot_iter, gen_noise=True , scheduler = args.scheduler, compute_kl=args.compute_kl) 
                folder_xp = os.path.join(folder_name, f"N{N_mixture}/xp{xp}", "bw")
                os.makedirs(folder_xp, exist_ok=True)
                vi.save(folder_xp)

            
            if args.ngd: 
                print("Optim ngd")
                vi = VI_GMM(target, mode = "iso", learning_rate= args.lr, n_iterations= args.n_iter,
                            n_components = N_mixture, scale = vgmm_scale_cov, BG = args.B_gradients, 
                            BKL = args.B_kls, s = vgmm_sample_boule, 
                            means = init_means, covs = init_covs) 
                vi.optimize(bw = False, md  = False, ngd = True,  means_only=False, plot_iter=plot_iter, gen_noise=True, scheduler=args.scheduler, compute_kl=args.compute_kl) 
                folder_xp = os.path.join(folder_name, f"N{N_mixture}/xp{xp}", "ngd")
                os.makedirs(folder_xp, exist_ok=True)
                vi.save(folder_xp)
    

            if args.means_only: 
                print("Optim GD")
                vi = VI_GMM(target, mode = "iso", learning_rate= args.lr, n_iterations= args.n_iter,
                            n_components = N_mixture, scale = vgmm_scale_cov, BG = args.B_gradients, 
                            BKL = args.B_kls, s = vgmm_sample_boule, 
                            means = init_means, covs = init_covs) 
                vi.optimize(bw = False, md  = False, ngd = False,  means_only=True, plot_iter=plot_iter, gen_noise=True, scheduler=args.scheduler, compute_kl=args.compute_kl) 
                folder_xp = os.path.join(folder_name, f"N{N_mixture}/xp{xp}", "gd")
                os.makedirs(folder_xp, exist_ok=True)
                vi.save(folder_xp)

    if args.nf:
        print("Optim Normalizing Flow")
        from code_nf.models import NormalizingFlow
        from src.normalizing_flow import train
        import torch 

        hp_nf = {
                "hidden_dim": args.hidden_dim_nf,
                "n_iter" : args.n_iter_nf,
                "lr" : args.lr_nf,
                "B_gradients" : args.B_gradients
                }
        with open(
            os.path.join(folder_name , "nf_hp.json"), "w") as outfile:
            json.dump(hp_nf, outfile, indent=4)

        for n_blocks in args.n_blocks:
            folder_xp = os.path.join(folder_name, f"N{n_blocks}", "nf")
            model = NormalizingFlow(d, n_blocks, hidden_dim=args.hidden_dim_nf)
            start = time.time()
            losses = train(model, target.model, n_epochs=args.n_iter_nf, lr=args.lr_nf, B = args.B_gradients*100 ) 
            time_NF = time.time() - start
            os.makedirs(folder_xp, exist_ok=True)
            torch.save(model.state_dict(), f"{folder_xp}/model.pth")
            np.save(f"{folder_xp}/kls.npy", losses)
            np.save(f"{folder_xp}/time.npy", time_NF)



    print("FOLDER NAME", folder_name)






  










if __name__ == "__main__":
    args = parse_args()
    main(args)








