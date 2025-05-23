
from src.optim import VI_GMM
from src.target import Target

import json
import argparse
import os
from datetime import datetime
import numpy as np 
import time

# np.random.seed(1)

import torch 
import tqdm 



def parse_args():
    """ Parse command-line arguments. """
    parser = argparse.ArgumentParser(description="Run Gaussian mixture optimization experiments.")
    parser.add_argument("--lr", type=float, default=1, help="Learning rate for mu")
    parser.add_argument("--B_gradients", type=int, default=10, help="Batch size for Monte Carlo estimation.")
    parser.add_argument("--B_kls", type=int, default=1000, help="Batch size for Monte Carlo estimation.")
    parser.add_argument("--n_iter", type=int, default=1000, help="Number of iterations")
    parser.add_argument("--compute_kl", type=int, default=1000, help="compute KL every iter")
    parser.add_argument("--nxp", type=int, default=1, help="Number of time to do the same xp")
    parser.add_argument("--seed", type=int, default=None, help="seed to create the target")
    parser.add_argument("--vgmm_sample_boule", type=int, default=5, help="vgmm sample boule")
    parser.add_argument("--vgmm_scale_cov", type=int, default=5, help="vgmm scale cov")
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
    parser.add_argument("--n_blocks",  nargs="+", type=int, default=[2], 
                        help="List of values for the number of blocks in NF")



    return parser.parse_args()



def main(args):
   
    n_values = args.n_values
    exp_name = args.exp_name
    plot_iter = 10000



    current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M")
    folder_name = os.path.join(exp_name, current_datetime)

    vgmm_sample_boule = args.vgmm_sample_boule
    vgmm_scale_cov  = args.vgmm_scale_cov

    targets = ["diana1", "diana2", "diana3"]
    hyperparam = {
                  "lr" : args.lr,
                  "n_iter" : args.n_iter,
                  "B_kls" : args.B_kls,
                  "B_gradients" : args.B_gradients,
                  "vgmm_sample_boule" : vgmm_sample_boule,
                  "vgmm_scale_cov" : vgmm_scale_cov,
    }

    for targ in targets:
        folder_targ = os.path.join(exp_name, targ, current_datetime)
        os.makedirs(folder_targ, exist_ok=True)
        target = Target( "ht" , targ)


        # Save the hyperparameters as a JSON file
        with open(os.path.join(folder_targ , "hp.json"), "w") as outfile:
            json.dump(hyperparam, outfile, indent=4)

        vi = None
        for N_mixture in n_values:  
            init_means, init_covs = None, None
            for xp in range(args.nxp) :
                print("Optim IBW")
                vi = VI_GMM(target, mode = "iso", learning_rate= args.lr, n_iterations= args.n_iter,
                            n_components = N_mixture, scale = vgmm_scale_cov, BG = args.B_gradients, 
                            BKL = args.B_kls, s = vgmm_sample_boule, 
                            means = init_means, covs = init_covs) 
                os.makedirs(f"{folder_targ}/N{N_mixture}/xp{xp}", exist_ok=True)
                if not os.path.exists((f"{folder_targ}/N{N_mixture}/vgmm_mean.npy")):
                    init_means = vi.vgmm.means
                    init_covs = vi.vgmm.covariances
                    np.save( f"{folder_targ}/N{N_mixture}/vgmm_mean.npy", init_means)
                    np.save( f"{folder_targ}/N{N_mixture}/vgmm_cov.npy", init_covs )
                vi.optimize(bw = True, md  = False, ngd = False,  means_only=False, plot_iter=plot_iter, gen_noise=True, scheduler = args.scheduler, compute_kl=args.compute_kl) 
                folder_xp = os.path.join(folder_targ, f"N{N_mixture}/xp{xp}", "ibw")
                os.makedirs(folder_xp, exist_ok=True)
                vi.save(folder_xp)



                print("Optim MD")
                vi = VI_GMM(target, mode = "iso", learning_rate= args.lr, n_iterations= args.n_iter,
                            n_components = N_mixture, scale = vgmm_scale_cov, BG = args.B_gradients, 
                            BKL = args.B_kls, s = vgmm_sample_boule,
                            means = init_means,
                            covs = init_covs) 
                vi.optimize(bw = False, md  = True, ngd = False,  means_only=False, plot_iter=plot_iter, gen_noise=True,  scheduler = args.scheduler, compute_kl=args.compute_kl) 
                folder_xp = os.path.join(folder_targ, f"N{N_mixture}/xp{xp}", "md")
                os.makedirs(folder_xp, exist_ok=True)
                vi.save(folder_xp)

                if args.full: 
                    print("Optim BW")
                    vi = VI_GMM(target, mode = "full", learning_rate= args.lr, n_iterations= args.n_iter,
                                n_components = N_mixture, scale = vgmm_scale_cov, BG = args.B_gradients, 
                                BKL = args.B_kls, s = vgmm_sample_boule, 
                                means = init_means, covs = init_covs) 
                    vi.optimize(bw = True, md  = False, ngd = False,  means_only=False, plot_iter=plot_iter, gen_noise=True , scheduler = args.scheduler, compute_kl=args.compute_kl) 
                    folder_xp = os.path.join(folder_targ, f"N{N_mixture}/xp{xp}", "bw")
                    os.makedirs(folder_xp, exist_ok=True)
                    vi.save(folder_xp)

                if args.ngd: 
                    print("Optim NGD")
                    vi = VI_GMM(target, mode = "iso", learning_rate= args.lr, n_iterations= args.n_iter,
                                n_components = N_mixture, scale = vgmm_scale_cov, BG = args.B_gradients, 
                                BKL = args.B_kls, s = vgmm_sample_boule, 
                                means = init_means, covs = init_covs) 
                    vi.optimize(bw = False, md  = False, ngd = True,  means_only=False, plot_iter=plot_iter, gen_noise=True, scheduler=args.scheduler, compute_kl=args.compute_kl) 
                    folder_xp = os.path.join(folder_targ, f"N{N_mixture}/xp{xp}", "ngd")
                    os.makedirs(folder_xp, exist_ok=True)
                    vi.save(folder_xp)


                if args.means_only: 
                    print("Optim GD")
                    vi = VI_GMM(target, mode = "iso", learning_rate= args.lr, n_iterations= args.n_iter,
                                n_components = N_mixture, scale = vgmm_scale_cov, BG = args.B_gradients, 
                                BKL = args.B_kls, s = vgmm_sample_boule, 
                                means = init_means, covs = init_covs) 
                    vi.optimize(bw = False, md  = False, ngd = False,  means_only=True, plot_iter=plot_iter, gen_noise=True, scheduler=args.scheduler, compute_kl=args.compute_kl) 
                    folder_xp = os.path.join(folder_targ, f"N{N_mixture}/xp{xp}", "gd")
                    os.makedirs(folder_xp, exist_ok=True)
                    vi.save(folder_xp)
        if args.nf:
            print("Optim Normalizing Flow")
            d = 2
            from code_nf.models import NormalizingFlow
            import torch 
            hp_nf = {
                    "hidden_dim": args.hidden_dim_nf,
                    "n_iter" : args.n_iter_nf,
                    "lr" : args.lr_nf,
                    "B_gradients" : args.B_gradients
                    }
            
            with open(
                os.path.join(folder_targ , "nf_hp.json"), "w") as outfile:
                json.dump(hp_nf, outfile, indent=4)
                
            target.model.tau = torch.as_tensor(target.model.tau)
            target.model.s = torch.as_tensor(target.model.s)
            target.model.Sigma = torch.as_tensor(target.model.Sigma)

            for n_blocks in args.n_blocks:
                folder_xp = os.path.join(folder_targ, f"N{n_blocks}", "nf")
                model = NormalizingFlow(d, n_blocks, hidden_dim=args.hidden_dim_nf)
                start = time.time()
                losses = train(model, target.model, n_epochs=args.n_iter_nf, lr=args.lr_nf ) 
                time_NF = time.time() - start
                os.makedirs(folder_xp, exist_ok=True)
                torch.save(model.state_dict(), f"{folder_xp}/model.pth")
                np.save(f"{folder_xp}/kls.npy", losses)
                np.save(f"{folder_xp}/time.npy", time_NF)



    print("FOLDER NAME", folder_name)




def sas_elliptical(target, z):
    ### z is a tensor
    inv_tau = 1.0 / target.tau
    a = torch.arcsinh(z)                     # shape (..., d)
    b = inv_tau * (a + target.s)                 # shape (..., d)
    z0 = torch.sinh(b)                       # shape (..., d)
    d = torch.cosh(b) * inv_tau / torch.sqrt(1 + z*z)
    return z0, b, d

def prob(target,  z):
    z0, b, d = sas_elliptical(target, z)
    d_logdet = torch.linalg.slogdet(target.Sigma)[1]
    invS = torch.linalg.inv(target.Sigma)
    
    # quadratic form
    # flatten samples into rows if needed
    z0_flat = z0.reshape(-1, z0.shape[-1])
    qf = torch.einsum("ni,ij,nj->n", z0_flat, invS, z0_flat)
    

    # Jacobian factor per sample
    jac = torch.prod(d.reshape(-1, d.shape[-1]), dim=1)
    
    # multivariate normal density at z0
    d_dim = z0.shape[-1]
    norm_const = (2*torch.pi)**(-0.5*d_dim) * torch.exp(-0.5*d_logdet)
    dens_flat = norm_const * torch.exp(-0.5*qf) * jac
    
    return dens_flat.reshape(z0.shape[:-1])
    

def kl_divergence_ht(model, target ):

    x = model.sample(1000)
    target_log_prob = prob(target, x).log()

    return (model.log_prob(x) - target_log_prob).mean()

def train(model, target,  n_epochs=100, lr=1e-3):
    losses = []
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in tqdm.tqdm(range(n_epochs)):
        optimizer.zero_grad()
        loss = kl_divergence_ht(model, target)

        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return losses
                    


  










if __name__ == "__main__":
    args = parse_args()
    main(args)








