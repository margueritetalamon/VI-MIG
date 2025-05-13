from src.prepare_dataset import prepare_dataset
from src.optim import VI_GMM
from src.target import Target


import json
import argparse
import os
from datetime import datetime
import numpy as np 
import time 
from einops import rearrange






def parse_args():
    """ Parse command-line arguments. """
    parser = argparse.ArgumentParser(description="Run Gaussian mixture optimization experiments.")
    parser.add_argument("--d", type=int, default=10, help="Dimensionality of the data (d).")
    parser.add_argument("--target", type=str, default="gmm", help="Target type")
    parser.add_argument("--dataset_name", type=str, default="sythetic", help="Dataset name, please choose between [breast_cancer, wine, toxicity, boston and synthetic]")
    parser.add_argument("--train_ratio", type=float, default=0.5, help="Datas training ratio")
    parser.add_argument("--prior_eps", type=float, default=100, help="Espilon prior")
    parser.add_argument("--lr", type=float, default=1, help="Learning rate")
    parser.add_argument("--B_gradients", type=int, default=100, help="Batch size for Monte Carlo estimation.")
    parser.add_argument("--compute_kls", type=int, default=1000, help="Batch size for Monte Carlo estimation.")
    parser.add_argument("--hidden_units", type=int, default=10, help="If BNN hidden units")
    parser.add_argument("--B_kls", type=int, default=1000, help="Batch size for Monte Carlo estimation.")
    parser.add_argument("--n_iter", type=int, default=1000, help="Number of iterations")
    parser.add_argument("--nxp", type=int, default=1, help="Number of time to do the same xp")
    parser.add_argument("--n_values", nargs="+", type=int, default=[1, 10, 20], 
                        help="List of values for the number of mixture components (N_mixture).")
    parser.add_argument("--exp_name", type=str, default="", help="Name for the parent folder of the experiment.")
    parser.add_argument("--plot_iter", type=int, default=10000, help="When to plot")

    # parser.add_argument("--nf", action="store_true", help="Optim normalizing flow")
    parser.add_argument("--full", action="store_true", help="Optim full BW")
    parser.add_argument("--lin", action="store_true", help="Optim Lin et al")
    parser.add_argument("--mcmc", action="store_true", help="Optim mcmc")
    parser.add_argument("--advi", action="store_true", help="Optim advi ")

    return parser.parse_args()



def main(args):

    n_values = args.n_values
    exp_name = args.exp_name

 

    current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M")
    folder_name = os.path.join(exp_name, current_datetime)

   
    dataset_train , dataset_test = prepare_dataset(args.dataset_name, args.train_ratio)

    n_samples, d = dataset_train[0].shape

    # vgmm_sample_boule = 1 * np.sqrt(d) 
    vgmm_sample_boule = 10
    vgmm_scale_cov  =  10




    hyperparam = {
                  "d": d, 
                  "n_samples" : n_samples,
                  "lr" : args.lr,
                  "n_iter" : args.n_iter,
                  "B_kls" : args.B_kls,
                  "B_gradients" : args.B_gradients,
                  "target" :  args.target, 
                  "dataset": args.dataset_name,
                  "vgmm_sample_boule" : vgmm_sample_boule,
                  "vgmm_scale_cov" : vgmm_scale_cov,
                  "prior_eps" : args.prior_eps,
                  "hidden_units" : args.hidden_units,

    }




    target = Target(args.target, dataset_train = dataset_train, dataset_test = dataset_test,  prior_eps = args.prior_eps, hidden_units=args.hidden_units)


    if target.name in ["logreg", "mlogreg"]:
        hyperparam["n_classes"] = target.model.n_classes


    

    folder_name = os.path.join(folder_name)
    os.makedirs(folder_name, exist_ok=True)

    # Save the hyperparameters as a JSON file
    with open(os.path.join(folder_name , "hp.json"), "w") as outfile:
        json.dump(hyperparam, outfile, indent=4)

    if args.dataset_name == "synthetic":
        np.save( f"{folder_name}/pi_mean.npy", np.array(target.model.means))
        np.save( f"{folder_name}/pi_cov.npy", target.model.cov)



    vi = None
    # XPS = ["mirror", "optim_wo_eps", "ibw"]

    for N_mixture in n_values:  
            
                

       
        #### BASIC OPTIM  BW ISO 
        

        vi = VI_GMM(target, mode = "iso", learning_rate= args.lr, n_iterations= args.n_iter,
                    n_components = N_mixture, scale = vgmm_scale_cov, BG = args.B_gradients, 
                    BKL = args.B_kls, s = vgmm_sample_boule) 
        
        previous_init = vi.vgmm.means 
        


        os.makedirs(f"{folder_name}/N{N_mixture}", exist_ok=True)

        np.save( f"{folder_name}/N{N_mixture}/vgmm_mean.npy", vi.vgmm.means)
        np.save( f"{folder_name}/N{N_mixture}/vgmm_cov.npy", vi.vgmm.covariances)

        plot_iter = args.plot_iter
        vi.optimize(bw = True, md  = False, lin = False,  means_only=False, plot_iter=plot_iter, gen_noise=True, compute_kl=args.compute_kls) 
        folder_xp = os.path.join(folder_name, f"N{N_mixture}", "ibw")
        os.makedirs(folder_xp, exist_ok=True)

        vi.save(folder_xp)

        B = 1000
        component_indices = np.random.choice(vi.vgmm.n_components, size=B, p=vi.vgmm.weights)
        noise  = np.random.randn(B, vi.dim) 
        vi.evaluate(folder_xp=folder_xp, noise = noise, component_indices=component_indices)
   





        vi = VI_GMM(target, mode = "iso", learning_rate= args.lr, n_iterations= args.n_iter,
                    n_components = N_mixture, scale = vgmm_scale_cov, BG = args.B_gradients, 
                    BKL = args.B_kls, s = vgmm_sample_boule) 


        #### BASIC OPTIM  MD

        vi.optimize(bw = False, md  = True, lin = False,  means_only=False, plot_iter=plot_iter, gen_noise=True, compute_kl=args.compute_kls) 
        folder_xp = os.path.join(folder_name, f"N{N_mixture}", "md")
        os.makedirs(folder_xp, exist_ok=True)

        vi.save(folder_xp)
        vi.evaluate(folder_xp=folder_xp, noise = noise, component_indices=component_indices)



        #### FULL COV OPTIM
        if args.full: 
            vi = VI_GMM(target, mode = "full", learning_rate= args.lr, n_iterations= args.n_iter,
                        n_components = N_mixture, scale = vgmm_scale_cov, BG = args.B_gradients, 
                        BKL = args.B_kls, s = vgmm_sample_boule, means = previous_init) 


            #### BASIC OPTIM  BW FULL 
            vi.optimize(bw = True, md  = False, lin = False,  means_only=False, plot_iter=plot_iter, gen_noise=True, compute_kl=args.compute_kls) 
            folder_xp = os.path.join(folder_name, f"N{N_mixture}", "bw")
            os.makedirs(folder_xp, exist_ok=True)

            vi.save(folder_xp)
            vi.evaluate(folder_xp=folder_xp, noise = noise, component_indices=component_indices)


        #### LIN
        if args.lin: 
            vi = VI_GMM(target, mode = "iso", learning_rate= args.lr, n_iterations= args.n_iter,
                        n_components = N_mixture, scale = vgmm_scale_cov, BG = args.B_gradients, 
                        BKL = args.B_kls, s = vgmm_sample_boule, means = previous_init) 


            #### BASIC OPTIM  LIN
            vi.optimize(bw = False, md  = False, lin = True,  means_only=False, plot_iter=plot_iter, gen_noise=True, compute_kl=args.compute_kls) 
            folder_xp = os.path.join(folder_name, f"N{N_mixture}", "lin")
            os.makedirs(folder_xp, exist_ok=True)

            vi.save(folder_xp)  
            vi.evaluate(folder_xp=folder_xp, noise = noise, component_indices=component_indices)




    ### MCMC METHODS #### HMC, ADVI
    if args.mcmc:

        
        folder_name_baseline = os.path.join("baseline", target.name, args.dataset_name, "mcmc")
        os.makedirs(folder_name_baseline, exist_ok=True)
        print(f"{folder_name_baseline}/lll_train.npy")
        if not os.path.exists(f"{folder_name_baseline}/lll_train.npy"):
            



        
            
            
            from cmdstanpy import CmdStanModel
            
            N, d = dataset_train[0].shape

  

            if target.name == "logreg":
                stan_file = "logreg_file.stan"
                stan_data = {
                    'N': N,
                    'y': target.model.y,
                    'X': target.model.X, 
                    'P': d
                    }
            
            elif target.name == "mlogreg":
                stan_file = "mlogreg_file.stan"
                stan_data = {
                                'N': N,
                                'y': target.model.y+1,
                                'X': target.model.X, 
                                'P': d,
                                "C": target.model.n_classes
                                }

            elif target.name == "linreg":
                raise ValueError("Not done yet")

            model = CmdStanModel(stan_file=stan_file)


            hyperparam_mcmc = {"chains": 4, 
                            "parallel_chains": 4, 
                            "iter_warmup": 10000, 
                            "iter_sampling":1000,
                            "seed" : 123 ,
                            "inits" : 50
                            }


            with open(os.path.join(folder_name_baseline , "hp.json"), "w") as outfile:
                json.dump(hyperparam_mcmc, outfile, indent=4)

            start = time.time()
            fit_mcmc = model.sample(
                data=stan_data,
                chains= hyperparam_mcmc["chains"],
                parallel_chains=hyperparam_mcmc["parallel_chains"],
                iter_warmup=hyperparam_mcmc["iter_warmup"],
                iter_sampling=hyperparam_mcmc["iter_sampling"],
                seed = hyperparam_mcmc["seed"], 
                show_progress=False
                )
            
            mcmc_time = time.time() - start
            np.save(f"{folder_name_baseline}/time.npy", mcmc_time)

            beta = fit_mcmc.stan_variable('beta')     
            np.save(f"{folder_name_baseline}/beta.npy", beta)


            #### EVALUATION ON TEST

            if target.name == "logreg":
                logits = np.einsum("nd,bd->nb" , target.model.X_test, beta)
                probs = (1/(1 + np.exp(-logits)) )
                acc_mcmc = ((probs.mean(axis = -1) > 0.5)== target.model.y_test).mean()
                

            elif target.name == "mlogreg":
                logits = np.einsum("nd,bcd->nbc",target.model.X_test,  beta)
                probs = np.exp(logits)/(np.exp(logits).sum(axis = -1))[...,None]
                y_hat = probs.mean(axis = 1).argmax(axis = -1)
                acc_mcmc = (y_hat == target.model.y_test).mean()

            lll_mcmc = target.model.log_likelihood(rearrange(beta, "b k d -> b d k"), split = "test")
            np.save(f"{folder_name_baseline}/accuracy_test.npy", acc_mcmc)
            np.save(f"{folder_name_baseline}/lll_test.npy", lll_mcmc)


            #### EVALUATION ON TRAIN
            if target.name == "logreg":
                logits = np.einsum("nd,bd->nb" , target.model.X,beta)
                probs = (1/(1 + np.exp(-logits)) )
                acc_mcmc = ((probs.mean(axis = -1) > 0.5)== target.model.y).mean()

            elif target.name == "mlogreg":
                logits = np.einsum("nd,bcd->nbc",target.model.X,  beta)
                probs = np.exp(logits)/(np.exp(logits).sum(axis = -1))[...,None]
                y_hat = probs.mean(axis = 1).argmax(axis = -1)
                acc_mcmc = (y_hat == target.model.y).mean()

            np.save(f"{folder_name_baseline}/accuracy_train.npy", acc_mcmc)

            lll_mcmc = target.model.log_likelihood(rearrange(beta, "b k d -> b d k"), split = "train")
            np.save(f"{folder_name_baseline}/lll_train.npy", lll_mcmc)
                        

    if args.advi:

        folder_name_baseline = os.path.join("baseline", target.name, args.dataset_name, "advi")
        os.makedirs(folder_name_baseline, exist_ok=True)
        if not os.path.exists(f"{folder_name_baseline}/lll_train.npy"):
            
                        
            
            from cmdstanpy import CmdStanModel
            
            N, d = dataset_train[0].shape



            if target.name == "logreg":
                stan_file = "logreg_file.stan"
                stan_data = {
                        'N': N,
                        'y': target.model.y,
                        'X': target.model.X, 
                        'P': d
                        }
            
            elif target.name == "mlogreg":
                stan_file = "mlogreg_file.stan"
                stan_data = {
                            'N': N,
                            'y': target.model.y+1,
                            'X': target.model.X, 
                            'P': d,
                            "C": target.model.n_classes
                            }

            elif target.name == "linreg":
                raise ValueError("Not done yet")
            

            model = CmdStanModel(stan_file=stan_file)


            hyperparam_advi = {"iter": 20000, 
                            "grad_samples": 10, 
                            "elbo_samples": 100, 
                            "eval_elbo":1000,
                            "algorithm" : "meanfield",
                            "require_converged" : False,
                            "seed" : 12,
                            "inits" : 50
                            }


            with open(os.path.join(folder_name_baseline , "hp.json"), "w") as outfile:
                json.dump(hyperparam_advi, outfile, indent=4)

            start = time.time()
                    
            fit_advi = model.variational(
                data=stan_data,
                iter=hyperparam_advi["iter"],
                grad_samples=hyperparam_advi["grad_samples"],
                elbo_samples=hyperparam_advi["elbo_samples"],
                eval_elbo = hyperparam_advi["eval_elbo"],
                algorithm=hyperparam_advi["algorithm"],
                require_converged=hyperparam_advi["require_converged"], 
                seed = hyperparam_advi["seed"], 
                inits = hyperparam_advi["inits"]
            )
            
            advi_time = time.time() - start
            np.save(f"{folder_name_baseline}/time.npy", advi_time)

            beta = fit_advi.stan_variable(var = "beta", mean = False)     
            np.save(f"{folder_name_baseline}/beta.npy", beta)


            #### EVALUATION ON TEST

            if target.name == "logreg":
                logits = np.einsum("nd,bd->nb" , target.model.X_test,beta)
                probs = (1/(1 + np.exp(-logits)) )
                acc_advi = ((probs.mean(axis = -1) > 0.5)== target.model.y_test).mean()

            elif target.name == "mlogreg":
                logits = np.einsum("nd,bcd->nbc",target.model.X_test,  beta)
                probs = np.exp(logits)/(np.exp(logits).sum(axis = -1))[...,None]
                y_hat = probs.mean(axis = 1).argmax(axis = -1)
                acc_advi = (y_hat == target.model.y_test).mean()

            np.save(f"{folder_name_baseline}/accuracy_test.npy", acc_advi)

            lll_advi = target.model.log_likelihood(rearrange(beta, "b k d -> b d k"), split = "test")
            np.save(f"{folder_name_baseline}/lll_test.npy", lll_advi)


            if target.name == "logreg":
                logits = np.einsum("nd,bd->nb" , target.model.X,beta)
                probs = (1/(1 + np.exp(-logits)) )
                acc_advi = ((probs.mean(axis = -1) > 0.5)== target.model.y).mean()

            elif target.name == "mlogreg":
                logits = np.einsum("nd,bcd->nbc",target.model.X,  beta)
                probs = np.exp(logits)/(np.exp(logits).sum(axis = -1))[...,None]
                y_hat = probs.mean(axis = 1).argmax(axis = -1)
                acc_advi = (y_hat == target.model.y).mean()

            np.save(f"{folder_name_baseline}/accuracy_train.npy", acc_advi)

            lll_advi = target.model.log_likelihood(rearrange(beta, "b k d -> b d k"), split = "train")
            np.save(f"{folder_name_baseline}/lll_train.npy", lll_advi)
                        



    print("---FOLDER---", folder_name)
  










if __name__ == "__main__":
    args = parse_args()
    main(args)








