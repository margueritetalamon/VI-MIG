from src.gmm import GMM, IGMM, FGMM, STGMM
from src.logreg import LogReg

import numpy as np
import tqdm
from matplotlib import pyplot as plt
import math 
from scipy.stats import norm 
import time
import torch 
import torch.distributions as dist



class VI_GMM:
    def __init__(self, target , mode = "iso", n_iterations = 1000, learning_rate = 0.1, BKL = 1000, BG = 1, num_stab  = 0,  **kwargs):
        
        self.target = target

        self.dim = self.target.model.dim
        self.target_family = self.target.name
        self.mode = mode
        self.num_stab = num_stab


        if mode == "full":
            self.vgmm = FGMM(d = self.dim, **kwargs)

        elif  mode == "iso":
            self.vgmm = IGMM(d = self.dim, **kwargs)


        elif mode == "stan":
            self.vgmm = STGMM(**kwargs)

        self.dim = self.vgmm.dim
        

        self.vgmm.num_stab = self.num_stab
        self.target.model.num_stab = self.num_stab

        self.n_iterations = n_iterations
        self.learning_rate = learning_rate
        self.kls = []
        self.rmse_train = []
        self.rmse_test = []
        self.log_likelihood_train = []
        self.log_likelihood_test = []
        self.BKL = BKL
        self.BG = BG
        self.GM = []
        self.GE = []

        self.drop_rate = 0.8
        self.epochs_drop = 1000

    
    def lr_step_based_decay(self, epoch, initial_lr =  1):
        

        decay_factor = math.pow(self.drop_rate, math.floor(epoch / self.epochs_drop))
        new_learning_rate = initial_lr * decay_factor

        return new_learning_rate
    
    def lr_sqrt(self, epoch):
        new_learning_rate = 1 - np.sqrt( epoch / self.n_iterations)
        return new_learning_rate

    def lr_ln(self, epoch):
        new_learning_rate = 1 -  np.log(epoch+1)/np.log(self.n_iterations) 

        return new_learning_rate

    def lr_sqrt_ln(self, epoch):
        new_learning_rate = 1 - np.sqrt( np.log(epoch+1)/np.log(self.n_iterations) )

        return new_learning_rate
    
    def lr_linear(self, epoch):
        new_learning_rate = self.learning_rate /(epoch +1)

        return new_learning_rate
    
    def optimize(self, bw = True, md = False, lin = False,  means_only = False,  plot_iter = 1000, gen_noise = True, scheduler  = False, save_grads = False, compute_kl = 1000):


        initial_lr =  self.learning_rate
        learning_rate = initial_lr

        if not gen_noise :
            noise_grads = np.random.randn(self.BG, self.dim) 
        else:
            noise_grads = None

        noise_KL = np.random.randn(self.BKL, self.dim) 
        component_indices = np.random.choice(self.vgmm.n_components, size=self.BKL, p=self.vgmm.weights)


        start = time.time()
        for iter in tqdm.tqdm(range(self.n_iterations), leave = False):
            grad_means, grad_covs = self.vgmm.compute_grads(self.target.model, noise_grads,  B = self.BG, optim_epsilon = not means_only)

            
            if save_grads :
                self.GM.append(grad_means)
                self.GE.append(grad_covs)

            if scheduler:

                learning_rate = self.lr_linear(_)
                # print(learning_rate)

            new_means = self.vgmm.means - learning_rate * self.vgmm.n_components * grad_means

            
            if bw: 
                if self.mode == "iso":
                    new_epsilons = (1 - (2*self.vgmm.n_components*learning_rate/self.dim)  * grad_covs)**2 * self.vgmm.epsilons 
                

                elif self.mode == "full":

                    M = np.eye(self.dim) - 2*self.vgmm.n_components*learning_rate*grad_covs
                    new_epsilons = M * self.vgmm.covariances * M 
 
            elif md:

                new_epsilons = self.vgmm.epsilons * np.exp(-(2*self.vgmm.n_components*learning_rate/self.dim) * grad_covs )

            

            elif lin : 
                new_means = self.vgmm.means - self.vgmm.epsilons[:, None] * self.vgmm.n_components * learning_rate * grad_means

                inv_new_epsilons = (1/self.vgmm.epsilons) + (2 *  self.vgmm.n_components * learning_rate * grad_covs / self.dim)
                new_epsilons = (inv_new_epsilons)**(-1)

                    
            elif means_only:
                new_epsilons = None

            else:
                raise ValueError("No optim performed.")



            self.vgmm.update(new_means, new_epsilons)

            # self.kls.append(self.target.model.compute_KL(vgmm = self.vgmm, noise =  noise_KL, B = self.BKL, component_indices = component_indices))
            if iter % compute_kl == 0:
                self.kls.append(self.target.model.compute_KL(vgmm = self.vgmm, noise =  noise_KL, B = self.BKL, component_indices = component_indices))

            if iter % plot_iter == 0:
                print("LR" , learning_rate)
                print("KL ",self.kls[-1])
                B = 1000
                vi_samples = self.vgmm.sample(B)
                if self.target.name in ["linreg", "linreg_bnn"]:
                    X_ = torch.from_numpy(self.target.model.X)
                    y_ = torch.from_numpy(self.target.model.y)
                    res_train = np.zeros_like(self.target.model.y)
                    lll_train = 0
                    #
                    X_test_ = torch.from_numpy(self.target.model.X_test)
                    y_test_ = torch.from_numpy(self.target.model.y_test)
                    res_test = np.zeros_like(self.target.model.y_test)
                    lll_test = 0
                    #
                    sigma = self.target.model.sigma
                    for i in range(B):
                        theta = vi_samples[i]
                        theta_ = torch.from_numpy(theta)
                        self.target.model.neural_network.set_weights_from_vector(theta_)
                        with torch.no_grad():
                            y_hat_train_ = self.target.model.neural_network.forward(X_).squeeze()
                            diff_train_ = y_ - y_hat_train_
                            lll_train += ((-torch.dot(diff_train_, diff_train_) / (2 * sigma) - torch.log(torch.tensor(2 * np.pi * sigma))/2).sum()).numpy()
                            res_train += y_hat_train_.numpy()
                            #
                            y_hat_test_ = self.target.model.neural_network.forward(X_test_).squeeze()
                            diff_test_ = y_test_ - y_hat_test_
                            lll_test += ((-torch.dot(diff_test_, diff_test_) / (2 * sigma) - torch.log(torch.tensor(2 * np.pi * sigma))/2).sum()).numpy()
                            res_test += y_hat_test_.numpy()
                    res_train /= B
                    lll_train /= B
                    res_test /= B
                    lll_test /= B
                    #
                    diff_train = res_train - self.target.model.y
                    diff_test = res_test - self.target.model.y_test
                    #
                    rmse_train = np.sqrt(diff_train.dot(diff_train) / len(diff_train))
                    rmse_test = np.sqrt(diff_test.dot(diff_test) / len(diff_test))
                    #
                    self.rmse_train.append(rmse_train)
                    self.rmse_test.append(rmse_test)
                    print(f"RMSE train: ", rmse_train)
                    print(f"RMSE test: ", rmse_test)
                    #
                    self.log_likelihood_train.append(lll_train)
                    self.log_likelihood_test.append(lll_test)
                    print(f"Log likelihood train: ", lll_train)
                    print(f"Log likelihood test: ", lll_test)
                elif self.target.name in ["logreg", "mlogreg"]:
                    N, _ = self.target.model.X.shape
                    X_ = torch.from_numpy(self.target.model.X)
                    y_ = torch.from_numpy(self.target.model.y)
                    #
                    M_test = min(1000, self.target.model.X_test.shape[0])
                    indices = np.random.permutation(len(self.target.model.X_test))[:M_test]
                    X_test = self.target.model.X_test[indices]
                    y_test = self.target.model.y_test[indices]
                    N_test, _ = X_test.shape
                    assert N_test == M_test
                    X_test_ = torch.from_numpy(X_test)
                    y_test_ = torch.from_numpy(y_test)
                    #
                    K = self.target.model.n_classes
                    probs_train = np.zeros((B, N, K))
                    probs_test = np.zeros((B, N_test, K))
                    lll_train = 0
                    lll_test = 0
                    for i in range(B):
                        theta = vi_samples[i]
                        theta_ = torch.from_numpy(theta)
                        nn = self.target.model.neural_network
                        nn.set_weights_from_vector(theta_)
                        # forward
                        with torch.no_grad():
                            probs_train_ = nn.forward(X_) # logits come from softmax of nn
                            lll_train += (((y_ * torch.log(torch.clip(probs_train_, min=1e-3))).sum(dim = 1)).sum()).numpy()
                            probs_train[i] = probs_train_.numpy()
                            #
                            probs_test_ = nn.forward(X_test_)
                            lll_test += (((y_test_ * torch.log(torch.clip(probs_test_, min=1e-3))).sum(dim = 1)).sum()).numpy()
                            probs_test[i] = probs_test_.numpy()
                    lll_train /= B
                    lll_test /= B

                    y_hat_train = probs_train.mean(axis = 0).argmax(axis = -1) # mean over all samples B, then get class with argmax
                    assert y_hat_train.shape[0] == N
                    assert y_hat_train.size == N
                    #
                    y = self.target.model.y.argmax(axis = -1)
                    assert y.shape[0] == N
                    assert y.size == N
                    #
                    acc_train = (y_hat_train == y).mean()
                    self.rmse_train.append(acc_train)
                    self.log_likelihood_train.append(lll_train)
                    print(f"Mean train accuracy: ", acc_train)
                    ##
                    ##
                    y_hat_test = probs_test.mean(axis = 0).argmax(axis = -1) # mean over all samples B, then get class with argmax
                    assert y_hat_test.shape[0] == N_test
                    assert y_hat_test.size == N_test
                    #
                    y_test = y_test.argmax(axis = -1)
                    assert y_test.shape[0] == N_test
                    assert y_test.size == N_test
                    #
                    acc_test = (y_hat_test == y_test).mean()
                    self.rmse_test.append(acc_test)
                    self.log_likelihood_test.append(lll_test)
                    print(f"Mean test accuracy: ", acc_test)

            # Plot for linreg - for debug purposes
            '''
            if iter % plot_iter == 0 and self.target.name in ["linreg", "linreg_bnn"] and iter > 0:
                B = 10000
                vi_samples = self.vgmm.sample(B)
                X_ = torch.from_numpy(self.target.model.X)
                res = np.zeros_like(self.target.model.y)
                for i in range(B):
                    theta = vi_samples[i]
                    theta_ = torch.from_numpy(theta)
                    self.target.model.neural_network.set_weights_from_vector(theta_)
                    with torch.no_grad():
                        y_hat_ = self.target.model.neural_network.forward(X_).squeeze()
                        res += y_hat_.numpy()
                res /= B
                plt.clf()
                plt.plot(res,  label = "ibw")
                plt.plot(self.target.model.y, label = "real")
                plt.legend()
                plt.savefig("optim.pdf")
            '''

        self.time = time.time() - start




    
    def plot_target_and_circles(self,jump = 1000, bound = 20, grid_size = 100):

        ax = self.target.plot(bound = bound, grid_size=grid_size)
        self.vgmm.plot_evolution(0, ax, jump, bound=bound)

    

    def save(self, folder):
               
        np.save(f"{folder}/optimized_means.npy", self.vgmm.optimized_means)
        np.save(f"{folder}/optimized_epsilons.npy", self.vgmm.optimized_epsilons)
        if self.mode == "full":
            np.save(f"{folder}/optimized_covariances.npy", self.vgmm.optimized_covs)

        np.save(f"{folder}/kls.npy", self.kls)
        #
        np.save(f"{folder}/rmse_train.npy", self.rmse_train)
        np.save(f"{folder}/rmse_test.npy", self.rmse_test)
        #
        np.save(f"{folder}/log_likelihood_train.npy", self.log_likelihood_train)
        np.save(f"{folder}/log_likelihood_test.npy", self.log_likelihood_test)
        #
        np.save(f"{folder}/time.npy", self.time)

        


   

    def plot_estimated(self,  axes = None):
        

        if self.dim == 2:
            return 
        

        elif self.dim>2:

            if axes is None:

                bound = 60
                grid_size = 100
                x_grid = np.linspace(-bound, bound, grid_size)

                grid_rows = 3
                grid_cols = 4

                fig, axes = plt.subplots(4, self.dim//4, figsize=(5 * grid_cols, 4 * grid_rows))
                axes = axes.flatten()  # Flatten for easy indexing

                mu_final = self.vgmm.optimized_means[0]
                epsilon_final = self.vgmm.optimized_epsilons[0]
                pi_mean = self.target.model.means
                pi_cov = self.target.model.covariances



            for j in range(self.dim):
                estimated_marginals = [
                    norm.pdf(x_grid, loc=mu[j], scale=np.sqrt(epsilon))  # 1D Gaussian PDF
                    for mu, epsilon in zip(mu_final, epsilon_final)
                ]
                Z_estimated = np.sum(estimated_marginals, axis=0) / len(estimated_marginals)

                target_marginals = [
                    norm.pdf(x_grid, loc=pim[j], scale=np.sqrt(pic[j, j]))
                    for pim, pic in zip(pi_mean, pi_cov)
                ]
                Z_target = np.sum(target_marginals, axis=0) / len(target_marginals)

                axes[j].plot(x_grid, Z_estimated, label="MD", color="red", linewidth=3)
                axes[j].plot(x_grid, Z_target, label="target", color="blue", linestyle="--", linewidth=3)
                # axes[j].set_title(f"dim {j}", fontweight='bold')
                axes[j].set_xticks([-50,0, 50])
                axes[j].set_yticks([])

                
            plt.legend()

    
    def evaluate(self, folder_xp, B = 1000, noise = None, component_indices = None):

        if noise is None:
            noise = np.random.randn(B, self.dim) 
        
        if component_indices is None:
            component_indices = np.random.choice(self.vgmm.n_components, size=B, p=self.vgmm.weights)

        lll , acc  = [], []
        if self.target.name in ["logreg", "mlogreg"]:
            for split in ["train", "test"]:
                acc = self.target.model.evaluate_accuracy(self.vgmm,noise = noise, component_indices=component_indices, jump = 5, split = split )
                np.save(f"{folder_xp}/accuracy_{split}.npy", acc)
       
                lll = self.target.model.evaluate_lll(self.vgmm,noise = noise, component_indices=component_indices, jump = 5, split = split ).mean(axis = 1 )
                np.save(f"{folder_xp}/lll_{split}.npy", lll)


        elif self.target.name == "linreg":
            raise


    def recompute_KLS(self, jump = 1, B = 1000):
        kls = []

        noise = np.random.randn(B, self.dim) 
        component_indices = np.random.choice(self.vgmm.n_components, size=B, p=self.vgmm.weights)

        for t in tqdm.tqdm(range(0, self.n_iterations, jump)):
            samples = self.vgmm.sample(B, noise, component_indices, t = t)
            covariances = self.vgmm.optimized_epsilons[t][:,None,None] * np.eye(self.vgmm.dim)
            gaussians = dist.MultivariateNormal(torch.as_tensor(self.vgmm.optimized_means[t]), covariance_matrix=torch.as_tensor(covariances)) 
            prob = gaussians.log_prob(torch.as_tensor(samples[:,None])).exp().numpy()
            l_prob = np.log((self.vgmm.weights[None] * prob).sum(axis = -1))
            
            kls.append((l_prob - self.target.model.log_prob(samples)).mean())
        
        self.kls = kls

        


                            


                                    


                            


