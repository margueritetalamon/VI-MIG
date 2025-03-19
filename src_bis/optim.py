from src_bis.gmm import GMM, IGMM
from src_bis.logreg import LogReg
import numpy as np
import tqdm
from matplotlib import pyplot as plt
import math 

class VI_IGMM:
    def __init__(self, target , n_iterations = 1000, learning_rate = 0.1, BKL = 1000, BG = 1,  **kwargs):
        
        self.target = target
        self.target_family = self.target.name
        self.vgmm = IGMM(**kwargs)
        self.dim = self.vgmm.dim
        


        self.n_iterations = n_iterations
        self.learning_rate = learning_rate
        self.kls = []
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
    
    def optimize(self, ibw = True, md = False, means_only = False, full = False, plot_iter = 1000, gen_noise = True, scheduler  = False, save_grads = False):


        initial_lr =  self.learning_rate
        learning_rate = initial_lr

        if not gen_noise :
            noise_grads = np.random.randn(self.BG, self.dim) 
        else:
            noise_grads = None

        noise_KL = np.random.randn(self.BKL, self.dim) 
        component_indices = np.random.choice(self.vgmm.n_components, size=self.BKL, p=self.vgmm.weights)


        for _ in tqdm.tqdm(range(self.n_iterations)):

            grad_means, grad_covs = self.vgmm.compute_grads_iso(self.target.model, noise_grads,  B = self.BG, optim_epsilon = not means_only)

            
            if save_grads :
                self.GM.append(grad_means)
                self.GE.append(grad_covs)

            if scheduler:

                learning_rate = self.lr_step_based_decay(_, initial_lr)
                print(learning_rate)

            new_means = self.vgmm.means - learning_rate * grad_means
            
            if ibw : 
                new_epsilons = (1 - (2/self.dim) * learning_rate * grad_covs)**2 * self.vgmm.epsilons 

            elif md :
                new_epsilons = self.vgmm.epsilons * np.exp(-learning_rate * grad_covs)

            elif means_only:
                new_epsilons = self.vgmm.epsilons
            
            elif full:
                raise ValueError("Optim not available yet.")

            else:
                raise ValueError("No optim performed.")



            self.vgmm.update(new_means, new_epsilons)

            self.kls.append(self.target.model.compute_KL(vgmm = self.vgmm, noise =  noise_KL, B = self.BKL, component_indices = component_indices))
            if _ % plot_iter == 0:
                print("LR" , learning_rate)
                print("KL ",self.kls[-1])

    
    def plot_target_and_circles(self,jump = 1000, bound = 20, grid_size = 100):

        ax = self.target.plot(bound = bound, grid_size=grid_size)
        self.vgmm.plot_evolution(0, ax, jump, bound=bound)

    

    def save(self, folder, file_name):
        if self.target_family == "gmm":
            np.save(f"{folder}/pi_means.npy", self.target.model.means)
            np.save(f"{folder}/pi_covs.npy", self.target.model.covariances)
        
        np.save(f"{folder}/optimized_means.npy", self.vgmm.optimized_means)
        np.save(f"{folder}/optimized_epsilons.npy", self.vgmm.optimized_epsilons)
        np.save(f"{folder}/kls.npy", self.kls)

        


   



            


                    


            


