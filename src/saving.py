import os 
import torch 

def save_results(folder_name, file_prefix, mu_optim, M, kls, epsilon_optim = None, E  = None):
    """ Save the results in the appropriate folder structure. """
    os.makedirs(folder_name, exist_ok=True)

    torch.save(mu_optim, f"{file_prefix}_mus.pt")
    torch.save(M, f"{file_prefix}_mus_evolution.pt")
    torch.save(kls, f"{file_prefix}_kls_evolution.pt")



    if not epsilon_optim is None:
        torch.save(epsilon_optim, f"{file_prefix}_epsilons.pt")
        torch.save(E, f"{file_prefix}_epsilons_evolution.pt")
    