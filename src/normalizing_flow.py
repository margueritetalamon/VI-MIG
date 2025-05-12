import torch 
import tqdm 



def kl_divergence(model, target , B):

    x = model.sample(B)
    log_prob = target.gaussians.log_prob(x[:,None])
    log_prob_c = torch.clip(log_prob, -700, 700)
    prob_c = torch.exp(log_prob_c)
    target_log_prob = (torch.as_tensor(target.weights[None]) * prob_c).sum(axis = -1).log()
    return (model.log_prob(x) - target_log_prob).mean()

def train(model, target,  n_epochs=100, lr=1e-3, B = 100):
    losses = []
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in tqdm.tqdm(range(n_epochs)):
        optimizer.zero_grad()
        loss = kl_divergence(model, target, B)

        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return losses
                    
