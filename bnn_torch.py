import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal, Categorical
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import os

# Set random seed for reproducibility
torch.set_default_dtype(torch.float64)
torch.manual_seed(42)

# Load MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

download_mnist = True
if os.path.exists("./data/MNIST"):
    download_mnist = False
train_dataset = datasets.MNIST('./data', train=True, download=download_mnist, transform=transform)
test_dataset = datasets.MNIST('./data', train=False, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False)

# Define a Bayesian Layer with Sampled Mixture of Gaussians
class SampledMixtureLinear(nn.Module):
    def __init__(self, in_features, out_features, n_components=2, n_samples=5):
        super(SampledMixtureLinear, self).__init__()
        
        # Initialize mixture components parameters
        self.in_features = in_features
        self.out_features = out_features
        self.n_components = n_components
        self.n_samples = n_samples  # Number of components to sample during forward pass
        
        # Mixture weights (probabilities)
        self.mix_logits = nn.Parameter(torch.zeros(n_components))
        
        # Mean parameters for each Gaussian component
        self.weight_mu = nn.Parameter(torch.Tensor(n_components, out_features, in_features).normal_(0, 0.1))
        self.bias_mu = nn.Parameter(torch.Tensor(n_components, out_features).normal_(0, 0.1))
        
        # Log variance parameters for each Gaussian component
        self.weight_logvar = nn.Parameter(torch.Tensor(n_components, out_features, in_features).fill_(-5))
        self.bias_logvar = nn.Parameter(torch.Tensor(n_components, out_features).fill_(-5))
        
    def forward(self, x, sample=True):
        batch_size = x.size(0)
        
        # Get mixture weights
        mix_weights = F.softmax(self.mix_logits, dim=0)
        
        # Determine how many components to sample
        n_samples = min(self.n_samples, self.n_components)
        
        if self.training or sample:
            # Create categorical distribution for sampling components
            mixture_dist = Categorical(mix_weights)
            
            # Sample component indices based on their weights
            if self.n_components <= self.n_samples:
                # If we have fewer components than samples, use all components
                sampled_indices = torch.arange(self.n_components, device=x.device)
                sampled_weights = mix_weights
            else:
                # Sample components according to mixture weights
                sampled_indices = mixture_dist.sample((n_samples,))
                
                # To avoid duplicate samples, use unique indices
                sampled_indices = torch.unique(sampled_indices)
                
                # If we ended up with fewer unique samples, sample more
                while len(sampled_indices) < n_samples and len(sampled_indices) < self.n_components:
                    new_indices = mixture_dist.sample((n_samples - len(sampled_indices),))
                    sampled_indices = torch.unique(torch.cat([sampled_indices, new_indices]))
                
                # Get the weights of the sampled components
                sampled_weights = mix_weights[sampled_indices]
                
                # Normalize the sampled weights to sum to 1
                sampled_weights = sampled_weights / sampled_weights.sum()
            
            # Initialize output
            output = torch.zeros(batch_size, self.out_features, device=x.device)
            
            # Process sampled components
            for _ in range(5):
                for i, idx in enumerate(sampled_indices):
                        # Sample weights from the component
                        weight = self.weight_mu[idx] + torch.exp(0.5 * self.weight_logvar[idx]) * torch.randn_like(self.weight_mu[idx])
                        bias = self.bias_mu[idx] + torch.exp(0.5 * self.bias_logvar[idx]) * torch.randn_like(self.bias_mu[idx])
                        
                        # Compute output for this component
                        component_output = F.linear(x, weight, bias)
                        
                        # Add to total output, weighted by this component's weight
                        output += sampled_weights[i] * component_output
            output /= 5
        else:
            # During evaluation without sampling, use expected values weighted by mixture weights
            output = torch.zeros(batch_size, self.out_features, device=x.device)
            for _ in range(5):
                for k in range(self.n_components):
                    component_output = F.linear(x, self.weight_mu[k], self.bias_mu[k])
                    output += mix_weights[k] * component_output
            output /= 5
                
        return output
    
    def kl_divergence(self, prior_mu=0, prior_var=1, mc_samples=10):
        """
        Calculate KL divergence using Monte Carlo sampling from the mixture components.
        """
        mix_weights = F.softmax(self.mix_logits, dim=0)
        
        # Determine number of components to sample for KL calculation
        n_kl_samples = min(mc_samples, self.n_components)
        
        # Create categorical distribution for sampling components
        mixture_dist = Categorical(mix_weights)
        
        # Sample component indices for KL calculation
        if self.n_components <= n_kl_samples:
            # If we have fewer components than KL samples, use all components with their exact weights
            sampled_indices = torch.arange(self.n_components, device=self.mix_logits.device)
            sampled_weights = mix_weights
        else:
            # Sample components based on mixture weights
            sampled_indices = mixture_dist.sample((n_kl_samples,))
            sampled_counts = torch.bincount(sampled_indices, minlength=self.n_components)
            sampled_weights = sampled_counts.float() / n_kl_samples
        
        # Calculate KL only for sampled components
        kl_total = 0
        for idx in sampled_indices.unique():
            weight = mix_weights[idx]
            
            # KL for weights
            kl_weights = 0.5 * weight * torch.sum(
                self.weight_logvar[idx].exp() / prior_var +
                (self.weight_mu[idx] - prior_mu)**2 / prior_var -
                1 - self.weight_logvar[idx]
            )
            
            # KL for bias
            kl_bias = 0.5 * weight * torch.sum(
                self.bias_logvar[idx].exp() / prior_var +
                (self.bias_mu[idx] - prior_mu)**2 / prior_var -
                1 - self.bias_logvar[idx]
            )
            
            kl_total += kl_weights + kl_bias
        
        # Scale the KL by the ratio of total components to sampled components
        if self.n_components > n_kl_samples:
            kl_total = kl_total * (self.n_components / n_kl_samples)
            
        return kl_total

# Define Bayesian Neural Network
class SampledMixtureNN(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=256, output_dim=10, n_components=2, n_samples=5):
        super(SampledMixtureNN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_components = n_components
        self.layer1 = SampledMixtureLinear(input_dim, hidden_dim, n_components, n_samples)
        self.layer2 = SampledMixtureLinear(hidden_dim, output_dim, n_components, n_samples)
        
    def forward(self, x, sample=True):
        x = x.view(-1, 784)
        x = F.relu(self.layer1(x, sample))
        x = self.layer2(x, sample)
        return x
    
    def kl_divergence(self, mc_samples=10):
        return self.layer1.kl_divergence(mc_samples=mc_samples) + self.layer2.kl_divergence(mc_samples=mc_samples)
    
    def get_mixture_weights(self):
        layer1_weights = F.softmax(self.layer1.mix_logits, dim=0)
        layer2_weights = F.softmax(self.layer2.mix_logits, dim=0)
        return layer1_weights, layer2_weights

# Define a Bayesian Layer with Sampled Mixture of Isotropic Gaussians
class IsotropicSampledMixtureLinear(nn.Module):
    def __init__(self, in_features, out_features, n_components=2, n_samples=5):
        super(IsotropicSampledMixtureLinear, self).__init__()
        
        # Initialize mixture components parameters
        self.in_features = in_features
        self.out_features = out_features
        self.n_components = n_components
        self.n_samples = n_samples  # Number of components to sample during forward pass
        
        # Mixture weights (probabilities)
        self.mix_logits = nn.Parameter(torch.zeros(n_components))
        
        # Mean parameters for each Gaussian component
        self.weight_mu = nn.Parameter(torch.Tensor(n_components, out_features, in_features).normal_(0, 0.1))
        self.bias_mu = nn.Parameter(torch.Tensor(n_components, out_features).normal_(0, 0.1))
        
        # Single scalar log variance parameter for each component (isotropic)
        # One for weights and one for biases per component
        self.weight_logvar = nn.Parameter(torch.Tensor(n_components).fill_(-5))
        self.bias_logvar = nn.Parameter(torch.Tensor(n_components).fill_(-5))
        
    def forward(self, x, sample=True):
        batch_size = x.size(0)
        
        # Get mixture weights
        mix_weights = F.softmax(self.mix_logits, dim=0)
        
        # Determine how many components to sample
        n_samples = min(self.n_samples, self.n_components)
        
        if self.training or sample:
            # Create categorical distribution for sampling components
            mixture_dist = Categorical(mix_weights)
            
            # Sample component indices based on their weights
            if self.n_components <= self.n_samples:
                # If we have fewer components than samples, use all components
                sampled_indices = torch.arange(self.n_components, device=x.device)
                sampled_weights = mix_weights
            else:
                # Sample components according to mixture weights
                sampled_indices = mixture_dist.sample((n_samples,))
                
                # To avoid duplicate samples, use unique indices
                sampled_indices = torch.unique(sampled_indices)
                
                # If we ended up with fewer unique samples, sample more
                while len(sampled_indices) < n_samples and len(sampled_indices) < self.n_components:
                    new_indices = mixture_dist.sample((n_samples - len(sampled_indices),))
                    sampled_indices = torch.unique(torch.cat([sampled_indices, new_indices]))
                
                # Get the weights of the sampled components
                sampled_weights = mix_weights[sampled_indices]
                
                # Normalize the sampled weights to sum to 1
                sampled_weights = sampled_weights / sampled_weights.sum()
            
            # Initialize output
            output = torch.zeros(batch_size, self.out_features, device=x.device)
            
            # Process sampled components
            for i, idx in enumerate(sampled_indices):
                # For isotropic Gaussian, we use the same scalar variance for all weights
                # We need to broadcast the scalar variance to all weights
                weight_std = torch.exp(0.5 * self.weight_logvar[idx])
                bias_std = torch.exp(0.5 * self.bias_logvar[idx])
                
                # Sample weights using isotropic variance
                weight = self.weight_mu[idx] + weight_std * torch.randn_like(self.weight_mu[idx])
                bias = self.bias_mu[idx] + bias_std * torch.randn_like(self.bias_mu[idx])
                
                # Compute output for this component
                component_output = F.linear(x, weight, bias)
                
                # Add to total output, weighted by this component's weight
                output += sampled_weights[i] * component_output
        else:
            # During evaluation without sampling, use expected values weighted by mixture weights
            output = torch.zeros(batch_size, self.out_features, device=x.device)
            for k in range(self.n_components):
                component_output = F.linear(x, self.weight_mu[k], self.bias_mu[k])
                output += mix_weights[k] * component_output
                
        return output
    
    def kl_divergence(self, prior_mu=0, prior_var=1, mc_samples=10):
        """
        Calculate KL divergence for isotropic Gaussian mixture components.
        """
        mix_weights = F.softmax(self.mix_logits, dim=0)
        
        # Determine number of components to sample for KL calculation
        n_kl_samples = min(mc_samples, self.n_components)
        
        # Create categorical distribution for sampling components
        mixture_dist = Categorical(mix_weights)
        
        # Sample component indices for KL calculation
        if self.n_components <= n_kl_samples:
            # If we have fewer components than KL samples, use all components with their exact weights
            sampled_indices = torch.arange(self.n_components, device=self.mix_logits.device)
            sampled_weights = mix_weights
        else:
            # Sample components based on mixture weights
            sampled_indices = mixture_dist.sample((n_kl_samples,))
            sampled_counts = torch.bincount(sampled_indices, minlength=self.n_components)
            sampled_weights = sampled_counts.float() / n_kl_samples
        
        # Calculate KL only for sampled components
        kl_total = 0
        for idx in sampled_indices.unique():
            weight = mix_weights[idx]
            
            # KL for weights - note we're using a scalar variance for all weights
            # We need to account for the number of weight parameters
            n_weight_params = self.weight_mu[idx].numel()
            kl_weights = 0.5 * weight * (
                n_weight_params * (self.weight_logvar[idx].exp() / prior_var - 1 - self.weight_logvar[idx]) +
                torch.sum((self.weight_mu[idx] - prior_mu)**2) / prior_var
            )
            
            # KL for bias - note we're using a scalar variance for all biases
            # We need to account for the number of bias parameters
            n_bias_params = self.bias_mu[idx].numel()
            kl_bias = 0.5 * weight * (
                n_bias_params * (self.bias_logvar[idx].exp() / prior_var - 1 - self.bias_logvar[idx]) +
                torch.sum((self.bias_mu[idx] - prior_mu)**2) / prior_var
            )
            
            kl_total += kl_weights + kl_bias
        
        # Scale the KL by the ratio of total components to sampled components
        if self.n_components > n_kl_samples:
            kl_total = kl_total * (self.n_components / n_kl_samples)
            
        return kl_total

# Define Bayesian Neural Network with Isotropic Gaussian Mixtures
class IsotropicSampledMixtureNN(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=256, output_dim=10, n_components=2, n_samples=5):
        super(IsotropicSampledMixtureNN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_components = n_components
        self.layer1 = IsotropicSampledMixtureLinear(input_dim, hidden_dim, n_components, n_samples)
        self.layer2 = IsotropicSampledMixtureLinear(hidden_dim, output_dim, n_components, n_samples)
        
    def forward(self, x, sample=True):
        x = x.view(-1, self.input_dim)
        x = F.relu(self.layer1(x, sample))
        x = F.softmax(self.layer2(x, sample))
        return x
    
    def kl_divergence(self, mc_samples=10):
        return self.layer1.kl_divergence(mc_samples=mc_samples) + self.layer2.kl_divergence(mc_samples=mc_samples)
    
    def get_mixture_weights(self):
        layer1_weights = F.softmax(self.layer1.mix_logits, dim=0)
        layer2_weights = F.softmax(self.layer2.mix_logits, dim=0)
        return layer1_weights, layer2_weights

class SuperIsotropicSampledMixtureNN(nn.Module):
    def __init__(self, input_dim=784, hidden_dim1=256, hidden_dim2=128, output_dim=10, n_components=2, n_samples=5):
        super(SuperIsotropicSampledMixtureNN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_components = n_components
        self.layer1 = IsotropicSampledMixtureLinear(input_dim, hidden_dim1, n_components, n_samples)
        self.layer2 = IsotropicSampledMixtureLinear(hidden_dim1, hidden_dim2, n_components, n_samples)
        self.layer3 = IsotropicSampledMixtureLinear(hidden_dim2, output_dim, n_components, n_samples)
        
    def forward(self, x, sample=True):
        x = x.view(-1, self.input_dim)
        x = F.relu(self.layer1(x, sample))
        x = F.relu(self.layer2(x, sample))
        x = self.layer3(x, sample)
        return x
    
    def kl_divergence(self, mc_samples=10):
        return self.layer1.kl_divergence(mc_samples=mc_samples) + self.layer2.kl_divergence(mc_samples=mc_samples) + self.layer3.kl_divergence(mc_samples=mc_samples)
    
    def get_mixture_weights(self):
        layer1_weights = F.softmax(self.layer1.mix_logits, dim=0)
        layer2_weights = F.softmax(self.layer2.mix_logits, dim=0)
        layer3_weights = F.softmax(self.layer3.mix_logits, dim=0)
        return layer1_weights, layer2_weights, layer3_weights

# Initialize model and optimizer
n_components = 5  # Try with more components
n_samples = 5      # Sample only 3 components during forward pass

# model = SampledMixtureNN(n_components=n_components, n_samples=n_samples)
model = IsotropicSampledMixtureNN(n_components=n_components, n_samples=n_samples, hidden_dim=512)
# model = SuperIsotropicSampledMixtureNN(n_components=n_components, n_samples=n_samples, hidden_dim1=128, hidden_dim2=128)
lr = 1e-3

# ELBO loss function
def elbo_loss(output, target, kl_div, n_samples):
    # Negative log likelihood
    nll = F.cross_entropy(output, target, reduction='sum')
    # Scale KL divergence by dataset size
    kl_div = kl_div / n_samples
    # Return negative ELBO (we minimize this)
    return nll + kl_div

# Custom gradient descent for Bayesian Neural Networks with special logvar update
def bayesian_gradient_descent(model, learning_rate=0.001,
                              # mean_lr_factor=1.0, 
                              # mixture_lr_factor=1.0,
                              # weight_decay=1e-5,
                              max_norm=5.0,
                              # logvar_update_factor=0.1,
                              eps=1e-6,
                              method = "ibw"):
    """
    Custom gradient descent optimizer for Bayesian Neural Networks with a special update rule for logvar.
    
    Parameters:
    - model: The BNN model with mixture components
    - learning_rate: Base learning rate for all parameters
    - mean_lr_factor: Learning rate multiplier for mean parameters
    - mixture_lr_factor: Learning rate multiplier for mixture weights
    - weight_decay: L2 regularization strength
    - max_norm: Maximum gradient norm for clipping
    - logvar_update_factor: Scaling factor for the logvar update rule
    - eps: Small constant for numerical stability
    """
    # Separate parameters by type
    mean_params = []
    logvar_params = []
    mixture_params = []
    
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
            
        # Apply gradient clipping to avoid explosive gradients
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(param, max_norm)
            
        # Separate parameters based on their type
        if 'mu' in name:
            mean_params.append(param)
        elif 'logvar' in name:
            logvar_params.append(param)
        elif 'mix_logits' in name:
            mixture_params.append(param)
    
    # Update means using standard gradient descent
    with torch.no_grad():
        d = model.input_dim
        n = model.n_components
        assert len(mean_params) == len(logvar_params)
        for p, param in enumerate(mean_params):
            # if weight_decay > 0:
            #     param.grad.add_(weight_decay * param)
            if method == "lin":
                param.data.add_(param.grad, alpha=-n * learning_rate * logvar_params[p].data)
            else:
                # print(param.grad)
                param.data.add_(param.grad, alpha=-n * learning_rate)
                # print(param.data)
        
        # Update mixture weights using standard gradient descent
        # for param in mixture_params:
        #     param.data.add_(param.grad, alpha=-learning_rate * mixture_lr_factor)

        # Update logvars using variance gradients
        for param in logvar_params:
            # Convert logvar gradients to variance gradients
            # If we have logvar, then var = exp(logvar)
            # The gradient w.r.t variance is: dL/dvar = dL/dlogvar * dlogvar/dvar = dL/dlogvar * (1/var)
            
            # Current variance (from logvar)
            variance = torch.exp(param.data)
            
            # Convert logvar gradient to variance gradient
            # dL/dvar = dL/dlogvar * (1/var)
            var_grad = param.grad / variance
            
            # Apply your update rule in variance space
            if method == "ibw":
                # var = var + var_update_factor * var_grad^2
                var_update = (1.0 - (2.0 * n * learning_rate / d) * var_grad) ** 2
                new_variance = var_update * variance
            elif method == "md":
                var_update = torch.exp((-2.0 * n * learning_rate / d) * var_grad)
                new_variance = var_update * variance
            elif method == "lin":
                inv_new_variance = (1 / variance) + (2.0 * n * learning_rate * var_grad / d)
                new_variance = 1.0 / inv_new_variance
            else:
                raise NotImplementedError
            
            # Convert back to logvar
            new_logvar = torch.log(new_variance + eps)
            
            # Update the parameter (logvar)
            param.data.copy_(new_logvar)

# Training function
def train(model, train_loader, epoch):
    model.train()
    train_loss = 0
    n_samples = len(train_loader.dataset)
    
    for batch_idx, (data, target) in enumerate(train_loader):
        # optimizer.zero_grad()
        # Zero gradients from previous step
        for param in model.parameters():
            if param.grad is not None:
                param.grad.zero_()
        output = model(data)
        kl_div = model.kl_divergence(mc_samples=5)  # Use MC sampling for KL
        loss = elbo_loss(output, target, kl_div, n_samples)
        loss.backward()
        bayesian_gradient_descent(model, learning_rate=lr)
        # optimizer.step()
        train_loss += loss.item()
        
        if batch_idx % 10 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}'
                  f' ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
    
    print(f'====> Epoch: {epoch} Average loss: {train_loss / len(train_loader.dataset):.4f}')

# Evaluation function with uncertainty estimation
def test(model, test_loader, n_samples=10):
    model.eval()
    test_loss = 0
    correct = 0
    n_test = len(test_loader.dataset)
    uncertainties = []
    
    with torch.no_grad():
        for data, target in test_loader:
            # Get multiple predictions
            outputs = []
            for _ in range(n_samples):
                outputs.append(F.softmax(model(data, sample=True), dim=1))
            
            # Stack predictions
            outputs = torch.stack(outputs)
            
            # Mean prediction
            mean_output = outputs.mean(0)
            
            # Calculate predictive entropy (uncertainty)
            entropy = -torch.sum(mean_output * torch.log(mean_output + 1e-6), dim=1)
            uncertainties.extend(entropy.cpu().numpy())
            
            # Get predictions
            pred = mean_output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            
            # Calculate loss (use mean output for simplicity)
            loss = F.cross_entropy(mean_output.log(), target, reduction='sum')
            test_loss += loss.item()
    
    test_loss /= n_test
    print(f'Test set: Average loss: {test_loss:.4f}, '
          f'Accuracy: {correct}/{n_test} ({100. * correct / n_test:.2f}%)')
    
    return uncertainties

# Train the model
epochs = 10
l1_weights_history = []
l2_weights_history = []

for epoch in range(1, epochs + 1):
    # train(model, train_loader, optimizer, epoch)
    train(model, train_loader, epoch)
    uncertainties = test(model, test_loader)

# Plot how mixture weights change during training
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
for i in range(min(5, n_components)):  # Plot top 5 components
    plt.plot([w[i] for w in l1_weights_history], label=f'Component {i+1}')
plt.title('Layer 1 Top Component Weights')
plt.xlabel('Epoch')
plt.ylabel('Weight')
plt.legend()

plt.subplot(1, 2, 2)
for i in range(min(5, n_components)):  # Plot top 5 components
    plt.plot([w[i] for w in l2_weights_history], label=f'Component {i+1}')
plt.title('Layer 2 Top Component Weights')
plt.xlabel('Epoch')
plt.ylabel('Weight')
plt.legend()

plt.tight_layout()
plt.show()

# Plot component weight distribution for final model
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.bar(range(n_components), l1_weights_history[-1])
plt.title('Layer 1 Final Component Weights')
plt.xlabel('Component')
plt.ylabel('Weight')

plt.subplot(1, 2, 2)
plt.bar(range(n_components), l2_weights_history[-1])
plt.title('Layer 2 Final Component Weights')
plt.xlabel('Component')
plt.ylabel('Weight')

plt.tight_layout()
plt.show()
