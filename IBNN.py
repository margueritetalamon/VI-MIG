import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

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
        x = F.softmax(self.layer2(x, sample), dim=1)
        return x
    
    def kl_divergence(self, mc_samples=10):
        return self.layer1.kl_divergence(mc_samples=mc_samples) + self.layer2.kl_divergence(mc_samples=mc_samples)

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
        x = F.softmax(self.layer3(x, sample), dim=1)
        return x
    
    def kl_divergence(self, mc_samples=10):
        return self.layer1.kl_divergence(mc_samples=mc_samples) + self.layer2.kl_divergence(mc_samples=mc_samples) + self.layer3.kl_divergence(mc_samples=mc_samples)
