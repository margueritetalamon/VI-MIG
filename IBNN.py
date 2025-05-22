import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

METHOD_IBW = 0
METHOD_MD = 1
METHOD_LIN = 2
METHOD_GD = 3

# Define a Bayesian Layer with Sampled Mixture of Isotropic Gaussians
class IsotropicSampledMixtureLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, n_components: int = 2, n_samples: int = 5):
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
        
    def forward(self, x: torch.Tensor, sample: bool = True):
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
    
    def kl_divergence(self, prior_mu: int = 0, prior_var: int = 1, mc_samples: int = 10):
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

# Multi-Layer Bayesian Neural Network
class IGMMBayesianMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list, output_dim: int,
                 n_components: int = 2, n_samples: int = 5, dropout_rate: float = 0.0):
        """
        Multi-layer Bayesian Neural Network using IsotropicSampledMixtureLinear layers.

        Args:
            input_dim: Input dimension
            hidden_dims: List of hidden layer dimensions (e.g., [512, 256, 128])
            output_dim: Output dimension (number of classes)
            n_components: Number of mixture components per layer
            n_samples: Number of components to sample during forward pass
            dropout_rate: Dropout rate between layers (optional regularization)
        """
        super(IGMMBayesianMLP, self).__init__()

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.n_components = n_components
        self.n_samples = n_samples
        self.dropout_rate = dropout_rate

        # Build the network layers
        self.layers = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        # Create all dimensions list (input -> hidden layers -> output)
        all_dims = [input_dim] + hidden_dims + [output_dim]

        # Create Bayesian layers
        for i in range(len(all_dims) - 1):
            layer = IsotropicSampledMixtureLinear(
                in_features=all_dims[i],
                out_features=all_dims[i + 1],
                n_components=n_components,
                n_samples=n_samples
            )
            self.layers.append(layer)

            # Add dropout between hidden layers (not after output layer)
            if i < len(all_dims) - 2 and dropout_rate > 0:
                self.dropouts.append(nn.Dropout(dropout_rate))
            else:
                self.dropouts.append(nn.Identity())

        # Separate parameters based on their type
        self.mean_params = []
        self.logvar_params = []
        self.mixture_params = []
        for name, param in self.named_parameters():
            if 'mu' in name:
                self.mean_params.append(param)
            elif 'logvar' in name:
                self.logvar_params.append(param)
            elif 'mix_logits' in name:
                self.mixture_params.append(param)
        assert len(self.mean_params) == len(self.logvar_params)

    def forward(self, x: torch.Tensor, sample: bool = True):
        """
        Forward pass through the Bayesian MLP.

        Args:
            x: Input tensor
            sample: Whether to sample from the posterior (True) or use mean (False)
        """
        # Flatten input if needed (for image data)
        if len(x.shape) > 2:
            x = x.view(x.size(0), -1)

        # Forward pass through all layers
        for i, (layer, dropout) in enumerate(zip(self.layers, self.dropouts)):
            x = layer(x, sample=sample)

            # Apply activation function to all layers except the last (output) layer
            if i < len(self.layers) - 1:
                x = F.relu(x)
                x = dropout(x)

        # Final softmax layer
        x = F.softmax(x, dim=1)
        return x

    def kl_divergence(self, prior_mu: float = 0, prior_var: float = 1, mc_samples: int = 10):
        """
        Calculate total KL divergence across all layers.
        """
        total_kl = 0
        for layer in self.layers:
            total_kl += layer.kl_divergence(prior_mu, prior_var, mc_samples)
        return total_kl

    def predict_with_uncertainty(self, x: torch.Tensor, n_samples: int = 100):
        """
        Make predictions with uncertainty estimates using multiple forward passes.

        Args:
            x: Input tensor
            n_samples: Number of forward passes for uncertainty estimation

        Returns:
            mean_pred: Mean prediction across samples
            std_pred: Standard deviation across samples (epistemic uncertainty)
        """
        self.eval()
        predictions = []

        with torch.no_grad():
            for _ in range(n_samples):
                pred = self.forward(x, sample=True)
                predictions.append(pred)

        # Stack predictions and compute statistics
        predictions = torch.stack(predictions)  # (n_samples, batch_size, n_classes)

        mean_pred = predictions.mean(dim=0)
        std_pred = predictions.std(dim=0)

        return mean_pred, std_pred

    def get_model_info(self):
        """
        Get information about the model architecture.
        """
        total_params = sum(p.numel() for p in self.parameters())
        bayesian_params = sum(p.numel() for layer in self.layers for p in layer.parameters())

        info = {
            "architecture": [self.input_dim] + self.hidden_dims + [self.output_dim],
            "n_layers": len(self.layers),
            "n_components_per_layer": self.n_components,
            "n_samples_per_forward": self.n_samples,
            "total_parameters": total_params,
            "bayesian_parameters": bayesian_params,
            "dropout_rate": self.dropout_rate
        }
        return info

    def step(self,
             learning_rate: float = 0.001,
             max_norm: float = 5.0,
             eps: float = 1e-6,
             method: int = METHOD_IBW) -> None:
        """
        Custom gradient descent optimizer for Bayesian Neural Networks with a special update rule for logvar.

        Parameters:
        - model: The BNN model with mixture components
        - learning_rate: Base learning rate for all parameters
        - max_norm: Maximum gradient norm for clipping
        - eps: Small constant for numerical stability
        - method: Update method for logvar (METHOD_IBW, METHOD_MD, METHOD_LIN)
        """
        # Update means using standard gradient descent
        with torch.no_grad():
            d = self.input_dim
            n = self.n_components
            for p, param in enumerate(self.mean_params):
                if param.grad is None:
                    continue

                # Apply gradient clipping to avoid explosive gradients
                torch.nn.utils.clip_grad_norm_(param, max_norm)

                if method == METHOD_LIN:
                    mu = param.data
                    ek = self.logvar_params[p].data.unsqueeze(1)
                    if mu.ndim == 3:
                        ek = ek.unsqueeze(1)
                    new_mu = mu - n * learning_rate * torch.exp(ek) * param.grad
                    param.data.copy_(new_mu)
                else:
                    param.data.add_(param.grad, alpha=-n * learning_rate)

            # Update logvars using variance gradients
            for param in self.logvar_params:
                # Convert logvar gradients to variance gradients
                if method == METHOD_GD:
                    param.data.add_(param.grad, alpha=-n * learning_rate / d)
                else:
                    # If we have logvar, then var = exp(logvar)
                    # The gradient w.r.t variance is: dL/dvar = dL/dlogvar * dlogvar/dvar = dL/dlogvar * (1/var)

                    # Current variance (from logvar)
                    variance = torch.exp(param.data)

                    # Convert logvar gradient to variance gradient
                    # dL/dvar = dL/dlogvar * (1/var)
                    var_grad = param.grad / variance

                    # Apply your update rule in variance space
                    if method == METHOD_IBW:
                        # var = var + var_update_factor * var_grad^2
                        var_update = (1.0 - (2.0 * n * learning_rate / d) * var_grad) ** 2
                        new_variance = var_update * variance
                    elif method == METHOD_MD:
                        var_update = torch.exp((-2.0 * n * learning_rate / d) * var_grad)
                        new_variance = var_update * variance
                    elif method == METHOD_LIN:
                        inv_new_variance = (1 / variance) + (2.0 * n * learning_rate * var_grad / d)
                        new_variance = 1.0 / inv_new_variance
                    else:
                        # no update
                        new_variance = variance

                    # Convert back to logvar
                    new_logvar = torch.log(new_variance + eps)

                    # Update the parameter (logvar)
                    param.data.copy_(new_logvar)
