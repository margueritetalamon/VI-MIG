import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import math

METHOD_IBW = 0
METHOD_MD = 1
METHOD_LIN = 2
METHOD_GD = 3

# Define a Bayesian Layer with Sampled Mixture of Isotropic Gaussians
class IsotropicSampledMixtureLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, n_components: int=2,
                 n_samples: int=5, mu_scale_init: float=1.0, var_init: float=1.0,
                 prior_mu: float=0.0, prior_var: float=1.0):
        super(IsotropicSampledMixtureLinear, self).__init__()

        # NLL = Negative log likelihood
        # Internally, the parameters z of the network follow a distrib q(z).
        # The forward of the network computes q(D|z) (the likelihood).
        # We minimize the Negative ELBO(z) = NELBO(z) = NLL(q(D|z)) + KL(q(z) || p(z)),
        # where p(z) is a **constant** prior on the parameters of the model.
        # The flatter the gaussian, the less we care about it.
        #
        # In our case, because we choose both normal distrib for q(z) and the prior p(z),
        # there is a closed form for the KL.
        
        # Initialize mixture components parameters
        self.in_features = in_features
        self.out_features = out_features
        self.n_components = n_components
        self.n_samples = n_samples  # Number of components to sample during forward pass
        # The prior is a gaussian
        self.prior_mu = prior_mu # mean of the prior
        self.prior_var = prior_var # mean of the variance
        
        # Mixture weights (probabilities)
        self.mix_logits = nn.Parameter(torch.zeros(n_components)) # will be passed through softmax -> 1/n_components
        
        # Mean parameters for each Gaussian component
        self.weight_mu = nn.Parameter(torch.Tensor(n_components, out_features, in_features).uniform(-mu_scale_init, mu_scale_init))
        self.bias_mu = nn.Parameter(torch.Tensor(n_components, out_features).normal_(-mu_scale_init, mu_scale_init))
        
        # Single scalar log variance parameter for each component (isotropic)
        # One for weights and one for biases per component
        self.weight_logvar = nn.Parameter(torch.Tensor(n_components).fill_(math.log(var_init)))
        self.bias_logvar = nn.Parameter(torch.Tensor(n_components).fill_(math.log(var_init)))
        
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
    
    def kl_divergence(self, mc_samples: int = 10):
        """
        Calculate KL divergence for isotropic Gaussian mixture components.
        If both the prior and the weights come from gaussians, there is a closed form for the KL(q(z) || p(z)).
        However, for a Mixture of Gaussians, it's not the case.
        So we need to use MC sampling to estimate it.
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
                n_weight_params * (self.weight_logvar[idx].exp() / self.prior_var - 1 - self.weight_logvar[idx]) +
                torch.sum((self.weight_mu[idx] - self.prior_mu)**2) / self.prior_var
            )
            
            # KL for bias - note we're using a scalar variance for all biases
            # We need to account for the number of bias parameters
            n_bias_params = self.bias_mu[idx].numel()
            kl_bias = 0.5 * weight * (
                n_bias_params * (self.bias_logvar[idx].exp() / self.prior_var - 1 - self.bias_logvar[idx]) +
                torch.sum((self.bias_mu[idx] - self.prior_mu)**2) / self.prior_var
            )
            
            kl_total += kl_weights + kl_bias
        
        # Scale the KL by the ratio of total components to sampled components
        if self.n_components > n_kl_samples:
            kl_total = kl_total * (self.n_components / n_kl_samples)
            
        return kl_total

# Multi-Layer Bayesian Neural Network
class IGMMBayesianMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list, output_dim: int,
                 n_components: int = 2, n_samples: int = 5, dropout_rate: float = 0.0,
                 mu_scale_init: float=1.0, var_init: float=1.0, prior_mu: float=0.0, prior_var: float=1.0):
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
                n_samples=n_samples,
                mu_scale_init=mu_scale_init,
                var_init=var_init,
                prior_mu=prior_mu,
                prior_var=prior_var,
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

    def kl_divergence(self, mc_samples: int = 10):
        """
        Calculate total KL divergence across all layers.
        """
        total_kl = 0
        for layer in self.layers:
            total_kl += layer.kl_divergence(mc_samples)
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
             grad_clip: float = 5.0,
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
                torch.nn.utils.clip_grad_norm_(param, grad_clip)

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

# Define a Bayesian Convolutional Layer with Sampled Mixture of Isotropic Gaussians
class IsotropicSampledMixtureConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, 
                 stride: int=1, padding: int=0, n_components: int=2, n_samples: int=5,
                 mu_scale_init: float=0.0, var_init: float=1.0,
                 prior_mu: float=0.0, prior_var: float=1.0):
        super(IsotropicSampledMixtureConv2d, self).__init__()
        
        # Initialize mixture components parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride
        self.padding = padding
        self.n_components = n_components
        self.n_samples = n_samples
        # The prior is a gaussian
        self.prior_mu = prior_mu # mean of the prior
        self.prior_var = prior_var # mean of the variance
        
        # Mixture weights (probabilities)
        self.mix_logits = nn.Parameter(torch.zeros(n_components))
        
        # Mean parameters for each Gaussian component
        self.weight_mu = nn.Parameter(torch.Tensor(n_components, out_channels, in_channels, *self.kernel_size).uniform(-mu_scale_init, mu_scale_init))
        self.bias_mu = nn.Parameter(torch.Tensor(n_components, out_channels).normal_(-mu_scale_init, mu_scale_init))
        
        # Single scalar log variance parameter for each component (isotropic)
        self.weight_logvar = nn.Parameter(torch.Tensor(n_components).fill_(math.log(var_init)))
        self.bias_logvar = nn.Parameter(torch.Tensor(n_components).fill_(math.log(var_init)))
        
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
                sampled_indices = torch.arange(self.n_components, device=x.device)
                sampled_weights = mix_weights
            else:
                sampled_indices = mixture_dist.sample((n_samples,))
                sampled_indices = torch.unique(sampled_indices)
                
                while len(sampled_indices) < n_samples and len(sampled_indices) < self.n_components:
                    new_indices = mixture_dist.sample((n_samples - len(sampled_indices),))
                    sampled_indices = torch.unique(torch.cat([sampled_indices, new_indices]))
                
                sampled_weights = mix_weights[sampled_indices]
                sampled_weights = sampled_weights / sampled_weights.sum()
            
            # Get output shape for convolution
            # We need to calculate the output dimensions
            output_height = (x.size(2) + 2 * self.padding - self.kernel_size[0]) // self.stride + 1
            output_width = (x.size(3) + 2 * self.padding - self.kernel_size[1]) // self.stride + 1
            output = torch.zeros(batch_size, self.out_channels, output_height, output_width, device=x.device)
            
            # Process sampled components
            for i, idx in enumerate(sampled_indices):
                # Sample weights and biases using isotropic variance
                weight_std = torch.exp(0.5 * self.weight_logvar[idx])
                bias_std = torch.exp(0.5 * self.bias_logvar[idx])
                
                weight = self.weight_mu[idx] + weight_std * torch.randn_like(self.weight_mu[idx])
                bias = self.bias_mu[idx] + bias_std * torch.randn_like(self.bias_mu[idx])
                
                # Compute convolution for this component
                component_output = F.conv2d(x, weight, bias, stride=self.stride, padding=self.padding)
                
                # Add to total output, weighted by this component's weight
                output += sampled_weights[i] * component_output
        else:
            # During evaluation without sampling, use expected values weighted by mixture weights
            output_height = (x.size(2) + 2 * self.padding - self.kernel_size[0]) // self.stride + 1
            output_width = (x.size(3) + 2 * self.padding - self.kernel_size[1]) // self.stride + 1
            output = torch.zeros(batch_size, self.out_channels, output_height, output_width, device=x.device)
            
            for k in range(self.n_components):
                component_output = F.conv2d(x, self.weight_mu[k], self.bias_mu[k], 
                                          stride=self.stride, padding=self.padding)
                output += mix_weights[k] * component_output
                
        return output
    
    def kl_divergence(self, mc_samples: int = 10):
        """Calculate KL divergence for isotropic Gaussian mixture components."""
        mix_weights = F.softmax(self.mix_logits, dim=0)
        
        n_kl_samples = min(mc_samples, self.n_components)
        mixture_dist = Categorical(mix_weights)
        
        if self.n_components <= n_kl_samples:
            sampled_indices = torch.arange(self.n_components, device=self.mix_logits.device)
            sampled_weights = mix_weights
        else:
            sampled_indices = mixture_dist.sample((n_kl_samples,))
            sampled_counts = torch.bincount(sampled_indices, minlength=self.n_components)
            sampled_weights = sampled_counts.float() / n_kl_samples
        
        kl_total = 0
        for idx in sampled_indices.unique():
            weight = mix_weights[idx]
            
            # KL for weights
            n_weight_params = self.weight_mu[idx].numel()
            kl_weights = 0.5 * weight * (
                n_weight_params * (self.weight_logvar[idx].exp() / self.prior_var - 1 - self.weight_logvar[idx]) +
                torch.sum((self.weight_mu[idx] - self.prior_mu)**2) / self.prior_var
            )
            
            # KL for bias
            n_bias_params = self.bias_mu[idx].numel()
            kl_bias = 0.5 * weight * (
                n_bias_params * (self.bias_logvar[idx].exp() / self.prior_var - 1 - self.bias_logvar[idx]) +
                torch.sum((self.bias_mu[idx] - self.prior_mu)**2) / self.prior_var
            )
            
            kl_total += kl_weights + kl_bias
        
        if self.n_components > n_kl_samples:
            kl_total = kl_total * (self.n_components / n_kl_samples)
            
        return kl_total

# Bayesian CNN Architecture
class IGMMBayesianCNN(nn.Module):
    def __init__(self, device, input_channels: int, input_width: int, input_height: int, output_dim: int,
                 conv_configs: list = None, fc_dims: list = None,
                 n_components: int = 2, n_samples: int = 5, dropout_rate: float = 0.0,
                 mu_scale_init: float=1.0, var_init: float=1.0, prior_mu: float=0.0, prior_var: float=1.0):
        """
        Bayesian CNN using IsotropicSampledMixture layers.

        Args:
            input_channels: Number of input channels (e.g., 3 for RGB images)
            output_dim: Number of output classes
            conv_configs: List of conv layer configs [(out_channels, kernel_size, stride, padding), ...]
            fc_dims: List of fully connected layer dimensions
            n_components: Number of mixture components per layer
            n_samples: Number of components to sample during forward pass
            dropout_rate: Dropout rate between layers
        """
        super(IGMMBayesianCNN, self).__init__()

        self.input_channels = input_channels
        self.input_height = input_height
        self.input_width = input_width
        self.output_dim = output_dim
        self.n_components = n_components
        self.n_samples = n_samples
        self.dropout_rate = dropout_rate

        # Default conv configurations if not provided
        if conv_configs is None:
            conv_configs = [
                (32, 3, 1, 1),   # 32 filters, 3x3 kernel, stride 1, padding 1
                (64, 3, 1, 1),   # 64 filters, 3x3 kernel, stride 1, padding 1
                (128, 3, 1, 1),  # 128 filters, 3x3 kernel, stride 1, padding 1
            ]
        
        if fc_dims is None:
            fc_dims = [256, 128]

        self.conv_configs = conv_configs
        self.fc_dims = fc_dims

        # Build convolutional layers
        self.conv_layers = nn.ModuleList()
        self.conv_dropouts = nn.ModuleList()
        self.pool_layers = nn.ModuleList()

        current_channels = input_channels
        for out_channels, kernel_size, stride, padding in conv_configs:
            # Bayesian conv layer
            conv_layer = IsotropicSampledMixtureConv2d(
                in_channels=current_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                n_components=n_components,
                n_samples=n_samples,
                mu_scale_init=mu_scale_init,
                var_init=var_init,
                prior_mu=prior_mu,
                prior_var=prior_var
            )
            self.conv_layers.append(conv_layer)
            
            # Max pooling layer
            self.pool_layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            
            # Dropout
            if dropout_rate > 0:
                self.conv_dropouts.append(nn.Dropout2d(dropout_rate))
            else:
                self.conv_dropouts.append(nn.Identity())
            
            current_channels = out_channels

        # Adaptive pooling to handle variable input sizes
        if device != torch.device("mps"): # adaptive pooling not working on macos
            self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
            # Calculate the flattened size after conv layers
            # This will be current_channels * 4 * 4 due to adaptive pooling
            flattened_size = current_channels * 4 * 4
        else:
            self.adaptive_pool = None
            flattened_size = self._calculate_flattened_size(input_channels, conv_configs, input_height, input_width)

        # Build fully connected layers
        self.fc_layers = nn.ModuleList()
        self.fc_dropouts = nn.ModuleList()

        # Create FC dimensions list
        all_fc_dims = [flattened_size] + fc_dims + [output_dim]

        for i in range(len(all_fc_dims) - 1):
            fc_layer = IsotropicSampledMixtureLinear(
                in_features=all_fc_dims[i],
                out_features=all_fc_dims[i + 1],
                n_components=n_components,
                n_samples=n_samples
            )
            self.fc_layers.append(fc_layer)

            # Add dropout between hidden layers (not after output layer)
            if i < len(all_fc_dims) - 2 and dropout_rate > 0:
                self.fc_dropouts.append(nn.Dropout(dropout_rate))
            else:
                self.fc_dropouts.append(nn.Identity())

        # Separate parameters based on their type for custom optimizer
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

    def _calculate_flattened_size(self, input_channels, conv_configs, input_height, input_width):
        """Calculate flattened size for any CNN architecture."""
        current_channels = input_channels
        height = input_height
        width = input_width
        
        for out_channels, kernel_size, stride, padding in conv_configs:
            # You might want to validate channel progression here
            if current_channels <= 0:
                raise ValueError(f"Invalid channel count: {current_channels}")
            
            current_channels = out_channels  # Update for next layer
            
            # Calculate spatial dimensions
            height = (height + 2 * padding - kernel_size) // stride + 1
            width = (width + 2 * padding - kernel_size) // stride + 1
            height = height // 2
            width = width // 2
        
        return current_channels * height * width

    def forward(self, x: torch.Tensor, sample: bool = True):
        """Forward pass through the Bayesian CNN."""
        # Convolutional layers
        for conv_layer, pool_layer, dropout in zip(self.conv_layers, self.pool_layers, self.conv_dropouts):
            x = conv_layer(x, sample=sample)
            x = F.relu(x)
            x = pool_layer(x)
            x = dropout(x)

        # Adaptive pooling and flatten
        if self.adaptive_pool is not None:
            x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)

        # Fully connected layers
        for i, (fc_layer, dropout) in enumerate(zip(self.fc_layers, self.fc_dropouts)):
            x = fc_layer(x, sample=sample)

            # Apply ReLU to all FC layers except the last (output) layer
            if i < len(self.fc_layers) - 1:
                x = F.relu(x)
                x = dropout(x)

        # Apply softmax for classification
        x = F.softmax(x, dim=1)
        return x

    def kl_divergence(self, mc_samples: int = 10):
        """Calculate total KL divergence across all layers."""
        total_kl = 0
        
        # KL from conv layers
        for layer in self.conv_layers:
            total_kl += layer.kl_divergence(mc_samples)
        
        # KL from FC layers
        for layer in self.fc_layers:
            total_kl += layer.kl_divergence(mc_samples)
        
        return total_kl

    def predict_with_uncertainty(self, x: torch.Tensor, n_samples: int = 100):
        """Make predictions with uncertainty estimates using multiple forward passes."""
        self.eval()
        predictions = []

        with torch.no_grad():
            for _ in range(n_samples):
                pred = self.forward(x, sample=True)
                predictions.append(pred)

        predictions = torch.stack(predictions)
        mean_pred = predictions.mean(dim=0)
        std_pred = predictions.std(dim=0)

        return mean_pred, std_pred

    def get_model_info(self):
        """Get information about the model architecture."""
        total_params = sum(p.numel() for p in self.parameters())
        
        conv_params = sum(p.numel() for layer in self.conv_layers for p in layer.parameters())
        fc_params = sum(p.numel() for layer in self.fc_layers for p in layer.parameters())

        info = {
            "input_channels": self.input_channels,
            "output_dim": self.output_dim,
            "conv_configs": self.conv_configs,
            "fc_dims": self.fc_dims,
            "n_components_per_layer": self.n_components,
            "n_samples_per_forward": self.n_samples,
            "total_parameters": total_params,
            "conv_parameters": conv_params,
            "fc_parameters": fc_params,
            "dropout_rate": self.dropout_rate,
            "n_conv_layers": len(self.conv_layers),
            "n_fc_layers": len(self.fc_layers)
        }
        return info

    def step(self, learning_rate: float = 0.001, grad_clip: float = 5.0,
             eps: float = 1e-6, method: int = METHOD_IBW) -> None:
        """Custom gradient descent optimizer with special update rule for logvar."""
        with torch.no_grad():
            # Calculate input dimension for normalization
            # For CNN, we'll use the total number of parameters as a proxy
            d = sum(p.numel() for p in self.parameters())
            n = self.n_components

            # Update means using standard gradient descent
            for p, param in enumerate(self.mean_params):
                if param.grad is None:
                    continue

                torch.nn.utils.clip_grad_norm_(param, grad_clip)

                if method == METHOD_LIN:
                    mu = param.data
                    ek = self.logvar_params[p].data
                    # Handle different tensor shapes
                    if mu.ndim > 1:
                        # Reshape ek to broadcast correctly
                        shape = [1] * mu.ndim
                        shape[0] = ek.size(0)
                        ek = ek.view(shape)
                    
                    new_mu = mu - n * learning_rate * torch.exp(ek) * param.grad
                    param.data.copy_(new_mu)
                else:
                    param.data.add_(param.grad, alpha=-n * learning_rate)

            # Update logvars using variance gradients
            for param in self.logvar_params:
                if param.grad is None:
                    continue
                    
                if method == METHOD_GD:
                    param.data.add_(param.grad, alpha=-n * learning_rate / d)
                else:
                    variance = torch.exp(param.data)
                    var_grad = param.grad / variance

                    if method == METHOD_IBW:
                        var_update = (1.0 - (2.0 * n * learning_rate / d) * var_grad) ** 2
                        new_variance = var_update * variance
                    elif method == METHOD_MD:
                        var_update = torch.exp((-2.0 * n * learning_rate / d) * var_grad)
                        new_variance = var_update * variance
                    elif method == METHOD_LIN:
                        inv_new_variance = (1 / variance) + (2.0 * n * learning_rate * var_grad / d)
                        new_variance = 1.0 / inv_new_variance
                    else:
                        new_variance = variance

                    new_logvar = torch.log(new_variance + eps)
                    param.data.copy_(new_logvar)
