import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import math

METHOD_IBW = 0
METHOD_MD = 1
METHOD_NGD = 2
METHOD_GD = 3
METHOD_LAPLACE_DIAG = 4
METHOD_LAPLACE_KFAC = 5

# Define a Bayesian Layer with Sampled Mixture of Isotropic Gaussians
class IsotropicSampledMixtureLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, n_components: int=2, shared_logeps = None):
        super(IsotropicSampledMixtureLinear, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.n_components = n_components        
        self.weights = torch.ones(n_components)/n_components ## uniform weights

        # Mean parameters for each Gaussian component
        
        self.weight_mu = nn.Parameter(torch.Tensor(n_components, out_features, in_features).normal_(0, 1))
        self.bias_mu = nn.Parameter(torch.Tensor(n_components, out_features).normal_(0, 1))
        
        # Single scalar log variance parameter for each component (isotropic)
        # One for weights and one for biases per component
        if shared_logeps is not None:
            self.logeps = shared_logeps
        else:
            self.logeps = nn.Parameter(torch.Tensor(n_components).fill_(-5))
        
        
    def forward(self):
        ### the forward is done in IGMMBayesianMLP
        return 

# Multi-Layer Bayesian Neural Network
class IGMMBayesianMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list, output_dim: int,
                 n_components: int = 2, dropout_rate: float = 0.0, prior_var: float=1.0):
        """
        Multi-layer Bayesian Neural Network using IsotropicSampledMixtureLinear layers.

        Args:
            input_dim: Input dimension
            hidden_dims: List of hidden layer dimensions (e.g., [512, 256, 128])
            output_dim: Output dimension (number of classes)
            n_components: Number of mixture components per layer
            dropout_rate: Dropout rate between layers (optional regularization)
        """
        super(IGMMBayesianMLP, self).__init__()

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.n_components = n_components
        self.dropout_rate = dropout_rate
        self.prior_var = prior_var
        self.training = True
        self.kl_weight = None

        # Build the network layers
        self.layers = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        # Create all dimensions list (input -> hidden layers -> output)
        # we only have one hidden layers
        all_dims = [input_dim] + hidden_dims + [output_dim]
        total_params = sum(all_dims[i]*all_dims[i+1] + all_dims[i+1] 
                   for i in range(len(all_dims)-1))
        self.overall_dim = total_params

        # Create Bayesian layers
        shared_logeps = None
        for i in range(len(all_dims) - 1):
            layer = IsotropicSampledMixtureLinear(
                in_features=all_dims[i],
                out_features=all_dims[i + 1],
                n_components=n_components,
                shared_logeps = shared_logeps
            )
            shared_logeps = layer.logeps ## each time it gives the eps of the previous layer which is what we want since eps is the same accross dim
            self.layers.append(layer)

            # Add dropout between hidden layers (not after output layer)
            if i < len(all_dims) - 2 and dropout_rate > 0:
                self.dropouts.append(nn.Dropout(dropout_rate))
            else:
                self.dropouts.append(nn.Identity())

        # Separate parameters based on their type
        self.mean_params = []
        self.logvar_params = []
        for name, param in self.named_parameters():
            if 'mu' in name:
                self.mean_params.append(param)
            elif 'logeps' in name:
                self.logvar_params.append(param)
        
        self.N_layers = len(self.layers)

        print("LOG VAR PARAM",len(self.logvar_params)) ### has to be equal to n_components
        print(len(self.mean_params)) ### has to be equal to n_components
        # assert len(self.mean_params) == len(self.logvar_params)

    def forward(self, x: torch.Tensor, sample: bool = True, S = 5):
        """
        Forward pass through the Bayesian MLP.

        Args:
            x: Input tensor
            sample: Whether to sample from the posterior (True) or use mean (False)
        """

        if self.training or sample:
            # S must be divisible by n_components
            assert S % self.n_components == 0
            samples_per_component = S // self.n_components
            
            x = x.view(x.size(0), -1)
            x = x.unsqueeze(0).expand(S, -1, -1)  # (S, Batch, input_dim)
            
            
            for L in range(self.N_layers):
                layer = self.layers[L]
                param_std = torch.exp(0.5 * layer.logeps)  # (n_components,)
                
                d_out, d_in = layer.weight_mu.shape[1:]
                component_ids = torch.arange(self.n_components, device=x.device).repeat_interleave(samples_per_component) 
                
                noise_weight = torch.randn(S, d_out, d_in, device=x.device)
                noise_bias = torch.randn(S, d_out, device=x.device)
                
                weight_means = layer.weight_mu[component_ids] # (S, d_out, d_in)
                bias_means = layer.bias_mu[component_ids]  # (S, d_out)
                
                weight_std = param_std[component_ids].unsqueeze(-1).unsqueeze(-1).expand(S, d_out, d_in)
                bias_std = param_std[component_ids].unsqueeze(-1).expand(S, d_out)
                
                # Sample weights and biases
                weights = weight_means + weight_std * noise_weight  # (S, d_out, d_in)
                biases = bias_means + bias_std * noise_bias  # (S, d_out)
                
                # Forward pass
                x = torch.bmm(x, weights.transpose(1, 2))  # (S, Batch, d_out)
                x = x + biases.unsqueeze(1)
                
                if L < self.N_layers - 1:
                    x = F.relu(x)
                    x = self.dropouts[L](x)
            
            return x
                

    def compute_expected_neg_loglikelihood(self, outputs, target):

        S = outputs.shape[0]  # outputs of shape S, B, C
        labels_expanded = target.unsqueeze(0).expand(S, -1)
        # Compute log probabilities
        log_probs = F.log_softmax(outputs, dim=2)  # (S, Batch, n_classes)
        nll_per_sample = F.nll_loss(log_probs.view(S * target.size(0), -1),
                                    labels_expanded.reshape(-1),
                                    reduction='none').view(S, -1)
        # Average over samples, sum over batchcan 
        expected_nll = nll_per_sample.mean(dim=0).sum()
        return expected_nll 

    def loss_function(self, output: torch.Tensor,
                target: torch.Tensor):
        nll = self.compute_expected_neg_loglikelihood(output, target)
        kl = self.compute_KL_vi_prior()
        return nll  + self.kl_weight * kl , nll, kl
            

    def compute_KL_vi_prior(self):
        kl_total = 0
        ### This is an upper bound on the true KL 
        prior_sigma_sq = self.prior_var
        for layer in self.layers:
            sigma_sq = torch.exp(layer.logeps)  # (n_components,) 

            # Weights: (n_components, d_out, d_in)
            n_w = layer.weight_mu[0].numel()
            kl_w = 0.5 * (
                sigma_sq/prior_sigma_sq  * n_w + 
                (layer.weight_mu ** 2).view(self.n_components, -1).sum(dim=1)/prior_sigma_sq - 
                n_w - 
                n_w * torch.log(sigma_sq/prior_sigma_sq)
            ).sum() / self.n_components
            
            # Biases: (n_components, d_out)
            n_b = layer.bias_mu[0].numel()
            kl_b = 0.5 * (
                sigma_sq/prior_sigma_sq * n_b + 
                (layer.bias_mu ** 2).sum(dim=1)/prior_sigma_sq- 
                n_b - 
                n_b * torch.log(sigma_sq/prior_sigma_sq)
            ).sum() / self.n_components
            
            kl_total += kl_w + kl_b
    
        return kl_total

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
            "total_parameters": total_params,
            "bayesian_parameters": bayesian_params,
            "dropout_rate": self.dropout_rate
        }
        return info

    def step(self,
             learning_rate: float = 0.001,
             grad_clip: float = 5.0,
             num_stab: float = 1e-6,
             method: int = METHOD_IBW) -> None:
        """
        Custom gradient descent optimizer for Bayesian Neural Networks with a special update rule for logvar.

        Parameters:
        - model: The BNN model with mixture components
        - learning_rate: Base learning rate for all parameters
        - max_norm: Maximum gradient norm for clipping
        - method: Update method for logvar (METHOD_IBW, METHOD_MD, METHOD_NGD)
        """
        # Update means using standard gradient descent
        with torch.no_grad():
            d = self.overall_dim
            n = self.n_components
            for p, param in enumerate(self.mean_params):
                if param.grad is None:
                    continue

                # Apply gradient clipping to avoid explosive gradients
                torch.nn.utils.clip_grad_norm_(param, grad_clip)

                if method == METHOD_NGD:
                    mu = param.data
                    ek = self.logvar_params[p].data.unsqueeze(1)
                    if mu.ndim == 3:
                        ek = ek.unsqueeze(1)
                    new_mu = mu - n * learning_rate * torch.exp(ek) * param.grad
                    param.data.copy_(new_mu)
                else:
                    param.data.add_(param.grad, alpha=-n * learning_rate)

            for param in self.logvar_params:
                # Convert logvar gradients to variance gradients
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
                elif method == METHOD_NGD:
                    inv_new_variance = (1 / variance) + (2.0 * n * learning_rate * var_grad / d)
                    new_variance = 1.0 / inv_new_variance
                elif method == METHOD_GD:
                    # no update on variance
                    new_variance = variance
                else:
                    raise ValueError("Not implemented method")

                # Convert back to logvar
                new_logvar = torch.log(new_variance + num_stab)

                # Update the parameter (logvar)
                param.data.copy_(new_logvar)
