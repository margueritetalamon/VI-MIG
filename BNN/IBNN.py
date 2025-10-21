import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import math

METHOD_IBW = 0
METHOD_MD = 1
METHOD_LIN = 2
METHOD_GD = 3
METHOD_LAPLACE_DIAG = 4
METHOD_LAPLACE_KFAC = 5

# Define a Bayesian Layer with Sampled Mixture of Isotropic Gaussians
class IsotropicSampledMixtureLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, n_components: int=2,
                 n_samples: int=5, shared_logeps = None):
        super(IsotropicSampledMixtureLinear, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.n_components = n_components
        self.n_samples = n_samples  # Number of components to sample during forward pass       
        
        # Mixture weights (probabilities)
        self.weights = torch.ones(n_components)/n_components

        # Mean parameters for each Gaussian component
        self.weight_mu = nn.Parameter(torch.Tensor(n_components, out_features, in_features).normal_(0, 0.1))
        self.bias_mu = nn.Parameter(torch.Tensor(n_components, out_features).uniform_(0, 0.1))
        
        # Single scalar log variance parameter for each component (isotropic)
        # One for weights and one for biases per component
        if shared_logeps is not None:
            self.logeps = shared_logeps
        else:
            self.logeps = nn.Parameter(torch.Tensor(n_components).fill_(-5))
        
        
    def forward(self, x: torch.Tensor, sample: bool = True, sampled_indices = None):
       
        return 

# Multi-Layer Bayesian Neural Network
class IGMMBayesianMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list, output_dim: int,
                 n_components: int = 2, n_samples: int = 5, dropout_rate: float = 0.0,
                 mu_scale_init: float=1.0, prior_mean: float=0.0, prior_var: float=1.0):
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
        self.prior_var = prior_var
        self.prior_mean = prior_mean
        self.training = True
        self.kl_weight = None

        # Build the network layers
        self.layers = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        # Create all dimensions list (input -> hidden layers -> output)
        # we only have one hidden layers
        all_dims = [input_dim] + hidden_dims + [output_dim]
        self.overall_dim = (hidden_dims[0] * input_dim) + hidden_dims[0] + (output_dim*hidden_dims[0]) + output_dim
        print(all_dims)

        # Create Bayesian layers
        shared_logeps = None
        for i in range(len(all_dims) - 1):
            layer = IsotropicSampledMixtureLinear(
                in_features=all_dims[i],
                out_features=all_dims[i + 1],
                n_components=n_components,
                n_samples=n_samples,
                shared_logeps = shared_logeps
            )
            shared_logeps = layer.logeps ## each time it gives the eps of the previous layer which is what i want since eps is the same accross dim
            self.layers.append(layer)

            # Add dropout between hidden layers (not after output layer)
            if i < len(all_dims) - 2 and dropout_rate > 0:
                self.dropouts.append(nn.Dropout(dropout_rate))
            else:
                self.dropouts.append(nn.Identity())

        # Separate parameters based on their type
        self.mean_params = []
        self.logvar_params = []
        self.mixture_params = [] ### deleted now 
        for name, param in self.named_parameters():
            if 'mu' in name:
                # self.mean_params.append(param)
                self.mean_params.append(param)
            elif 'logeps' in name:
                # self.logvar_params.append(param)
                self.logvar_params.append(param)
        
        self.N_layers = len(self.layers)

   
            
    

        print("LOG VAR PARAM",len(self.logvar_params)) ### has to be equal to n_components
        print(len(self.mean_params)) ### has to be equal to n_components
        # assert len(self.mean_params) == len(self.logvar_params)

    def forward(self, x: torch.Tensor, sample: bool = True, S = 10):
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
                
                # Create component indices: [0,0,...,0, 1,1,...,1, ..., n_comp-1,...]
                # Shape: (S,) where each component appears samples_per_component times
                component_ids = torch.arange(self.n_components, device=x.device).repeat_interleave(samples_per_component)
                
                # Sample noise for all S samples
                noise_weight = torch.randn(S, d_out, d_in, device=x.device)
                noise_bias = torch.randn(S, d_out, device=x.device)
                
                # Get means and stds by component
                # weight_mu: (n_components, d_out, d_in) -> (S, d_out, d_in)
                weight_means = layer.weight_mu[component_ids]  # (S, d_out, d_in)
                bias_means = layer.bias_mu[component_ids]  # (S, d_out)
                
                # Broadcast stds: (n_components,) -> (S, d_out, d_in)
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
        
        # Gather the log probability of the true class
        # (S, Batch, n_classes) -> (S, Batch)
        nll_per_sample = F.nll_loss(log_probs.view(S * target.size(0), -1),
                                    labels_expanded.reshape(-1),
                                    reduction='none').view(S, -1)
        # Average over samples, sum over batchcan 
        expected_loss = nll_per_sample.mean(dim=0).sum()
        return expected_loss / target.size(0)




    # posterior distribution loss function
    ### can go to IBNN too
    def loss_function(self, output: torch.Tensor,
                target: torch.Tensor):
        
        nll = self.compute_expected_neg_loglikelihood(output, target)
        kl = self.compute_KL_vi_prior()
        return nll  + self.kl_weight * kl , nll, kl
            

    def compute_KL_vi_prior(self):
        kl_total = 0
        ### This is an upper bound on the true KL 

        for layer in self.layers:
            sigma_sq = torch.exp(layer.logeps)  # (n_components,) - note: no 0.5 factor
            
            # Weights: (n_components, d_out, d_in)
            n_w = layer.weight_mu[0].numel()
            kl_w = 0.5 * (
                sigma_sq * n_w + 
                (layer.weight_mu ** 2).view(self.n_components, -1).sum(dim=1) - 
                n_w - 
                n_w * torch.log(sigma_sq)
            ).sum() / self.n_components
            
            # Biases: (n_components, d_out)
            n_b = layer.bias_mu[0].numel()
            kl_b = 0.5 * (
                sigma_sq * n_b + 
                (layer.bias_mu ** 2).sum(dim=1) - 
                n_b - 
                n_b * torch.log(sigma_sq)
            ).sum() / self.n_components
            
            kl_total += kl_w + kl_b
    
        return kl_total


    # def compute_expected_log_prior(self):
    #     ### we assume that piror_mean = 0
    #     return None

    #     norm_means_summed = 0


    #     for layer in self.layers: 
    #         for w, b in zip(layer.weight_mu, layer.bias_mu):
    #             norm_means_summed+= (w**2).sum() + (b**2).sum()  ### \sum_i=1^N d*\eps^i + ||m^i||^2 = sum sur tout

    #     eps_sum = ((self.layers[0].logeps.exp()**2)).sum()*self.overall_dim

    #     return (norm_means_summed + eps_sum) / (2*self.prior_var) ### just need to add this term no put - 
    
    # def compute_negentropy(self, samples):

    #     means_flat = torch.cat([torch.cat([layer.weight_mu.reshape(self.n_components, -1),layer.bias_mu.reshape(self.n_components, -1) ], dim=1)for layer in self.layers ], dim=1)
    #     eps = torch.exp(0.5 * self.layers[0].logeps)**2

    #     diff = samples[:, None, :] - means_flat[None, :, :] 
    #     log_comp = -0.5 * self.overall_dim * torch.log(math.pi*2*eps) - 0.5 * (diff**2).sum(dim = -1)/eps
    #     neg_entropy = torch.logsumexp(log_comp - math.log(self.n_components), dim=1).mean()   

    #     return neg_entropy ### just need to add this term no - 



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
            d = self.overall_dim
            n = self.n_components
            for p, param in enumerate(self.mean_params):
                # print(f"{param.grad=}")
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
                    # print(f"{var_grad=}")
                    # print(f"{(2.0 * n * learning_rate / d) * var_grad=}")
                    # print(f"{variance=}")
                    new_variance = var_update * variance
                    # print(f"{new_variance=}")

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














############

















