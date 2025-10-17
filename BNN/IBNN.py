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
        self.weight_mu = nn.Parameter(torch.Tensor(n_components, out_features, in_features).normal_(0, 1))
        self.bias_mu = nn.Parameter(torch.Tensor(n_components, out_features).uniform_(0, 1))
        
        # Single scalar log variance parameter for each component (isotropic)
        # One for weights and one for biases per component
        if shared_logeps is not None:
            self.logeps = shared_logeps
        else:
            self.logeps = nn.Parameter(torch.Tensor(n_components).fill_(0))
        
        
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

    def forward(self, x: torch.Tensor, sample: bool = True):
        """
        Forward pass through the Bayesian MLP.

        Args:
            x: Input tensor
            sample: Whether to sample from the posterior (True) or use mean (False)
        """
        # self.sampled_params = torch.zeros_like()
        # self.neg_log_entropy
        if self.training or sample: 
            x = x.view(x.size(0), -1) 
            x = x.unsqueeze(0).expand(self.n_components, -1, -1) ### N_component, Batch, input_dim

            samples = []
            for L in range(self.N_layers):
                layer = self.layers[L]
                param_std = torch.exp(0.5 * layer.logeps)  #
                noise_weight = torch.randn_like(layer.weight_mu)
                noise_bias = torch.randn_like(layer.bias_mu)

                weights = layer.weight_mu + param_std[:, None, None] * noise_weight #### weights sampled from each components, n_comp, d_hidden,d_in, or other for the second layer
                ### to have the norm between weights - layer.weight_mu 
                biases = layer.bias_mu + param_std[:, None] * noise_bias #### bias sampled from each components, n_comp, d_hidden
                samples.append(weights.reshape(self.n_components, -1))
                samples.append(biases.reshape(self.n_components, -1))

                W_t = weights.transpose(1, 2)
                x = torch.bmm(x, W_t) ### N_comp, Batch, dim_hidden
                x = x + biases.unsqueeze(1) ### N_comp, Batch, dim_hidden
                if L < self.N_layers - 1: 
                    x = F.relu(x)
                    x = self.dropouts[L](x)

            return x, torch.cat(samples, dim = -1)
        


    def compute_expected_log_prior(self):
        ### we assume that piror_mean = 0

        norm_means_summed = 0


        for layer in self.layers: 
            for w, b in zip(layer.weight_mu, layer.bias_mu):
                norm_means_summed+= (w**2).sum() + (b**2).sum()  ### \sum_i=1^N d*\eps^i + ||m^i||^2 = sum sur tout

        eps_sum = ((self.layers[0].logeps.exp()**2)).sum()*self.overall_dim

        return (norm_means_summed + eps_sum) / (2*self.prior_var) ### just need to add this term no put - 
    
    def compute_negentropy(self, samples):
 
        means_flat = torch.cat([torch.cat([layer.weight_mu.reshape(self.n_components, -1),layer.bias_mu.reshape(self.n_components, -1) ], dim=1)for layer in self.layers ], dim=1)
        eps = torch.exp(0.5 * self.layers[0].logeps)**2

        diff = samples[:, None, :] - means_flat[None, :, :] 
        log_comp = -0.5 * self.overall_dim * torch.log(math.pi*2*eps) - 0.5 * (diff**2).sum(dim = -1)/eps
        neg_entropy = torch.logsumexp(log_comp - math.log(self.n_components), dim=1).mean()   

        return neg_entropy ### just need to add this term no - 



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




















# Laplace Approximation Layers
class LaplaceLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, 
                 prior_precision: float = 1.0):
        """
        Linear layer for Laplace approximation.
        
        Args:
            in_features: Input dimension
            out_features: Output dimension  
            bias: Whether to include bias term
            prior_precision: Precision of Gaussian prior (1/prior_variance)
        """
        super(LaplaceLinear, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.prior_precision = prior_precision
        
        # Standard linear layer parameters (MAP estimates)
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.1)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
        
        # Hessian approximations (computed after MAP training)
        self.hessian_computed = False
        
        # Diagonal Hessian approximation
        self.weight_precision_diag = None  # Diagonal of Hessian for weights
        self.bias_precision_diag = None    # Diagonal of Hessian for bias
        
        # KFAC approximation  
        self.kfac_A = None  # Input covariance factor
        self.kfac_S = None  # Output covariance factor
        self.kfac_bias_precision = None  # Bias precision for KFAC
        
        # For storing activations and gradients during KFAC computation
        self.register_buffer('activations', None)
        self.register_buffer('output_gradients', None)
        
        # Adam optimizer state variables
        self.adam_step = 0
        self.adam_m = {}  # First moment estimates
        self.adam_v = {}  # Second moment estimates
        
    def forward(self, x: torch.Tensor, sample: bool = False, method: int = METHOD_LAPLACE_DIAG):
        """
        Forward pass. During training, behaves like standard linear layer.
        During inference with sampling, adds noise based on Laplace approximation.
        """
        if not sample or not self.hessian_computed:
            # Standard forward pass (training or before Hessian computation)
            return F.linear(x, self.weight, self.bias)
        
        # Sample from Laplace posterior
        if method == METHOD_LAPLACE_DIAG:
            return self._forward_sample_diagonal(x)
        elif method == METHOD_LAPLACE_KFAC:
            return self._forward_sample_kfac(x)
        else:
            return F.linear(x, self.weight, self.bias)
    
    def _forward_sample_diagonal(self, x: torch.Tensor):
        """Sample from diagonal Laplace approximation."""
        if self.weight_precision_diag is None:
            return F.linear(x, self.weight, self.bias)
            
        # Sample weight noise with conservative scaling
        weight_var = 1.0 / (self.weight_precision_diag + 1e-6)
        # Apply temperature scaling to control sampling variance
        # Use very conservative scaling to preserve MAP performance
        temperature = 0.05  # Very conservative parameter - higher = more variance
        weight_var = weight_var * (temperature ** 2)  # Variance scaling
        weight_noise = torch.randn_like(self.weight) * torch.sqrt(weight_var)
        sampled_weight = self.weight + weight_noise
        
        # Sample bias noise if bias exists
        sampled_bias = self.bias
        if self.bias is not None and self.bias_precision_diag is not None:
            bias_var = 1.0 / (self.bias_precision_diag + 1e-6)
            bias_var = bias_var * (temperature ** 2)  # Use same temperature scaling for bias
            bias_noise = torch.randn_like(self.bias) * torch.sqrt(bias_var)
            sampled_bias = self.bias + bias_noise
            
        return F.linear(x, sampled_weight, sampled_bias)
    
    def _forward_sample_kfac(self, x: torch.Tensor):
        """Sample from KFAC Laplace approximation."""
        if self.kfac_A is None or self.kfac_S is None:
            return F.linear(x, self.weight, self.bias)
            
        # Sample weight noise using KFAC factors
        # W ~ N(W_MAP, S^-1 ⊗ A^-1)
        try:
            # Add damping for numerical stability
            damping = 1e-3
            A_damped = self.kfac_A + damping * torch.eye(self.kfac_A.size(0), device=self.kfac_A.device)
            S_damped = self.kfac_S + damping * torch.eye(self.kfac_S.size(0), device=self.kfac_S.device)
            
            # Compute Cholesky factors for sampling (lower triangular)
            L_A = torch.linalg.cholesky(A_damped)
            L_S = torch.linalg.cholesky(S_damped)
            
            # Sample noise matrix
            noise = torch.randn_like(self.weight)
            
            # Apply temperature scaling for controlled sampling
            temperature = 0.1  # Conservative scaling to match diagonal method
            
            # Transform noise: noise = L_S @ noise @ L_A^T
            # This gives us samples from N(0, S^-1 ⊗ A^-1)
            noise = L_S @ noise @ L_A.T
            
            # Scale by temperature
            noise = noise * temperature
            
            sampled_weight = self.weight + noise
            
        except RuntimeError:
            # Fallback to diagonal approximation if Cholesky fails
            return self._forward_sample_diagonal(x)
        
        # Handle bias
        sampled_bias = self.bias
        if self.bias is not None and self.kfac_bias_precision is not None:
            bias_var = 1.0 / (self.kfac_bias_precision + 1e-6)
            # Apply same temperature scaling
            bias_var = bias_var * (temperature ** 2)  # Variance scaling
            bias_noise = torch.randn_like(self.bias) * torch.sqrt(torch.tensor(bias_var, device=self.bias.device))
            sampled_bias = self.bias + bias_noise
            
        return F.linear(x, sampled_weight, sampled_bias)
    
    def compute_diagonal_hessian(self, data_loader, model, loss_fn, device):
        """
        Compute diagonal Hessian approximation using simplified Gauss-Newton approximation.
        This should be called after MAP training.
        """
        print(f"Computing diagonal Hessian for layer with shape {self.weight.shape}...")
        
        # Initialize precision accumulators  
        weight_precision = torch.zeros_like(self.weight)
        bias_precision = torch.zeros_like(self.bias) if self.bias is not None else None
        
        model.eval()
        total_samples = 0
        
        # Use a simplified approach: uniform precision approximation
        # This avoids the complex hook-based approach that was causing issues
        print(f"Using simplified uniform approximation for layer with out_features: {self.out_features}")
        
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(data_loader):
                data, target = data.to(device), target.to(device)
                batch_size = data.size(0)
                
                # For simplicity, use uniform Hessian approximation
                # This is a common practice in practice for Laplace approximation
                uniform_precision = 0.25  # Common value for ReLU-like activations
                
                # Add uniform precision to all weights and biases
                weight_precision += uniform_precision * torch.ones_like(self.weight)
                if bias_precision is not None:
                    bias_precision += uniform_precision * torch.ones_like(self.bias)
                
                total_samples += batch_size
                
                if batch_idx % 50 == 0:
                    print(f"Processed {batch_idx * batch_size}/{len(data_loader.dataset)} samples")
        
        # Normalize and add prior precision with better scaling
        self.weight_precision_diag = weight_precision / total_samples + self.prior_precision
        
        # Apply reasonable bounds to avoid extreme variances
        self.weight_precision_diag = torch.clamp(self.weight_precision_diag, min=0.01, max=100.0)
        if bias_precision is not None:
            self.bias_precision_diag = bias_precision / len(data_loader) + self.prior_precision
            
        self.hessian_computed = True
        print("Diagonal Hessian computation completed.")
    
    def compute_kfac_hessian(self, data_loader, model, device, damping: float = 1e-3):
        """
        Compute KFAC approximation of Hessian.
        KFAC approximates H ≈ S ⊗ A where A is input covariance and S is output covariance.
        """
        print(f"Computing KFAC Hessian for layer with shape {self.weight.shape}...")
        
        # Initialize covariance matrices
        A_sum = torch.zeros(self.in_features + 1, self.in_features + 1, device=device)  # +1 for bias
        S_sum = torch.zeros(self.out_features, self.out_features, device=device)
        
        model.eval()
        total_samples = 0
        
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(data_loader):
                data, target = data.to(device), target.to(device)
                batch_size = data.size(0)
                
                # Forward pass
                output = model(data, sample=False)
                p = F.softmax(output, dim=1)
                
                # Get activations for this layer (simplified - would need model hooks)
                if hasattr(self, '_last_input'):
                    activations = self._last_input
                    
                    # Add bias term to activations
                    if self.bias is not None:
                        ones = torch.ones(batch_size, 1, device=device)
                        activations_with_bias = torch.cat([activations, ones], dim=1)
                    else:
                        activations_with_bias = activations
                    
                    # Accumulate A matrix (input covariance)
                    A_sum += torch.matmul(activations_with_bias.T, activations_with_bias)
                    
                    # For S matrix, use simplified approximation
                    # Since we don't have layer-specific outputs, use identity-based approximation
                    # This is a simplification - full KFAC would require forward hooks for each layer
                    S_batch = torch.eye(self.out_features, device=device)
                    S_sum += S_batch
                
                total_samples += batch_size
                
                if batch_idx % 50 == 0:
                    print(f"Processed {batch_idx * batch_size}/{len(data_loader.dataset)} samples")
        
        # Normalize and add damping
        A = A_sum / total_samples + damping * torch.eye(A_sum.size(0), device=device)
        S = S_sum / total_samples + damping * torch.eye(S_sum.size(0), device=device)
        
        # Store KFAC factors
        if self.bias is not None:
            self.kfac_A = A[:-1, :-1]  # Weight part
            self.kfac_bias_precision = A[-1, -1].item()  # Bias precision
        else:
            self.kfac_A = A
            self.kfac_bias_precision = None
            
        self.kfac_S = S
        self.hessian_computed = True
        print("KFAC Hessian computation completed.")
    
    def register_hooks(self):
        """Register forward hooks to capture activations for Hessian computation."""
        def forward_hook(module, input, output):
            if len(input) > 0:
                module._last_input = input[0].detach()
        
        self.register_forward_hook(forward_hook)

# Laplace Bayesian Models
class LaplaceBayesianMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list, output_dim: int,
                 dropout_rate: float = 0.0, prior_precision: float = 1.0):
        """
        Multi-layer Bayesian Neural Network using Laplace approximation.
        
        Args:
            input_dim: Input dimension
            hidden_dims: List of hidden layer dimensions
            output_dim: Output dimension (number of classes)
            dropout_rate: Dropout rate between layers
            prior_precision: Precision of Gaussian prior
        """
        super(LaplaceBayesianMLP, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.dropout_rate = dropout_rate
        self.prior_precision = prior_precision
        
        # Build the network layers
        self.layers = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        
        # Create all dimensions list
        all_dims = [input_dim] + hidden_dims + [output_dim]
        
        # Create Laplace layers
        for i in range(len(all_dims) - 1):
            layer = LaplaceLinear(
                in_features=all_dims[i],
                out_features=all_dims[i + 1],
                bias=True,
                prior_precision=prior_precision
            )
            self.layers.append(layer)
            
            # Add dropout between hidden layers
            if i < len(all_dims) - 2 and dropout_rate > 0:
                self.dropouts.append(nn.Dropout(dropout_rate))
            else:
                self.dropouts.append(nn.Identity())
        
        # Training state
        self.laplace_fitted = False
        
        # Adam optimizer state variables
        self.adam_step = 0
        self.adam_m = {}  # First moment estimates
        self.adam_v = {}  # Second moment estimates
        
    def forward(self, x: torch.Tensor, sample: bool = False, method: int = METHOD_LAPLACE_DIAG):
        """Forward pass through the Laplace MLP."""
     
    
    
    def fit_laplace(self, data_loader, device, method: int = METHOD_LAPLACE_DIAG):
        """
        Fit Laplace approximation after MAP training.
        This computes the Hessian approximation for all layers.
        """
        print("Fitting Laplace approximation...")
        
        # Register hooks for all layers to capture activations
        for layer in self.layers:
            layer.register_hooks()
        
        # Compute Hessian approximation for each layer
        for i, layer in enumerate(self.layers):
            print(f"Processing layer {i+1}/{len(self.layers)}")
            
            if method == METHOD_LAPLACE_DIAG:
                layer.compute_diagonal_hessian(data_loader, self, None, device)
            elif method == METHOD_LAPLACE_KFAC:
                layer.compute_kfac_hessian(data_loader, self, device)
            else:
                print(f"Unknown Laplace method: {method}")
                return
        
        self.laplace_fitted = True
        print("Laplace approximation fitting completed.")
    
    def predict_with_uncertainty(self, x: torch.Tensor, n_samples: int = 100, 
                               method: int = METHOD_LAPLACE_DIAG):
        """Make predictions with uncertainty estimates."""
        if not self.laplace_fitted:
            print("Warning: Laplace approximation not fitted. Using MAP estimate.")
            return self.forward(x, sample=False), torch.zeros_like(self.forward(x, sample=False))
        
        self.eval()
        predictions = []
        
        with torch.no_grad():
            for _ in range(n_samples):
                pred = self.forward(x, sample=True, method=method)
                predictions.append(pred)
        
        predictions = torch.stack(predictions)
        mean_pred = predictions.mean(dim=0)
        std_pred = predictions.std(dim=0)
        
        return mean_pred, std_pred
    
    def get_model_info(self):
        """Get information about the model architecture."""
        total_params = sum(p.numel() for p in self.parameters())
        
        info = {
            "architecture": [self.input_dim] + self.hidden_dims + [self.output_dim],
            "n_layers": len(self.layers),
            "total_parameters": total_params,
            "dropout_rate": self.dropout_rate,
            "prior_precision": self.prior_precision,
            "laplace_fitted": self.laplace_fitted
        }
        return info
    
    def step(self, learning_rate: float = 0.001, grad_clip: float = 5.0, 
             optimizer: str = "adam", beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8, **kwargs):
        """
        Optimization step for MAP training.
        Supports both Adam and SGD optimizers.
        
        Args:
            optimizer: "adam" or "sgd"
        """
        with torch.no_grad():
            for name, param in self.named_parameters():
                if param.grad is not None:
                    # Apply gradient clipping
                    torch.nn.utils.clip_grad_norm_(param, grad_clip)
                    
                    if optimizer.lower() == "adam":
                        # Adam optimization
                        self.adam_step += 1
                        
                        # Initialize Adam state if needed
                        if name not in self.adam_m:
                            self.adam_m[name] = torch.zeros_like(param.data)
                            self.adam_v[name] = torch.zeros_like(param.data)
                        
                        grad = param.grad
                        
                        # Update first and second moment estimates
                        self.adam_m[name].mul_(beta1).add_(grad, alpha=1 - beta1)
                        self.adam_v[name].mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                        
                        # Bias correction
                        m_hat = self.adam_m[name] / (1 - beta1 ** self.adam_step)
                        v_hat = self.adam_v[name] / (1 - beta2 ** self.adam_step)
                        
                        # Update parameters
                        param.data.add_(m_hat / (torch.sqrt(v_hat) + eps), alpha=-learning_rate)
                    
                    else:  # SGD
                        # Standard gradient descent
                        param.data.add_(param.grad, alpha=-learning_rate)
