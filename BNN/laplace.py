
import torch
import torch.nn as nn
import torch.nn.functional as F
import math 

METHOD_LAPLACE_DIAG = 4
METHOD_LAPLACE_KFAC = 5



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
        print(prior_precision)
        
        # Standard linear layer parameters (MAP estimates)
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.1)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
        
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
        self.num_stab = 1e-10
        
    def forward(self, x: torch.Tensor, sample: bool = False, method: int = METHOD_LAPLACE_DIAG, S = 10):
        """
        Forward pass. During training, behaves like standard linear layer.
        During inference with sampling, adds noise based on Laplace approximation.
        """

        if not sample or not self.hessian_computed:
            # Standard forward pass (training or before Hessian computation)
            return F.linear(x, self.weight, self.bias)
        
        # Sample from Laplace posterior
        if method == METHOD_LAPLACE_DIAG:
            return self._forward_sample_diagonal(x, S = S)
        
        elif method == METHOD_LAPLACE_KFAC:
            return self._forward_sample_kfac(x, S = S)
        
        else:
            return F.linear(x, self.weight, self.bias)
    
    def _forward_sample_diagonal(self, x: torch.Tensor, S:int = 10):
        """Sample from diagonal Laplace approximation."""

        if x.dim() == 2:  # [B, I]
            B, I = x.shape
            x = x.unsqueeze(0).expand(S, -1, -1) 

        S, B, I = x.shape

        O = self.out_features

        if self.weight_precision_diag is None:
            return F.linear(x.view(-1, I), self.weight, self.bias).view(S, B, O)
    
         
        weight_var = 1.0 / (self.weight_precision_diag + self.num_stab)
        weight_std = torch.sqrt(weight_var) 
        bias_var = 1.0 / (self.bias_precision_diag + self.num_stab)
        bias_std = torch.sqrt(bias_var)


        noise_weight = torch.randn(S, O, I, device=self.weight.device, dtype=self.weight.dtype)
        weights = self.weight.unsqueeze(0) + weight_std.unsqueeze(0) * noise_weight  # [S, O, I]

        noise_bias = torch.randn(S, O, device=self.bias.device, dtype=self.bias.dtype)
        biases = self.bias.unsqueeze(0) + bias_std.unsqueeze(0) * noise_bias
   
    
        output = torch.bmm(x, weights.transpose(1, 2))
        output = output + biases.unsqueeze(1)

       
            
        return output
    
    def _forward_sample_kfac(self, x: torch.Tensor, S: int = 5):
        # x: [S,B,I]

        if x.dim() == 2:  # [B, I]
            B, I = x.shape
            x = x.unsqueeze(0).expand(S, -1, -1) 

        S, B, I = x.shape
        O = self.out_features
        lam = self.prior_precision

        # eigh returns (eigenvalues, eigenvectors)
        a, UA = torch.linalg.eigh(self.kfac_A)     # a:[I], UA:[I,I]
        s, US = torch.linalg.eigh(self.kfac_S)     # s:[O], US:[O,O]

        # --- sample weights ---
        Z = torch.randn(S, O, I, device=self.weight.device, dtype=self.weight.dtype)   # [S,O,I]

        # Z' = U_S^T * Z * U_A   -> [S,O,I]
        Zp = torch.einsum('op,soi,iq->spq', US.transpose(-1, -2), Z, UA)  # [S,O,I]

        denom = s[:, None] * a[None, :] + lam                            # [O,I]
        Zs = Zp / torch.sqrt(denom)                                      # [S,O,I]

        # ΔW = U_S * Zs * U_A^T  -> [S,O,I]
        dW = torch.einsum('op,spq,qr->sor', US, Zs, UA.transpose(-1, -2))  # [S,O,I]
        W  = self.weight.unsqueeze(0) + dW                                  # [S,O,I]

        # --- sample bias: Cov(b) ≈ (S + lam I)^-1 ---
        Zb   = torch.randn(S, O, device=self.bias.device, dtype=self.bias.dtype)      # [S,O]
        Zb_p = Zb @ US                                                                # [S,O]
        Zb_s = Zb_p / torch.sqrt(s + lam)                                             # [S,O]
        b    = Zb_s @ US.transpose(-1, -2) + self.bias.unsqueeze(0)                   # [S,O]

        # --- forward: [S,B,I] @ [S,I,O] -> [S,B,O] ---
        y = torch.bmm(x, W.transpose(-1, -2)) + b.unsqueeze(1)                        # [S,B,O]
        return y



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
        self.kl_weight = None
        
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
        self.N_layers = len(self.layers)
        self.laplace_fitted = False
        
        # Adam optimizer state variables
        self.adam_step = 0
        self.adam_m = {}  # First moment estimates
        self.adam_v = {}  # Second moment estimates
        
    def forward(self, x: torch.Tensor, sample: bool = False, method: int = METHOD_LAPLACE_DIAG, S = 10):
        """Forward pass through the Laplace MLP."""

        if len(x.shape) > 2:
            x = x.view(x.size(0), -1)

        if not sample:
            for L in range(self.N_layers):
                x = self.layers[L](x, sample=sample, method=method)
                if L < self.N_layers - 1:
                    x = F.relu(x)
                    x = self.dropouts[L](x)
            
            return x 


        if sample: 
            x = x.unsqueeze(0).expand(S, -1, -1) ### n_sample, Batch, input_dim

            for L in range(self.N_layers):
                x = self.layers[L](x, sample=sample, method=method, S=S)
                if L < self.N_layers - 1: 
                    x = F.relu(x)
                    x = self.dropouts[L](x)
            return x  ## N_sample, B, out_dim

    def loss_function(self, outputs, targets):
       
        """
        U(z) = -log p(D|z) - log p(z)
        Likelihood:    Cross-Entropy on logits (no softmax in forward)
        Prior (Gaussian): (λ0/2) * ||z||^2   with λ0 = self.prior_precision

        Args:
            outputs: predicted by the model
            y: targets [B]
        """
        # print(outputs.shape)
        if outputs.dim() == 2:
            nll = F.cross_entropy(outputs, targets, reduction="sum")
        
        elif outputs.dim() == 3:
            S, B, C = outputs.shape
            # log p(y|x) ≈ log(1/S * sum_s p(y|x,w_s))
            log_probs = F.log_softmax(outputs, dim=-1)  # [S, B, C]
            
            # Gather correct class log probs: [S, B]
            target_idx = targets.view(1, B, 1).expand(S, B, 1)
            log_p_correct = torch.gather(log_probs, dim=2, index=target_idx).squeeze(-1)
            
            # Log-mean-exp over samples: log(mean_s p(y|w_s))
            log_mean_p = torch.logsumexp(log_p_correct, dim=0) - torch.log(torch.tensor(float(S)))
            
            # NLL = -sum_b log p(y_b|x_b)
            nll = -log_mean_p.sum()
        else: 
            raise ValueError("Problem")

        # Gaussian prior term: (λ0/2) * sum ||params||^2
        prior = 0.0
        for p in self.parameters():
            if p is not None:
                prior = prior + p.pow(2).sum()
        prior = 0.5 * self.prior_precision * prior

        return nll + self.kl_weight * prior , nll, prior
     
    
    
    def fit_laplace(self, data_loader, device, method: int = METHOD_LAPLACE_DIAG):
        """
        Fit Laplace approximation after MAP training.
        This computes the Hessian approximation for all layers.
        """
        print("Fitting Laplace approximation...") 

        self.training = False
            
        if method == METHOD_LAPLACE_DIAG:
            self.compute_diag_hessian(data_loader, device)
        elif method == METHOD_LAPLACE_KFAC:
                self.compute_KFAC_hessian(data_loader, device)
        else:
            print(f"Unknown Laplace method: {method}")
            return  
        self.laplace_fitted = True
        print("Laplace approximation fitting completed.")
      

    def compute_diag_hessian(self, data_loader, device):
        self.eval()
        stats = [{"a2_sum": None, "d2_sum": None, "N": 0} for _ in self.layers]
        hooks = []

        # Forward hook
        def fwd_hook(idx):
            def hook(mod, inp, out):
                mod._a = inp[0].detach()
            return hook

        # Backward hook
        def bwd_hook(idx):
            def hook(mod, grad_input, grad_output):
                mod._delta = grad_output[0].detach()
            return hook

        # Register hooks
        for i, layer in enumerate(self.layers):
            hooks.append(layer.register_forward_hook(fwd_hook(i)))
            hooks.append(layer.register_full_backward_hook(bwd_hook(i)))

        batch_count = 0

        for data, target in data_loader:
            data = data.to(device)
            target = target.to(device)

            self.zero_grad(set_to_none=True)
            logits  = self(data)
            loss = F.cross_entropy(logits, target, reduction='sum')
            loss.backward()

            B = data.size(0)

            # DEBUG: Only print first batch
            if batch_count == 0:
                for i, layer in enumerate(self.layers):
                    a = layer._a
                    d = layer._delta
                    print(f"\n=== Layer {i}, Batch {batch_count} ===")
                    print(f"  a shape: {a.shape}, range: [{a.min():.4f}, {a.max():.4f}]")
                    print(f"  δ shape: {d.shape}, range: [{d.min():.4f}, {d.max():.4f}]")
                    print(f"  a² sum: {(a**2).sum():.4e}")
                    print(f"  δ² sum: {(d**2).sum():.4e}")
                 

                print("\n=== Gradient Check ===")
                for name, param in self.named_parameters():
                    if param.grad is not None:
                        print(f"{name}: grad norm = {param.grad.norm():.4e}, grad mean = {param.grad.mean():.4e}")

            # Accumulate for ALL batches
            for i, layer in enumerate(self.layers):
                a = layer._a
                d = layer._delta
                
                a2 = (a ** 2).sum(dim=0)
                d2 = (d ** 2).sum(dim=0)

                if stats[i]["a2_sum"] is None:
                    stats[i]["a2_sum"] = a2
                    stats[i]["d2_sum"] = d2
                    stats[i]["N"] = B
                else:
                    stats[i]["a2_sum"] += a2
                    stats[i]["d2_sum"] += d2
                    stats[i]["N"] += B

            batch_count += 1
            # NO BREAK HERE!

        # AFTER loop - check accumulation
        print(f"\n=== After Processing All Batches ===")
        print(f"Total batches processed: {batch_count}")
        print(f"Total samples: {stats[0]['N']}")
        
        eps = 1e-12
        for i, layer in enumerate(self.layers):
            if stats[i]["N"] == 0:
                I = torch.full((layer.in_features,), eps, device=device, dtype=layer.weight.dtype)
                O = torch.full((layer.out_features,), eps, device=device, dtype=layer.weight.dtype)
            else:
                I = stats[i]["a2_sum"] / stats[i]["N"]
                O = stats[i]["d2_sum"] / stats[i]["N"]
                
                print(f"Layer {i}:")
                print(f"  I range: [{I.min():.4e}, {I.max():.4e}], mean: {I.mean():.4e}")
                print(f"  O range: [{O.min():.4e}, {O.max():.4e}], mean: {O.mean():.4e}")

            # Compute diagonal Hessian
            weight_diag = torch.outer(O, I)
            bias_diag = O.clone()
            
            print(f"  weight_diag range: [{weight_diag.min():.4e}, {weight_diag.max():.4e}], mean: {weight_diag.mean():.4e}")

            # Posterior precision
            layer.weight_precision_diag = weight_diag + layer.prior_precision
            if layer.bias is not None:
                layer.bias_precision_diag = bias_diag + layer.prior_precision

            layer.weight_precision_diag = torch.clamp(layer.weight_precision_diag, min=1e-12)
            if layer.bias is not None:
                layer.bias_precision_diag = torch.clamp(layer.bias_precision_diag, min=1e-12)

            layer.hessian_computed = True

        # Cleanup
        for h in hooks:
            h.remove()
        self._clear_tmp_io()

    
    def compute_KFAC_hessian(self, data_loader, device,
                         max_batches=None,
                         ema_decay: float = 0.95,
                         ridge: float = 1e-3):
        """
        Compute K-FAC factors A = E[a a^T], S = E[δ δ^T] for each LaplaceLinear layer.
        - Uses logits + CrossEntropyLoss (no softmax in forward).
        - Accumulates via EMA across batches.
        - Adds ridge damping to A and S.
        - Sets bias precision from diag(S) + prior.
        """
        self.eval()  # disable dropout/BN randomness; grads still flow
        hooks = []

        # We assume your self.layers contains only LaplaceLinear
        layers = self.layers

        # Initialize EMA slots on first call if needed
        for m in layers:
            I, O = m.in_features, m.out_features
            if m.kfac_A is None:
                m.kfac_A = torch.zeros(I, I, device=device, dtype=m.weight.dtype)
            if m.kfac_S is None:
                m.kfac_S = torch.zeros(O, O, device=device, dtype=m.weight.dtype)

        # Forward hook: capture activations a
        def fwd_hook(mod, inp, out):
            # inp[0]: [B, in_features]
            mod.activations = inp[0].detach()

        # Backward hook: capture pre-activation grads δ = dL/ds
        def bwd_hook(mod, grad_input, grad_output):
            # grad_output[0]: [B, out_features]
            mod.output_gradients = grad_output[0].detach()

        # Register hooks
        for m in layers:
            hooks.append(m.register_forward_hook(fwd_hook))
            hooks.append(m.register_full_backward_hook(bwd_hook))

        total_samples = 0

        for x, y in data_loader:
            x = x.to(device)
            y = y.to(device)
            B = x.size(0)

            # Forward + backward with summed CE to keep scaling consistent
            self.zero_grad(set_to_none=True)
            logits  = self(x)  # no softmax here
            loss = F.cross_entropy(logits, y, reduction='sum')
            loss.backward()

            # Per-layer accumulation
            for m in layers:
                a = m.activations          # [B, I]
                d = m.output_gradients     # [B, O]
                if a is None or d is None:  # safety
                    continue

                # second moments (not centered): [I,I], [O,O]
                m.kfac_A += a.T @ a        # [I, I]
                m.kfac_S += d.T @ d     
        
            total_samples += B
            

        for m in layers:
            I, O = m.in_features, m.out_features
            # Ridge damping for numerical stability
            m.kfac_A = m.kfac_A / total_samples
            m.kfac_S = m.kfac_S / total_samples

            m.kfac_A = m.kfac_A + ridge * torch.eye(I, device=device, dtype=m.kfac_A.dtype)
            m.kfac_S = m.kfac_S + ridge * torch.eye(O, device=device, dtype=m.kfac_S.dtype)


            bias_prec = torch.diag(m.kfac_S).clone() + m.prior_precision
            m.kfac_bias_precision = torch.clamp(bias_prec, min=1e-12)

            m.hessian_computed = True  # mark available

        # Remove hooks, clear temps
        for h in hooks:
            h.remove()
        for m in layers:
            m.activations = None
            m.output_gradients = None

    
    @torch.no_grad()
    def _clear_tmp_io(self):
        for m in self.layers:  # only LaplaceLinear modules in your model
            if hasattr(m, "_a"):     delattr(m, "_a")
            if hasattr(m, "_delta"): delattr(m, "_delta")
            if hasattr(m, "activations"):       m.activations = None
            if hasattr(m, "output_gradients"):  m.output_gradients = None


                
    
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

    def get_laplace_state(self):
        """
        Collect the MAP params + diagonal and/or K-FAC curvature you computed.
        Everything is detached and moved to CPU for saving.
        """
        state = {
            "prior_precision": float(self.prior_precision),
            "layers": []
        }
        for m in self.layers:
            d = {
                "weight": m.weight.detach().cpu(),
                "bias":   (m.bias.detach().cpu() if m.bias is not None else None),
                # diagonal Laplace
                "weight_precision_diag": (getattr(m, "weight_precision_diag", None).detach().cpu()
                                        if getattr(m, "weight_precision_diag", None) is not None else None),
                "bias_precision_diag":   (getattr(m, "bias_precision_diag", None).detach().cpu()
                                        if getattr(m, "bias_precision_diag", None) is not None else None),
                # K-FAC
                "kfac_A": (m.kfac_A.detach().cpu() if getattr(m, "kfac_A", None) is not None else None),
                "kfac_S": (m.kfac_S.detach().cpu() if getattr(m, "kfac_S", None) is not None else None),
            }
            state["layers"].append(d)
        return state

