# Variational Inference with Mixture of Isotropic Gaussians

This repository provides code for variational inference using mixtures of isotropic Gaussians. It implements optimization schemes for different approaches to estimating variational parameters.

## Repository Structure

- **`src/`**: Contains the main Python files for optimization, utilities, and saving results.
- **`optim_hd.py`**: Implements optimization for:
  - *Mirror Descent (MD)*
  - *Isotropic-Bures Wasserstein (IBW)*
  - Optimization of means only
- **`xp_dim.py`**: The main script to launch optimization experiments.
- **`target.py`**: Defines models (GMM, LogReg) as target distributions.
- **`utils_hd.py`**: Contains utilities, including gradients and KL divergence computation.
- **`saving.py`**: Handles saving optimization results.

## Running Experiments with `xp_dim.py`

`xp_dim.py` is the main script to run optimization experiments on Gaussian mixture models. It supports multiple optimization schemes and different configurations.

### Command Example:

```bash
python xp_dim.py --d 10 --lr_mu 0.1 --lr_eps 0.01 --n_iter 1000 --nxp 5 --n_values 1 10 50 100 --exp_name "experiment_1"
```


| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--d` | `int` | `10` | Dimensionality of the data (d). |
| `--lr_mu` | `float` | `0.1` | Learning rate for the optimization of `mu`. |
| `--lr_eps` | `float` | `0.1` | Learning rate for the optimization of `epsilon`. |
| `--B_gradients` | `int` | `100` | Batch size for Monte Carlo estimation used in optimization. |
| `--B_kls` | `int` | `100` | Batch size for Monte Carlo estimation used in KL computation. |
| `--n_iter` | `int` | `1000` | Number of iterations for the optimization process. |
| `--nxp` | `int` | `5` | Number of times to repeat the same experiment. |
| `--n_values` | `list of int` | `[1, 10, 50, 100]` | List of values for the number of mixture components (N_mixture). |
| `--exp_name` | `str` | `""` | Name for the parent folder where experiment results will be saved. |

