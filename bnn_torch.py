import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal, Categorical
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import os
import tap
import json
import datetime
from IBNN import IsotropicSampledMixtureNN

class MargArgs(tap.Tap):
    method: str = "ibw" # method: ibw, md, lin
    lr: float = 1e-3 # learning rate
    n_components: int = 5 # number of gaussians in MOG
    epochs: int = 10 # number of times we go through the dataset
    hidden_dim: int = 256
    save_interval: int = 1  # Save metrics every N epochs
    save_dir: str = "./results_mnist"  # Directory to save results

args = MargArgs().parse_args()

# Create directory to save results if it doesn't exist
os.makedirs(args.save_dir, exist_ok=True)

# Create a timestamp for this run
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
run_dir = os.path.join(args.save_dir, f"{args.method}_{timestamp}")
os.makedirs(run_dir, exist_ok=True)

# Save hyperparameters
hyperparams = {
    "method": args.method,
    "learning_rate": args.lr,
    "n_components": args.n_components,
    "epochs": args.epochs,
    "hidden_dim": args.hidden_dim,
    "save_interval": args.save_interval,
    "timestamp": timestamp,
    "pytorch_seed": 42
}

with open(os.path.join(run_dir, "hyperparameters.json"), "w") as f:
    json.dump(hyperparams, f, indent=4)

# Set random seed for reproducibility
torch.set_default_dtype(torch.float64)
torch.manual_seed(hyperparams["pytorch_seed"])

# Save the model configuration
model_config = {
    "input_dim": 28 * 28,  # MNIST image size
    "hidden_dim": args.hidden_dim,
    "output_dim": 10,      # MNIST has 10 classes
    "n_components": args.n_components,
    "n_samples": args.n_components
}

with open(os.path.join(run_dir, "model_config.json"), "w") as f:
    json.dump(model_config, f, indent=4)

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

# Initialize model and optimizer
n_components = args.n_components
n_samples = n_components

model = IsotropicSampledMixtureNN(n_components=n_components, n_samples=n_samples, hidden_dim=args.hidden_dim)
lr = args.lr

# Initialize metrics storage
metrics = {
    'train_accuracy': [],
    'train_kl_div': [],
    'train_nll': [],
    'train_elbo': [],
    'test_accuracy': [],
    'test_kl_div': [],
    'test_nll': [],
    'test_elbo': [],
    'epochs': []  # Store epoch numbers for plotting
}

# Function to evaluate model without training
def evaluate_model(model, data_loader, n_samples=5):
    """Evaluate model metrics without updating weights"""
    model.eval()
    total_loss = 0
    total_nll = 0
    total_kl_div = 0
    correct = 0
    n_data = len(data_loader.dataset)
    
    with torch.no_grad():
        for data, target in data_loader:
            # Get multiple predictions for robust evaluation
            outputs = []
            kl_divs = []
            
            for _ in range(n_samples):
                outputs.append(model(data, sample=True))
                kl_divs.append(model.kl_divergence(mc_samples=1))
            
            # Stack predictions and average
            outputs = torch.stack(outputs)
            mean_output = outputs.mean(0)
            kl_div = torch.stack(kl_divs).mean()
            
            # Calculate accuracy
            batch_accuracy, batch_correct = calculate_accuracy(mean_output, target)
            correct += batch_correct
            
            # Calculate loss components
            nll = F.cross_entropy(mean_output, target, reduction='sum')
            kl_div_scaled = kl_div / n_data
            loss = nll + kl_div_scaled
            
            # Accumulate metrics
            total_loss += loss.item()
            total_nll += nll.item()
            total_kl_div += kl_div.item()
    
    # Calculate average metrics
    avg_loss = total_loss / n_data
    avg_nll = total_nll / n_data
    avg_kl_div = total_kl_div / n_data
    accuracy = correct / n_data
    
    return avg_loss, avg_nll, avg_kl_div, accuracy

# Custom gradient descent for Bayesian Neural Networks with special logvar update
def bayesian_gradient_descent(model, learning_rate=0.001,
                              max_norm=5.0,
                              eps=1e-6,
                              method="ibw"):
    """
    Custom gradient descent optimizer for Bayesian Neural Networks with a special update rule for logvar.
    
    Parameters:
    - model: The BNN model with mixture components
    - learning_rate: Base learning rate for all parameters
    - max_norm: Maximum gradient norm for clipping
    - eps: Small constant for numerical stability
    - method: Update method for logvar ("ibw", "md", "lin", "gd")
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
            if method == "lin":
                mu = param.data
                ek = logvar_params[p].data.unsqueeze(1)
                if mu.ndim == 3:
                    ek = ek.unsqueeze(1)
                new_mu = mu - n * learning_rate * torch.exp(ek) * param.grad
                param.data.copy_(new_mu)
            else:
                param.data.add_(param.grad, alpha=-n * learning_rate)
        
        # Update logvars using variance gradients
        for param in logvar_params:
            # Convert logvar gradients to variance gradients
            if method == "gd":
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

# ELBO loss function
def elbo_loss(output, target, kl_div, n_samples):
    # Negative log likelihood
    nll = F.cross_entropy(output, target, reduction='sum')
    # Scale KL divergence by dataset size
    kl_div = kl_div / n_samples
    # Return negative ELBO (we minimize this)
    return nll + kl_div, nll, kl_div

# Calculate accuracy
def calculate_accuracy(output, target):
    pred = output.argmax(dim=1, keepdim=True)
    correct = pred.eq(target.view_as(pred)).sum().item()
    accuracy = correct / len(target)
    return accuracy, correct

# Training function
def train(model, train_loader, epoch, method):
    model.train()
    train_loss = 0
    train_nll_total = 0
    train_kl_div_total = 0
    n_samples = len(train_loader.dataset)
    correct = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        # Zero gradients from previous step
        for param in model.parameters():
            if param.grad is not None:
                param.grad.zero_()

        output = model(data)
        kl_div = model.kl_divergence(mc_samples=5)  # Use MC sampling for KL
        loss, nll, kl_div_scaled = elbo_loss(output, target, kl_div, n_samples)
        
        # Calculate accuracy
        batch_accuracy, batch_correct = calculate_accuracy(output, target)
        correct += batch_correct
        
        # Accumulate metrics
        train_loss += loss.item()
        train_nll_total += nll.item()
        train_kl_div_total += kl_div.item()
        
        loss.backward()
        bayesian_gradient_descent(model, learning_rate=lr, method=method)
        
        if batch_idx % 10 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}'
                  f' ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
    
    # Calculate average metrics
    avg_loss = train_loss / n_samples
    avg_nll = train_nll_total / n_samples
    avg_kl_div = train_kl_div_total / n_samples
    accuracy = correct / n_samples
    
    # Store metrics
    metrics['train_accuracy'].append(accuracy)
    metrics['train_kl_div'].append(avg_kl_div)
    metrics['train_nll'].append(avg_nll)
    metrics['train_elbo'].append(-(avg_nll + avg_kl_div))  # Negative loss is ELBO
    
    print(f'====> Epoch: {epoch} Average loss: {avg_loss:.4f}, '
          f'NLL: {avg_nll:.4f}, KL: {avg_kl_div:.4f}, '
          f'Accuracy: {accuracy:.4f}')
    
    return avg_loss, avg_nll, avg_kl_div, accuracy

# Evaluation function with uncertainty estimation
def test(model, test_loader, n_samples=10):
    model.eval()
    test_loss = 0
    test_nll_total = 0
    test_kl_div_total = 0
    correct = 0
    n_test = len(test_loader.dataset)
    uncertainties = []
    
    with torch.no_grad():
        for data, target in test_loader:
            # Get multiple predictions
            outputs = []
            kl_divs = []
            
            for _ in range(n_samples):
                outputs.append(model(data, sample=True))
                kl_divs.append(model.kl_divergence(mc_samples=1))
            
            # Stack predictions
            outputs = torch.stack(outputs)
            kl_div = torch.stack(kl_divs).mean()
            
            # Mean prediction
            mean_output = outputs.mean(0)
            
            # Calculate predictive entropy (uncertainty)
            entropy = -torch.sum(mean_output * torch.log(mean_output + 1e-6), dim=1)
            uncertainties.extend(entropy.cpu().numpy())
            
            # Get predictions and calculate accuracy
            batch_accuracy, batch_correct = calculate_accuracy(mean_output, target)
            correct += batch_correct
            
            # Calculate loss components
            nll = F.cross_entropy(mean_output, target, reduction='sum')
            kl_div_scaled = kl_div / n_test
            loss = nll + kl_div_scaled
            
            # Accumulate metrics
            test_loss += loss.item()
            test_nll_total += nll.item()
            test_kl_div_total += kl_div.item()
    
    # Calculate average metrics
    avg_loss = test_loss / n_test
    avg_nll = test_nll_total / n_test
    avg_kl_div = test_kl_div_total / n_test
    accuracy = correct / n_test
    
    # Store metrics
    metrics['test_accuracy'].append(accuracy)
    metrics['test_kl_div'].append(avg_kl_div)
    metrics['test_nll'].append(avg_nll)
    metrics['test_elbo'].append(-(avg_nll + avg_kl_div))  # Negative loss is ELBO
    
    print(f'Test set: Average loss: {avg_loss:.4f}, '
          f'NLL: {avg_nll:.4f}, KL: {avg_kl_div:.4f}, '
          f'Accuracy: {accuracy:.4f} ({correct}/{n_test})')
    
    return avg_loss, avg_nll, avg_kl_div, accuracy, uncertainties

# Save metrics function
def save_metrics(epoch):
    # Save metrics as numpy files
    for metric_name, values in metrics.items():
        np.save(os.path.join(run_dir, f"{metric_name}.npy"), np.array(values))
    
    # Also save the current state of all metrics in one file for convenience
    np.save(os.path.join(run_dir, f"metrics_epoch_{epoch}.npy"), metrics)
    
    # Save the latest metrics values to a JSON file for easy inspection
    latest_metrics = {metric: values[-1] for metric, values in metrics.items()}
    latest_metrics["epoch"] = epoch
    
    with open(os.path.join(run_dir, "latest_metrics.json"), "w") as f:
        json.dump(latest_metrics, f, indent=4)
    
    print(f"Metrics saved for epoch {epoch}")

# Function to save model checkpoints
def save_model_checkpoint(model, epoch):
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "hyperparams": hyperparams,
        "latest_metrics": {metric: values[-1] for metric, values in metrics.items()}
    }
    torch.save(checkpoint, os.path.join(run_dir, f"model_checkpoint_epoch_{epoch}.pt"))
    # Also save as latest checkpoint
    torch.save(checkpoint, os.path.join(run_dir, "model_latest.pt"))
    print(f"Model checkpoint saved for epoch {epoch}")

# Train the model
epochs = args.epochs

print(f"Starting training with hyperparameters: {hyperparams}")
print(f"Saving results to: {run_dir}")

# Evaluate initial model performance (epoch 0) before any training
print("Evaluating initial model performance (pre-training)...")
_, train_nll, train_kl_div, train_accuracy = evaluate_model(model, train_loader)
_, test_nll, test_kl_div, test_accuracy = evaluate_model(model, test_loader)

# Store initial metrics (epoch 0)
metrics['epochs'].append(0)
metrics['train_accuracy'].append(train_accuracy)
metrics['train_kl_div'].append(train_kl_div)
metrics['train_nll'].append(train_nll)
metrics['train_elbo'].append(-(train_nll + train_kl_div))
metrics['test_accuracy'].append(test_accuracy)
metrics['test_kl_div'].append(test_kl_div)
metrics['test_nll'].append(test_nll)
metrics['test_elbo'].append(-(test_nll + test_kl_div))

print(f"Initial metrics before training:")
print(f"  Train accuracy: {train_accuracy:.4f}, ELBO: {-(train_nll + train_kl_div):.4f}")
print(f"  Test accuracy: {test_accuracy:.4f}, ELBO: {-(test_nll + test_kl_div):.4f}")

# Save initial metrics
save_metrics(0)
save_model_checkpoint(model, 0)

# Start training loop
for epoch in range(1, epochs + 1):
    metrics['epochs'].append(epoch)
    train_metrics = train(model, train_loader, epoch, args.method)
    test_metrics = test(model, test_loader)
    
    # Save metrics every save_interval epochs and on the last epoch
    if epoch % args.save_interval == 0 or epoch == epochs:
        save_metrics(epoch)
        save_model_checkpoint(model, epoch)

# Final save of all metrics
print("Training completed. Saving final metrics...")
np.save(os.path.join(run_dir, "metrics_all.npy"), metrics)
print(f"All metrics saved to {run_dir}")

# Plot training and testing curves
plt.figure(figsize=(15, 10))

# Get epoch numbers for x-axis (including epoch 0)
epochs_list = metrics['epochs']

# Accuracy plot
plt.subplot(2, 2, 1)
plt.plot(epochs_list, metrics['train_accuracy'], label='Train')
plt.plot(epochs_list, metrics['test_accuracy'], label='Test')
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)

# ELBO plot
plt.subplot(2, 2, 2)
plt.plot(epochs_list, metrics['train_elbo'], label='Train')
plt.plot(epochs_list, metrics['test_elbo'], label='Test')
plt.title('ELBO')
plt.xlabel('Epoch')
plt.ylabel('ELBO')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)

# KL Divergence plot
plt.subplot(2, 2, 3)
plt.plot(epochs_list, metrics['train_kl_div'], label='Train')
plt.plot(epochs_list, metrics['test_kl_div'], label='Test')
plt.title('KL Divergence')
plt.xlabel('Epoch')
plt.ylabel('KL Divergence')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)

# Negative Log Likelihood plot
plt.subplot(2, 2, 4)
plt.plot(epochs_list, metrics['train_nll'], label='Train')
plt.plot(epochs_list, metrics['test_nll'], label='Test')
plt.title('Negative Log Likelihood')
plt.xlabel('Epoch')
plt.ylabel('NLL')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig(os.path.join(run_dir, "training_curves.png"))

# Save a summary text file with the training results
with open(os.path.join(run_dir, "training_summary.txt"), "w") as f:
    f.write(f"Training Summary for {args.method} Method\n")
    f.write("=" * 50 + "\n\n")
    f.write(f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    
    f.write("Hyperparameters:\n")
    for key, value in hyperparams.items():
        f.write(f"  {key}: {value}\n")
    f.write("\n")
    
    f.write("Initial Metrics (Pre-Training):\n")
    f.write(f"  train_accuracy: {metrics['train_accuracy'][0]:.6f}\n")
    f.write(f"  train_nll: {metrics['train_nll'][0]:.6f}\n")
    f.write(f"  train_kl_div: {metrics['train_kl_div'][0]:.6f}\n")
    f.write(f"  train_elbo: {metrics['train_elbo'][0]:.6f}\n")
    f.write(f"  test_accuracy: {metrics['test_accuracy'][0]:.6f}\n")
    f.write(f"  test_nll: {metrics['test_nll'][0]:.6f}\n")
    f.write(f"  test_kl_div: {metrics['test_kl_div'][0]:.6f}\n")
    f.write(f"  test_elbo: {metrics['test_elbo'][0]:.6f}\n\n")
    
    f.write("Final Metrics (After Training):\n")
    for metric, values in metrics.items():
        if metric != 'epochs' and len(values) > 1:
            f.write(f"  {metric}: {values[-1]:.6f}\n")
    f.write("\n")
    
    # Calculate improvements from initial to final
    f.write("Improvements (Final - Initial):\n")
    for metric in ['train_accuracy', 'test_accuracy', 'train_elbo', 'test_elbo']:
        if len(metrics[metric]) > 1:
            improvement = metrics[metric][-1] - metrics[metric][0]
            f.write(f"  {metric}: {improvement:.6f}\n")
    f.write("\n")
    
    # Find best metrics
    best_train_acc = max(metrics['train_accuracy'])
    best_train_acc_epoch = metrics['epochs'][metrics['train_accuracy'].index(best_train_acc)]
    
    best_test_acc = max(metrics['test_accuracy'])
    best_test_acc_epoch = metrics['epochs'][metrics['test_accuracy'].index(best_test_acc)]
    
    best_train_elbo = max(metrics['train_elbo'])
    best_train_elbo_epoch = metrics['epochs'][metrics['train_elbo'].index(best_train_elbo)]
    
    best_test_elbo = max(metrics['test_elbo'])
    best_test_elbo_epoch = metrics['epochs'][metrics['test_elbo'].index(best_test_elbo)]
    
    f.write("Best Metrics:\n")
    f.write(f"  Best Train Accuracy: {best_train_acc:.4f} (Epoch {best_train_acc_epoch})\n")
    f.write(f"  Best Test Accuracy: {best_test_acc:.4f} (Epoch {best_test_acc_epoch})\n")
    f.write(f"  Best Train ELBO: {best_train_elbo:.4f} (Epoch {best_train_elbo_epoch})\n")
    f.write(f"  Best Test ELBO: {best_test_elbo:.4f} (Epoch {best_test_elbo_epoch})\n")

print(f"Training summary saved to {os.path.join(run_dir, 'training_summary.txt')}")

