import os
import tap
import json
import datetime
import torch
import torch.nn.functional as F
import ast

from utils_bnn_torch import (
    get_device,
    load_dataset,
    save_and_plot_metrics,
    save_metrics,
    save_model_checkpoint,
    LearningRateScheduler)

from IBNN import (
    METHOD_IBW,
    METHOD_MD,
    METHOD_LIN,
    METHOD_GD,
    IGMMBayesianMLP,
    IGMMBayesianCNN)

def parse_tuple(s: str) -> tuple[int, int, int, int]:
    return ast.literal_eval(s)

class MargArgs(tap.Tap):
    dataset: str = "" # mnist, cifar10, cifar10_flat
    device: str = "cpu" # whether to use CPU or GPU (if available)
    seed: int = 42
    save_interval: int = 1  # Save metrics every N epochs
    save_dir: str = "./results"  # Directory to save results
    method: str = "ibw" # method: ibw, md, lin
    model: str = "mlp" # model: mlp, cnn
    bs: int = 128 # batch size
    lr: float = 1e-3 # learning rate
    lr_scheduler: str = "cosine" # cosine, cosine_restart, step, none
    lr_decay_factor: float = 0.1 # decay factor (step scheduler)
    lr_decay_epochs: list[int] = [100, 150] # epochs at which to apply the decay factor (step scheduler)
    lr_min: float = 1e-6 # minimum value for lr (cosine schedulers)
    lr_restart: int = 50 # for cosine restart
    epochs: int = 10 # number of times we go through the dataset
    n_components: int = 5 # number of gaussians in MOG
    fc_dims: list[int] = [256] # fully-connect layers dimensions
    conv_configs: list[tuple[int, int, int, int]] = [(32, 3, 1, 1), (64, 3, 1, 1), (128, 3, 1, 1)] # config of convolutionnal layers (#filters, filter size, stride, padding)
    dropout: float = 0.0
    grad_clip: float = 1.0 # clip gradient norm
    warmup_epochs: int = 10 # number of epochs to slowly bring KL weight up from kl_start to kl_end
    kl_start: float = 0.0 # starting value of KL weight
    kl_end: float = 0.001 # KL weight during training (after warmup period)
    compile: int = 0 # Whether or not to compile the BNN
    skip_pretraining: bool = False # Skip initial evaluation of model

    def configure(self):
        # Usage: --conv_configs "(32, 3, 1, 1)" "(64, 3, 1, 1)" "(128, 3, 1, 1)"
        self.add_argument('--conv_configs', type=parse_tuple, nargs='+')

args = MargArgs().parse_args()

# First thing: get device.
# This is important to be first because this sets the default dtype for torch
force_cpu = True if args.device == "cpu" else False
device = get_device(force_cpu)
# Set random seed for reproducibility
torch.manual_seed(args.seed)

# get method id
method: int = -1
if args.method == "ibw":
    method = METHOD_IBW
if args.method == "md":
    method = METHOD_MD
if args.method == "lin":
    method = METHOD_LIN
if args.method == "gd":
    method = METHOD_GD
if method < 0:
    print(f"Unsupported method: {args.method}")
    raise NotImplementedError

# Create directory to save results if it doesn't exist
save_dir = os.path.join(args.save_dir, args.dataset)
os.makedirs(save_dir, exist_ok=True)

# Create a timestamp for this run
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
run_dir = os.path.join(save_dir, f"{args.method}_{timestamp}")
os.makedirs(run_dir, exist_ok=True)

# Save hyperparameters
hyperparams = args.as_dict()
with open(os.path.join(run_dir, "hyperparameters.json"), "w") as f:
    json.dump(hyperparams, f, indent=4)

# Load dataset
train_loader, test_loader = load_dataset(args.dataset, args.bs)
sample_batch, sample_labels = next(iter(train_loader))
input_dim = sample_batch.view(sample_batch.size(0), -1).size(1)  # Flatten and get feature count
output_dim = len(torch.unique(sample_labels))  # Number of unique classes in first batch

# Initialize model and optimizer
n_components = args.n_components
n_samples = n_components

# Define the model and the learning rate scheduler
if args.model == "mlp":
    model = IGMMBayesianMLP(input_dim=input_dim, output_dim=output_dim, n_components=n_components, n_samples=n_samples, hidden_dims=args.fc_dims, dropout_rate=args.dropout)
elif args.model == "cnn":
    model = IGMMBayesianCNN(conv_configs=args.conv_configs, fc_dims=args.fc_dims, output_dim=output_dim, n_components=n_components, n_samples=n_samples, dropout_rate=args.dropout)
# Save the model configuration
model_config = model.get_model_info()
print(f"--> Model info:\n {model_config}")
with open(os.path.join(run_dir, "model_config.json"), "w") as f:
    json.dump(model_config, f, indent=4)

# Learning rate scheduler
lr_scheduler = LearningRateScheduler(args.lr, args.epochs, args.lr_scheduler, min_lr=args.lr_min, lr_decay_epochs=args.lr_decay_epochs, lr_decay_factor=args.lr_decay_factor, restart_period=args.lr_restart)

# Put model to GPU if needed and if possible
model = model.to(device)
if args.compile:
    model = torch.compile(model)

# Initialize metrics storage
metrics = {
    'epochs': [],  # Store epoch numbers for plotting
    'lr': [],  # Current lr for the epoch
    'kl_weight': [],
    'train_accuracy': [],
    'train_kl_div': [],
    'train_nll': [],
    'train_elbo': [],
    'test_accuracy': [],
    'test_kl_div': [],
    'test_nll': [],
    'test_elbo': []
}

# KL warmup
warmup_epochs = args.warmup_epochs
kl_start = args.kl_start
kl_end = args.kl_start

def kl_weight_scheduler(epoch):
    """Gradually increase KL weight during warmup."""
    if epoch < warmup_epochs:
        return kl_start + (kl_end - kl_start) * (epoch / warmup_epochs)
    else:
        return kl_end

# ELBO loss function
def elbo_loss(output: torch.Tensor,
              target: torch.Tensor,
              kl_div: torch.Tensor,
              kl_weight: float,
              n_samples: int):
    # Negative log likelihood
    nll = F.cross_entropy(output, target, reduction='sum')
    # Scale KL divergence by dataset size
    kl_div = kl_div / n_samples
    # Return negative ELBO (we minimize this)
    return nll + kl_weight * kl_div, nll, kl_div

# Calculate accuracy
def calculate_accuracy(output: torch.Tensor, target: torch.Tensor):
    pred = output.argmax(dim=1, keepdim=True)
    correct = pred.eq(target.view_as(pred)).sum().item()
    accuracy = correct / len(target)
    return accuracy, correct

# Function to evaluate model without training
def evaluate_model(model,
                   data_loader,
                   n_samples: int=5):
    """Evaluate model metrics without updating weights"""
    model.eval()
    total_loss = 0
    total_nll = 0
    total_kl_div = 0
    correct = 0
    n_data = len(data_loader.dataset)

    kl_weight = kl_weight_scheduler(0)
    
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)

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
            loss = nll + kl_weight * kl_div_scaled
            
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

# Training function
def train(model, train_loader, epoch, method):
    model.train()
    train_loss = 0
    train_nll_total = 0
    train_kl_div_total = 0
    n_samples = len(train_loader.dataset)
    correct = 0

    kl_weight = kl_weight_scheduler(epoch)
    current_lr = lr_scheduler.step(epoch)

    print("\n===============================================================================")
    print(f'Train Epoch: {epoch}/{args.epochs}\tKL weight: {kl_weight}\tLR: {current_lr}')
    print("===============================================================================")
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        # Zero gradients from previous step
        for param in model.parameters():
            if param.grad is not None:
                param.grad.zero_()

        output = model(data)
        kl_div = model.kl_divergence(mc_samples=5)  # Use MC sampling for KL
        loss, nll, kl_div_scaled = elbo_loss(output, target, kl_div, kl_weight, n_samples)
        
        # Calculate accuracy
        batch_accuracy, batch_correct = calculate_accuracy(output, target)
        correct += batch_correct
        
        # Accumulate metrics
        train_loss += loss.item()
        train_nll_total += nll.item()
        train_kl_div_total += kl_div.item()
        
        loss.backward()
        model.step(learning_rate=current_lr, method=method, grad_clip=args.grad_clip)
        
        if batch_idx % 10 == 0:
            print(f'Train Epoch: {epoch}/{args.epochs} [{batch_idx * len(data)}/{len(train_loader.dataset)}'
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
    
    print(f'====> Epoch: {epoch}/{args.epochs} Average loss: {avg_loss:.4f}, '
          f'NLL: {avg_nll:.4f}, KL: {avg_kl_div:.4f}, '
          f'Accuracy: {accuracy:.4f}')
    
    return avg_loss, avg_nll, avg_kl_div, accuracy

# Evaluation function with uncertainty estimation
# Takes epoch as input because needs to determine the kl_weight for the loss
def test(model, test_loader, epoch, n_samples=10):
    model.eval()
    test_loss = 0
    test_nll_total = 0
    test_kl_div_total = 0
    correct = 0
    n_test = len(test_loader.dataset)
    uncertainties = []

    kl_weight = kl_weight_scheduler(epoch)
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

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
            loss = nll + kl_weight * kl_div_scaled
            
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

# Train the model
epochs = args.epochs

print(f"--> Starting training with hyperparameters:\n {hyperparams}")
print(f"--> Saving results to: {run_dir}")

if not args.skip_pretraining:
    # Evaluate initial model performance (epoch 0) before any training
    print("Evaluating initial model performance (pre-training)...")
    _, train_nll, train_kl_div, train_accuracy = evaluate_model(model, train_loader)
    _, test_nll, test_kl_div, test_accuracy = evaluate_model(model, test_loader)

    # Store initial metrics (epoch 0)
    metrics['epochs'].append(0)
    metrics['lr'].append(lr_scheduler.step(0))
    metrics['kl_weight'].append(kl_weight_scheduler(0))
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
    save_metrics(0, metrics, run_dir)
    save_model_checkpoint(model, 0, hyperparams, metrics, run_dir)

# Start training loop
for epoch in range(1, epochs + 1):
    metrics['epochs'].append(epoch)
    metrics['lr'].append(lr_scheduler.step(epoch))
    metrics['kl_weight'].append(kl_weight_scheduler(epoch))
    train_metrics = train(model, train_loader, epoch, method)
    test_metrics = test(model, test_loader, epoch)
    
    # Save metrics every save_interval epochs and on the last epoch
    if epoch % args.save_interval == 0 or epoch == epochs:
        save_metrics(epoch, metrics, run_dir)
        save_model_checkpoint(model, epoch, hyperparams, metrics, run_dir)

save_and_plot_metrics(args.method, metrics, hyperparams, run_dir)
