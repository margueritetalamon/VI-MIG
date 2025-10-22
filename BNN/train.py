import os
import tap
import json
import datetime
import torch
import torch.nn.functional as F
import ast

from utils_bnn_torch import (
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
    )

from laplace import (
    METHOD_LAPLACE_DIAG,
    METHOD_LAPLACE_KFAC,
    LaplaceBayesianMLP
)


def setup_device(device_arg):
    """Setup device with optimal settings"""
    if device_arg == 'cpu':
        device = torch.device('cpu')
        
        # Get number of CPUs from SLURM or system
        num_cpus = int(os.environ.get('SLURM_CPUS_PER_TASK', os.cpu_count()))
        
        # Set threading for optimal CPU performance
        torch.set_num_threads(num_cpus)
        
        # Enable MKL optimizations
        if torch.backends.mkl.is_available():
            torch.backends.mkl.enabled = True
        
        print(f"Using CPU with {num_cpus} threads")
        print(f"MKL enabled: {torch.backends.mkl.is_available()}")
        
        return device, num_cpus
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        num_cpus = 4  # For data loading
        print(f"Using device: {device}")
        return device, num_cpus



def parse_tuple(s: str) -> tuple[int, int, int, int]:
    return ast.literal_eval(s)

class MargArgs(tap.Tap):
    dataset: str = "" # mnist, cifar10, boston
    device: str = "gpu" # whether to use CPU or GPU (if available)
    seed: int = 41
    save_interval: int = 1000  # Save metrics every N epochs
    save_dir: str = "./results"  # Directory to save results
    method: str = "ibw" # method: ibw, md, lin, gd, laplace_diag, laplace_kfac
    bs: int = 128 # batch size
    lr: float = 1e-3 # learning rate
    lr_scheduler: str = "none" # cosine, cosine_restart, step, none
    lr_decay_factor: float = 0.1 # decay factor (step scheduler)
    lr_decay_epochs: list[int] = [100, 150] # epochs at which to apply the decay factor (step scheduler)
    lr_min: float = 1e-6 # minimum value for lr (cosine schedulers)
    lr_restart: int = 50 # for cosine restart
    epochs: int = 10 # number of times we go through the dataset
    n_components: int = 5 # number of gaussians in MOG
    # NOTE: we call ELBO the Negative ELBO
    S: int = 5 # MC samples to estimate the KL divergence (ELBO = NLL(q(D|z)) + KL(q(z) || p(z)))
    mu_scale_init: float = 1.0 # mu weigths are initialized in a uniform distribu between [-a, a], a = mu_scale_init
    prior_mean: float = 0.0 # prior on mu weights
    prior_var: float = 1 # prior on var weights (higher means we care less about prior)
    fc_dims: list[int] = [256] # fully-connect layers dimensions
    dropout: float = 0.0
    grad_clip: float = 1.0 # clip gradient norm
    warmup_epochs: int = 10 # number of epochs to slowly bring KL weight up from kl_start to kl_end
    kl_start: float = 0.0 # starting value of KL weight
    kl_end: float = 0.001 # KL weight during training (after warmup period)
    compile: int = 0 # Whether or not to compile the BNN
    skip_pretraining: bool = True # Skip initial evaluation of model
    optimizer: str = "sgd" # Optimizer for Laplace methods: adam, sgd

args = MargArgs().parse_args()


# First thing: get device.
# This is important to be first because this sets the default dtype for torch
device, num_workers = setup_device(args.device)
non_blocking = (device.type == "cuda")


# Set random seed for reproducibility
torch.manual_seed(args.seed)

# get method id
method: int = -1
if args.method == "ibw":
    method = METHOD_IBW
elif args.method == "md":
    method = METHOD_MD
elif args.method == "lin":
    method = METHOD_LIN
elif args.method == "gd":
    method = METHOD_GD
elif args.method == "laplace_diag":
    method = METHOD_LAPLACE_DIAG
elif args.method == "laplace_kfac":
    method = METHOD_LAPLACE_KFAC
else:
    print(f"Unsupported method: {args.method}")
    raise NotImplementedError

# Create directory to save results if it doesn't exist
save_dir = os.path.join(args.save_dir, args.dataset)
os.makedirs(save_dir, exist_ok=True)

# Create a timestamp for this run
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
run_dir = os.path.join(save_dir, f"{args.method}_n{str(args.n_components)}_{str(args.lr)}_{str(args.epochs)}_{timestamp}")
os.makedirs(run_dir, exist_ok=True)

# Save hyperparameters
hyperparams = args.as_dict()
with open(os.path.join(run_dir, "hyperparameters.json"), "w") as f:
    json.dump(hyperparams, f, indent=4)


train_loader, test_loader = load_dataset(device, args.dataset, args.bs, num_workers)

# Initialize model and optimizer
n_components = args.n_components
n_samples = n_components

# Define the model and the learning rate scheduler
sample_batch, sample_labels = next(iter(train_loader))

# Check if we're using Laplace approximation methods
is_laplace = method in [METHOD_LAPLACE_DIAG, METHOD_LAPLACE_KFAC]


input_dim = sample_batch.view(sample_batch.size(0), -1).size(1)  # Flatten and get feature count
output_dim = len(torch.unique(sample_labels))  # Number of unique classes in first batch
print("input_dim", input_dim)
print("output_dim", output_dim)
print("args.fc_dims", args.fc_dims)
print("args.dropout", args.dropout)
print("args.prior_var",args.prior_var)



if is_laplace:
    model = LaplaceBayesianMLP(input_dim=input_dim, output_dim=output_dim,
                                hidden_dims=args.fc_dims, dropout_rate=args.dropout,
                                prior_precision=1.0/args.prior_var)  # Convert prior variance to precision
else:
    model = IGMMBayesianMLP(input_dim=input_dim, output_dim=output_dim,
                            n_components=n_components, n_samples=n_samples,
                            hidden_dims=args.fc_dims, dropout_rate=args.dropout,
                            mu_scale_init=args.mu_scale_init,
                            prior_mean=args.prior_mean, prior_var=args.prior_var)

N_train = len(train_loader)
model.kl_weight = 1/N_train

# Save the model configuration
model_config = model.get_model_info()
print(f"--> Model info:\n {model_config}")
with open(os.path.join(run_dir, "model_config.json"), "w") as f:
    json.dump(model_config, f, indent=4)

# Learning rate scheduler
lr_scheduler = LearningRateScheduler(args.lr, args.epochs, args.lr_scheduler,
                                     min_lr=args.lr_min, lr_decay_epochs=args.lr_decay_epochs,
                                     lr_decay_factor=args.lr_decay_factor, restart_period=args.lr_restart)

# Put model to GPU if needed and if possible
if args.compile:
    model = torch.compile(model)
model = model.to(device)

# Initialize metrics storage
metrics = {
    'epochs': [],  # Store epoch numbers for plotting
    'lr': [],  # Current lr for the epoch
    'train_accuracy': [],
    'train_nll': [],
    'train_loss': [],
    'test_accuracy': [],
    'test_nll': [],
    'test_loss': []
}



# Calculate accuracy
def calculate_accuracy(preds: torch.Tensor, target: torch.Tensor):
    
    correct = (preds == target).sum()
    accuracy = correct / len(target)
    return accuracy, correct



# Training function
def train(model, train_loader, epoch, method, is_laplace=False):
    model.train()
    train_loss = 0
    train_nll_total = 0
    n_samples = len(train_loader.dataset)
    correct = 0

    current_lr = lr_scheduler.step(epoch)

    print("\n==========================================================================================")
    if is_laplace:
        print(f'Train Epoch: {epoch}/{args.epochs} (MAP Training), LR: {current_lr:.7f}')
    else:
        print(f'Train Epoch: {epoch}/{args.epochs},, LR: {current_lr:.7f}')
    print("==========================================================================================")
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.to(device, non_blocking=non_blocking)
        target = target.to(device, non_blocking=non_blocking)

        # Zero gradients from previous step
        for param in model.parameters():
            if param.grad is not None:
                param.grad.zero_()

        # Forward pass
        if is_laplace:
            output  = model(data, sample=False)  # No sampling during MAP training
        else:
            output  = model(data, S = args.S)

        # loss, neg_entropy, nll, neg_log_prior = loss_function(output, target, model, samples)

        loss, nll, kl = model.loss_function(output, target ) ### maybe this is easier then 
        
        train_loss += loss.item()
        train_nll_total += nll.item()

        loss.backward()
        
        if is_laplace:
            model.step(learning_rate=current_lr, grad_clip=args.grad_clip, optimizer=args.optimizer)
        else:
            model.step(learning_rate=current_lr, method=method, grad_clip=args.grad_clip)

        if is_laplace:
            probs = torch.softmax(output, dim=-1)  # (B,C) ## no need to average 
        else: 
            probs = torch.softmax(output, dim=-1).mean(dim=0)  # (B,C)
        
        preds = probs.argmax(dim=-1)    
       

        batch_accuracy, batch_correct = calculate_accuracy(preds, target)
        correct += batch_correct
        
        

        
        # Use appropriate step method

        
      
    

    # Calculate average metrics
    avg_loss = train_loss / n_samples
    avg_nll = train_nll_total / n_samples
    accuracy = correct / n_samples
    
    # Store metrics
    metrics['train_accuracy'].append(accuracy)
    metrics['train_nll'].append(avg_nll)
    metrics['train_loss'].append(avg_loss)  
    
    if is_laplace:
        print(f'====> Epoch: {epoch}/{args.epochs} Average loss: {avg_loss:.4f}, '
              f'NLL: {avg_nll:.4f}, Accuracy: {accuracy:.4f}')
    else:
        print(f'====> Epoch: {epoch}/{args.epochs} Average loss: {avg_loss:.4f}, '
              f'Loss: {(avg_loss):.4f}, NLL: {avg_nll:.4f}, '
              f'Accuracy: {accuracy:.4f}')
    
    return avg_loss, avg_nll, accuracy

# Evaluation function with uncertainty estimation
# Takes epoch as input because needs to determine the kl_weight for the loss
def test(model, test_loader, epoch, S=10, is_laplace=False, method=None):
    model.eval()
    test_loss = 0
    test_nll_total = 0
    correct = 0
    samples = None
    n_test = len(test_loader.dataset)
    
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device, non_blocking=non_blocking)
            target = target.to(device, non_blocking=non_blocking)

            # Get multiple predictions
            
            if is_laplace:
                # For Laplace models, use sampling if Hessian is fitted
                if hasattr(model, 'laplace_fitted') and model.laplace_fitted:
                    outputs  = model(data, sample=True, method=method, S = S)
                else:
                    outputs  = model(data, sample=False)

            else:
                outputs  = model(data, sample=True, S = S )
                


            # Stack predictions
            loss, nll, kl = model.loss_function(outputs, target)

            if outputs.dim() == 3:
                probs = torch.softmax(outputs, dim=-1).mean(dim=0)  # (B,C)
            else:
                probs = torch.softmax(outputs, dim=-1)  # (B,C)

            preds = probs.argmax(dim=-1)    
       
            batch_accuracy, batch_correct = calculate_accuracy(preds, target)
            correct += batch_correct
        
            
            # Accumulate metrics
            test_loss += loss.item()
            test_nll_total += nll.item()
    
    # Calculate average metrics
    avg_loss = test_loss / n_test
    avg_nll = test_nll_total / n_test
    accuracy = correct / n_test
    
    # Store metrics
    metrics['test_accuracy'].append(accuracy)
    metrics['test_nll'].append(avg_nll)
    metrics['test_loss'].append(avg_loss)  # Negative loss is ELBO
    
    if is_laplace:
        laplace_status = "with Laplace sampling" if (hasattr(model, 'laplace_fitted') and model.laplace_fitted) else "MAP estimate only"
        print(f'Test set ({laplace_status}): Average loss: {avg_loss:.4f}, '
              f'NLL: {avg_nll:.4f}, Accuracy: {accuracy:.4f} ({correct}/{n_test})')
    else:
        print(f'Test set: Average loss: {avg_loss:.4f}, '
              f'Loss: {(avg_loss):.4f}, NLL: {avg_nll:.4f}, '
              f'Accuracy: {accuracy:.4f} ({correct}/{n_test})')
    
    return avg_loss, avg_nll, accuracy

# Train the model
epochs = args.epochs

print(f"--> Starting training with hyperparameters:\n {hyperparams}")
print(f"--> Saving results to: {run_dir}")

if not args.skip_pretraining:
    # Evaluate initial model performance (epoch 0) before any training
    print("Evaluating initial model performance (pre-training)...")
    train_loss, train_nll, train_accuracy = test(model, train_loader, is_laplace=is_laplace)
    test_loss, test_nll, test_accuracy = test(model, test_loader, is_laplace=is_laplace)

    # Store initial metrics (epoch 0)
    metrics['epochs'].append(0)
    metrics['lr'].append(lr_scheduler.step(0))
    metrics['train_accuracy'].append(train_accuracy)
    metrics['train_nll'].append(train_nll)
    metrics['train_loss'].append(train_loss)
    metrics['test_accuracy'].append(test_accuracy)
    metrics['test_nll'].append(test_nll)
    metrics['test_loss'].append(test_loss)

    print(f"Initial metrics before training:")
    print(f"  Train accuracy: {train_accuracy:.4f}, Loss: {train_loss:.4f}")
    print(f"  Test accuracy: {test_accuracy:.4f}, Loss: {test_loss:.4f}")

    # Save initial metrics
    save_metrics(0, metrics, run_dir)
    save_model_checkpoint(model, 0, hyperparams, metrics, run_dir)

# Start training loop
for epoch in range(1, epochs + 1):
    metrics['epochs'].append(epoch)
    metrics['lr'].append(lr_scheduler.step(epoch))
    
    loss, nll, accuracy = train(model, train_loader, epoch, method, is_laplace=is_laplace)
    print(f"Train accuracy: {accuracy:.4f}, Loss: {loss:.4f}")

    # For Laplace methods, fit the Laplace approximation after each training epoch
    if is_laplace and epoch == epochs:
        print(f"\nFitting Laplace approximation after epoch {epoch}...")
        model.fit_laplace(train_loader, device, method)
        print("\n=== Hessian Diagnostics ===")
        for i, layer in enumerate(model.layers):
            if method == METHOD_LAPLACE_DIAG:
                print(f"Layer {i}:")
                print(f"  Weight precision: min={layer.weight_precision_diag.min():.2e}, max={layer.weight_precision_diag.max():.2e}, mean={layer.weight_precision_diag.mean():.2e}")
                print(f"  Bias precision: min={layer.bias_precision_diag.min():.2e}, max={layer.bias_precision_diag.max():.2e}")
        
        # Check variance (inverse of precision)
        weight_var = 1.0 / (layer.weight_precision_diag + 1e-10)
        print(f"  Weight std: min={torch.sqrt(weight_var).min():.2e}, max={torch.sqrt(weight_var).max():.2e}, mean={torch.sqrt(weight_var).mean():.2e}")
        states = model.get_laplace_state()
        torch.save(states, os.path.join(run_dir, "model_latest.pt"))


    test_loss, test_nll, test_accuracy  = test(model = model, test_loader = test_loader,epoch =  epoch, is_laplace=is_laplace, method=method, S = args.S*2)

    print(f"Test metrics:")
    print(f"Test accuracy: {test_accuracy:.4f}, Loss: {test_loss:.4f}")
    
    # Save metrics every save_interval epochs and on the last epoch
    if epoch % args.save_interval == 0 or epoch == epochs:
        save_metrics(epoch, metrics, run_dir)
        save_model_checkpoint(model, epoch, hyperparams, metrics, run_dir)


# save_and_plot_metrics(args.method, metrics, hyperparams, run_dir)

metrics = {
    'test_accuracy': [],
    'test_nll': [],
    'test_loss': []
}
test(model, train_loader, -1, is_laplace=is_laplace, method=method)
test(model, test_loader, -1, is_laplace=is_laplace, method=method)
save_metrics(-1, metrics, run_dir)