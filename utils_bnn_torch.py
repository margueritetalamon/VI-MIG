import os
import datetime
import numpy as np
import matplotlib.pyplot as plt
import torch
import json

from torchvision import datasets, transforms

def load_mnist():
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
    return train_loader, test_loader


# Save metrics function
def save_metrics(epoch, metrics, run_dir):
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
def save_model_checkpoint(model, epoch, hyperparams, metrics, run_dir):
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

# Final save of all metrics
def save_and_plot_metrics(method, metrics, hyperparams, run_dir):
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
        f.write(f"Training Summary for {method} Method\n")
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
