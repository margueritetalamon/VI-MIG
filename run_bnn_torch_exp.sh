#!/bin/bash

BASE_DIR="./results_mnist"
mkdir -p $BASE_DIR

# Log file to keep track of all runs
LOG_FILE="${BASE_DIR}/experiment_log.txt"

echo "Starting BNN experiments at $(date)" | tee -a $LOG_FILE
echo "Base directory: ${BASE_DIR}" | tee -a $LOG_FILE
echo "---------------------------------------------" | tee -a $LOG_FILE

# Function to run an experiment with given parameters
run_experiment() {
    local method=$1
    local lr=$2
    local epochs=$3
    local hidden_dim=${4:-256}  # Default value is 256
    local n_components=${5:-5}  # Default value is 5
    
    # Create a descriptive name for this experiment
    exp_name="${method}_lr${lr}_e${epochs}"
    
    echo "Starting experiment: ${exp_name}" | tee -a $LOG_FILE
    echo "  Method: ${method}" | tee -a $LOG_FILE
    echo "  Learning rate: ${lr}" | tee -a $LOG_FILE
    echo "  Epochs: ${epochs}" | tee -a $LOG_FILE
    echo "  Hidden dimension: ${hidden_dim}" | tee -a $LOG_FILE
    echo "  Number of components: ${n_components}" | tee -a $LOG_FILE
    echo "  Started at: $(date)" | tee -a $LOG_FILE
    
    # Run the Python script with the specified parameters
    # Use the base directory as save_dir to organize all runs
    python bnn_torch.py \
        --method ${method} \
        --lr ${lr} \
        --epochs ${epochs} \
        --hidden_dim ${hidden_dim} \
        --n_components ${n_components} \
        --save_dir ${BASE_DIR} \
        --save_interval 1
        --bs 128
        --compile 1
        --warmup_epochs 0
        --kl_start 1
        --kl_end 1
        --device gpu
        --dataset mnist
        --dropout 0
        --model mlp
    
    echo "  Finished at: $(date)" | tee -a $LOG_FILE
    echo "---------------------------------------------" | tee -a $LOG_FILE
}

# Create a combined results directory
mkdir -p "${BASE_DIR}/combined_results"

# Run all combinations as specified
# Methods: ibw, md, lin
# Learning rates: 1e-3 (20 epochs), 1e-4 (50 epochs)

echo "Running all experiments sequentially..." | tee -a $LOG_FILE

# Method: ibw
echo "Running IBW experiments..." | tee -a $LOG_FILE
run_experiment "ibw" "1e-3" 100

# Method: md
echo "Running MD experiments..." | tee -a $LOG_FILE
run_experiment "md" "1e-3" 100

# Method: lin
echo "Running LIN experiments..." | tee -a $LOG_FILE
run_experiment "lin" "1e-3" 100

# Summary
echo "All experiments completed at $(date)" | tee -a $LOG_FILE
echo "Results saved in ${BASE_DIR}" | tee -a $LOG_FILE

# Create summary of best results
echo "Creating summary of results..." | tee -a $LOG_FILE
SUMMARY_FILE="${BASE_DIR}/combined_results/best_results_summary.txt"
echo "Best Results Summary" > $SUMMARY_FILE
echo "=====================" >> $SUMMARY_FILE
echo "" >> $SUMMARY_FILE

# Find all training_summary.txt files and extract the best test accuracy
echo "Method | Learning Rate | Epochs | Best Test Accuracy | Best ELBO | Final Test NLL | Final KL Div" >> $SUMMARY_FILE
echo "-------|---------------|--------|-------------------|-----------|----------------|------------" >> $SUMMARY_FILE

# Get all training_summary.txt files
find $BASE_DIR -name "training_summary.txt" | while read summary_file; do
    # Extract method and lr from the summary file
    method=$(grep "method:" $summary_file | head -1 | awk '{print $2}')
    lr=$(grep "learning_rate:" $summary_file | awk '{print $2}')
    epochs=$(grep "epochs:" $summary_file | awk '{print $2}')
    
    # Extract best test accuracy and ELBO
    best_acc=$(grep "Best Test Accuracy:" $summary_file | awk '{print $4}')
    best_elbo=$(grep "Best ELBO:" $summary_file | awk '{print $3}')
    
    # Extract final test NLL and KL divergence
    test_nll=$(grep "test_nll:" $summary_file | awk '{print $2}')
    test_kl=$(grep "test_kl_div:" $summary_file | awk '{print $2}')
    
    # Append to summary
    echo "$method | $lr | $epochs | $best_acc | $best_elbo | $test_nll | $test_kl" >> $SUMMARY_FILE
done

# Create comparison plots
echo "Creating comparison plots..." | tee -a $LOG_FILE

# Generate a Python script to create comparison plots
cat > "${BASE_DIR}/combined_results/create_comparison_plots.py" << 'EOL'
import os
import numpy as np
import matplotlib.pyplot as plt
import glob
import re
import json

# Find all experiment directories
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
method_runs = {}

# Collect all metrics files
for method in ['ibw', 'md', 'lin']:
    method_runs[method] = {'1e-3': {}, '1e-4': {}}
    
    # Find all directories for this method
    for metrics_file in glob.glob(f"{base_dir}/*/{method}_*/latest_metrics.json"):
        # Extract the learning rate from hyperparameters.json in the same directory
        hyperparam_file = os.path.join(os.path.dirname(metrics_file), "hyperparameters.json")
        
        if os.path.exists(hyperparam_file):
            with open(hyperparam_file, 'r') as f:
                hyperparams = json.load(f)
                
            # Get lr and epochs
            lr = str(hyperparams.get('learning_rate', ''))
            epochs = hyperparams.get('epochs', 0)
            
            # Only process if this is a valid learning rate
            if lr in ['1e-3', '1e-4']:
                # Get all metrics files for this run
                run_dir = os.path.dirname(metrics_file)
                
                # Load test accuracy, nll, kl, and elbo
                test_acc_file = os.path.join(run_dir, "test_accuracy.npy")
                test_nll_file = os.path.join(run_dir, "test_nll.npy")
                test_kl_file = os.path.join(run_dir, "test_kl_div.npy")
                test_elbo_file = os.path.join(run_dir, "test_elbo.npy")
                
                if os.path.exists(test_acc_file) and os.path.exists(test_nll_file) and \
                   os.path.exists(test_kl_file) and os.path.exists(test_elbo_file):
                    
                    method_runs[method][lr]['epochs'] = epochs
                    method_runs[method][lr]['test_accuracy'] = np.load(test_acc_file)
                    method_runs[method][lr]['test_nll'] = np.load(test_nll_file)
                    method_runs[method][lr]['test_kl_div'] = np.load(test_kl_file)
                    method_runs[method][lr]['test_elbo'] = np.load(test_elbo_file)

# Create comparison plots
metrics = ['test_accuracy', 'test_elbo', 'test_nll', 'test_kl_div']
titles = {
    'test_accuracy': 'Test Accuracy',
    'test_elbo': 'Test ELBO',
    'test_nll': 'Test Negative Log Likelihood',
    'test_kl_div': 'Test KL Divergence'
}
ylabels = {
    'test_accuracy': 'Accuracy',
    'test_elbo': 'ELBO',
    'test_nll': 'NLL',
    'test_kl_div': 'KL Divergence'
}

# Plot all metrics by learning rate
for lr in ['1e-3', '1e-4']:
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'Comparison of Methods (lr={lr})', fontsize=16)
    
    for i, metric in enumerate(metrics):
        row, col = i // 2, i % 2
        ax = axes[row, col]
        
        for method in ['ibw', 'md', 'lin']:
            if lr in method_runs[method] and metric in method_runs[method][lr]:
                data = method_runs[method][lr][metric]
                epochs = method_runs[method][lr]['epochs']
                x = np.arange(1, len(data) + 1)
                ax.plot(x, data, label=f'{method}')
        
        ax.set_title(titles[metric])
        ax.set_xlabel('Epoch')
        ax.set_ylabel(ylabels[metric])
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), f'comparison_lr_{lr}.png'))

# Plot comparison of different learning rates for each method
for method in ['ibw', 'md', 'lin']:
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'Comparison of Learning Rates for {method.upper()}', fontsize=16)
    
    for i, metric in enumerate(metrics):
        row, col = i // 2, i % 2
        ax = axes[row, col]
        
        for lr in ['1e-3', '1e-4']:
            if lr in method_runs[method] and metric in method_runs[method][lr]:
                data = method_runs[method][lr][metric]
                epochs = method_runs[method][lr]['epochs']
                
                # Normalize x-axis to percentage of training
                x = np.linspace(0, 100, len(data))
                ax.plot(x, data, label=f'lr={lr} ({epochs} epochs)')
        
        ax.set_title(titles[metric])
        ax.set_xlabel('Training Progress (%)')
        ax.set_ylabel(ylabels[metric])
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), f'comparison_{method}.png'))

print("Comparison plots created successfully!")
EOL

# Run the Python script to create the plots
echo "Generating comparison plots with Python..." | tee -a $LOG_FILE
cd "${BASE_DIR}/combined_results"
python create_comparison_plots.py

echo "Experiment batch completed successfully!" | tee -a $LOG_FILE
echo "Results and comparison plots are available in ${BASE_DIR}/combined_results/" | tee -a $LOG_FILE
